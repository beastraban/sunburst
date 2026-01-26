#!/usr/bin/env python3
"""
GreenDragonRisesFromWater v1.7 - MODULE 2: Peak Refinement (PURE GPU)
=====================================================================

"Green Dragon Rises from Water" (青龍出水) - The dragon emerges refined
and purified. Module 2 takes rough peak candidates from CarryTiger and
refines them to machine precision.

v1.1: Pure GPU implementation - no CPU transfers mid-computation
v1.2: Added trajectory banking (return_bank=True) for Module 3
v1.3: Two modes - default (2×D axis-aligned samples, richer bank) 
      and fast (20 random samples, faster)
      Both compute diag_H + low_rank correction
v1.5: Saddle point filtering - removes peaks where diag_H > 0
      (indicates saddle point, not maximum)
v1.6: BUGFIX - FD Hessian now uses actual delta after clipping
      Before: delta = 0.9 * width (intended perturbation)
      After:  delta = actual displacement after clipping to bounds
      This fixes huge errors when bounds are tight (e.g., cigar Gaussian)
v1.7: NEW - Trajectory storage for rotation detection
      TrajectoryBank now stores intermediate L-BFGS positions
      Used by Module 3 for perpendicular step fraction analysis
      to detect rotated Gaussians requiring full Hessian computation

Pipeline Position:
    Module 0 (StrikePalms) → Module 1 (CarryTiger) → MODULE 2 (GreenDragon)
    → Module 3 (evidence integration)

Input (from CarryTiger, GPU arrays):
    - peaks: [K, D] peak locations
    - widths: [K, 2] (longest, shortest) OR [K,] single width per peak
    - Likelihood function (log-space, GPU-pure from StrikePalms)

Output (GPU arrays):
    - Refined peaks [K', D]
    - Refined L_peaks [K']
    - diag_H [K', D] diagonal Hessian
    - low_rank_U, low_rank_V [K', D, r] low-rank Hessian correction
    - TrajectoryBank (if return_bank=True)

Modes:
    fast=False (default): 2×D axis-aligned samples at ±0.9w per dimension
        - Richer bank (2×D samples per peak)
        - diag_H is FREE from seed log_L values
        - Better for evidence calculation
        
    fast=True: 20 random samples per peak
        - Faster L-BFGS (fewer samples)
        - diag_H via finite differences (2×D extra evals)
        - Better when speed is critical
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional, Any
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# Import from ChiSao
try:
    # Package-relative import
    from ..utils.chisao import (
        lbfgs_batch,
        deduplicate_peaks_L_infinity,
    )
except ImportError:
    try:
        # Fallback for standalone usage
        from chisao3_3 import (
            lbfgs_batch,
            deduplicate_peaks_L_infinity,
        )
    except ImportError:
        raise ImportError(
            "ChiSao required for GreenDragon. "
            "Please install the sunburst package properly."
        )


# =============================================================================
# TRAJECTORY BANK - SAMPLE STORAGE FOR EVIDENCE CALCULATION
# =============================================================================

@dataclass
class HessianData:
    """Hessian approximation: H ≈ diag(diag_H) + U @ V.T"""
    diag_H: 'cp.ndarray'       # (K, D) diagonal
    low_rank_U: 'cp.ndarray'   # (K, D, r) 
    low_rank_V: 'cp.ndarray'   # (K, D, r)
    rank: int


class TrajectoryBank:
    """
    GPU-resident storage for refinement samples and trajectories.
    Used for evidence calculation in Module 3.
    
    v1.7: Added trajectories storage for rotation detection.
          Each trajectory is a list of positions from L-BFGS iterations.
          Used for perpendicular step fraction analysis.
    """
    
    def __init__(self, D: int, xp=None):
        self.xp = xp if xp is not None else (cp if GPU_AVAILABLE else np)
        self.D = D
        
        # Seeds
        self.seed_positions = None      # (N_seeds, D)
        self.seed_log_L = None          # (N_seeds,)
        self.seed_peak_ids = None       # (N_seeds,)
        self.seed_dims = None           # (N_seeds,) - which dimension (for axis-aligned)
        self.seed_signs = None          # (N_seeds,) - +1 or -1 (for axis-aligned)
        
        # Post L-BFGS
        self.final_positions = None     # (N_seeds, D)
        self.final_log_L = None         # (N_seeds,)
        self.converged = None           # (N_seeds,) bool
        
        # NEW v1.7: Full trajectories for rotation detection
        self.trajectories = None        # List of (n_steps, D) arrays per seed
        
        # Refined peaks
        self.refined_peaks = None       # (K, D)
        self.refined_log_L = None       # (K,)
        
        # Metadata
        self.n_seeds = 0
        self.n_seeds_per_peak = 0
        self.n_peaks = 0
        self.mode = None  # 'axis_aligned' or 'random'
    
    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        if self.seed_positions is None:
            return 0.0
        bytes_used = (
            self.n_seeds * self.D * 8 * 2 +  # seed + final positions
            self.n_seeds * 8 * 2 +            # log_L values
            self.n_seeds * 4 * 3 +            # peak_ids, dims, signs
            self.n_peaks * self.D * 8 +       # refined peaks
            self.n_peaks * 8                  # refined log_L
        )
        return bytes_used / (1024 * 1024)


# =============================================================================
# GREENDRAGON CLASS
# =============================================================================

class GreenDragonRisesFromWater:
    """
    Module 2: Peak refinement with Hessian estimation.
    PURE GPU implementation.
    
    "Green Dragon Rises from Water" (青龍出水) - Refines rough peak
    candidates to machine precision and computes Hessian approximation.
    
    Two modes:
        fast=False (default): 2×D axis-aligned samples, richer bank
        fast=True: 20 random samples, faster
    """
    
    def __init__(
        self,
        func: Callable,
        bounds: 'cp.ndarray',
        use_gpu: bool = True,
        fast: bool = False,
        n_seeds_per_peak: int = 20,
        low_rank_r: int = 5,
        maxiter: int = 100,
        convergence_tol: float = 1e-6,
        dedup_tolerance: float = 0.01,
        verbose: bool = True
    ):
        """
        Initialize Green Dragon.
        
        Args:
            func: Log-likelihood function (GPU-pure)
            bounds: [D, 2] parameter bounds (GPU array)
            use_gpu: Use GPU acceleration
            fast: Use fast mode (20 random samples instead of 2×D axis-aligned)
            n_seeds_per_peak: Seeds per peak in fast mode (ignored in default mode)
            low_rank_r: Rank for low-rank Hessian correction
            maxiter: Maximum L-BFGS iterations
            convergence_tol: Gradient tolerance for convergence
            dedup_tolerance: L∞ tolerance for deduplication
            verbose: Print progress
        """
        self.func = func
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Ensure bounds are on correct device
        if self.use_gpu and not isinstance(bounds, cp.ndarray):
            self.bounds = cp.asarray(bounds)
        else:
            self.bounds = bounds
            
        self.D = len(self.bounds)
        
        self.fast = fast
        self.n_seeds_per_peak = n_seeds_per_peak
        self.low_rank_r = low_rank_r
        self.maxiter = maxiter
        self.convergence_tol = convergence_tol
        self.dedup_tolerance = dedup_tolerance
        self.verbose = verbose
    
    def refine(
        self,
        peaks: 'cp.ndarray',
        widths: 'cp.ndarray',
        L_peaks: Optional['cp.ndarray'] = None,
        return_bank: bool = False
    ) -> Dict[str, Any]:
        """
        Refine peak candidates and compute Hessian approximation.
        
        Args:
            peaks: [K, D] peak locations from CarryTiger (GPU array)
            widths: [K, 2] or [K,] width estimates (GPU array)
            L_peaks: [K] optional likelihood values (GPU array)
            return_bank: If True, include TrajectoryBank in results
            
        Returns:
            Dictionary with:
                - peaks: [K', D] refined peaks
                - L_peaks: [K'] likelihood values
                - diag_H: [K', D] diagonal Hessian
                - low_rank_U: [K', D, r] low-rank factor
                - low_rank_V: [K', D, r] low-rank factor
                - n_peaks: int
                - timing: dict
                - trajectory_bank: TrajectoryBank (if return_bank=True)
        """
        import time
        xp = self.xp
        
        # Ensure inputs are on correct device
        if self.use_gpu:
            if not isinstance(peaks, cp.ndarray):
                peaks = cp.asarray(peaks)
            if not isinstance(widths, cp.ndarray):
                widths = cp.asarray(widths)
        
        K, D = peaks.shape
        
        # Handle width input - convert to per-dimension
        if widths.ndim == 1:
            widths_per_dim = xp.tile(widths[:, None], (1, D))  # (K, D)
        elif widths.shape[1] == 2:
            # Use longest width for all dimensions
            widths_per_dim = xp.tile(widths[:, 0:1], (1, D))  # (K, D)
        else:
            widths_per_dim = widths  # Already (K, D)
        
        if self.verbose:
            mode_str = "FAST (random)" if self.fast else "DEFAULT (axis-aligned)"
            print(f"\n{'='*70}")
            print(f"MODULE 2: GreenDragonRisesFromWater v1.7 (青龍出水)")
            print(f"Mode: {mode_str}")
            print(f"{'='*70}")
            print(f"  Input peaks: {K}")
            print(f"  Dimensions: {D}")
            if self.fast:
                print(f"  Seeds per peak: {self.n_seeds_per_peak}")
            else:
                print(f"  Seeds per peak: 2×D = {2*D}")
        
        timing = {}
        
        if K == 0:
            return self._empty_result(D, return_bank)
        
        # Dispatch to appropriate method
        if self.fast:
            result = self._refine_fast(peaks, widths_per_dim, return_bank, timing)
        else:
            result = self._refine_default(peaks, widths_per_dim, return_bank, timing)
        
        # Deduplication (scale-invariant: normalize to [0,1]^D first)
        t0 = time.perf_counter()
        if result['n_peaks'] > 1:
            # Normalize peaks to unit cube for scale-invariant comparison
            bounds_width = self.bounds[:, 1] - self.bounds[:, 0]
            bounds_lower = self.bounds[:, 0]
            peaks_normalized = (result['peaks'] - bounds_lower) / bounds_width
            
            dedup_peaks_norm, dedup_L = deduplicate_peaks_L_infinity(
                peaks=peaks_normalized,
                L_peaks=result['L_peaks'],
                tolerance=self.dedup_tolerance,
                keep_best=True,
                verbose=False
            )
            
            # Convert back to physical coordinates
            dedup_peaks = dedup_peaks_norm * bounds_width + bounds_lower
            
            # Apply same mask to Hessian data
            # Note: deduplicate returns sorted by L, need to match indices
            n_before = result['n_peaks']
            n_after = len(dedup_peaks)
            
            if n_after < n_before:
                result['peaks'] = dedup_peaks
                result['L_peaks'] = dedup_L
                # Recompute Hessian for deduped peaks
                # For now, keep first n_after entries (approximately correct)
                result['diag_H'] = result['diag_H'][:n_after]
                result['low_rank_U'] = result['low_rank_U'][:n_after]
                result['low_rank_V'] = result['low_rank_V'][:n_after]
                result['n_peaks'] = n_after
        
        timing['dedup'] = time.perf_counter() - t0
        timing['total'] = sum(timing.values())
        result['timing'] = timing
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Module 2 Complete]")
            print(f"  Output: {result['n_peaks']} refined peaks")
            print(f"  Total time: {timing['total']:.3f}s")
            if return_bank and 'trajectory_bank' in result:
                print(f"  TrajectoryBank: {result['trajectory_bank'].memory_mb():.2f} MB")
        
        return result
    
    def _empty_result(self, D, return_bank):
        """Return empty result for K=0 case."""
        xp = self.xp
        result = {
            'peaks': xp.array([]).reshape(0, D),
            'L_peaks': xp.array([]),
            'diag_H': xp.array([]).reshape(0, D),
            'low_rank_U': xp.array([]).reshape(0, D, self.low_rank_r),
            'low_rank_V': xp.array([]).reshape(0, D, self.low_rank_r),
            'n_peaks': 0,
            'timing': {}
        }
        if return_bank:
            result['trajectory_bank'] = TrajectoryBank(D, xp)
        return result
    
    # =========================================================================
    # DEFAULT MODE: 2×D AXIS-ALIGNED SAMPLES
    # =========================================================================
    
    def _refine_default(self, peaks, widths_per_dim, return_bank, timing):
        """
        Default mode: 2×D axis-aligned samples at ±0.9w per dimension.
        diag_H is FREE from seed log_L values.
        """
        import time
        xp = self.xp
        K, D = peaks.shape
        r = min(self.low_rank_r, D)
        
        # -----------------------------------------------------------------
        # Stage 1: Seed 2×D samples per peak at ±0.9w (VECTORIZED)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        # Unit vectors for each dimension
        unit_vectors = xp.eye(D)  # (D, D)
        
        # Perturbations: ±0.9w along each axis
        perturbations = 0.9 * widths_per_dim[:, :, None] * unit_vectors[None, :, :]  # (K, D, D)
        
        # Expand peaks for broadcasting
        peaks_expanded = peaks[:, None, :]  # (K, 1, D)
        
        # Plus and minus seeds
        seeds_plus = peaks_expanded + perturbations   # (K, D, D)
        seeds_minus = peaks_expanded - perturbations  # (K, D, D)
        
        # Interleave: [+d0, -d0, +d1, -d1, ...]
        all_seeds = xp.stack([seeds_plus, seeds_minus], axis=2)  # (K, D, 2, D)
        all_seeds = all_seeds.reshape(K, 2*D, D)
        all_seeds = all_seeds.reshape(K * 2 * D, D)
        
        # Clip to bounds
        all_seeds = xp.clip(all_seeds, self.bounds[:, 0], self.bounds[:, 1])
        
        # v1.6 FIX: Compute actual delta accounting for clipping
        # Fast path: check if any clipping would occur
        intended_delta = 0.9 * widths_per_dim  # (K, D)
        would_clip = xp.any(peaks + intended_delta > self.bounds[:, 1]) or \
                     xp.any(peaks - intended_delta < self.bounds[:, 0])
        
        if would_clip:
            # Slow path: compute actual delta from clipped positions
            # All O(K*D) operations, no fancy indexing
            clipped_plus = xp.minimum(peaks + intended_delta, self.bounds[:, 1])
            clipped_minus = xp.maximum(peaks - intended_delta, self.bounds[:, 0])
            actual_delta = xp.minimum(clipped_plus - peaks, peaks - clipped_minus)
            actual_delta = xp.maximum(actual_delta, 1e-10)
        else:
            # Fast path: no clipping, use intended delta
            actual_delta = intended_delta
        
        N_samples = K * 2 * D
        
        # Metadata arrays
        seed_peak_ids = xp.repeat(xp.arange(K), 2 * D)
        seed_dims = xp.tile(xp.repeat(xp.arange(D), 2), K)
        seed_signs = xp.tile(xp.array([1, -1] * D), K)
        
        # Evaluate log_L at seeds (FREE Hessian data!)
        L_seeds = self.func(all_seeds)
        
        timing['seeding'] = time.perf_counter() - t0
        
        if self.verbose:
            print(f"\n[Stage 1: Axis-Aligned Seeding]")
            print(f"  Generated {N_samples} samples ({K} peaks × {2*D} seeds)")
        
        # -----------------------------------------------------------------
        # Stage 2: L-BFGS (SINGLE CALL - all samples in lockstep)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        # Try with return_trajectories (v3.3+), fall back if not supported
        try:
            bfgs_result = lbfgs_batch(
                func=self.func,
                x0=all_seeds,
                maxiter=self.maxiter,
                tol=self.convergence_tol,
                bounds=self.bounds,
                verbose=False,
                return_trajectories=True  # NEW v1.7: For rotation detection
            )
            trajectories = bfgs_result.get('trajectories')
        except TypeError:
            # Fallback for older chisao versions without return_trajectories
            bfgs_result = lbfgs_batch(
                func=self.func,
                x0=all_seeds,
                maxiter=self.maxiter,
                tol=self.convergence_tol,
                bounds=self.bounds,
                verbose=False
            )
            trajectories = None
        
        final_positions = bfgs_result['x']
        converged = bfgs_result['converged']
        grad_norm_all = bfgs_result['grad_norm']  # (K * 2*D,) - FREE gradient info!
        
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        timing['lbfgs'] = time.perf_counter() - t0
        
        if self.verbose:
            n_conv = int(xp.sum(converged))
            print(f"\n[Stage 2: L-BFGS]")
            print(f"  Converged: {n_conv}/{N_samples}")
            print(f"  Time: {timing['lbfgs']:.3f}s")
        
        # -----------------------------------------------------------------
        # Stage 3: Extract refined peaks (VECTORIZED)
        # -----------------------------------------------------------------
        # Key insight: Seeds from peak k may converge to peak j (different peak!)
        # We must cluster by WHICH ORIGINAL PEAK each seed is closest to,
        # not just pick the highest L from each peak's seeds.
        t0 = time.perf_counter()
        
        L_final = self.func(final_positions)
        
        # Reshape to (K, 2*D, D)
        final_positions_reshaped = final_positions.reshape(K, 2*D, D)
        L_final_reshaped = L_final.reshape(K, 2*D)
        grad_norm_reshaped = grad_norm_all.reshape(K, 2*D)
        
        # For each converged position, find which ORIGINAL peak it's closest to
        # This handles the case where a seed from peak 0 converges to peak 1
        # Shape: final_positions_reshaped (K, 2*D, D), peaks (K, D)
        # We want distance from each of K*2*D final positions to each of K original peaks
        
        final_flat = final_positions_reshaped.reshape(K * 2 * D, D)  # (K*2*D, D)
        
        # Distance from each final position to each original peak
        # dist[i, j] = ||final_i - peak_j||
        dist_to_peaks = xp.linalg.norm(
            final_flat[:, None, :] - peaks[None, :, :],  # (K*2*D, K, D)
            axis=2
        )  # (K*2*D, K)
        
        # Which original peak is each final position closest to?
        closest_peak = xp.argmin(dist_to_peaks, axis=1)  # (K*2*D,)
        
        # Now for each original peak k, collect all final positions that are closest to k
        # and pick the one with highest L
        refined_peaks_list = []
        L_refined_list = []
        grad_norm_list = []
        peak_valid = []
        
        for k in range(K):
            # Mask: which final positions belong to peak k?
            mask_k = closest_peak == k  # (K*2*D,)
            
            if xp.sum(mask_k) == 0:
                # No seeds converged to this peak - it was absorbed by another
                peak_valid.append(False)
                continue
            
            # Get positions and likelihoods for this peak's cluster
            positions_k = final_flat[mask_k]  # (n_k, D)
            L_k = L_final.reshape(-1)[mask_k]  # (n_k,)
            grad_k = grad_norm_all.reshape(-1)[mask_k]  # (n_k,)
            
            # Pick the best one
            best_idx = xp.argmax(L_k)
            refined_peaks_list.append(positions_k[best_idx])
            L_refined_list.append(L_k[best_idx])
            grad_norm_list.append(grad_k[best_idx])
            peak_valid.append(True)
        
        # Stack results
        if len(refined_peaks_list) == 0:
            return self._empty_result(D, return_bank)
        
        refined_peaks = xp.stack(refined_peaks_list)
        L_refined = xp.array(L_refined_list)
        grad_norm_refined = xp.array(grad_norm_list)
        K = len(refined_peaks)  # May be less than original K if peaks merged
        
        # Update widths to match remaining peaks
        peak_valid_arr = xp.array(peak_valid)
        widths_per_dim = widths_per_dim[peak_valid_arr]
        actual_delta = actual_delta[peak_valid_arr]  # v1.6 FIX: filter actual_delta too
        
        # Also filter L_seeds to match remaining peaks (BUG FIX v1.5.1)
        # L_seeds has shape (K_original * 2 * D,) - need to keep only valid peaks' seeds
        K_original = len(peak_valid)
        L_seeds_reshaped_orig = L_seeds.reshape(K_original, D, 2)
        L_seeds_reshaped_orig = L_seeds_reshaped_orig[peak_valid_arr]  # (K_new, D, 2)
        
        timing['extraction'] = time.perf_counter() - t0
        
        # -----------------------------------------------------------------
        # Stage 4: Diagonal Hessian (FREE from seed log_L)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        # L_seeds already reshaped and filtered above
        L_seeds_reshaped = L_seeds_reshaped_orig
        f_plus = L_seeds_reshaped[:, :, 0]   # f(peak + 0.9w*e_d)
        f_minus = L_seeds_reshaped[:, :, 1]  # f(peak - 0.9w*e_d)
        
        # v1.6 FIX: Use actual_delta (accounts for clipping) instead of intended delta
        delta = actual_delta  # (K, D)
        
        diag_H = (f_plus - 2*L_refined[:, None] + f_minus) / (delta**2)
        
        timing['diag_hessian'] = time.perf_counter() - t0
        
        if self.verbose:
            print(f"\n[Stage 4: Diagonal Hessian]")
            print(f"  Cost: FREE (from seed evaluations)")
        
        # -----------------------------------------------------------------
        # Stage 4b: Saddle Point Filtering
        # For a maximum, all diag_H should be negative (concave down).
        # If any diag_H > 0, that peak is a saddle point.
        # -----------------------------------------------------------------
        saddle_threshold = 1e-6  # Small positive tolerance for numerical noise
        is_maximum = xp.all(diag_H < saddle_threshold, axis=1)  # (K,)
        n_saddles = int(xp.sum(~is_maximum))
        
        if n_saddles > 0:
            if self.verbose:
                print(f"\n[Stage 4b: Saddle Point Filtering]")
                print(f"  Removed {n_saddles} saddle points, kept {int(xp.sum(is_maximum))} maxima")
            
            # Filter out saddle points
            refined_peaks = refined_peaks[is_maximum]
            L_refined = L_refined[is_maximum]
            diag_H = diag_H[is_maximum]
            widths_per_dim = widths_per_dim[is_maximum]
            grad_norm_refined = grad_norm_refined[is_maximum]  # Also filter gradient norms
            K = int(xp.sum(is_maximum))
            
            # Handle case where all peaks were saddles
            if K == 0:
                if self.verbose:
                    print(f"  WARNING: All peaks were saddle points!")
                return self._empty_result(D, return_bank)
        
        # -----------------------------------------------------------------
        # Stage 4c: Gradient Filter (true maxima have zero gradient)
        # Uses FREE gradient norm from L-BFGS - no extra function evaluations!
        # -----------------------------------------------------------------
        grad_threshold = 1e-4  # Tolerance for "zero" gradient
        is_stationary = grad_norm_refined < grad_threshold
        n_not_stationary = int(xp.sum(~is_stationary))
        
        if n_not_stationary > 0:
            if self.verbose:
                print(f"\n[Stage 4c: Gradient Filter (FREE from L-BFGS)]")
                print(f"  Removed {n_not_stationary} non-stationary points, kept {int(xp.sum(is_stationary))}")
            
            # Filter out non-stationary points
            refined_peaks = refined_peaks[is_stationary]
            L_refined = L_refined[is_stationary]
            diag_H = diag_H[is_stationary]
            widths_per_dim = widths_per_dim[is_stationary]
            K = int(xp.sum(is_stationary))
            
            # Handle case where all peaks were non-stationary
            if K == 0:
                if self.verbose:
                    print(f"  WARNING: All peaks were non-stationary!")
                return self._empty_result(D, return_bank)
        
        # -----------------------------------------------------------------
        # Stage 5: Low-rank correction (from displacement structure)
        # NOTE: If peaks were filtered out, we skip low-rank since the
        # trajectory data no longer aligns with the filtered peaks.
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        low_rank_U = xp.zeros((K, D, r))
        low_rank_V = xp.zeros((K, D, r))
        
        # Only compute low-rank if no filtering happened
        # (original K equals current K)
        K_original = len(final_positions) // (2 * D)
        
        if K == K_original and K > 0:
            # Displacements: (K, 2D, D) -> (K, D, 2, D)
            displacements = final_positions.reshape(K, 2*D, D) - all_seeds.reshape(K, 2*D, D)
            displacements = displacements.reshape(K, D, 2, D)
            
            # Average +/- to get net displacement per starting axis: (K, D, D)
            mean_disp = displacements.mean(axis=2)
            
            # Normalize by width
            mean_disp_normalized = mean_disp / (0.9 * widths_per_dim[:, :, None] + 1e-10)
            
            # Extract top-r off-diagonal couplings
            for k in range(K):
                offdiag = mean_disp_normalized[k].copy()
                xp.fill_diagonal(offdiag, 0)
                
                flat = xp.abs(offdiag).ravel()
                top_indices = xp.argsort(flat)[-r:][::-1]
                
                for rank_idx, flat_idx in enumerate(top_indices):
                    i = int(flat_idx) // D
                    j = int(flat_idx) % D
                    if i != j:
                        coupling = float(offdiag[i, j])
                        low_rank_U[k, i, rank_idx] = coupling * diag_H[k, i] / (diag_H[k, j] + 1e-10)
                        low_rank_V[k, j, rank_idx] = 1.0
        # else: low-rank stays as zeros (filtered peaks lose this info)
        
        timing['low_rank'] = time.perf_counter() - t0
        
        # -----------------------------------------------------------------
        # Build result
        # -----------------------------------------------------------------
        result = {
            'peaks': refined_peaks,
            'L_peaks': L_refined,
            'diag_H': diag_H,
            'low_rank_U': low_rank_U,
            'low_rank_V': low_rank_V,
            'n_peaks': K
        }
        
        if return_bank:
            bank = TrajectoryBank(D, xp)
            bank.seed_positions = all_seeds
            bank.seed_log_L = L_seeds
            bank.seed_peak_ids = seed_peak_ids
            bank.seed_dims = seed_dims
            bank.seed_signs = seed_signs
            bank.final_positions = final_positions
            bank.final_log_L = L_final
            bank.converged = converged
            bank.trajectories = trajectories  # NEW v1.7: For rotation detection
            bank.refined_peaks = refined_peaks
            bank.refined_log_L = L_refined
            bank.n_seeds = N_samples
            bank.n_seeds_per_peak = 2 * D
            bank.n_peaks = K
            bank.mode = 'axis_aligned'
            result['trajectory_bank'] = bank
        
        return result
    
    # =========================================================================
    # FAST MODE: 20 RANDOM SAMPLES
    # =========================================================================
    
    def _refine_fast(self, peaks, widths_per_dim, return_bank, timing):
        """
        Fast mode: n_seeds_per_peak random samples.
        diag_H via finite differences (2×D extra evals).
        """
        import time
        xp = self.xp
        K, D = peaks.shape
        n_seeds = self.n_seeds_per_peak
        r = min(self.low_rank_r, D)
        eps = 1e-5
        
        # Use scalar width (first dimension or mean)
        width_scalar = widths_per_dim[:, 0]  # (K,)
        
        # -----------------------------------------------------------------
        # Stage 1: Radial seeding (VECTORIZED)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        # Random directions: (K, n_seeds, D)
        directions = xp.random.randn(K, n_seeds, D)
        norms = xp.linalg.norm(directions, axis=2, keepdims=True)
        directions = directions / norms
        
        # Seeds at 0.9w distance
        peaks_expanded = peaks[:, None, :]           # (K, 1, D)
        widths_expanded = width_scalar[:, None, None]  # (K, 1, 1)
        
        all_seeds = peaks_expanded + 0.9 * widths_expanded * directions  # (K, n_seeds, D)
        all_seeds = all_seeds.reshape(K * n_seeds, D)
        all_seeds = xp.clip(all_seeds, self.bounds[:, 0], self.bounds[:, 1])
        
        N_samples = K * n_seeds
        seed_peak_ids = xp.repeat(xp.arange(K), n_seeds)
        
        L_seeds = self.func(all_seeds)
        
        timing['seeding'] = time.perf_counter() - t0
        
        if self.verbose:
            print(f"\n[Stage 1: Random Seeding]")
            print(f"  Generated {N_samples} samples ({K} peaks × {n_seeds} seeds)")
        
        # -----------------------------------------------------------------
        # Stage 2: L-BFGS (SINGLE CALL)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        # Try with return_trajectories (v3.3+), fall back if not supported
        try:
            bfgs_result = lbfgs_batch(
                func=self.func,
                x0=all_seeds,
                maxiter=self.maxiter,
                tol=self.convergence_tol,
                bounds=self.bounds,
                verbose=False,
                return_trajectories=True  # NEW v1.7: For rotation detection
            )
            trajectories = bfgs_result.get('trajectories')
        except TypeError:
            # Fallback for older chisao versions without return_trajectories
            bfgs_result = lbfgs_batch(
                func=self.func,
                x0=all_seeds,
                maxiter=self.maxiter,
                tol=self.convergence_tol,
                bounds=self.bounds,
                verbose=False
            )
            trajectories = None
        
        final_positions = bfgs_result['x']
        converged = bfgs_result['converged']
        grad_norm_all = bfgs_result['grad_norm']  # (K * n_seeds,) - FREE gradient info!
        
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        timing['lbfgs'] = time.perf_counter() - t0
        
        if self.verbose:
            n_conv = int(xp.sum(converged))
            print(f"\n[Stage 2: L-BFGS]")
            print(f"  Converged: {n_conv}/{N_samples}")
            print(f"  Time: {timing['lbfgs']:.3f}s")
        
        # -----------------------------------------------------------------
        # Stage 3: Extract refined peaks (VECTORIZED)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        L_final = self.func(final_positions)
        
        L_final_reshaped = L_final.reshape(K, n_seeds)
        final_positions_reshaped = final_positions.reshape(K, n_seeds, D)
        grad_norm_reshaped = grad_norm_all.reshape(K, n_seeds)  # Also reshape grad_norm
        
        best_idx_per_peak = xp.argmax(L_final_reshaped, axis=1)
        
        refined_peaks = final_positions_reshaped[xp.arange(K), best_idx_per_peak, :]
        L_refined = L_final_reshaped[xp.arange(K), best_idx_per_peak]
        grad_norm_refined = grad_norm_reshaped[xp.arange(K), best_idx_per_peak]  # FREE!
        
        timing['extraction'] = time.perf_counter() - t0
        
        # -----------------------------------------------------------------
        # Stage 4: Diagonal Hessian via finite differences
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        eye_D = xp.eye(D)
        
        all_plus = refined_peaks[:, None, :] + eps * eye_D[None, :, :]   # (K, D, D)
        all_minus = refined_peaks[:, None, :] - eps * eye_D[None, :, :]  # (K, D, D)
        
        all_plus = all_plus.reshape(K * D, D)
        all_minus = all_minus.reshape(K * D, D)
        
        f_plus = self.func(all_plus).reshape(K, D)
        f_minus = self.func(all_minus).reshape(K, D)
        
        diag_H = (f_plus - 2*L_refined[:, None] + f_minus) / (eps**2)
        
        timing['diag_hessian'] = time.perf_counter() - t0
        
        if self.verbose:
            print(f"\n[Stage 4: Diagonal Hessian]")
            print(f"  Cost: {2*K*D} evaluations")
        
        # -----------------------------------------------------------------
        # Stage 4b: Saddle Point Filtering
        # For a maximum, all diag_H should be negative (concave down).
        # If any diag_H > 0, that peak is a saddle point.
        # -----------------------------------------------------------------
        saddle_threshold = 1e-6  # Small positive tolerance for numerical noise
        is_maximum = xp.all(diag_H < saddle_threshold, axis=1)  # (K,)
        n_saddles = int(xp.sum(~is_maximum))
        
        if n_saddles > 0:
            if self.verbose:
                print(f"\n[Stage 4b: Saddle Point Filtering]")
                print(f"  Removed {n_saddles} saddle points, kept {int(xp.sum(is_maximum))} maxima")
            
            # Filter out saddle points
            refined_peaks = refined_peaks[is_maximum]
            L_refined = L_refined[is_maximum]
            diag_H = diag_H[is_maximum]
            grad_norm_refined = grad_norm_refined[is_maximum]  # Also filter gradient norms
            # Also filter f_plus, f_minus for Stage 5
            f_plus = f_plus[is_maximum]
            f_minus = f_minus[is_maximum]
            K = int(xp.sum(is_maximum))
            
            # Handle case where all peaks were saddles
            if K == 0:
                if self.verbose:
                    print(f"  WARNING: All peaks were saddle points!")
                return self._empty_result(D, return_bank)
        
        # -----------------------------------------------------------------
        # Stage 4c: Gradient Filter (true maxima have zero gradient)
        # Uses FREE gradient norm from L-BFGS - no extra function evaluations!
        # -----------------------------------------------------------------
        grad_threshold = 1e-4  # Tolerance for "zero" gradient
        is_stationary = grad_norm_refined < grad_threshold
        n_not_stationary = int(xp.sum(~is_stationary))
        
        if n_not_stationary > 0:
            if self.verbose:
                print(f"\n[Stage 4c: Gradient Filter (FREE from L-BFGS)]")
                print(f"  Removed {n_not_stationary} non-stationary points, kept {int(xp.sum(is_stationary))}")
            
            # Filter out non-stationary points
            refined_peaks = refined_peaks[is_stationary]
            L_refined = L_refined[is_stationary]
            diag_H = diag_H[is_stationary]
            # Also filter f_plus, f_minus for Stage 5
            f_plus = f_plus[is_stationary]
            f_minus = f_minus[is_stationary]
            K = int(xp.sum(is_stationary))
            
            # Handle case where all peaks were non-stationary
            if K == 0:
                if self.verbose:
                    print(f"  WARNING: All peaks were non-stationary!")
                return self._empty_result(D, return_bank)
        
        # -----------------------------------------------------------------
        # Stage 5: Low-rank correction via probing
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        
        low_rank_U = xp.zeros((K, D, r))
        low_rank_V = xp.zeros((K, D, r))
        
        # Random probe directions
        probe_dirs = xp.random.randn(r, D)
        probe_dirs = probe_dirs / xp.linalg.norm(probe_dirs, axis=1, keepdims=True)
        
        # Compute gradient at refined peaks (reuse from diag_H computation)
        grad_at_x = (f_plus - f_minus) / (2*eps)  # (K, D)
        
        for i in range(r):
            v = probe_dirs[i]
            x_plus_v = refined_peaks + eps * v
            
            grad_pts_pv_plus = x_plus_v[:, None, :] + eps * eye_D[None, :, :]
            grad_pts_pv_minus = x_plus_v[:, None, :] - eps * eye_D[None, :, :]
            
            f_pv_plus = self.func(grad_pts_pv_plus.reshape(K*D, D)).reshape(K, D)
            f_pv_minus = self.func(grad_pts_pv_minus.reshape(K*D, D)).reshape(K, D)
            
            grad_at_xv = (f_pv_plus - f_pv_minus) / (2*eps)
            
            Hv = (grad_at_xv - grad_at_x) / eps
            residual = Hv - diag_H * v
            
            low_rank_U[:, :, i] = residual
            low_rank_V[:, :, i] = xp.tile(v, (K, 1))
        
        timing['low_rank'] = time.perf_counter() - t0
        
        # -----------------------------------------------------------------
        # Build result
        # -----------------------------------------------------------------
        result = {
            'peaks': refined_peaks,
            'L_peaks': L_refined,
            'diag_H': diag_H,
            'low_rank_U': low_rank_U,
            'low_rank_V': low_rank_V,
            'n_peaks': K
        }
        
        if return_bank:
            bank = TrajectoryBank(D, xp)
            bank.seed_positions = all_seeds
            bank.seed_log_L = L_seeds
            bank.seed_peak_ids = seed_peak_ids
            bank.seed_dims = None  # Not applicable for random
            bank.seed_signs = None
            bank.final_positions = final_positions
            bank.final_log_L = L_final
            bank.converged = converged
            bank.trajectories = trajectories  # NEW v1.7: For rotation detection
            bank.refined_peaks = refined_peaks
            bank.refined_log_L = L_refined
            bank.n_seeds = N_samples
            bank.n_seeds_per_peak = n_seeds
            bank.n_peaks = K
            bank.mode = 'random'
            result['trajectory_bank'] = bank
        
        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def refine_peaks(
    func: Callable,
    peaks: 'cp.ndarray',
    widths: 'cp.ndarray',
    bounds: 'cp.ndarray',
    fast: bool = False,
    return_bank: bool = False,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for peak refinement.
    
    Args:
        func: Log-likelihood function
        peaks: [K, D] peak locations
        widths: [K, 2] or [K,] width estimates
        bounds: [D, 2] parameter bounds
        fast: Use fast mode (20 random samples instead of 2×D)
        return_bank: Include TrajectoryBank in results
        verbose: Print progress
        **kwargs: Additional arguments to GreenDragonRisesFromWater
        
    Returns:
        Dictionary with refined peaks, Hessian approximation, timing
    """
    dragon = GreenDragonRisesFromWater(
        func=func,
        bounds=bounds,
        fast=fast,
        verbose=verbose,
        **kwargs
    )
    return dragon.refine(peaks, widths, return_bank=return_bank)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("="*70)
    print("GreenDragonRisesFromWater v1.7 - Test")
    print("="*70)
    
    xp = cp if GPU_AVAILABLE else np
    
    # Test function
    def gaussian_nd(theta):
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        return -0.5 * xp.sum(theta**2, axis=1)
    
    for D in [32, 128]:
        print(f"\n{'='*60}")
        print(f"D = {D}")
        print(f"{'='*60}")
        
        bounds = xp.array([[-5.0, 5.0]] * D)
        peaks = xp.random.randn(3, D) * 0.5
        widths = xp.ones(3) * 1.0
        
        # Default mode
        print("\n--- DEFAULT MODE (axis-aligned) ---")
        t0 = time.perf_counter()
        result_default = refine_peaks(
            func=gaussian_nd,
            peaks=peaks.copy(),
            widths=widths.copy(),
            bounds=bounds,
            fast=False,
            return_bank=True,
            verbose=True
        )
        time_default = time.perf_counter() - t0
        
        # Fast mode
        print("\n--- FAST MODE (random) ---")
        t0 = time.perf_counter()
        result_fast = refine_peaks(
            func=gaussian_nd,
            peaks=peaks.copy(),
            widths=widths.copy(),
            bounds=bounds,
            fast=True,
            return_bank=True,
            verbose=True
        )
        time_fast = time.perf_counter() - t0
        
        print(f"\n--- COMPARISON ---")
        print(f"  Default time: {time_default:.3f}s")
        print(f"  Fast time:    {time_fast:.3f}s")
        print(f"  Speedup:      {time_default/time_fast:.1f}x")
        print(f"  Default bank: {result_default['trajectory_bank'].n_seeds} samples")
        print(f"  Fast bank:    {result_fast['trajectory_bank'].n_seeds} samples")
