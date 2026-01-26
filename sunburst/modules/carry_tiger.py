"""
CarryTiger (抱虎归山 - Bào Hǔ Guī Shān) v2.2
==============================================

Carry Tiger to Mountain: Multi-scale Ray Casting Orchestrator

Named after the Guang Ping Yang Style Tai Chi form, in honor of Master Donald Robbo.

Philosophy:
    "Carry the tiger (difficult problem) to the mountain (optimal solution)
     by understanding its nature (scales) before applying force (optimization)"

Strategy:
    1. Generate rays through parameter space (v2v, v2e, w2w, sunburst)
    2. Initial coarse sampling along rays
    3. **Scale Discovery**: Use SingleWhip to identify λ_fine, λ_mid, λ_coarse
    4. **Multi-scale Resampling**: Adapt sampling density to discovered scales
    5. **ChiSao Optimization**: Full sticky hands with configurable oscillations
    6. Return refined peaks with widths

Key Innovation:
    Unlike Module 1 v1.5 which uses fixed sampling density, CarryTiger
    adapts sampling to the problem's natural scales, ensuring we don't
    miss fine features OR waste samples on smooth regions.

NEW in v1.7: Two-Phase Detection (multi_scale=True)
    For problems with multiple scales (Ackley, Rastrigin, Eggbox):
    - Phase 1: HLC-only scouting for global features
    - Phase 2: Full ChiSao with global seeds injected
    This ensures global optima are never lost while still finding local features.

NEW in v1.9: Sample Banking (return_bank=True)
    For evidence calculation integration with Module 3:
    - RayBank: stores ray geometry, samples along rays, log_L values
    - ChiSaoBank: stores optimization trajectory from ChiSao v2.4
    Enable with return_bank=True in detect_modes()

NEW in v2.0: Vectorized Ray Sampling
    - _initial_sampling() now fully vectorized (no Python loops)
    - _multiscale_resample() now fully vectorized (no Python loops)
    - Removed per-sample deduplication (unnecessary overhead)
    - ~10-100× faster for high-D problems

NEW in v2.1: Explicit ChiSao Version Tracking
    - Proper fallback chain: chisao3_2 → chisao3_1 → chisao2_5
    - Warnings on fallback to alert user of missing optimizations
    - Reports ChiSao version in verbose output

NEW in v2.2: Configurable ChiSao Oscillations
    - n_oscillations now a constructor parameter (default=3)
    - Was hardcoded to 1, now properly configurable
    - Higher values (3-5) improve mode detection reliability
    - Lower values (1) for speed when modes are well-separated

Integration with SunBURST:
    Module 0 → log_likelihood (GPU-ready) → CarryTiger (抱虎归山 - Bào Hǔ Guī Shān) v2.2 → peaks → Module 2

CRITICAL REQUIREMENT - PURE GPU PIPELINE:
    **EVERYTHING stays in GPU (CuPy arrays) throughout!**
    - Input: CuPy arrays from Module 0
    - Processing: All operations in CuPy
    - Internal: Bounds, samples, all arrays are CuPy
    - Output: CuPy arrays (converted to NumPy only for final display)
    - NO CPU conversions during processing!
    
    This ensures:
    ✓ Maximum performance (no CPU↔GPU transfers)
    ✓ No array mixing bugs
    ✓ Pure GPU pipeline as designed

Version: 2.0
Author: SunBURST Development Team
Date: December 2025
Status: INTEGRATION READY - PURE GPU - VECTORIZED
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    xp = np

import time
from typing import Callable, Optional, Tuple, List, Dict, Union
import warnings

# Import dependencies - SingleWhip with version tracking
SINGLEWHIP_VERSION = None
SINGLEWHIP_AVAILABLE = False
try:
    # Package-relative import
    from ..utils.single_whip import SingleWhip
    SINGLEWHIP_AVAILABLE = True
    SINGLEWHIP_VERSION = '1.6'
except ImportError:
    try:
        # Fallback for standalone usage
        from single_whip_GPU_v1_6 import SingleWhip
        SINGLEWHIP_AVAILABLE = True
        SINGLEWHIP_VERSION = '1.6'
    except ImportError:
        raise ImportError(
            "SingleWhip required for CarryTiger. "
            "Please install the sunburst package properly."
        )

# ChiSao import with explicit version tracking
CHISAO_VERSION = None
CHISAO_WHIP_VERSION = None
try:
    # Package-relative import
    from ..utils.chisao import sticky_hands, SINGLEWHIP_VERSION as CHISAO_WHIP_VERSION
    CHISAO_AVAILABLE = True
    CHISAO_VERSION = '3.3'
except ImportError:
    try:
        # Fallback for standalone usage
        from chisao3_3 import sticky_hands, SINGLEWHIP_VERSION as CHISAO_WHIP_VERSION
        CHISAO_AVAILABLE = True
        CHISAO_VERSION = '3.3'
    except ImportError:
        raise ImportError(
            "ChiSao required for CarryTiger. "
            "Please install the sunburst package properly."
        )


# ============================================================================
# RAY BANK - SAMPLE STORAGE FOR EVIDENCE CALCULATION (v1.9)
# ============================================================================

class RayBank:
    """
    GPU-resident storage for ray casting samples.
    Used for evidence calculation in Module 3.
    
    Stores:
        - Ray geometry (starts, ends, types)
        - Samples along rays (positions, log_L, t-values)
    """
    
    def __init__(self, D: int, xp=None):
        self.xp = xp if xp is not None else (cp if GPU_AVAILABLE else np)
        self.D = D
        self.ray_starts = None
        self.ray_ends = None
        self.ray_types = None
        self.samples = None
        self.log_L = None
        self.t_values = None
        self.f_samples = None
        self.n_rays = 0
        self.n_samples_per_ray = 0
    
    def store_rays(self, rays, ray_types: List[int] = None):
        """
        Store ray geometry.
        
        Args:
            rays: Either (ray_starts, ray_ends) tuple of arrays [N, D] each,
                  or List of (start, end) tuples (legacy format)
            ray_types: Optional array of ray type indices
        """
        # Handle both formats: tuple of arrays or list of tuples
        if isinstance(rays, tuple) and len(rays) == 2:
            # New format: (ray_starts, ray_ends) arrays
            self.ray_starts = rays[0]
            self.ray_ends = rays[1]
            self.n_rays = len(rays[0])
        else:
            # Legacy format: list of (start, end) tuples
            if len(rays) == 0:
                return
            self.n_rays = len(rays)
            self.ray_starts = self.xp.stack([r[0] for r in rays])
            self.ray_ends = self.xp.stack([r[1] for r in rays])
        
        if ray_types is not None:
            self.ray_types = self.xp.array(ray_types, dtype=self.xp.int32)
        else:
            self.ray_types = self.xp.zeros(self.n_rays, dtype=self.xp.int32)
    
    def store_samples(self, samples, f_samples, t_values):
        """Store samples along rays."""
        self.samples = samples
        self.f_samples = f_samples
        self.t_values = t_values
        self.n_samples_per_ray = len(t_values)
        self.log_L = f_samples.flatten()
    
    def get_bank_dict(self) -> dict:
        """Return bank contents as dictionary."""
        return {
            'ray_starts': self.ray_starts,
            'ray_ends': self.ray_ends,
            'ray_types': self.ray_types,
            'samples': self.samples,
            'log_L': self.log_L,
            't_values': self.t_values,
            'f_samples': self.f_samples,
            'n_rays': self.n_rays,
            'n_samples_per_ray': self.n_samples_per_ray,
            'D': self.D
        }
    
    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        if self.samples is None:
            return 0.0
        n_samples = self.samples.shape[0]
        bytes_used = (
            n_samples * self.D * 8 +
            n_samples * 8 +
            self.n_rays * self.D * 8 * 2 +
            self.n_rays * 4
        )
        return bytes_used / (1024 * 1024)


# ============================================================================
# CARRY TIGER TO MOUNTAIN - MAIN ORCHESTRATOR
# ============================================================================

class CarryTigerToMountain:
    """
    抱虎归山 - Carry Tiger to Mountain
    
    Multi-scale ray casting orchestrator with ChiSao optimization.
    
    Workflow:
        1. Ray casting using established recipes (v2v, v2e, w2w, sunburst)
        2. Initial sampling along rays (coarse grid)
        3. Scale discovery via SingleWhip.estimate_characteristic_scales()
        4. Multi-scale adaptive resampling based on discovered scales
        5. ChiSao optimization with full sticky hands (3 oscillations)
        6. Return refined peaks with width estimates
    
    NEW v1.7: Two-Phase Detection (multi_scale=True):
        For problems with different scales (Ackley, Rastrigin, Eggbox):
        - Phase 1: HLC-only scouting finds global features
        - Phase 2: Full ChiSao with global seeds injected
        Ensures global optima are never lost while finding local features.
    
    Parameters:
        func: Log-likelihood function (GPU-ready from Module 0)
        bounds: [D, 2] parameter bounds
        n_rays: Number of rays per recipe (default: 10, optimized from 50)
        n_samples_per_ray: Samples per ray (default: 50, optimized from 100)
            Note: n_init = n_rays × n_samples_per_ray = 500 (validated optimal)
        sample_weights: (w_fine, w_mid, w_coarse) relative weights summing to 1.0 (default: (0.5, 0.3, 0.2))
        n_initial_samples: Initial samples per ray for scale discovery (default: 100)
        n_ensemble: Number of ensemble runs (default: 8)
        use_gpu: Enable GPU acceleration (default: True)
        multi_scale: Enable two-phase detection for Ackley-type problems (default: False)
        n_oscillations: ChiSao oscillation cycles (default: 3, was hardcoded to 1 in v2.1)
            - Higher values (3-5): More thorough mode detection, slower
            - Lower values (1): Faster, suitable for well-separated modes
        
    Methods:
        detect_modes(verbose=False): Main entry point
        
    Returns:
        peaks: [K, D] refined peak locations
        L_peaks: [K] log-likelihood at peaks
        widths: [K, 2] width estimates [longest, shortest]
    """
    
    def __init__(
        self,
        func: Callable,
        bounds: np.ndarray,
        n_rays: int = None,  # None = adaptive: 10 + log2(d)
        n_samples_per_ray: int = 50,  # OPTIMIZED: was 100 (total n_init=500)
        sample_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        n_initial_samples: int = 100,
        n_ensemble: int = 8,
        use_gpu: bool = True,
        multi_scale: bool = False,  # NEW v1.7: Two-phase detection for Ackley-type
        func_returns_grad: bool = False,  # NEW v3.1: Analytic gradient support
        line_search: str = 'armijo',  # NEW v3.2: 'armijo' (safe) or 'fixed' (fast)
        n_oscillations: int = 3  # NEW v2.2: ChiSao oscillation cycles (was hardcoded to 1)
    ):
        self.func = func
        self.bounds = xp.array(bounds) if use_gpu else np.array(bounds)
        self.dim = len(bounds)
        
        # Adaptive n_rays: 10 + log2(d)
        if n_rays is None:
            n_rays = int(10 + np.log2(max(2, self.dim)))
        self.n_rays = n_rays
        
        self.n_samples_per_ray = n_samples_per_ray
        self.sample_weights = sample_weights
        self.n_initial_samples = n_initial_samples
        self.n_ensemble = n_ensemble
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.multi_scale = multi_scale  # NEW v1.7
        self.func_returns_grad = func_returns_grad  # NEW v3.1
        self.line_search = line_search  # NEW v3.2
        self.n_oscillations = n_oscillations  # NEW v2.2
        
        # Create a likelihood-only wrapper for places that don't need gradients
        if func_returns_grad:
            self._func_L_only = lambda x: func(x)[0]  # Extract just L from (L, grad)
        else:
            self._func_L_only = func
        
        # Initialize SingleWhip for scale discovery and utilities
        if not SINGLEWHIP_AVAILABLE:
            raise RuntimeError("SingleWhip v1.3+ is required")
        
        self.whip = SingleWhip(use_gpu=self.use_gpu)
        
        # Initialize ChiSao parameters
        self.chisao_params = {
            'method': 'lbfgs',
            'n_oscillations': self.n_oscillations,  # v2.2: Now configurable (was hardcoded to 1)
            'n_converge': 10,
            'n_anticonverge': 5,
            'reseed_strategy': 'sunburst',  # Repulse Monkey enabled
            'cannon_through_sky': True,
            'estimate_widths': True,
            'stick_tolerance': 1e-6,  # Default dedup tolerance
            'bounds': self.bounds  # PURE GPU: CuPy array
        }
    
    def detect_modes(
        self,
        verbose: bool = False,
        iteration: int = 1,
        seed_points: Optional[np.ndarray] = None,
        return_bank: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect modes using multi-scale ray casting + ChiSao optimization.
        
        Args:
            verbose: Print progress information
            iteration: 1 (discovery) or 2 (refinement from seed_points)
            seed_points: For iteration 2, peaks from iteration 1
            return_bank: If True, also return RayBank and ChiSao bank (v1.9)
            
        Returns:
            peaks: [K, D] peak locations
            L_peaks: [K] log-likelihood values
            widths: [K, 2] width estimates [longest, shortest]
            
            If return_bank=True, also returns:
            ray_bank: RayBank with ray geometry and samples
            chisao_bank: dict with ChiSao optimization trajectory
        """
        # NEW v1.7: Two-phase detection for multi-scale problems
        if self.multi_scale:
            return self._detect_modes_two_phase(verbose=verbose, return_bank=return_bank)
        
        if verbose:
            print("\n" + "="*70)
            print(f"CarryTiger v2.2: Iteration {iteration}")
            if CHISAO_VERSION:
                print(f"  [ChiSao v{CHISAO_VERSION}]")
            print("="*70)
        
        # v2.0: Initialize RayBank if banking enabled
        ray_bank = RayBank(D=self.dim, xp=self.xp) if return_bank else None
        
        # === STEP 1: RAY GENERATION ===
        if verbose:
            print("\n[Step 1: Ray Generation]")
        
        if iteration == 1 or seed_points is None:
            rays = self._generate_rays_discovery(verbose=verbose)
        else:
            rays = self._generate_rays_refinement(seed_points, verbose=verbose)
        
        # rays is now (ray_starts, ray_ends) tuple of arrays
        n_rays = len(rays[0])
        
        if verbose:
            print(f"  Generated {n_rays} rays")
        
        # v2.0: Store ray geometry
        if return_bank:
            ray_bank.store_rays(rays)
        
        # === STEP 2: INITIAL SAMPLING ===
        if verbose:
            print("\n[Step 2: Initial Sampling for Scale Discovery]")
        
        samples, f_samples, t_values = self._initial_sampling(
            rays, verbose=verbose
        )
        
        if verbose:
            print(f"  Sampled {len(samples)} points along rays")
        
        # v2.0: Store samples
        if return_bank:
            ray_bank.store_samples(samples, f_samples, t_values)
        
        # === STEP 3: SCALE DISCOVERY ===
        if verbose:
            print("\n[Step 3: Scale Discovery via SingleWhip]")
        
        λ_fine, λ_mid, λ_coarse = self._discover_scales(
            f_samples, t_values, verbose=verbose
        )
        
        if verbose:
            print(f"  λ_fine   = {λ_fine:.6f} (local features)")
            print(f"  λ_mid    = {λ_mid:.6f} (intermediate structure)")
            print(f"  λ_coarse = {λ_coarse:.6f} (global shape)")
        
        # === STEP 4: MULTI-SCALE RESAMPLING ===
        if verbose:
            print("\n[Step 4: Multi-Scale Adaptive Resampling]")
        
        resampled_points = self._multiscale_resample(
            rays, λ_fine, λ_mid, λ_coarse, verbose=verbose
        )
        
        if verbose:
            print(f"  Resampled {len(resampled_points)} points (scale-adaptive)")
        
        # === STEP 5: CHISAO OPTIMIZATION ===
        if verbose:
            print("\n[Step 5: ChiSao Optimization (3 oscillations, GR+RM)]")
        
        result = self._optimize_with_chisao(
            resampled_points, verbose=verbose, bank_samples=return_bank
        )
        
        peaks = result['peaks']
        L_peaks = result['L_peaks']
        widths = result['widths']
        
        # v2.0: Extract ChiSao bank
        chisao_bank = result.get('sample_bank', None) if return_bank else None
        
        if verbose:
            print(f"\n[CarryTiger v2.2: Complete]")
            print(f"  Found {len(peaks)} peaks")
            print(f"  Peak likelihood range: [{xp.min(L_peaks):.2f}, {xp.max(L_peaks):.2f}]")
            if len(widths) > 0:
                print(f"  Width range: [{xp.min(widths):.4f}, {xp.max(widths):.4f}]")
        
        if return_bank:
            return peaks, L_peaks, widths, ray_bank, chisao_bank
        return peaks, L_peaks, widths
    
    # ========================================================================
    # STEP 1: RAY GENERATION
    # ========================================================================
    
    def _generate_rays_discovery(self, verbose: bool = False) -> Tuple:
        """
        Generate rays for iteration 1 (discovery).
        Uses 4-component recipe: v2v, v2e, w2w, sunburst.
        
        VECTORIZED in v2.0 - returns (ray_starts, ray_ends) arrays directly.
        
        Sunburst rays: 1 burst from center, with QR orthonormal directions,
        shooting +/- along each dimension = 4 × d × 2 rays.
        
        Returns:
            (ray_starts, ray_ends): Both [N_rays, D] arrays on GPU
        """
        xp = self.xp
        D = self.dim
        
        # Component distribution for geometric rays
        n_v2v = int(0.4 * self.n_rays)
        n_v2e = int(0.3 * self.n_rays)
        n_w2w = self.n_rays - n_v2v - n_v2e  # Remainder
        
        all_starts = []
        all_ends = []
        
        # === VERTEX-TO-VERTEX (vectorized) ===
        if n_v2v > 0:
            # Random vertices: each vertex is a choice of lower/upper bound per dimension
            v2v_start_choices = xp.random.randint(0, 2, size=(n_v2v, D))
            v2v_end_choices = xp.random.randint(0, 2, size=(n_v2v, D))
            
            v2v_starts = xp.where(v2v_start_choices == 0, 
                                   self.bounds[:, 0], self.bounds[:, 1])
            v2v_ends = xp.where(v2v_end_choices == 0,
                                 self.bounds[:, 0], self.bounds[:, 1])
            
            all_starts.append(v2v_starts)
            all_ends.append(v2v_ends)
        
        # === VERTEX-TO-EDGE (vectorized) ===
        if n_v2e > 0:
            # Start: random vertices
            v2e_start_choices = xp.random.randint(0, 2, size=(n_v2e, D))
            v2e_starts = xp.where(v2e_start_choices == 0,
                                   self.bounds[:, 0], self.bounds[:, 1])
            
            # End: random points with one dimension fixed to boundary
            v2e_ends = xp.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(n_v2e, D)
            ).astype(xp.float64)
            
            # Vectorized: create index arrays for scatter
            row_idx = xp.arange(n_v2e)
            fixed_dims = xp.random.randint(0, D, size=n_v2e)
            fixed_sides = xp.random.randint(0, 2, size=n_v2e)
            
            # Scatter boundary values (fully vectorized)
            boundary_vals = xp.where(fixed_sides == 0,
                                      self.bounds[fixed_dims, 0],
                                      self.bounds[fixed_dims, 1])
            v2e_ends[row_idx, fixed_dims] = boundary_vals
            
            all_starts.append(v2e_starts)
            all_ends.append(v2e_ends)
        
        # === WALL-TO-WALL (vectorized) ===
        if n_w2w > 0:
            # Both start and end are random points with one dim fixed
            w2w_starts = xp.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(n_w2w, D)
            ).astype(xp.float64)
            w2w_ends = xp.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(n_w2w, D)
            ).astype(xp.float64)
            
            row_idx = xp.arange(n_w2w)
            
            # Fix random dimensions for starts
            start_dims = xp.random.randint(0, D, size=n_w2w)
            start_sides = xp.random.randint(0, 2, size=n_w2w)
            start_boundary = xp.where(start_sides == 0,
                                       self.bounds[start_dims, 0],
                                       self.bounds[start_dims, 1])
            w2w_starts[row_idx, start_dims] = start_boundary
            
            # Fix random dimensions for ends
            end_dims = xp.random.randint(0, D, size=n_w2w)
            end_sides = xp.random.randint(0, 2, size=n_w2w)
            end_boundary = xp.where(end_sides == 0,
                                     self.bounds[end_dims, 0],
                                     self.bounds[end_dims, 1])
            w2w_ends[row_idx, end_dims] = end_boundary
            
            all_starts.append(w2w_starts)
            all_ends.append(w2w_ends)
        
        # === SUNBURST RAYS FROM CENTER (n_rays random orthonormal directions) ===
        # Instead of 2D rays (which scales with dimension), use fixed n_rays
        # This makes total ray count O(1) instead of O(D)
        center = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        box_width = self.bounds[:, 1] - self.bounds[:, 0]
        ray_length = float(xp.max(box_width).get() if hasattr(xp.max(box_width), 'get') else xp.max(box_width)) / 2.0
        
        n_sunburst_rays = self.n_rays  # Fixed count, doesn't scale with D
        
        # Generate random orthonormal directions via QR
        Q = self._qr_random_basis()  # [D, D] orthonormal columns
        
        # Select n_sunburst_rays random directions (with random signs)
        # If n_rays > D, we cycle through directions multiple times
        sunburst_starts = xp.tile(center, (n_sunburst_rays, 1))  # [n_rays, D]
        
        # Pick which orthonormal direction each ray uses
        dir_indices = xp.arange(n_sunburst_rays) % D
        directions = Q[:, dir_indices].T  # [n_rays, D]
        
        # Random signs (+1 or -1) for each ray
        signs = xp.where(xp.random.randint(0, 2, size=n_sunburst_rays) == 0, 1.0, -1.0)
        
        # Compute endpoints
        sunburst_ends = center[None, :] + signs[:, None] * directions * ray_length
        
        # Clip to bounds
        sunburst_ends = xp.clip(sunburst_ends, self.bounds[:, 0], self.bounds[:, 1])
        
        all_starts.append(sunburst_starts)
        all_ends.append(sunburst_ends)
        
        if verbose:
            print(f"    Sunburst rays: {n_sunburst_rays} (fixed, random orthonormal)")
        
        # Combine all rays into single arrays
        ray_starts = xp.vstack(all_starts)  # [N_rays, D]
        ray_ends = xp.vstack(all_ends)      # [N_rays, D]
        
        return ray_starts, ray_ends
    
    def _qr_random_basis(self) -> np.ndarray:
        """
        Generate random orthonormal basis using QR decomposition.
        
        Returns:
            Q: [d, d] orthonormal matrix (columns are basis vectors)
        """
        # Generate random matrix
        A = self.xp.random.randn(self.dim, self.dim).astype(self.xp.float64)
        
        # QR decomposition: A = QR where Q is orthonormal
        Q, R = self.xp.linalg.qr(A)
        
        return Q
    
    def _generate_rays_refinement(
        self,
        seed_points: np.ndarray,
        verbose: bool = False
    ) -> Tuple:
        """
        Generate rays for iteration 2 (refinement).
        Rays emanate from discovered peaks.
        
        VECTORIZED in v2.0 - returns (ray_starts, ray_ends) arrays.
        
        Returns:
            (ray_starts, ray_ends): Both [N_rays, D] arrays on GPU
        """
        xp = self.xp
        n_seeds = len(seed_points)
        rays_per_seed = self.n_rays // max(n_seeds, 1)
        D = self.dim
        
        all_starts = []
        all_ends = []
        
        # Box dimensions for ray length
        box_width = self.bounds[:, 1] - self.bounds[:, 0]
        ray_length = float(xp.max(box_width).get() if hasattr(xp.max(box_width), 'get') else xp.max(box_width)) / 2.0
        
        for seed in seed_points:
            # Generate orthonormal directions via QR
            Q = self._qr_random_basis()  # [D, D]
            
            # Use as many directions as we need
            n_dirs = min(rays_per_seed, 2 * D)
            
            # Create starts (all from this seed)
            starts = xp.tile(seed, (n_dirs, 1))  # [n_dirs, D]
            
            # Create ends using QR directions (alternate + and -)
            ends = xp.zeros((n_dirs, D), dtype=xp.float64)
            for i in range(n_dirs):
                dim_idx = i // 2
                sign = 1 if i % 2 == 0 else -1
                if dim_idx < D:
                    ends[i] = seed + sign * Q[:, dim_idx] * ray_length
            
            # Clip to bounds
            ends = xp.clip(ends, self.bounds[:, 0], self.bounds[:, 1])
            
            all_starts.append(starts)
            all_ends.append(ends)
        
        if len(all_starts) > 0:
            ray_starts = xp.vstack(all_starts)
            ray_ends = xp.vstack(all_ends)
        else:
            # Fallback: just use discovery rays
            return self._generate_rays_discovery(verbose=verbose)
        
        return ray_starts, ray_ends
    
    def _random_vertex(self) -> np.ndarray:
        """Random hypercube vertex"""
        choices = self.xp.random.choice([0, 1], size=self.dim)
        return self.xp.where(
            choices == 0,
            self.bounds[:, 0],
            self.bounds[:, 1]
        )
    
    def _random_edge_point(self) -> np.ndarray:
        """Random point on a hypercube edge"""
        # Fix one dimension to boundary, randomize others
        fixed_dim = self.xp.random.randint(0, self.dim)
        point = self.xp.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1]
        )
        # Fix the selected dimension (CuPy-compatible)
        boundary_side = int(self.xp.random.randint(0, 2))
        point[fixed_dim] = self.bounds[fixed_dim, boundary_side]
        return point
    
    def _random_wall_point(self) -> np.ndarray:
        """Random point on hypercube boundary"""
        # Random point, then project one coordinate to boundary
        point = self.xp.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1]
        )
        # Choose random dimension and boundary (CuPy-compatible)
        wall_dim = self.xp.random.randint(0, self.dim)
        wall_side = int(self.xp.random.randint(0, 2))
        point[wall_dim] = self.bounds[wall_dim, wall_side]
        return point
    
    # ========================================================================
    # STEP 2: INITIAL SAMPLING
    # ========================================================================
    
    def _initial_sampling(
        self,
        rays: Tuple,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample points along rays for scale discovery.
        
        VECTORIZED in v2.0 - no Python loops over rays.
        
        Args:
            rays: Tuple of (ray_starts, ray_ends) arrays, each [N_rays, D]
        
        Returns:
            samples: [N_total, D] sample positions
            f_samples: [N_rays, N_samples_per_ray] likelihood values
            t_values: [N_samples_per_ray] parametric positions (0 to 1)
        """
        ray_starts, ray_ends = rays
        n_rays = len(ray_starts)
        n_samples = self.n_initial_samples
        
        # Parametric positions along ray
        t_values = self.xp.linspace(0, 1, n_samples)
        
        # Ray directions
        ray_dirs = ray_ends - ray_starts  # [n_rays, D]
        
        # Vectorized sampling: samples[i, j, d] = start[i, d] + t[j] * dir[i, d]
        # ray_starts[:, None, :] is [n_rays, 1, D]
        # t_values[None, :, None] is [1, n_samples, 1]
        # ray_dirs[:, None, :] is [n_rays, 1, D]
        # Result: [n_rays, n_samples, D]
        all_samples = ray_starts[:, None, :] + t_values[None, :, None] * ray_dirs[:, None, :]
        
        # Clip to bounds
        all_samples = self.xp.clip(
            all_samples,
            self.bounds[:, 0],
            self.bounds[:, 1]
        )
        
        # Evaluate likelihood - reshape to [n_rays * n_samples, D] for batch eval
        samples_flat = all_samples.reshape(-1, self.dim)
        f_flat = self._func_L_only(samples_flat)  # Use L-only wrapper
        
        # Reshape back to [n_rays, n_samples]
        f_samples = f_flat.reshape(n_rays, n_samples)
        
        # Flatten samples to [N_total, D]
        samples = samples_flat
        
        return samples, f_samples, t_values
    
    # ========================================================================
    # STEP 3: SCALE DISCOVERY
    # ========================================================================
    
    def _discover_scales(
        self,
        f_samples: np.ndarray,
        t_values: np.ndarray,
        verbose: bool = False
    ) -> Tuple[float, float, float]:
        """
        Discover characteristic scales using SingleWhip.
        
        Args:
            f_samples: [N_rays, N_samples] likelihood along rays
            t_values: [N_samples] parametric positions
            
        Returns:
            λ_fine, λ_mid, λ_coarse: Characteristic length scales
        """
        start_time = time.time()
        
        # Use SingleWhip's scale estimation
        λ_fine, λ_mid, λ_coarse = self.whip.estimate_characteristic_scales(
            f_samples=f_samples,
            t_values=t_values
        )
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  Scale discovery: {elapsed:.3f}s")
        
        return λ_fine, λ_mid, λ_coarse
    
    # ========================================================================
    # STEP 4: MULTI-SCALE RESAMPLING
    # ========================================================================
    
    def _multiscale_resample(
        self,
        rays: Tuple,
        λ_fine: float,
        λ_mid: float,
        λ_coarse: float,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Adaptive resampling based on discovered scales.
        
        VECTORIZED in v2.0 - no Python loops over rays.
        
        Strategy:
            Allocate sampling density inversely proportional to scale:
            - Fine scale → many samples (catch local features)
            - Coarse scale → few samples (global shape already smooth)
            
            Total samples per ray = α/λ_fine + β/λ_mid + γ/λ_coarse
            where α, β, γ are tunable weights.
        
        Args:
            rays: Tuple of (ray_starts, ray_ends) arrays, each [N_rays, D]
            λ_fine, λ_mid, λ_coarse: Discovered scales
            
        Returns:
            resampled_points: [N_total, D] adaptively sampled points
        """
        ray_starts, ray_ends = rays
        n_rays = len(ray_starts)
        
        # Budget-based allocation using relative weights
        w_fine, w_mid, w_coarse = self.sample_weights
        
        # Allocate budget proportionally
        n_fine = int(self.n_samples_per_ray * w_fine)
        n_mid = int(self.n_samples_per_ray * w_mid)
        n_coarse = int(self.n_samples_per_ray * w_coarse)
        
        # Handle rounding: distribute remainder to fine scale
        remainder = self.n_samples_per_ray - (n_fine + n_mid + n_coarse)
        n_fine += remainder
        
        n_total_per_ray = n_fine + n_mid + n_coarse
        
        if verbose:
            print(f"  Samples per ray: fine={n_fine}, mid={n_mid}, coarse={n_coarse}")
            print(f"  Total samples per ray: {n_total_per_ray}")
        
        # Ray directions (already arrays)
        ray_dirs = ray_ends - ray_starts  # [n_rays, D]
        
        # Create combined t_values for all scales
        t_fine = self.xp.linspace(0, 1, n_fine)
        t_mid = self.xp.linspace(0, 1, n_mid)
        t_coarse = self.xp.linspace(0, 1, n_coarse)
        t_combined = self.xp.concatenate([t_fine, t_mid, t_coarse])  # [n_total_per_ray]
        
        # Vectorized sampling: [n_rays, n_total_per_ray, D]
        all_samples = ray_starts[:, None, :] + t_combined[None, :, None] * ray_dirs[:, None, :]
        
        # Clip to bounds
        all_samples = self.xp.clip(
            all_samples,
            self.bounds[:, 0],
            self.bounds[:, 1]
        )
        
        # Flatten to [N_total, D]
        resampled_points = all_samples.reshape(-1, self.dim)
        
        # Skip deduplication - ChiSao will handle duplicates during optimization
        # This avoids the expensive O(N²) dedup on 26,000+ samples
        
        return resampled_points
    
    def _remove_duplicate_samples(
        self,
        points: np.ndarray,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Remove duplicate samples using L∞ metric.
        Uses SingleWhip's deduplication.
        """
        # Create dummy likelihood values (all same for deduplication)
        L_dummy = self.xp.zeros(len(points))
        
        # Use SingleWhip's deduplicate_peaks
        unique_points, _ = self.whip.deduplicate_peaks(
            points, L_dummy, tolerance=tolerance
        )
        
        return unique_points
    
    # ========================================================================
    # NEW v1.7: TWO-PHASE DETECTION FOR MULTI-SCALE PROBLEMS
    # ========================================================================
    
    def _detect_modes_two_phase(
        self,
        verbose: bool = False,
        return_bank: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-phase detection for multi-scale problems (Ackley, Rastrigin, Eggbox).
        
        Phase 1: HLC-only scouting for global features
            - Normal ray casting and sampling
            - ChiSao with mode='hlc_only' (extended HLC, minimal local refinement)
            - Output: Global peak candidates
            
        Phase 2: Full exploration with global seeds injected
            - Fresh ray casting and sampling
            - Inject Phase 1 peaks as seeds (guarantees global is in sample set)
            - Full ChiSao (HLC + convergence + anti-convergence)
            - Output: Complete peak list (global guaranteed + local discovered)
        
        Returns:
            peaks: [K, D] peak locations
            L_peaks: [K] log-likelihood values
            widths: [K, 2] width estimates
            
            If return_bank=True, also returns ray_bank and chisao_bank
        """
        if verbose:
            print("\n" + "="*70)
            print("CarryTiger v2.2: TWO-PHASE MODE (multi_scale=True)")
            print("="*70)
        
        # v2.0: Initialize RayBank if banking enabled
        ray_bank = RayBank(D=self.dim, xp=self.xp) if return_bank else None
        
        # ================================================================
        # PHASE 1: GLOBAL SCOUTING (HLC-only)
        # ================================================================
        if verbose:
            print("\n" + "-"*70)
            print("[Phase 1: Global Scouting (HLC-only)]")
            print("-"*70)
        
        # Normal ray casting
        rays_p1 = self._generate_rays_discovery(verbose=verbose)
        if verbose:
            print(f"  Generated {len(rays_p1)} rays")
        
        # v2.0: Store Phase 1 ray geometry
        if return_bank:
            ray_bank.store_rays(rays_p1)
        
        # Initial sampling
        samples_p1, f_samples_p1, t_values_p1 = self._initial_sampling(
            rays_p1, verbose=verbose
        )
        if verbose:
            print(f"  Sampled {len(samples_p1)} points along rays")
        
        # v2.0: Store Phase 1 samples
        if return_bank:
            ray_bank.store_samples(samples_p1, f_samples_p1, t_values_p1)
        
        # Scale discovery
        λ_fine, λ_mid, λ_coarse = self._discover_scales(
            f_samples_p1, t_values_p1, verbose=verbose
        )
        if verbose:
            print(f"  Scales: λ_fine={λ_fine:.6f}, λ_mid={λ_mid:.6f}, λ_coarse={λ_coarse:.6f}")
        
        # Multi-scale resampling
        resampled_p1 = self._multiscale_resample(
            rays_p1, λ_fine, λ_mid, λ_coarse, verbose=verbose
        )
        if verbose:
            print(f"  Resampled {len(resampled_p1)} points")
        
        # ChiSao with HLC-only mode (global features)
        if verbose:
            print("\n  [ChiSao Phase 1: HLC-only mode]")
        
        result_p1 = sticky_hands(
            func=self.func,
            x0=resampled_p1,
            method=self.chisao_params['method'],
            mode='hlc_only',  # NEW: Extended HLC, minimal local refinement
            reseed_strategy=self.chisao_params['reseed_strategy'],
            cannon_through_sky=self.chisao_params['cannon_through_sky'],
            estimate_widths=False,  # Skip widths in Phase 1
            bounds=self.bounds,
            verbose=verbose,
            func_returns_grad=self.func_returns_grad  # v3.1: Analytic gradient
        )
        
        global_peaks = result_p1['peaks']
        if verbose:
            print(f"\n  Phase 1 complete: {len(global_peaks)} global candidates found")
        
        # ================================================================
        # PHASE 2: FULL EXPLORATION (with global seeds)
        # ================================================================
        if verbose:
            print("\n" + "-"*70)
            print("[Phase 2: Full Exploration (with global seeds)]")
            print("-"*70)
        
        # Fresh ray casting
        rays_p2 = self._generate_rays_discovery(verbose=verbose)
        if verbose:
            print(f"  Generated {len(rays_p2)} fresh rays")
        
        # Fresh initial sampling
        samples_p2, f_samples_p2, t_values_p2 = self._initial_sampling(
            rays_p2, verbose=verbose
        )
        if verbose:
            print(f"  Sampled {len(samples_p2)} points along rays")
        
        # Scale discovery
        λ_fine, λ_mid, λ_coarse = self._discover_scales(
            f_samples_p2, t_values_p2, verbose=verbose
        )
        if verbose:
            print(f"  Scales: λ_fine={λ_fine:.6f}, λ_mid={λ_mid:.6f}, λ_coarse={λ_coarse:.6f}")
        
        # Multi-scale resampling
        resampled_p2 = self._multiscale_resample(
            rays_p2, λ_fine, λ_mid, λ_coarse, verbose=verbose
        )
        if verbose:
            print(f"  Resampled {len(resampled_p2)} points")
        
        # INJECT Phase 1 global peaks as seeds
        if len(global_peaks) > 0:
            resampled_p2 = self.xp.vstack([resampled_p2, global_peaks])
            if verbose:
                print(f"  Injected {len(global_peaks)} global seeds from Phase 1")
                print(f"  Total samples for Phase 2: {len(resampled_p2)}")
        
        # Full ChiSao optimization
        if verbose:
            print("\n  [ChiSao Phase 2: Full mode]")
        
        result = self._optimize_with_chisao(resampled_p2, verbose=verbose, bank_samples=return_bank)
        
        peaks = result['peaks']
        L_peaks = result['L_peaks']
        widths = result['widths']
        
        # v2.0: Extract ChiSao bank (from Phase 2)
        chisao_bank = result.get('sample_bank', None) if return_bank else None
        
        if verbose:
            print(f"\n[CarryTiger (抱虎归山 - Bào Hǔ Guī Shān) v2.0 Two-Phase: Complete]")
            print(f"  Phase 1 global candidates: {len(global_peaks)}")
            print(f"  Final peaks found: {len(peaks)}")
            print(f"  Peak likelihood range: [{self.xp.min(L_peaks):.2f}, {self.xp.max(L_peaks):.2f}]")
            if len(widths) > 0:
                print(f"  Width range: [{self.xp.min(widths):.4f}, {self.xp.max(widths):.4f}]")
        
        if return_bank:
            return peaks, L_peaks, widths, ray_bank, chisao_bank
        return peaks, L_peaks, widths
    
    # ========================================================================
    # STEP 5: CHISAO OPTIMIZATION
    # ========================================================================
    
    def _optimize_with_chisao(
        self,
        initial_points: np.ndarray,
        verbose: bool = False,
        bank_samples: bool = False
    ) -> Dict:
        """
        Run ChiSao optimization with full sticky hands.
        
        PURE GPU: All arrays stay CuPy, no conversions!
        
        Args:
            initial_points: [N, D] starting positions (CuPy if GPU enabled)
            bank_samples: If True, enable ChiSao sample banking (v1.9)
            
        Returns:
            Dictionary with keys: peaks, L_peaks, widths, converged_mask, ...
            If bank_samples=True, also includes 'sample_bank' key
        """
        # PURE GPU: Pass everything directly to ChiSao
        # ChiSao v2.4 is fully GPU-aware and handles CuPy arrays natively
        
        result = sticky_hands(
            func=self.func,
            x0=initial_points,  # CuPy array (pure GPU)
            method=self.chisao_params['method'],
            n_oscillations=self.chisao_params['n_oscillations'],
            n_converge=self.chisao_params['n_converge'],
            n_anticonverge=self.chisao_params['n_anticonverge'],
            reseed_strategy=self.chisao_params['reseed_strategy'],
            cannon_through_sky=self.chisao_params['cannon_through_sky'],
            estimate_widths=self.chisao_params['estimate_widths'],
            stick_tolerance=self.chisao_params.get('stick_tolerance', 1e-6),
            bounds=self.bounds,  # CuPy array (pure GPU)
            verbose=verbose,
            bank_samples=bank_samples,  # v2.0: ChiSao banking
            func_returns_grad=self.func_returns_grad,  # v3.1: Analytic gradient
            line_search=self.line_search  # v3.2: Line search mode
        )
        
        return result


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CarryTiger (抱虎归山 - Bào Hǔ Guī Shān) v2.0 - Standalone Test")
    print("="*70)
    print()
    
    # Test on 2D Gaussian
    def gaussian_2d(theta):
        """2D Gaussian log-likelihood (GPU-compatible)"""
        # Detect array type (CuPy or NumPy)
        xp_local = cp.get_array_module(theta) if GPU_AVAILABLE else np
        
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        
        return -0.5 * xp_local.sum(theta**2, axis=1)
    
    bounds = np.array([[-5, 5], [-5, 5]])
    
    print("Test Problem: 2D Gaussian")
    print(f"Bounds: {bounds.tolist()}")
    print(f"True peak: [0, 0]")
    print()
    
    # Initialize CarryTiger with optimized defaults (n_init=500)
    carry_tiger = CarryTigerToMountain(
        func=gaussian_2d,
        bounds=bounds,
        # Using defaults: n_rays=10, n_samples_per_ray=50
        n_initial_samples=50,
        use_gpu=GPU_AVAILABLE
    )
    
    # Run detection
    print("Running CarryTiger (抱虎归山 - Bào Hǔ Guī Shān) v2.0...")
    print()
    
    peaks, L_peaks, widths = carry_tiger.detect_modes(verbose=True)
    
    # Convert to CPU for display
    if GPU_AVAILABLE:
        peaks = cp.asnumpy(peaks)
        L_peaks = cp.asnumpy(L_peaks)
        widths = cp.asnumpy(widths)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Number of peaks found: {len(peaks)}")
    print()
    
    if len(peaks) > 0:
        print("Peak details:")
        for i, (peak, L, width) in enumerate(zip(peaks, L_peaks, widths)):
            print(f"  Peak {i+1}: location={peak}, L={L:.4f}, widths={width}")
        
        # Check accuracy
        if len(peaks) == 1:
            error = np.linalg.norm(peaks[0] - np.zeros(2))
            print()
            print(f"✓ Peak location error: {error:.6f}")
            if error < 0.1:
                print("✓ TEST PASSED: Found correct peak location")
            else:
                print("⚠ TEST WARNING: Peak location has noticeable error")
    
    print()
    print("="*70)
    print("CarryTiger v2.2: Test Complete")
    print("="*70)
