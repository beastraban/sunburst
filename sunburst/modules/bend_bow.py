#!/usr/bin/env python3
"""
BendTheBowShootTheTiger v4.7 - PRODUCTION
==========================================

GPU-accelerated Bayesian evidence calculation via Laplace + Importance Sampling correction.

Key Features:
1. SUNBURST from each peak - radial rays emanating outward
2. BISECT rays to find iso-likelihood crossings (log-spaced levels)
3. ANALYZE iso-curve shapes for Gaussianity diagnostic
4. LAPLACE + IMPORTANCE SAMPLING for heavy-tailed/skewed posteriors
5. ADAPTIVE HESSIAN: Detects rotated posteriors, computes full Hessian when needed

Evidence Methods:
    - laplace_full_hessian: For rotated/ill-conditioned posteriors (auto-detected)
    - laplace_offdiag: For correlated Gaussians
    - laplace_shell_corrected: Laplace + IS correction (default for non-Gaussian)

Importance Sampling Algorithm:
    1. Detect non-Gaussianity: tail_alpha < 1.8, curvature > 0.1, or asymmetry > 0.2
    2. Sample from wider Gaussian proposal (1.5-3.0× width based on tail heaviness)
    3. Weight samples by 1/q(x) to get unbiased volume estimate
    4. Z = E_q[L(x)/q(x)] via log-sum-exp
    5. Correction = log_Z_IS - log_Z_Laplace

Changes in v4.7:
    - IMPROVED: Rotation detection using Perpendicular Step Fraction in Whitened Space
      * Previous curvature-based method failed at high dimensions (32D+)
      * New method: whiten trajectories, check if steps point toward origin
      * For axis-aligned: steps are radial (perpendicular fraction < 1%)
      * For rotated: steps have tangential component (perpendicular fraction 15-80%)
      * Robust at ALL dimensions (tested 8D-128D with 48x separation at 128D)
      * Still ZERO extra likelihood evaluations - reuses TrajectoryBank data
    - FIXES: rotated_cigar now correctly detected at 32D, 64D, 128D

Changes in v4.6:
    - BUGFIX: Removed RayBank-based rotation detection (Strategy 1)
      * CarryTiger's rays are for peak FINDING, not from peaks
      * They traverse domain from boundaries (v2v, v2e, w2w) or prior box center
      * Using them as if from peaks caused false rotation detection (CV=0.85)
      * This was causing wrong full Hessian computation -> wrong evidence
    - KEPT: TrajectoryBank-based rotation detection (Strategy 2)
      * GreenDragon's L-BFGS trajectories DO emanate from peaks
      * Curved paths in whitened space correctly indicate rotation

Changes in v4.5:
    - BUGFIX: Rotation detection now ALWAYS includes adjacent pairs (i, i+1)
    - This fixes 80% error on rotated Gaussians where correlation was in (0,1) plane
    - Vectorized pair generation (no Python loops)
    - Adjacent pairs catch most real-world correlations between neighboring dimensions
    - NEW: Bank-based rotation detection using TrajectoryBank (ZERO extra evals!)

Changes in v4.4:
    - Replaced shell rejection with proper importance sampling
    - Uses diagnostics (tail_alpha, curvature, asymmetry) to trigger IS
    - Width factor adapts to tail heaviness: 3.0 for heavy, 2.0 for moderate, 1.5 for skew
    - Disabled off-diagonal probe for heavy tails (unreliable)

Performance (validated on 8D):
    - Gaussian: 0.00% error (Laplace exact)
    - Student-t ν=3: ~1-2% error (IS correction)
    - Skewed Gaussian: ~1-2% error (IS correction)

Author: SunBURST Development Team
Date: January 2026
Version: 4.6 PRODUCTION
"""

import numpy as np
from typing import Callable, Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
import time

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = np
    HAS_GPU = False

from scipy.special import gammaln as gammaln_cpu


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_xp(use_gpu: bool = True):
    """Get array module."""
    return cp if (use_gpu and HAS_GPU) else np


def to_cpu(x):
    """Convert to numpy."""
    if HAS_GPU and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def to_gpu(x):
    """Convert to cupy if available."""
    if HAS_GPU:
        return cp.asarray(x)
    return np.asarray(x)


# =============================================================================
# SUNBURST RAY GENERATION
# =============================================================================

def generate_sunburst_directions(d: int, n_rays: int, xp) -> 'xp.ndarray':
    """
    Generate uniformly distributed ray directions on unit sphere.
    
    Uses Gram-Schmidt orthonormalization for first d directions,
    then random directions for the rest.
    
    Args:
        d: Dimension
        n_rays: Number of rays to generate
        xp: Array module (numpy or cupy)
        
    Returns:
        directions: (n_rays, d) unit vectors
    """
    directions = []
    
    # First d directions: axis-aligned (±e_i)
    for i in range(min(d, n_rays // 2)):
        e_plus = xp.zeros(d, dtype=xp.float64)
        e_plus[i] = 1.0
        directions.append(e_plus)
        
        if len(directions) < n_rays:
            e_minus = xp.zeros(d, dtype=xp.float64)
            e_minus[i] = -1.0
            directions.append(e_minus)
    
    # Remaining: random directions
    n_remaining = n_rays - len(directions)
    if n_remaining > 0:
        random_dirs = xp.random.randn(n_remaining, d)
        random_dirs = random_dirs / xp.linalg.norm(random_dirs, axis=1, keepdims=True)
        for i in range(n_remaining):
            directions.append(random_dirs[i])
    
    return xp.stack(directions[:n_rays])


def generate_sunburst_rays(
    peak: 'xp.ndarray',
    bounds: 'xp.ndarray',
    n_rays: int = 100,
    xp = np
) -> Tuple['xp.ndarray', 'xp.ndarray']:
    """
    Generate sunburst rays from peak to boundary.
    
    Args:
        peak: (d,) peak location
        bounds: (d, 2) parameter bounds
        n_rays: Number of rays
        xp: Array module
        
    Returns:
        directions: (n_rays, d) unit direction vectors
        max_t: (n_rays,) maximum t before hitting boundary
    """
    d = len(peak)
    directions = generate_sunburst_directions(d, n_rays, xp)
    
    # Compute max_t for each ray (where it hits boundary) - VECTORIZED
    # peak + t * direction = bound
    # t = (bound - peak) / direction
    
    # For each direction component, compute t to hit lower and upper bound
    # t_lower[i,j] = (bounds[j,0] - peak[j]) / directions[i,j]
    # t_upper[i,j] = (bounds[j,1] - peak[j]) / directions[i,j]
    
    # Avoid division by zero
    dirs_safe = xp.where(xp.abs(directions) > 1e-10, directions, xp.inf)
    
    t_lower = (bounds[:, 0] - peak) / dirs_safe  # (n_rays, d)
    t_upper = (bounds[:, 1] - peak) / dirs_safe  # (n_rays, d)
    
    # We want positive t values (ray goes forward)
    # For positive direction: use t_upper
    # For negative direction: use t_lower
    t_hit = xp.where(directions > 1e-10, t_upper, 
                     xp.where(directions < -1e-10, t_lower, xp.inf))
    
    # max_t is the minimum positive t across all dimensions
    max_t = xp.min(t_hit, axis=1)
    
    # Clamp to reasonable range
    max_t = xp.clip(max_t, 0.1, 100.0)
    
    return directions, max_t


def evaluate_sunburst(
    log_L_func: Callable,
    peak: 'xp.ndarray',
    directions: 'xp.ndarray',
    max_t: 'xp.ndarray',
    n_samples_per_ray: int = 50,
    xp = np
) -> Tuple['xp.ndarray', 'xp.ndarray']:
    """
    Evaluate log-likelihood along sunburst rays.
    
    Args:
        log_L_func: Vectorized log-likelihood function
        peak: (d,) peak location
        directions: (n_rays, d) ray directions
        max_t: (n_rays,) maximum t per ray
        n_samples_per_ray: Samples per ray
        xp: Array module
        
    Returns:
        t_values: (n_rays, n_samples) parametric positions
        log_L: (n_rays, n_samples) log-likelihood values
    """
    n_rays = len(directions)
    d = len(peak)
    
    # Generate t values - VECTORIZED (no for loop!)
    # t_values[i, j] = 0.01 + (max_t[i] - 0.01) * j / (n_samples - 1)
    t_frac = xp.linspace(0, 1, n_samples_per_ray)  # (n_samples,)
    t_values = 0.01 + (max_t[:, None] - 0.01) * t_frac[None, :]  # (n_rays, n_samples)
    
    # Generate all sample points
    # points[i, j] = peak + t_values[i, j] * directions[i]
    points = peak + t_values[:, :, None] * directions[:, None, :]
    points_flat = points.reshape(-1, d)
    
    # Evaluate
    log_L_flat = log_L_func(points_flat)
    log_L = log_L_flat.reshape(n_rays, n_samples_per_ray)
    
    return t_values, to_cpu(log_L)


# =============================================================================
# ISO-LIKELIHOOD BISECTION
# =============================================================================

def define_iso_levels(log_L_peak: float, log_L_min: float, n_levels: int = 500) -> np.ndarray:
    """
    Define iso-likelihood levels for contour extraction.
    
    Log-spaced from log_L_peak down to log_L_min (prior boundary).
    Log spacing gives more resolution near the peak where the data is dense,
    and fewer levels in the far tails where rays hit the boundary.
    
    Args:
        log_L_peak: Log-likelihood at peak
        log_L_min: Minimum log-likelihood (at prior boundary)
        n_levels: Number of iso-levels (default 500)
        
    Returns:
        iso_levels: (n_levels,) threshold values in descending order
    """
    # Ensure we have a reasonable range
    max_drop = max(log_L_peak - log_L_min, 10.0)  # At least 10 log units
    
    # Log spacing from peak to prior boundary
    # More levels near peak (where shoulder effect is visible), fewer at boundary
    drops = np.logspace(np.log10(0.5), np.log10(max_drop), n_levels)
    return log_L_peak - drops


def find_brackets(
    t_values: np.ndarray,
    log_L: np.ndarray,
    iso_levels: np.ndarray
) -> np.ndarray:
    """
    Find t-intervals that bracket each iso-level crossing.
    
    FULLY VECTORIZED - no Python loops!
    
    Args:
        t_values: (n_rays, n_samples) parametric positions
        log_L: (n_rays, n_samples) log-likelihood values
        iso_levels: (n_levels,) threshold values (descending)
        
    Returns:
        brackets: (n_rays, n_levels, 2) with [t_lo, t_hi]
                  NaN if ray doesn't cross that level
    """
    n_rays, n_samples = log_L.shape
    n_levels = len(iso_levels)
    
    # Expand for broadcasting: log_L (n_rays, n_samples, 1) vs iso_levels (1, 1, n_levels)
    log_L_exp = log_L[:, :, np.newaxis]  # (n_rays, n_samples, 1)
    iso_exp = iso_levels[np.newaxis, np.newaxis, :]  # (1, 1, n_levels)
    
    # above[i, j, l] = True if log_L[i, j] >= iso_levels[l]
    above = log_L_exp >= iso_exp  # (n_rays, n_samples, n_levels)
    
    # Crossing: above at j but not at j+1 (going outward from peak)
    crossing = above[:, :-1, :] & ~above[:, 1:, :]  # (n_rays, n_samples-1, n_levels)
    
    # Find first crossing for each (ray, level) pair
    # Use argmax on crossing - returns first True index (or 0 if none)
    first_crossing_idx = np.argmax(crossing, axis=1)  # (n_rays, n_levels)
    
    # Check if there actually was a crossing (argmax returns 0 for all-False)
    has_crossing = np.any(crossing, axis=1)  # (n_rays, n_levels)
    
    # Build brackets
    brackets = np.full((n_rays, n_levels, 2), np.nan)
    
    # Get indices for advanced indexing
    ray_idx = np.arange(n_rays)[:, np.newaxis]  # (n_rays, 1)
    
    # Where we have crossings, fill in the brackets
    brackets[:, :, 0] = np.where(has_crossing, 
                                  t_values[ray_idx, first_crossing_idx],
                                  np.nan)
    brackets[:, :, 1] = np.where(has_crossing,
                                  t_values[ray_idx, first_crossing_idx + 1],
                                  np.nan)
    
    return brackets


def bisect_to_crossings(
    log_L_func: Callable,
    peak: np.ndarray,
    directions: np.ndarray,
    brackets: np.ndarray,
    iso_levels: np.ndarray,
    n_bisect: int = 10,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Bisect to find exact iso-likelihood crossing points.
    
    GPU-OPTIMIZED: Fully vectorized across all rays and levels simultaneously.
    
    Args:
        log_L_func: Vectorized log-likelihood function
        peak: (d,) peak location
        directions: (n_rays, d) ray directions
        brackets: (n_rays, n_levels, 2) initial brackets
        iso_levels: (n_levels,) threshold values
        n_bisect: Number of bisection steps
        use_gpu: Use GPU acceleration
        
    Returns:
        crossings: (n_rays, n_levels) t-values at crossings
                   NaN where no crossing
    """
    xp = get_xp(use_gpu)
    
    n_rays, n_levels, _ = brackets.shape
    d = len(peak)
    
    peak_gpu = xp.asarray(peak)
    directions_gpu = xp.asarray(directions)
    
    # Work with valid brackets only
    t_lo = xp.asarray(brackets[:, :, 0].copy())
    t_hi = xp.asarray(brackets[:, :, 1].copy())
    thresholds = xp.asarray(iso_levels)
    
    valid = ~xp.isnan(t_lo)
    
    for _ in range(n_bisect):
        t_mid = (t_lo + t_hi) / 2
        
        # Flatten all valid (ray, level) pairs for batch evaluation
        valid_mask = valid.flatten()
        n_valid = int(xp.sum(valid_mask))
        
        if n_valid == 0:
            break
        
        # Get indices of valid pairs
        ray_indices, level_indices = xp.where(valid)
        
        # Compute all midpoints at once: (n_valid, d)
        t_mid_valid = t_mid[ray_indices, level_indices]
        points = peak_gpu + t_mid_valid[:, None] * directions_gpu[ray_indices]
        
        # Single batched likelihood evaluation
        log_L_vals = log_L_func(points)
        
        # Compare against thresholds
        thresh_valid = thresholds[level_indices]
        above = log_L_vals >= thresh_valid
        
        # Update brackets using advanced indexing
        # If above threshold, crossing is further out
        t_lo[ray_indices[above], level_indices[above]] = t_mid_valid[above]
        # If below threshold, crossing is closer in
        t_hi[ray_indices[~above], level_indices[~above]] = t_mid_valid[~above]
    
    # Return midpoint as best estimate
    crossings = to_cpu((t_lo + t_hi) / 2)
    crossings[~to_cpu(valid)] = np.nan
    
    return crossings


# =============================================================================
# GAUSSIANITY DIAGNOSTICS
# =============================================================================

def estimate_covariance_from_radii(
    crossings: np.ndarray,
    directions: np.ndarray,
    iso_levels: np.ndarray,
    log_L_peak: float,
    d: int
) -> Tuple[np.ndarray, float, bool]:
    """
    Estimate covariance matrix from iso-likelihood contour radii.
    
    For an ellipsoid at likelihood level L:
        log_L = log_L_peak - 0.5 * x^T Σ^{-1} x
    
    At iso-level with drop Δ = log_L_peak - L:
        x^T Σ^{-1} x = 2Δ
    
    For a ray in direction v, the crossing point is x = r*v, so:
        r² * v^T Σ^{-1} v = 2Δ
        v^T Σ^{-1} v = 2Δ / r²
    
    With multiple directions, we can estimate the diagonal of Σ^{-1},
    and detect if there's significant off-diagonal structure.
    
    Args:
        crossings: (n_rays, n_levels) t-values at crossings (= radii from peak)
        directions: (n_rays, d) unit ray directions  
        iso_levels: (n_levels,) log-likelihood levels
        log_L_peak: log-likelihood at peak
        d: dimension
        
    Returns:
        diag_precision: (d,) estimated diagonal of precision matrix
        ellipticity: scalar measure of how non-spherical (0 = sphere)
        has_correlation: bool, True if significant off-diagonal structure
    """
    n_rays, n_levels = crossings.shape
    
    # Use a middle iso-level where we have good data
    # (not too close to peak, not too far in tails)
    level_idx = min(n_levels // 4, 10)  # ~25th percentile or 10th level
    
    # Get radii and directions at this level
    radii_at_level = crossings[:, level_idx]
    valid = ~np.isnan(radii_at_level) & (radii_at_level > 0)
    
    if np.sum(valid) < d:
        # Not enough data, return default
        return np.ones(d), 0.0, False
    
    radii = radii_at_level[valid]
    dirs = directions[valid]  # (n_valid, d)
    
    # Drop at this level
    delta = log_L_peak - iso_levels[level_idx]
    if delta <= 0:
        return np.ones(d), 0.0, False
    
    # For each direction v with radius r:
    # v^T Σ^{-1} v = 2Δ / r²
    # 
    # If Σ^{-1} = diag(λ), then:
    # sum_i λ_i v_i² = 2Δ / r²
    #
    # We can estimate λ by least squares
    
    # Build system: A @ λ = b where A[k,i] = v_k[i]², b[k] = 2Δ/r_k²
    A = dirs ** 2  # (n_valid, d)
    b = 2 * delta / (radii ** 2)  # (n_valid,)
    
    # Least squares solve for diagonal precision
    try:
        diag_precision, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Ensure positive
        diag_precision = np.maximum(diag_precision, 1e-10)
        
        # Compute ellipticity: std(λ) / mean(λ)
        ellipticity = float(np.std(diag_precision) / np.mean(diag_precision))
        
        # Check for correlation by looking at residuals
        # Large residuals indicate off-diagonal structure
        predicted = A @ diag_precision
        rss = np.sum((b - predicted)**2)
        tss = np.sum((b - np.mean(b))**2)
        r_squared = 1 - rss / tss if tss > 0 else 1.0
        
        # If diagonal model doesn't fit well (R² < 0.9), there's correlation
        has_correlation = r_squared < 0.9
        
        return diag_precision, ellipticity, has_correlation
        
    except Exception:
        return np.ones(d), 0.0, False


def analyze_iso_curves(
    crossings: np.ndarray,
    directions: np.ndarray,
    peak: np.ndarray,
    H_diag: np.ndarray
) -> Dict:
    """
    Analyze iso-likelihood curves for Gaussianity.
    
    For a Gaussian in whitened space:
    - All iso-curves are SPHERES
    - Radii at each level should be equal in all directions
    
    For non-Gaussian:
    - Iso-curves are NOT spheres
    - Radii vary with direction
    - Shape changes with level (e.g., heavy tails)
    
    Args:
        crossings: (n_rays, n_levels) t-values at crossings
        directions: (n_rays, d) ray directions
        peak: (d,) peak location
        H_diag: (d,) diagonal Hessian
        
    Returns:
        diagnostics dict with:
        - radii_per_level: (n_levels,) mean radius at each level
        - radii_std_per_level: (n_levels,) std of radii (0 for Gaussian)
        - is_gaussian: bool
        - tail_alpha: estimated tail heaviness
    """
    n_rays, n_levels = crossings.shape
    d = len(peak)
    
    # Whitening scale from Hessian
    H_diag_safe = np.minimum(H_diag, -1e-10)
    sigma = np.sqrt(-1.0 / H_diag_safe)
    
    # Compute whitened radii at each crossing
    # crossing point = peak + t * direction
    # whitened distance = ||(crossing - peak) / sigma||
    
    radii = np.zeros((n_rays, n_levels))
    
    # VECTORIZED radii computation
    # crossing_point[i,l] = crossings[i,l] * directions[i] (relative to peak)
    # whitened[i,l] = crossing_point[i,l] / sigma
    # radii[i,l] = ||whitened[i,l]||
    
    # crossings: (n_rays, n_levels), directions: (n_rays, d), sigma: (d,)
    # crossing_points: (n_rays, n_levels, d)
    crossing_points = crossings[:, :, np.newaxis] * directions[:, np.newaxis, :]
    
    # whitened: (n_rays, n_levels, d)
    whitened = crossing_points / sigma[np.newaxis, np.newaxis, :]
    
    # radii: (n_rays, n_levels) = norm over d dimension
    radii = np.sqrt(np.sum(whitened**2, axis=2))
    
    # Where crossings were NaN, radii should be NaN
    radii = np.where(np.isnan(crossings), np.nan, radii)
    
    # Statistics per level
    radii_mean = np.nanmean(radii, axis=0)
    radii_std = np.nanstd(radii, axis=0)
    
    # Relative std (normalized by mean)
    with np.errstate(divide='ignore', invalid='ignore'):
        radii_std_ratio = radii_std / radii_mean
        radii_std_ratio = np.nan_to_num(radii_std_ratio, nan=0.0)
    
    # Gaussianity check
    # For Gaussian: radii should be constant across directions at each level
    max_std_ratio = np.nanmax(radii_std_ratio)
    is_gaussian_shape = max_std_ratio < 0.3
    
    # Tail behavior: compare inner vs outer radii
    # For Gaussian: radii grow as sqrt(2 * delta_logL)
    # For heavy tails: radii grow faster
    
    # Expected radii for Gaussian at each level
    # If log_L = log_L_peak - delta, then r² = 2 * delta (in whitened space)
    # So r = sqrt(2 * delta)
    
    # Compare actual vs expected
    tail_alpha = 2.0  # Default to Gaussian
    
    if n_levels >= 3 and np.sum(~np.isnan(radii_mean)) >= 3:
        # Fit: log(r) vs log(delta) should have slope 0.5 for Gaussian
        valid_levels = ~np.isnan(radii_mean) & (radii_mean > 0)
        n_valid = np.sum(valid_levels)
        
        if n_valid >= 3:
            # delta = level drop (0.5, 1, 2, 4, ...)
            deltas = 0.5 * (2.0 ** np.arange(n_levels))
            
            log_r = np.log(radii_mean[valid_levels])
            log_delta = np.log(deltas[valid_levels])
            
            # Linear fit with error checking
            try:
                slope, intercept = np.polyfit(log_delta, log_r, 1)
                
                # Sanity check: slope should be positive and reasonable
                if 0.1 < slope < 2.0:
                    # For Gaussian: slope ≈ 0.5, so tail_alpha = 1/0.5 = 2.0
                    # For heavy tails: slope > 0.5, so tail_alpha < 2.0
                    tail_alpha = 1.0 / slope
                else:
                    # Unreasonable slope, default to Gaussian
                    tail_alpha = 2.0
                    
                # Additional check: compute R² to see if fit is good
                predicted = slope * log_delta + intercept
                ss_res = np.sum((log_r - predicted)**2)
                ss_tot = np.sum((log_r - np.mean(log_r))**2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                # If fit is poor (R² < 0.8), don't trust tail_alpha
                if r_squared < 0.8:
                    tail_alpha = 2.0  # Default to Gaussian when uncertain
                    
            except (np.linalg.LinAlgError, ValueError):
                tail_alpha = 2.0
        
        # ADDITIONAL CHECK: Compare actual radii to Gaussian prediction
        # For a Gaussian, r_whitened = sqrt(2 * delta)
        # If actual radii are LARGER than predicted, we have heavy tails
        if n_valid >= 2:
            deltas = 0.5 * (2.0 ** np.arange(n_levels))
            expected_radii = np.sqrt(2 * deltas)  # Gaussian prediction
            
            # Compare at outer levels (where tails matter most)
            outer_levels = valid_levels & (deltas >= 4.0)  # delta >= 4 (~2 sigma)
            if np.sum(outer_levels) >= 2:
                actual_outer = radii_mean[outer_levels]
                expected_outer = expected_radii[outer_levels]
                
                # Ratio > 1 means heavier tails than Gaussian
                ratio = np.mean(actual_outer / expected_outer)
                
                # If radii are >20% larger than Gaussian prediction, mark as heavy-tailed
                if ratio > 1.2:
                    tail_alpha = min(tail_alpha, 1.4)  # Force heavy-tail detection
    
    is_light_tails = tail_alpha >= 1.7
    
    # ASYMMETRY CHECK: Compare radii in opposite directions
    # For symmetric distribution (Gaussian), r(dir) ≈ r(-dir)
    # For shoulder/bulge, r(dir) >> r(-dir) in some direction
    
    asymmetry = 0.0
    
    if n_rays >= 10:
        # For each ray, find the ray closest to opposite direction
        # Compute dot products: directions @ directions.T
        # Opposite ray has most negative dot product
        dot_products = directions @ directions.T  # (n_rays, n_rays)
        
        # For each ray i, find ray j with most negative dot product (closest to -dir[i])
        opposite_idx = np.argmin(dot_products, axis=1)  # (n_rays,)
        
        # Compare radii at middle iso-levels (not too close to peak, not at boundary)
        mid_levels = slice(n_levels // 4, 3 * n_levels // 4)
        
        asymmetry_per_ray = np.zeros(n_rays)
        for i in range(n_rays):
            j = opposite_idx[i]
            
            # Get radii for ray i and its opposite j at middle levels
            r_i = radii[i, mid_levels]
            r_j = radii[j, mid_levels]
            
            # Filter valid (non-nan) pairs
            valid = ~np.isnan(r_i) & ~np.isnan(r_j) & (r_i > 0) & (r_j > 0)
            if np.sum(valid) >= 5:
                # Asymmetry = |r_i - r_j| / (r_i + r_j)
                # 0 = symmetric, 1 = completely asymmetric
                asym = np.abs(r_i[valid] - r_j[valid]) / (r_i[valid] + r_j[valid])
                asymmetry_per_ray[i] = np.mean(asym)
        
        # Take max asymmetry across all ray pairs
        asymmetry = float(np.max(asymmetry_per_ray))
    
    # Asymmetry > 0.1 means shoulder/bulge present
    is_symmetric = asymmetry < 0.1
    
    return {
        'radii_mean': radii_mean,
        'radii_std': radii_std,
        'radii_std_ratio': radii_std_ratio,
        'max_std_ratio': max_std_ratio,
        'is_gaussian_shape': is_gaussian_shape,
        'tail_alpha': tail_alpha,
        'is_light_tails': is_light_tails,
        'asymmetry': asymmetry,
        'is_symmetric': is_symmetric,
        'is_gaussian': is_gaussian_shape and is_light_tails and is_symmetric,
    }


def bisect_bank_rays(
    log_L_func: Callable,
    ray_bank,
    iso_levels: np.ndarray,
    n_bisect: int = 10,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bisect M1 bank rays to find iso-likelihood crossings along GLOBAL rays.
    
    Unlike peak-centered sunburst rays, M1 rays traverse the entire domain
    (vertex-to-vertex, wall-to-wall, etc.). Bisection on these rays reveals
    iso-likelihood contours around ALL modes, not just the ones we found.
    
    This is the KEY to detecting modes missed during M1 optimization.
    
    Args:
        log_L_func: Vectorized log-likelihood function
        ray_bank: RayBank with ray_starts, ray_ends, f_samples
        iso_levels: (n_levels,) threshold values (descending)
        n_bisect: Number of bisection steps
        use_gpu: Use GPU acceleration
        
    Returns:
        crossings: (n_rays, n_levels) t-values at crossings (t ∈ [0,1])
        crossing_points: (n_crossings, d) actual positions of crossings
        crossing_log_L: (n_crossings,) log-likelihood at crossings
    """
    xp = get_xp(use_gpu)
    
    if ray_bank is None or ray_bank.ray_starts is None:
        return np.array([]), np.array([]), np.array([])
    
    # Get ray geometry from bank
    ray_starts = to_cpu(ray_bank.ray_starts)  # (n_rays, d)
    ray_ends = to_cpu(ray_bank.ray_ends)      # (n_rays, d)
    f_samples = to_cpu(ray_bank.f_samples)    # (n_rays, n_samples_per_ray)
    t_values_bank = to_cpu(ray_bank.t_values) # (n_samples_per_ray,)
    
    n_rays, d = ray_starts.shape
    n_levels = len(iso_levels)
    n_samples_per_ray = len(t_values_bank)
    
    # Convert to (n_rays, n_samples) format for find_brackets
    t_values_2d = np.tile(t_values_bank, (n_rays, 1))  # (n_rays, n_samples)
    
    # Find brackets using existing function
    brackets = find_brackets(t_values_2d, f_samples, iso_levels)
    
    # Convert rays to origin + direction format
    # point = ray_start + t * (ray_end - ray_start), t ∈ [0, 1]
    origins = ray_starts
    directions = ray_ends - ray_starts  # NOT normalized - t ∈ [0, 1]
    
    origins_gpu = xp.asarray(origins)
    directions_gpu = xp.asarray(directions)
    
    # Work with valid brackets only
    t_lo = xp.asarray(brackets[:, :, 0].copy())
    t_hi = xp.asarray(brackets[:, :, 1].copy())
    thresholds = xp.asarray(iso_levels)
    
    valid = ~xp.isnan(t_lo)
    
    for _ in range(n_bisect):
        t_mid = (t_lo + t_hi) / 2
        
        # Flatten all valid (ray, level) pairs for batch evaluation
        valid_mask = valid.flatten()
        n_valid = int(xp.sum(valid_mask))
        
        if n_valid == 0:
            break
        
        # Get indices of valid pairs
        ray_indices, level_indices = xp.where(valid)
        
        # Compute all midpoints at once: (n_valid, d)
        t_mid_valid = t_mid[ray_indices, level_indices]
        points = origins_gpu[ray_indices] + t_mid_valid[:, None] * directions_gpu[ray_indices]
        
        # Single batched likelihood evaluation
        log_L_vals = log_L_func(points)
        
        # Compare against thresholds
        thresh_valid = thresholds[level_indices]
        above = log_L_vals >= thresh_valid
        
        # Update brackets
        t_lo[ray_indices[above], level_indices[above]] = t_mid_valid[above]
        t_hi[ray_indices[~above], level_indices[~above]] = t_mid_valid[~above]
    
    # Return midpoint as best estimate
    crossings = to_cpu((t_lo + t_hi) / 2)
    crossings[~to_cpu(valid)] = np.nan
    
    # Compute actual crossing points for valid crossings
    valid_crossings = ~np.isnan(crossings)
    crossing_points_list = []
    crossing_log_L_list = []
    
    for i in range(n_rays):
        for l in range(n_levels):
            if valid_crossings[i, l]:
                t = crossings[i, l]
                point = origins[i] + t * directions[i]
                crossing_points_list.append(point)
                crossing_log_L_list.append(iso_levels[l])
    
    if len(crossing_points_list) > 0:
        crossing_points = np.array(crossing_points_list)
        crossing_log_L = np.array(crossing_log_L_list)
    else:
        crossing_points = np.zeros((0, d))
        crossing_log_L = np.array([])
    
    return crossings, crossing_points, crossing_log_L


def analyze_raybank_tails(
    ray_bank,
    peak: np.ndarray,
    H_diag: np.ndarray,
    log_L_peak: float
) -> Dict:
    """
    Analyze tail behavior using RayBank samples (FREE - already computed).
    
    For each sample in RayBank:
    1. Compute whitened distance from peak
    2. Compare log_L vs distance relationship
    
    Gaussian: log_L = log_L_peak - 0.5 * r_whitened^2
    Student-t: log_L decays slower (heavier tails)
    
    Args:
        ray_bank: RayBank with samples and log_L
        peak: (d,) peak location  
        H_diag: (d,) diagonal Hessian
        log_L_peak: log-likelihood at peak
        
    Returns:
        diagnostics dict with tail_alpha
    """
    if ray_bank is None or ray_bank.samples is None:
        return {'tail_alpha': 2.0, 'has_raybank': False}
    
    # Get samples and log_L from bank
    samples = to_cpu(ray_bank.samples)  # (n_samples, d)
    log_L = to_cpu(ray_bank.log_L) if hasattr(ray_bank, 'log_L') and ray_bank.log_L is not None else to_cpu(ray_bank.f_samples.flatten())
    
    if len(samples) == 0:
        return {'tail_alpha': 2.0, 'has_raybank': False}
    
    # Whitening scale from Hessian
    H_diag_safe = np.minimum(H_diag, -1e-10)
    sigma = np.sqrt(-1.0 / H_diag_safe)
    
    # Compute whitened distance from peak for each sample
    diff = samples - peak  # (n_samples, d)
    whitened = diff / sigma  # (n_samples, d)
    r_whitened = np.linalg.norm(whitened, axis=1)  # (n_samples,)
    
    # Compute log_L drop from peak
    delta_log_L = log_L_peak - log_L  # (n_samples,)
    
    # Filter to valid samples:
    # 1. Positive drop from this peak's log_L
    # 2. Not too close to peak (r > 0.1)
    # 3. Not too far from peak (r < 10) - avoids samples near other peaks
    # 4. Finite values
    valid = (delta_log_L > 0.5) & (r_whitened > 0.1) & (r_whitened < 10.0) & np.isfinite(delta_log_L) & np.isfinite(r_whitened)
    
    if np.sum(valid) < 10:
        return {'tail_alpha': 2.0, 'has_raybank': True, 'n_valid': int(np.sum(valid))}
    
    r_valid = r_whitened[valid]
    delta_valid = delta_log_L[valid]
    
    # For Gaussian: delta_log_L = 0.5 * r^2, so log(delta) = log(0.5) + 2*log(r)
    # slope of log(delta) vs log(r) should be 2.0 for Gaussian
    # For Student-t: slope < 2.0 (delta grows slower with r)
    
    log_r = np.log(r_valid)
    log_delta = np.log(delta_valid)
    
    try:
        slope, intercept = np.polyfit(log_r, log_delta, 1)
        
        # tail_alpha = slope (should be 2.0 for Gaussian, <2.0 for heavy tails)
        tail_alpha = float(slope)
        
        # Compute R^2
        predicted = slope * log_r + intercept
        ss_res = np.sum((log_delta - predicted)**2)
        ss_tot = np.sum((log_delta - np.mean(log_delta))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
    except (np.linalg.LinAlgError, ValueError):
        tail_alpha = 2.0
        r_squared = 0.0
    
    return {
        'tail_alpha': tail_alpha,
        'has_raybank': True,
        'n_valid': int(np.sum(valid)),
        'r_squared': r_squared
    }


def compute_trajectory_curvature(
    trajectory_bank: Any,
    peak: np.ndarray,
    peak_idx: int = 0,
    verbose: bool = False
) -> float:
    """
    Compute curvature of L-BFGS trajectories.
    
    Straight paths → Gaussian
    Curved paths → Non-Gaussian (banana, etc.)
    """
    if trajectory_bank is None or trajectory_bank.seed_positions is None:
        return 0.0
    
    seed_positions = to_cpu(trajectory_bank.seed_positions)
    final_positions = to_cpu(trajectory_bank.final_positions)
    
    # Get trajectories that CONVERGED TO this peak (not just started near it)
    # A trajectory converged to this peak if final position is within 0.1 of peak
    final_distances = np.linalg.norm(final_positions - peak, axis=1)
    mask = (final_distances < 0.1)
    n_trajs = np.sum(mask)
    
    if verbose:
        print(f"      [CurvDebug] peak={peak}, n_trajs converged={n_trajs}")
    
    if n_trajs < 3:
        return 0.0
    
    starts = seed_positions[mask]
    ends = final_positions[mask]
    
    if verbose:
        print(f"      [CurvDebug] starts range: {starts.min(axis=0)} to {starts.max(axis=0)}")
        print(f"      [CurvDebug] ends range: {ends.min(axis=0)} to {ends.max(axis=0)}")
    
    # Compute spread of endpoints (should all be at peak for Gaussian)
    end_spread = np.std(ends, axis=0)
    mean_spread = np.mean(end_spread)
    
    # Normalize by average path length
    distances = np.linalg.norm(ends - starts, axis=1)
    mean_dist = np.mean(distances)
    
    if verbose:
        print(f"      [CurvDebug] end_spread={end_spread}, mean_spread={mean_spread:.6f}")
        print(f"      [CurvDebug] mean_dist={mean_dist:.4f}")
    
    if mean_dist < 1e-10:
        return 0.0
    
    curvature = mean_spread / mean_dist
    return float(min(1.0, curvature))


# =============================================================================
# EVIDENCE CALCULATION METHODS
# =============================================================================

def probe_offdiag_hessian(
    log_L_func: Callable,
    peak: np.ndarray,
    H_diag: np.ndarray,
    logL_peak: float,
    h: float = 1e-5,
    use_gpu: bool = True
) -> Tuple[float, bool]:
    """
    Probe off-diagonal Hessian structure with O(1) extra evaluations.
    
    For uniform correlation: H = aI + b(J-I) where J = ones matrix
    
    We probe the all-ones direction to get:
        v^T H v = a + (d-1)*b  where v = [1,1,...,1]/sqrt(d)
    
    Since we have H_diag (diagonal = a), we can solve for b:
        b = (v^T H v - a) / (d-1)
    
    Args:
        log_L_func: Log-likelihood function
        peak: Peak location
        H_diag: Diagonal Hessian (already computed)
        logL_peak: Log-likelihood at peak
        h: Finite difference step size
        use_gpu: Use GPU
        
    Returns:
        b: Off-diagonal Hessian value (0 if no correlation)
        has_correlation: True if significant off-diagonal structure detected
    """
    xp = get_xp(use_gpu)
    d = len(peak)
    
    if d < 2:
        return 0.0, False
    
    # Mean diagonal (= a for uniform structure)
    a = float(np.mean(H_diag))
    
    # Probe all-ones direction: v = [1,1,...,1] / sqrt(d)
    peak_gpu = xp.asarray(peak)
    v = xp.ones(d, dtype=xp.float64) / np.sqrt(d)
    
    # v^T H v via finite differences: (f(x+hv) - 2f(x) + f(x-hv)) / h^2
    # Batch both evaluations into ONE call
    points = xp.stack([peak_gpu + h * v, peak_gpu - h * v])
    f_vals = to_cpu(log_L_func(points))
    f_plus, f_minus = f_vals[0], f_vals[1]
    vHv = (f_plus - 2 * logL_peak + f_minus) / (h * h)
    
    # Solve for b: v^T H v = a + (d-1)*b
    b = (vHv - a) / (d - 1) if d > 1 else 0.0
    
    # Detect significant correlation: |b| should be comparable to |a|
    # For uniform correlation ρ, the ratio |b/a| ≈ ρ / (1 + (d-1)*ρ)
    # At high d, this scales as ~1/d, so threshold must decrease with dimension
    # Threshold: 0.1 at d=8, scaling as 0.1 * 8 / d = 0.8 / d
    threshold = 0.8 / d
    has_correlation = abs(b) > threshold * abs(a) if abs(a) > 1e-10 else False
    
    return b, has_correlation


def laplace_evidence_with_offdiag(
    log_L_peak: float,
    H_diag: np.ndarray,
    d: int,
    b: float = 0.0
) -> Tuple[float, Dict]:
    """
    Laplace approximation with uniform off-diagonal correction.
    
    For H = aI + b(J-I), the eigenvalues are:
        λ₁ = a + (d-1)*b  (multiplicity 1, eigenvector [1,1,...,1])
        λ₂ = a - b        (multiplicity d-1)
    
    log|det(-H)| = log(-λ₁) + (d-1)*log(-λ₂)
    
    Args:
        log_L_peak: Log-likelihood at peak
        H_diag: Diagonal Hessian
        d: Dimension
        b: Off-diagonal Hessian value (0 for diagonal-only)
        
    Returns:
        log_Z: Evidence estimate
        info: Dictionary with method info
    """
    a = float(np.mean(H_diag))
    
    # Eigenvalues of H = aI + b(J-I)
    lambda1 = a + (d - 1) * b
    lambda2 = a - b
    
    # Check negative definiteness
    if lambda1 >= 0 or lambda2 >= 0:
        # Fall back to diagonal-only if off-diagonal makes H non-negative-definite
        H_diag_safe = np.minimum(H_diag, -1e-10)
        log_det_neg_H = np.sum(np.log(-H_diag_safe))
        log_Z = log_L_peak + (d / 2) * np.log(2 * np.pi) - 0.5 * log_det_neg_H
        return log_Z, {'method': 'laplace', 'offdiag_fallback': True}
    
    # log|det(-H)| = log(-λ₁) + (d-1)*log(-λ₂)
    log_det_neg_H = np.log(-lambda1) + (d - 1) * np.log(-lambda2)
    
    log_Z = log_L_peak + (d / 2) * np.log(2 * np.pi) - 0.5 * log_det_neg_H
    
    return log_Z, {'method': 'laplace_offdiag', 'a': a, 'b': b, 
                   'lambda1': lambda1, 'lambda2': lambda2}


def laplace_evidence(log_L_peak: float, H_diag: np.ndarray, d: int) -> Tuple[float, Dict]:
    """
    Laplace approximation for evidence (diagonal Hessian only).
    
    log Z = log L(peak) + (d/2) * log(2π) - (1/2) * log|det(-H)|
    """
    H_diag_safe = np.minimum(H_diag, -1e-10)
    log_det_neg_H = np.sum(np.log(-H_diag_safe))
    
    log_Z = log_L_peak + (d / 2) * np.log(2 * np.pi) - 0.5 * log_det_neg_H
    
    return log_Z, {'method': 'laplace'}


def laplace_evidence_full_hessian(log_L_peak: float, H_full: np.ndarray, d: int, use_gpu: bool = True) -> Tuple[float, Dict]:
    """
    Laplace approximation using full Hessian matrix.
    
    For rotated/ill-conditioned posteriors where diagonal Hessian fails.
    GPU-accelerated eigendecomposition.
    
    log Z = log L(peak) + (d/2) * log(2π) - (1/2) * log|det(-H)|
    """
    xp = get_xp(use_gpu)
    
    # Eigendecompose -H on GPU (should be positive definite at maximum)
    H_gpu = xp.asarray(-H_full)
    eigvals = xp.linalg.eigvalsh(H_gpu)
    
    # Ensure positive (numerical stability)
    n_nonpositive = int(xp.sum(eigvals <= 0))
    eigvals = xp.maximum(eigvals, 1e-10)
    
    log_det_neg_H = float(xp.sum(xp.log(eigvals)))
    
    log_Z = log_L_peak + (d / 2) * np.log(2 * np.pi) - 0.5 * log_det_neg_H
    
    return log_Z, {
        'method': 'laplace_full_hessian',
        'eigval_min': float(to_cpu(eigvals.min())),
        'eigval_max': float(to_cpu(eigvals.max())),
        'n_nonpositive_clipped': n_nonpositive
    }


def shell_rejection_sampling(
    log_L_func: Callable,
    peak: np.ndarray,
    H_diag: np.ndarray,
    log_L_peak: float,
    n_levels: int = 50,
    n_samples: int = None,
    width_factor: float = 1.1,
    use_gpu: bool = True,
    tail_alpha: float = 2.0,
    curvature: float = 0.0,
    asymmetry: float = 0.0,
    crossings: np.ndarray = None,
    directions: np.ndarray = None,
    iso_levels: np.ndarray = None
) -> Tuple[float, Dict]:
    """
    Evidence calculation via layer cake from crossings OR importance sampling fallback.
    
    PRIMARY METHOD: Layer cake using existing crossings
    - Uses iso-curve radii to estimate volume at each likelihood level
    - Z = Σ L_j × ΔV_j where ΔV_j is shell volume
    - No new samples needed!
    
    FALLBACK: Importance sampling with Gaussian proposal
    - Used when crossings unavailable or insufficient
    
    Args:
        log_L_func: Log-likelihood function
        peak: Peak location (mode)
        H_diag: Diagonal Hessian at peak (negative definite)
        log_L_peak: Log-likelihood at peak
        n_levels: Unused, kept for API compatibility
        n_samples: Number of samples for IS fallback
        width_factor: Multiplier for proposal width
        use_gpu: Use GPU acceleration
        tail_alpha: Tail decay exponent from diagnostics (2.0 = Gaussian)
        curvature: Trajectory curvature from diagnostics
        asymmetry: Asymmetry measure from diagnostics (0 = symmetric)
        crossings: Iso-curve crossing distances (n_rays, n_iso_levels)
        directions: Ray directions (n_rays, d)
        iso_levels: Log-likelihood values at each iso-level
        crossings: Iso-curve crossing distances (n_rays, n_iso_levels)
        directions: Ray directions (n_rays, d)
        
    Returns:
        log_Z_correction: Correction to Laplace evidence
        info: Dict with statistics
    """
    xp = get_xp(use_gpu)
    d = len(peak)
    
    # === Decide if correction is needed ===
    # Skip correction for Gaussian tails regardless of asymmetry
    # Asymmetry in multimodal cases is handled by summing modes, not per-mode correction
    is_gaussian = (tail_alpha >= 1.8) and (curvature < 0.1)
    
    if is_gaussian:
        return 0.0, {
            'method': 'importance_sampling_skipped',
            'reason': 'gaussian_detected',
            'tail_alpha': tail_alpha,
            'curvature': curvature,
            'asymmetry': asymmetry,
            'correction_applied': False
        }
    
    # === Set base width based on tail heaviness ===
    if tail_alpha < 1.5:
        w = 3.0  # Very heavy tails
    elif tail_alpha < 1.8:
        w = 2.0  # Moderate heavy tails
    else:
        w = 1.5  # Near-Gaussian with curvature/asymmetry
    
    # Scale samples with dimension
    if n_samples is None:
        n_samples = min(500000, max(50000, 2000 * d))
    
    # Whitening transform from Hessian (baseline)
    H_diag_gpu = xp.asarray(H_diag)
    H_diag_safe = xp.minimum(H_diag_gpu, -1e-10)
    sigma_hessian = xp.sqrt(-1.0 / H_diag_safe)  # Laplace width from Hessian
    peak_gpu = xp.asarray(peak)
    
    # === PROPOSAL FROM HESSIAN (isotropic in whitened space) ===
    use_anisotropic = False
    use_asymmetric_shift = False
    use_layer_cake = False
    
    # === TRY LAYER CAKE FROM EXISTING CROSSINGS ===
    # We already have crossings at multiple iso-levels - use them directly!
    # No new samples needed.
    # NOTE: Only works for symmetric distributions. Asymmetric ones need IS.
    
    if asymmetry < 0.3 and crossings is not None and directions is not None and iso_levels is not None and len(crossings) > 0:
        try:
            crossings_np = to_cpu(crossings)
            iso_levels_np = to_cpu(iso_levels) if hasattr(iso_levels, '__array__') else np.array(iso_levels)
            n_rays, n_iso = crossings_np.shape
            
            # For each iso-level, estimate enclosed volume from crossing radii
            # Volume ∝ (geometric mean of radii)^d
            # In log space: log(V) = d × mean(log(radii)) + const
            
            log_volumes = np.full(n_iso, np.nan)
            
            for level_idx in range(n_iso):
                radii = crossings_np[:, level_idx]
                valid_rays = ~np.isnan(radii) & (radii > 1e-10)
                
                if np.sum(valid_rays) >= max(d // 2, 5):
                    # Geometric mean of radii as effective radius
                    log_radii = np.log(radii[valid_rays])
                    log_r_eff = np.mean(log_radii)
                    
                    # log(Volume) = d × log(r_eff) + const
                    # const = log(V_d) where V_d is d-ball volume coefficient
                    # V_d = π^(d/2) / Γ(d/2 + 1)
                    from scipy.special import gammaln
                    log_V_d = (d/2) * np.log(np.pi) - gammaln(d/2 + 1)
                    log_volumes[level_idx] = log_V_d + d * log_r_eff
            
            # Need at least a few valid levels
            valid_levels = ~np.isnan(log_volumes)
            if np.sum(valid_levels) >= 5:
                # Shell volumes: ΔV_j = V_j - V_{j+1}
                # Levels go from high (near peak) to low (in tails)
                # So V increases with level index (larger iso-surfaces as we go out)
                
                # Layer cake: Z = Σ L_j × ΔV_j
                # where L_j is likelihood at level j (midpoint between j and j+1)
                
                log_Z_terms = []
                
                for j in range(n_iso - 1):
                    if np.isnan(log_volumes[j]) or np.isnan(log_volumes[j+1]):
                        continue
                    
                    # Shell volume = V[j+1] - V[j] (outer - inner)
                    # Since log_volumes increase outward, V[j+1] > V[j]
                    log_V_outer = log_volumes[j+1]
                    log_V_inner = log_volumes[j]
                    
                    # ΔV = V_outer - V_inner = V_outer × (1 - exp(log_V_inner - log_V_outer))
                    if log_V_outer > log_V_inner:
                        log_delta_V = log_V_outer + np.log(1 - np.exp(log_V_inner - log_V_outer))
                    else:
                        continue  # Skip if volumes not ordered correctly
                    
                    # Likelihood at shell midpoint (geometric mean)
                    log_L_mid = 0.5 * (iso_levels_np[j] + iso_levels_np[j+1])
                    
                    # L × ΔV in log space
                    log_term = log_L_mid + log_delta_V
                    log_Z_terms.append(log_term)
                
                if len(log_Z_terms) >= 3:
                    # Sum using log-sum-exp
                    log_Z_terms = np.array(log_Z_terms)
                    log_Z_max = np.max(log_Z_terms)
                    log_Z_layer_cake = log_Z_max + np.log(np.sum(np.exp(log_Z_terms - log_Z_max)))
                    
                    # Compute Laplace for comparison
                    H_diag_safe = np.minimum(to_cpu(H_diag), -1e-10)
                    log_det_H = float(-np.sum(np.log(-H_diag_safe)))
                    log_Z_laplace = float(log_L_peak) + (d/2) * np.log(2 * np.pi) + 0.5 * log_det_H
                    
                    log_Z_correction = log_Z_layer_cake - log_Z_laplace
                    
                    use_layer_cake = True
                    
                    return float(log_Z_correction), {
                        'method': 'layer_cake_crossings',
                        'n_valid_levels': len(log_Z_terms),
                        'log_Z': log_Z_layer_cake,
                        'log_Z_laplace': log_Z_laplace,
                        'log_Z_correction': float(log_Z_correction),
                        'tail_alpha': tail_alpha,
                        'curvature': curvature,
                        'asymmetry': asymmetry,
                        'correction_applied': True
                    }
                    
        except Exception as e:
            pass  # Fall through to IS
    
    # === FALLBACK: IMPORTANCE SAMPLING ===
    sigma_proposal = sigma_hessian * w
    
    # Sample from Gaussian proposal
    z = xp.random.randn(n_samples, d)
    samples = peak_gpu + z * sigma_proposal
    
    # Evaluate TRUE log-likelihood
    log_L_true = log_L_func(samples)
    
    # Importance sampling: Z = E_q[L(x)/q(x)]
    # q(x) is N(peak, diag(sigma_proposal²))
    # log(1/q(x)) = (d/2)*log(2π) + Σlog(σ_proposal) + 0.5*Σz²
    log_det_sigma_proposal = xp.sum(xp.log(sigma_proposal))
    r_sq = xp.sum(z**2, axis=1)
    
    log_inv_q = (d/2) * np.log(2 * np.pi) + log_det_sigma_proposal + 0.5 * r_sq
    
    # log(L/q) = log_L + log(1/q)
    log_L_over_q = log_L_true + log_inv_q
    
    # Z = E_q[L/q] = (1/N) * Σ exp(log_L_over_q)
    log_L_over_q_max = float(xp.max(log_L_over_q))
    log_Z = log_L_over_q_max + np.log(float(xp.mean(xp.exp(log_L_over_q - log_L_over_q_max))))
    
    # Compute Laplace evidence for comparison
    log_det_H = float(-xp.sum(xp.log(-H_diag_safe)))
    log_Z_laplace = float(log_L_peak) + (d/2) * np.log(2 * np.pi) + 0.5 * log_det_H
    
    log_Z_correction = log_Z - log_Z_laplace
    
    # Effective sample size
    log_weights = log_L_over_q - log_L_over_q_max
    weights = xp.exp(log_weights)
    weights_norm = weights / xp.sum(weights)
    ess = float(1.0 / xp.sum(weights_norm**2))
    ess_frac = ess / n_samples
    
    return float(log_Z_correction), {
        'method': 'importance_sampling',
        'n_samples': n_samples,
        'log_Z': log_Z,
        'log_Z_laplace': log_Z_laplace,
        'log_Z_correction': float(log_Z_correction),
        'ess': ess,
        'ess_frac': ess_frac,
        'width_factor': w,
        'use_anisotropic': use_anisotropic,
        'use_asymmetric_shift': use_asymmetric_shift,
        'tail_alpha': tail_alpha,
        'curvature': curvature,
        'asymmetry': asymmetry,
        'correction_applied': True
    }




# =============================================================================
# MAIN EVIDENCE CALCULATOR
# =============================================================================

class BendTheBowShootTheTiger:
    """
    Evidence calculator with proper sunburst + bisection implementation.
    """
    
    def __init__(
        self,
        log_L_func: Callable,
        bounds: np.ndarray,
        use_gpu: bool = True,
        verbose: bool = True,
        n_sunburst_rays: int = 100,
        n_samples_per_ray: int = 50,
        n_iso_levels: int = 500,
        n_bisect_steps: int = 10
    ):
        self.log_L_func = log_L_func
        self.use_gpu = use_gpu and HAS_GPU
        self.xp = get_xp(self.use_gpu)
        self.bounds = self.xp.asarray(bounds)
        self.d = len(bounds)
        self.verbose = verbose
        
        # Sunburst parameters
        self.n_sunburst_rays = n_sunburst_rays
        self.n_samples_per_ray = n_samples_per_ray
        self.n_iso_levels = n_iso_levels
        self.n_bisect_steps = n_bisect_steps
        
        if self.verbose:
            print(f"[BendBow v4.6] d={self.d}, GPU={'ON' if self.use_gpu else 'OFF'}")
            print(f"  Sunburst: {n_sunburst_rays} rays × {n_samples_per_ray} samples")
            print(f"  Bisection: {n_iso_levels} levels × {n_bisect_steps} steps")
    
    def compute_evidence(
        self,
        peaks: np.ndarray,
        diag_H: Optional[np.ndarray] = None,
        ray_bank: Optional[Any] = None,
        trajectory_bank: Optional[Any] = None,
        logL_peaks: Optional[np.ndarray] = None,
        force_method: Optional[str] = None
    ) -> Dict:
        """
        Compute evidence with full sunburst + bisection analysis.
        
        Uses BOTH:
        1. Peak-originated sunburst rays (local structure)
        2. M1 bank rays (global structure - reveals missed modes)
        """
        peaks = to_cpu(np.atleast_2d(peaks))
        K = len(peaks)
        
        if diag_H is not None:
            diag_H = to_cpu(np.atleast_2d(diag_H))
        
        K = len(peaks)
        
        if self.verbose:
            print(f"\n[BendBow v4.6] Computing evidence for {K} peak(s)")
        
        results = []
        log_Z_terms = []
        
        for k in range(K):
            peak = peaks[k]
            t_start = time.time()
            
            if self.verbose:
                print(f"\n  Peak {k+1}/{K}:")
            
            # === Get Hessian ===
            # Even if M2 provides diagonal Hessian, we need to check for rotation
            # M2 only computes diagonal - it can't detect rotated covariances
            if diag_H is not None and k < len(diag_H) and not np.any(np.isnan(diag_H[k])):
                H_diag_m2 = diag_H[k]
                
                # First: try bank-based rotation detection (FREE - no extra evals!)
                bank_rotated, bank_confidence, rotation_dims = self._detect_rotation_from_banks(
                    peak, ray_bank, trajectory_bank, H_diag_m2
                )
                
                if bank_rotated and bank_confidence > 0.3:
                    # Bank detected rotation - trust it
                    is_rotated = True
                    ratio = bank_confidence
                    if self.verbose:
                        print(f"    [Hessian] Bank-based rotation detected (confidence={bank_confidence:.2f})")
                        if rotation_dims is not None:
                            print(f"    [Hessian] Likely correlated dims: {rotation_dims}")
                else:
                    # Fallback: FD-based rotation detection
                    is_rotated, ratio = self._detect_rotation(peak, H_diag_m2)
                
                if is_rotated:
                    if self.verbose and not (bank_rotated and bank_confidence > 0.3):
                        print(f"    [Hessian] M2 diagonal provided, but rotation detected (ratio={ratio:.4f})")
                    if self.verbose:
                        print(f"    [Hessian] Computing full Hessian...")
                    H_diag = self._compute_full_hessian_from_diagonal(peak, H_diag_m2)
                else:
                    H_diag = H_diag_m2
            else:
                H_diag = self._compute_hessian_fd(peak)
            
            # === Get log_L at peak ===
            if logL_peaks is not None and k < len(logL_peaks):
                logL_peak = float(to_cpu(logL_peaks[k]))
            else:
                peak_gpu = self.xp.asarray(peak.reshape(1, -1))
                logL_peak = float(to_cpu(self.log_L_func(peak_gpu))[0])
            
            # === SUNBURST: Generate and evaluate radial rays ===
            t0 = time.time()
            
            bounds_gpu = self.xp.asarray(self.bounds)
            peak_gpu = self.xp.asarray(peak)
            
            directions, max_t = generate_sunburst_rays(
                peak_gpu, bounds_gpu, self.n_sunburst_rays, self.xp
            )
            
            t_values, log_L_rays = evaluate_sunburst(
                self.log_L_func, peak_gpu, directions, max_t,
                self.n_samples_per_ray, self.xp
            )
            
            time_sunburst = time.time() - t0
            n_evals_sunburst = self.n_sunburst_rays * self.n_samples_per_ray
            
            if self.verbose:
                print(f"    [Sunburst] {n_evals_sunburst} evals ({time_sunburst:.3f}s)")
            
            # === BISECTION: Find iso-likelihood crossings ===
            t0 = time.time()
            
            # Get minimum log_L from sunburst rays (at prior boundary)
            # Use per-ray minimum, then take a low percentile (not absolute min)
            # This avoids wasting levels on the few rays that extend very far
            per_ray_min = np.nanmin(log_L_rays, axis=1)  # min per ray
            valid_mins = per_ray_min[~np.isnan(per_ray_min)]
            if len(valid_mins) > 0:
                # Use 10th percentile - covers 90% of rays, clips the outliers
                log_L_min = float(np.percentile(valid_mins, 10))
            else:
                log_L_min = logL_peak - 10.0  # fallback
            
            iso_levels = define_iso_levels(logL_peak, log_L_min, self.n_iso_levels)
            brackets = find_brackets(to_cpu(t_values), log_L_rays, iso_levels)
            
            crossings = bisect_to_crossings(
                self.log_L_func, peak, to_cpu(directions), brackets,
                iso_levels, self.n_bisect_steps, self.use_gpu
            )
            
            time_bisect = time.time() - t0
            n_valid_brackets = np.sum(~np.isnan(brackets[:, :, 0]))
            n_evals_bisect = n_valid_brackets * self.n_bisect_steps
            
            if self.verbose:
                print(f"    [Bisection] {int(n_evals_bisect)} evals ({time_bisect:.3f}s)")
            
            # === DIAGNOSTICS: Analyze iso-curves ===
            t0 = time.time()
            
            # Extract diagonal for diagnostics (works for both 1D and 2D Hessian)
            H_diag_for_analysis = np.diag(H_diag) if H_diag.ndim == 2 else H_diag
            
            diagnostics = analyze_iso_curves(
                crossings, to_cpu(directions), peak, H_diag_for_analysis
            )
            
            # RayBank-based tail detection (FREE - uses precomputed samples)
            raybank_diag = analyze_raybank_tails(ray_bank, peak, H_diag_for_analysis, logL_peak)
            
            # Use RayBank tail_alpha if available and reliable
            # BUT: skip raybank for multimodal - samples may be contaminated by other modes
            if raybank_diag['has_raybank'] and raybank_diag.get('n_valid', 0) >= 100 and K == 1:
                diagnostics['tail_alpha'] = raybank_diag['tail_alpha']
                diagnostics['tail_source'] = 'raybank'
                diagnostics['raybank_r_squared'] = raybank_diag.get('r_squared', 0)
                diagnostics['raybank_n_valid'] = raybank_diag.get('n_valid', 0)
            else:
                diagnostics['tail_source'] = 'sunburst'
            
            # Trajectory curvature (if available)
            curvature = compute_trajectory_curvature(trajectory_bank, peak, k, verbose=self.verbose)
            diagnostics['curvature'] = curvature
            
            time_diag = time.time() - t0
            
            if self.verbose:
                asym = diagnostics.get('asymmetry', 0)
                print(f"    [Diagnostic] tail_α={diagnostics['tail_alpha']:.2f}, "
                      f"asym={asym:.2f}, curv={curvature:.3f}, "
                      f"src={diagnostics['tail_source']}")
            
            # === PROBE FOR CORRELATION (before method selection) ===
            # Use iso-curve radii to detect correlation - ZERO extra function evals!
            # This uses the sunburst data we already computed
            diag_precision_est, ellipticity, has_correlation_radii = estimate_covariance_from_radii(
                crossings, to_cpu(directions), iso_levels, logL_peak, self.d
            )
            
            # Also probe with finite differences for uniform correlation (2 extra evals)
            # This catches uniform correlation that radii-based method might miss
            b_offdiag, has_correlation_probe = probe_offdiag_hessian(
                self.log_L_func, peak, H_diag, logL_peak, use_gpu=self.use_gpu
            )
            
            # Combine both signals
            # BUT: disable off-diagonal for heavy tails - the probe is unreliable
            tail_alpha = diagnostics.get('tail_alpha', 2.0)
            if tail_alpha < 1.8:
                # Heavy tails detected - don't trust off-diagonal probe
                has_correlation = False
            else:
                has_correlation = has_correlation_radii or has_correlation_probe
            
            if self.verbose and has_correlation:
                print(f"    [Correlation] ellipticity={ellipticity:.3f}, b_offdiag={b_offdiag:.4f}, "
                      f"radii={has_correlation_radii}, probe={has_correlation_probe}")
            
            # === COMPUTE EVIDENCE ===
            t0 = time.time()
            
            # Check if we have full Hessian (2D) or diagonal only (1D)
            is_full_hessian = H_diag.ndim == 2
            curvature = diagnostics.get('curvature', 0.0)
            
            if is_full_hessian:
                # Full Hessian case - use direct Laplace with eigendecomposition
                if self.verbose:
                    print(f"    [Evidence] Using full Hessian Laplace (rotated posterior)")
                log_Z_k, method_info = laplace_evidence_full_hessian(logL_peak, H_diag, self.d, use_gpu=self.use_gpu)
                method_info['hessian_type'] = 'full'
            else:
                # Standard case: Laplace + shell rejection correction
                
                # Use off-diagonal correction if correlation detected (and not heavy tails)
                if has_correlation:
                    log_Z_laplace, method_info = laplace_evidence_with_offdiag(logL_peak, H_diag, self.d, b_offdiag)
                    if self.verbose:
                        print(f"    [OffDiag] b={b_offdiag:.4f}, using laplace_offdiag")
                    method_info['method'] = 'laplace_offdiag'
                else:
                    log_Z_laplace, method_info = laplace_evidence(logL_peak, H_diag, self.d)
                
                # Apply importance sampling correction for non-Gaussian shapes
                asymmetry = diagnostics.get('asymmetry', 0.0)
                log_Z_correction, shell_info = shell_rejection_sampling(
                    self.log_L_func, peak, H_diag, logL_peak,
                    use_gpu=self.use_gpu,
                    tail_alpha=tail_alpha,
                    curvature=curvature,
                    asymmetry=asymmetry,
                    crossings=crossings,
                    directions=to_cpu(directions),
                    iso_levels=iso_levels
                )
                
                # Apply correction
                log_Z_uncorrected = log_Z_laplace
                log_Z_k = log_Z_laplace + log_Z_correction
                
                if self.verbose and abs(log_Z_correction) > 0.001:
                    method_str = shell_info.get('method', 'unknown')
                    if method_str == 'layer_cake_crossings':
                        n_levels_used = shell_info.get('n_valid_levels', 0)
                        print(f"    [LayerCake] Laplace={log_Z_uncorrected:.4f} + correction={log_Z_correction:.4f} = {log_Z_k:.4f}")
                        print(f"                using {n_levels_used} iso-levels from crossings")
                    else:
                        aniso_str = "aniso" if shell_info.get('use_anisotropic', False) else "iso"
                        shift_str = "+shift" if shell_info.get('use_asymmetric_shift', False) else ""
                        print(f"    [IS] Laplace={log_Z_uncorrected:.4f} + correction={log_Z_correction:.4f} = {log_Z_k:.4f}")
                        print(f"         width={shell_info.get('width_factor', 1.0):.1f}, ESS={shell_info.get('ess', 0):.0f}, {aniso_str}{shift_str}")
                elif self.verbose and shell_info.get('correction_applied', False):
                    print(f"    [Shell] Ran with width={shell_info.get('width_factor', 1.0):.1f}")
                
                method_info['log_Z_correction'] = log_Z_correction
                method_info['log_Z_uncorrected'] = log_Z_uncorrected
                method_info['shell_info'] = shell_info
                method_info['method'] = 'laplace_shell_corrected'
            
            time_evidence = time.time() - t0
            
            if self.verbose:
                corr_str = f", corr={method_info.get('log_Z_correction', 0):.3f}" if 'log_Z_correction' in method_info else ""
                print(f"    [Evidence] method={method_info.get('method', 'laplace')}, log_Z={log_Z_k:.4f}{corr_str} ({time_evidence:.3f}s)")
            
            # Collect results
            info = {
                'peak_idx': k,
                'log_Z': log_Z_k,
                'diagnostics': diagnostics,
                'n_evals_sunburst': n_evals_sunburst,
                'n_evals_bisect': int(n_evals_bisect),
                'time_total': time.time() - t_start,
            }
            info.update(method_info)
            
            results.append(info)
            log_Z_terms.append(log_Z_k)
        
        # Combine evidence from all peaks
        if len(log_Z_terms) > 1:
            # Debug: show individual mode contributions
            if self.verbose:
                print(f"    [Multimodal] Individual log_Z: {log_Z_terms}")
                print(f"    [Multimodal] logaddexp.reduce = {float(np.logaddexp.reduce(log_Z_terms)):.6f}")
        log_Z_total = float(np.logaddexp.reduce(log_Z_terms))
        
        if self.verbose:
            print(f"\n[BendBow v4.6] Total log_Z = {log_Z_total:.6f}")
        
        return {
            'log_evidence': log_Z_total,
            'peak_results': results,
            'n_peaks': K,
        }
    
    def _detect_rotation_from_banks(
        self,
        peak: np.ndarray,
        ray_bank,
        trajectory_bank,
        H_diag: np.ndarray
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect rotation/correlation from existing bank data - ZERO extra evals!
        
        Strategy: Perpendicular Step Fraction in Whitened Space
           - Whiten trajectories using diagonal Hessian: y = x / sigma
           - For axis-aligned Gaussian: steps point toward origin (radial)
           - For rotated Gaussian: steps have perpendicular (tangential) component
           - Perpendicular fraction > threshold indicates rotation
        
        This method is robust at ALL dimensions (tested 8D-128D):
           - Axis-aligned: perpendicular fraction < 1%
           - Rotated: perpendicular fraction 15-80%
           - Clear separation even at 128D (48x ratio)
        
        Returns:
            is_rotated: bool
            confidence: float (0-1, how confident we are)
            rotation_dims: Optional indices of likely correlated dimensions
        """
        d = len(peak)
        
        # Default: no rotation detected
        is_rotated = False
        confidence = 0.0
        rotation_dims = None
        
        # Whitening scale from diagonal Hessian
        H_diag_cpu = to_cpu(H_diag) if hasattr(H_diag, 'get') else np.asarray(H_diag)
        H_diag_safe = np.minimum(H_diag_cpu, -1e-10)
        sigma = np.sqrt(-1.0 / H_diag_safe)
        
        # Peak location (origin of radial directions)
        peak_cpu = to_cpu(peak) if hasattr(peak, 'get') else np.asarray(peak)
        
        # === TrajectoryBank analysis: Perpendicular Step Fraction ===
        if trajectory_bank is not None and hasattr(trajectory_bank, 'trajectories'):
            try:
                trajectories = trajectory_bank.trajectories
                
                if trajectories is not None and len(trajectories) > 0:
                    perp_fractions = []
                    
                    for traj in trajectories[:20]:  # Check first 20 trajectories
                        if traj is None or len(traj) < 3:
                            continue
                        traj = to_cpu(traj)
                        
                        # Whiten trajectory (relative to peak)
                        traj_centered = traj - peak_cpu
                        traj_whitened = traj_centered / sigma
                        
                        # Analyze each step
                        for i in range(len(traj_whitened) - 1):
                            y = traj_whitened[i]           # Current position in whitened space
                            step = traj_whitened[i+1] - y  # Step in whitened space
                            
                            y_norm = np.linalg.norm(y)
                            step_norm = np.linalg.norm(step)
                            
                            # Skip if too close to origin or step too small
                            if y_norm < 1e-6 or step_norm < 1e-8:
                                continue
                            
                            # Radial direction (toward origin/peak)
                            radial = -y / y_norm
                            
                            # Radial component of step
                            radial_component = np.dot(step, radial)
                            
                            # Perpendicular component
                            step_perp = step - radial_component * radial
                            perp_norm = np.linalg.norm(step_perp)
                            
                            # Perpendicular fraction: 0 = purely radial, 1 = purely tangential
                            perp_fraction = perp_norm / step_norm
                            perp_fractions.append(perp_fraction)
                    
                    if len(perp_fractions) >= 5:
                        max_perp = np.max(perp_fractions)
                        mean_perp = np.mean(perp_fractions)
                        
                        # Detection threshold:
                        # - Axis-aligned: max_perp < 0.02 (steps are radial)
                        # - Rotated: max_perp > 0.10 (steps have tangential component)
                        # Use 0.05 as threshold with good margin
                        if max_perp > 0.05:
                            is_rotated = True
                            confidence = min(max_perp / 0.3, 1.0)
                        
                        if self.verbose:
                            print(f"    [BankRotation] Perpendicular fraction: max={max_perp:.4f}, mean={mean_perp:.4f}")
                            
            except Exception as e:
                if self.verbose:
                    print(f"    [BankRotation] Trajectory analysis failed: {e}")
        
        return is_rotated, confidence, rotation_dims
    
    def _detect_rotation(self, peak: np.ndarray, H_diag: np.ndarray, h: float = 1e-5) -> Tuple[bool, float]:
        """
        Detect if Hessian has significant off-diagonal terms.
        Uses adjacent pairs (i, i+1) ALWAYS + random pairs for coverage.
        
        Returns:
            is_rotated: bool
            ratio: max|H_ij| / mean|H_ii|
        """
        d = len(peak)
        xp = self.xp
        peak_gpu = xp.asarray(peak)
        
        # Adjacent pairs - CRITICAL for detecting neighbor correlations
        adjacent_pairs = np.column_stack([np.arange(d - 1), np.arange(1, d)])  # (d-1, 2)
        
        # Random pairs
        np.random.seed(42)
        n_random = min(2 * d, d * (d - 1) // 2)
        rand_i = np.random.randint(0, d, n_random * 3)
        rand_j = np.random.randint(0, d, n_random * 3)
        valid = rand_i != rand_j
        rand_i, rand_j = rand_i[valid][:n_random], rand_j[valid][:n_random]
        random_pairs = np.column_stack([np.minimum(rand_i, rand_j), np.maximum(rand_i, rand_j)])
        random_pairs = np.unique(random_pairs, axis=0)
        
        # Combine and deduplicate
        all_pairs = np.vstack([adjacent_pairs, random_pairs])
        pairs_arr = np.unique(all_pairs, axis=0)
        
        if len(pairs_arr) == 0:
            return False, 0.0
        
        # Build off-diagonal perturbations - vectorized
        pairs_gpu = xp.asarray(pairs_arr)
        idx_i = pairs_gpu[:, 0]
        idx_j = pairs_gpu[:, 1]
        n_pairs = len(pairs_arr)
        
        points = xp.zeros((4 * n_pairs, d), dtype=xp.float64)
        points[:] = peak_gpu
        
        offdiag_idx = xp.arange(n_pairs)
        points[4*offdiag_idx + 0, idx_i] += h
        points[4*offdiag_idx + 0, idx_j] += h
        points[4*offdiag_idx + 1, idx_i] += h
        points[4*offdiag_idx + 1, idx_j] -= h
        points[4*offdiag_idx + 2, idx_i] -= h
        points[4*offdiag_idx + 2, idx_j] += h
        points[4*offdiag_idx + 3, idx_i] -= h
        points[4*offdiag_idx + 3, idx_j] -= h
        
        # Single batched evaluation
        log_L_all = self.log_L_func(points)
        
        # Compute off-diagonal elements - vectorized
        f_pp = log_L_all[4*offdiag_idx + 0]
        f_pm = log_L_all[4*offdiag_idx + 1]
        f_mp = log_L_all[4*offdiag_idx + 2]
        f_mm = log_L_all[4*offdiag_idx + 3]
        H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
        
        max_offdiag = float(to_cpu(xp.max(xp.abs(H_ij))))
        mean_diag = float(np.mean(np.abs(to_cpu(H_diag))))
        
        ratio = max_offdiag / mean_diag if mean_diag > 0 else float('inf')
        is_rotated = ratio > 0.1
        
        return is_rotated, ratio
    
    def _compute_full_hessian_from_diagonal(self, peak: np.ndarray, H_diag: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute full Hessian given diagonal is already known.
        Only computes off-diagonal terms.
        """
        d = len(peak)
        xp = self.xp
        peak_gpu = xp.asarray(peak)
        
        # Off-diagonal pairs
        triu_i, triu_j = np.triu_indices(d, k=1)
        n_offdiag = len(triu_i)
        
        # Build all off-diagonal perturbations
        points = xp.zeros((4 * n_offdiag, d), dtype=xp.float64)
        points[:] = peak_gpu
        
        triu_i_gpu = xp.asarray(triu_i)
        triu_j_gpu = xp.asarray(triu_j)
        offdiag_idx = xp.arange(n_offdiag)
        
        points[4*offdiag_idx + 0, triu_i_gpu] += h
        points[4*offdiag_idx + 0, triu_j_gpu] += h
        points[4*offdiag_idx + 1, triu_i_gpu] += h
        points[4*offdiag_idx + 1, triu_j_gpu] -= h
        points[4*offdiag_idx + 2, triu_i_gpu] -= h
        points[4*offdiag_idx + 2, triu_j_gpu] += h
        points[4*offdiag_idx + 3, triu_i_gpu] -= h
        points[4*offdiag_idx + 3, triu_j_gpu] -= h
        
        # Single batched evaluation
        log_L_all = self.log_L_func(points)
        
        # Build full Hessian matrix on GPU
        H_full = xp.diag(xp.asarray(H_diag))
        
        # Compute off-diagonal elements
        f_pp = log_L_all[4*offdiag_idx + 0]
        f_pm = log_L_all[4*offdiag_idx + 1]
        f_mp = log_L_all[4*offdiag_idx + 2]
        f_mm = log_L_all[4*offdiag_idx + 3]
        H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
        
        H_full[triu_i_gpu, triu_j_gpu] = H_ij
        H_full[triu_j_gpu, triu_i_gpu] = H_ij
        
        return to_cpu(H_full)

    def _compute_hessian_fd(self, peak: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian via finite differences - ADAPTIVE, FULLY GPU.
        
        1. Compute diagonal Hessian (O(d) evals)
        2. Sample off-diagonal terms to detect rotation (O(d) evals)
        3. If rotated: compute full Hessian (O(d²) evals)
        4. If not: return diagonal only
        
        Returns:
            H_diag if not rotated (1D array)
            H_full if rotated (2D array)
        """
        d = len(peak)
        xp = self.xp
        peak_gpu = xp.asarray(peak)
        
        # === Step 1: Diagonal Hessian + rotation detection in ONE batch ===
        # Sample off-diagonal pairs for rotation detection
        # Adjacent pairs (i, i+1) ALWAYS included + random pairs for coverage
        
        # Adjacent pairs - CRITICAL for detecting neighbor correlations
        adjacent_pairs = np.column_stack([np.arange(d - 1), np.arange(1, d)])  # (d-1, 2)
        
        # Random pairs
        np.random.seed(42)
        n_random = min(2 * d, d * (d - 1) // 2)
        rand_i = np.random.randint(0, d, n_random * 3)
        rand_j = np.random.randint(0, d, n_random * 3)
        valid = rand_i != rand_j
        rand_i, rand_j = rand_i[valid][:n_random], rand_j[valid][:n_random]
        # Normalize to (min, max) order and remove duplicates
        random_pairs = np.column_stack([np.minimum(rand_i, rand_j), np.maximum(rand_i, rand_j)])
        random_pairs = np.unique(random_pairs, axis=0)
        
        # Combine and deduplicate
        all_pairs = np.vstack([adjacent_pairs, random_pairs])
        pairs_arr = np.unique(all_pairs, axis=0)  # (n_pairs, 2)
        
        # Build ALL perturbations on GPU: center + 2d diagonal + 4*n_pairs off-diagonal
        n_total = 1 + 2 * d + 4 * len(pairs_arr)
        points = xp.zeros((n_total, d), dtype=xp.float64)
        
        # Center point
        points[0] = peak_gpu
        
        # Diagonal perturbations: vectorized
        # points[1:d+1] = peak + h*e_i
        # points[d+1:2d+1] = peak - h*e_i
        points[1:2*d+1] = peak_gpu  # Copy peak to all diagonal slots
        diag_idx = xp.arange(d)
        points[1 + diag_idx, diag_idx] += h       # +h perturbations
        points[1 + d + diag_idx, diag_idx] -= h   # -h perturbations
        
        # Off-diagonal perturbations: vectorized
        base = 1 + 2 * d
        if len(pairs_arr) > 0:
            pairs_gpu = xp.asarray(pairs_arr)
            idx_i = pairs_gpu[:, 0]
            idx_j = pairs_gpu[:, 1]
            
            # Broadcast peak to all off-diagonal slots
            points[base:base + 4*len(pairs_arr)] = peak_gpu
            
            # Apply perturbations: ++, +-, -+, --
            offdiag_idx = xp.arange(len(pairs_arr))
            points[base + 4*offdiag_idx + 0, idx_i] += h
            points[base + 4*offdiag_idx + 0, idx_j] += h
            points[base + 4*offdiag_idx + 1, idx_i] += h
            points[base + 4*offdiag_idx + 1, idx_j] -= h
            points[base + 4*offdiag_idx + 2, idx_i] -= h
            points[base + 4*offdiag_idx + 2, idx_j] += h
            points[base + 4*offdiag_idx + 3, idx_i] -= h
            points[base + 4*offdiag_idx + 3, idx_j] -= h
        
        # Single batched evaluation - stays on GPU
        log_L_all = self.log_L_func(points)
        
        # Extract values - stay on GPU
        logL_peak = log_L_all[0]
        logL_plus = log_L_all[1:d+1]
        logL_minus = log_L_all[d+1:2*d+1]
        
        # Diagonal Hessian - GPU
        H_diag = (logL_plus - 2 * logL_peak + logL_minus) / (h * h)
        mean_diag = xp.mean(xp.abs(H_diag))
        
        # Check off-diagonal magnitude - GPU
        max_offdiag = xp.array(0.0)
        if len(pairs_arr) > 0:
            base = 1 + 2 * d
            for idx in range(len(pairs_arr)):
                f_pp = log_L_all[base + 4*idx + 0]
                f_pm = log_L_all[base + 4*idx + 1]
                f_mp = log_L_all[base + 4*idx + 2]
                f_mm = log_L_all[base + 4*idx + 3]
                H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                max_offdiag = xp.maximum(max_offdiag, xp.abs(H_ij))
        
        # Rotation detection threshold
        rotation_threshold = 0.1
        ratio = float(to_cpu(max_offdiag / mean_diag)) if float(to_cpu(mean_diag)) > 0 else float('inf')
        is_rotated = ratio > rotation_threshold
        
        if self.verbose:
            print(f"    [Hessian] off-diag/diag ratio: {ratio:.4f}, rotated: {is_rotated}")
        
        if not is_rotated:
            # Return diagonal only (as CPU numpy)
            return to_cpu(H_diag)
        
        # === Step 2: Compute full Hessian (rotated case) ===
        if self.verbose:
            print(f"    [Hessian] Computing full Hessian (O(d²) = {2*d + 4*d*(d-1)//2} evals)...")
        
        return self._compute_full_hessian_fd(peak_gpu, h, float(to_cpu(logL_peak)), to_cpu(H_diag))
    
    def _compute_full_hessian_fd(
        self, 
        peak_gpu,
        h: float, 
        logL_peak: float,
        H_diag: np.ndarray
    ) -> np.ndarray:
        """
        Compute full Hessian matrix via finite differences - FULLY GPU.
        
        Called only when rotation is detected.
        """
        d = len(H_diag)
        xp = self.xp
        
        # Off-diagonal pairs
        triu_i, triu_j = np.triu_indices(d, k=1)
        n_offdiag = len(triu_i)
        
        # Build all off-diagonal perturbations on GPU - vectorized
        points = xp.zeros((4 * n_offdiag, d), dtype=xp.float64)
        points[:] = peak_gpu  # Broadcast peak to all
        
        # Convert indices to GPU
        triu_i_gpu = xp.asarray(triu_i)
        triu_j_gpu = xp.asarray(triu_j)
        offdiag_idx = xp.arange(n_offdiag)
        
        # Apply perturbations: ++, +-, -+, --
        points[4*offdiag_idx + 0, triu_i_gpu] += h
        points[4*offdiag_idx + 0, triu_j_gpu] += h
        points[4*offdiag_idx + 1, triu_i_gpu] += h
        points[4*offdiag_idx + 1, triu_j_gpu] -= h
        points[4*offdiag_idx + 2, triu_i_gpu] -= h
        points[4*offdiag_idx + 2, triu_j_gpu] += h
        points[4*offdiag_idx + 3, triu_i_gpu] -= h
        points[4*offdiag_idx + 3, triu_j_gpu] -= h
        
        # Single batched evaluation - stays on GPU
        log_L_all = self.log_L_func(points)
        
        # Build full Hessian matrix on GPU
        H_full = xp.diag(xp.asarray(H_diag))
        
        # Compute off-diagonal elements - vectorized
        f_pp = log_L_all[4*offdiag_idx + 0]
        f_pm = log_L_all[4*offdiag_idx + 1]
        f_mp = log_L_all[4*offdiag_idx + 2]
        f_mm = log_L_all[4*offdiag_idx + 3]
        H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
        
        # Fill symmetric matrix
        H_full[triu_i_gpu, triu_j_gpu] = H_ij
        H_full[triu_j_gpu, triu_i_gpu] = H_ij
        
        # Return as CPU numpy (evidence computation expects this)
        return to_cpu(H_full)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BendBow v4.6 - Full Pipeline Test (with Banks)")
    print("="*70)
    
    # Try to import CarryTiger and GreenDragon
    try:
        from CarryTiger_v2_0 import CarryTigerToMountain
        HAS_CARRYTIGER = True
        print("CarryTiger v2.0 loaded")
    except ImportError:
        HAS_CARRYTIGER = False
        print("WARNING: CarryTiger not available")
    
    try:
        from GreenDragonRisesFromWater_v1_5 import GreenDragonRisesFromWater
        HAS_GREENDRAGON = True
        print("GreenDragon v1.5 loaded")
    except ImportError:
        HAS_GREENDRAGON = False
        print("WARNING: GreenDragon not available")
    
    xp = cp if HAS_GPU else np
    
    def run_full_pipeline(log_L_func, bounds, name, true_log_Z):
        """Run full CarryTiger → GreenDragon → BendTheBow pipeline."""
        print(f"\n--- {name} ---")
        d = len(bounds)
        
        ray_bank = None
        trajectory_bank = None
        peaks = np.zeros((1, d))  # Default: origin
        L_peaks = None
        diag_H = None
        
        # Module 1: CarryTiger (if available)
        if HAS_CARRYTIGER:
            tiger = CarryTigerToMountain(
                func=log_L_func,
                bounds=bounds,
                use_gpu=HAS_GPU
            )
            m1_result = tiger.detect_modes(verbose=False, return_bank=True)
            peaks, L_peaks, widths, ray_bank, chisao_bank = m1_result
            print(f"  CarryTiger: {len(peaks)} peak(s) found")
            
            if len(peaks) == 0:
                peaks = np.zeros((1, d))
        
        # Module 2: GreenDragon (if available)
        if HAS_GREENDRAGON and len(peaks) > 0:
            dragon = GreenDragonRisesFromWater(
                func=log_L_func,
                bounds=bounds,
                use_gpu=HAS_GPU,
                verbose=False
            )
            m2_result = dragon.refine(peaks, widths if HAS_CARRYTIGER else None, 
                                       L_peaks=L_peaks, return_bank=True)
            peaks = m2_result['peaks']
            L_peaks = m2_result['L_peaks']
            diag_H = m2_result.get('diag_H')
            trajectory_bank = m2_result.get('trajectory_bank')
            print(f"  GreenDragon: {len(peaks)} peak(s) refined")
        
        # Module 3: BendTheBow
        bow = BendTheBowShootTheTiger(log_L_func, bounds, verbose=True)
        result = bow.compute_evidence(
            peaks=peaks,
            logL_peaks=L_peaks,
            diag_H=diag_H,
            ray_bank=ray_bank,
            trajectory_bank=trajectory_bank
        )
        
        log_Z = result['log_evidence']
        error_pct = 100 * (np.exp(log_Z - true_log_Z) - 1)
        print(f"True: {true_log_Z:.6f}, Computed: {log_Z:.6f}, Error: {error_pct:+.4f}%")
        return result, error_pct
    
    # Test 1: Gaussian (should skip shell rejection, use pure Laplace)
    d = 8
    sigma = 0.2
    
    def log_L_gaussian(x):
        x = xp.atleast_2d(xp.asarray(x))
        return -0.5 * xp.sum(x**2, axis=1) / (sigma**2)
    
    bounds_gauss = np.array([[-5.0, 5.0]] * d)
    log_Z_true_gauss = (d / 2) * np.log(2 * np.pi * sigma**2)
    
    run_full_pipeline(log_L_gaussian, bounds_gauss, "8D Gaussian", log_Z_true_gauss)
    
    # Test 2: Student-t (should detect heavy tails, run shell rejection)
    nu = 3.0
    
    def log_L_student(x):
        x = xp.atleast_2d(xp.asarray(x))
        r2 = xp.sum(x**2, axis=1)
        return -((nu + d) / 2) * xp.log(1 + r2 / nu)
    
    bounds_student = np.array([[-10.0, 10.0]] * d)
    log_Z_true_student = float(gammaln_cpu(nu/2) + (d/2)*np.log(nu*np.pi) - gammaln_cpu((nu+d)/2))
    
    run_full_pipeline(log_L_student, bounds_student, "8D Student-t ν=3", log_Z_true_student)
    
    # Test 3: Skewed distribution (should detect asymmetry)
    print("\n--- 8D Skewed Gaussian ---")
    
    def log_L_skewed(x):
        x = xp.atleast_2d(xp.asarray(x))
        # Asymmetric: narrower on positive side, wider on negative
        mask_pos = x > 0
        sigma_arr = xp.where(mask_pos, 0.1, 0.3)
        return -0.5 * xp.sum((x / sigma_arr)**2, axis=1)
    
    bounds_skewed = np.array([[-5.0, 5.0]] * d)
    # Approximate true Z for skewed (not exact, but reasonable estimate)
    # Each dimension: half is sigma=0.1, half is sigma=0.3
    # Integral per dim ≈ 0.5 * sqrt(2π) * 0.1 + 0.5 * sqrt(2π) * 0.3 = sqrt(2π) * 0.2
    log_Z_true_skewed = (d / 2) * np.log(2 * np.pi) + d * np.log(0.2)
    
    run_full_pipeline(log_L_skewed, bounds_skewed, "8D Skewed Gaussian", log_Z_true_skewed)
    
    print("\n" + "="*70)
    print("Tests complete")
    print("="*70)
