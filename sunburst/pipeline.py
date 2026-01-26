"""
SunBURST Pipeline — Main Public API

Seeded Universe Navigation — Bayesian Unification via Radial Shooting Techniques

This module provides the clean public interface for SunBURST evidence calculation.
"""

import numpy as np
import time
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field

from .utils.gpu import gpu_available, get_array_module, to_cpu, to_gpu

# Module availability tracking
_MODULES_AVAILABLE = {
    'carry_tiger': False,
    'green_dragon': False,
    'bend_bow': False,
    'grasp_tail': False,
}

try:
    from .modules.carry_tiger import CarryTigerToMountain, CHISAO_VERSION, SINGLEWHIP_VERSION
    _MODULES_AVAILABLE['carry_tiger'] = True
except ImportError as e:
    CHISAO_VERSION = None
    SINGLEWHIP_VERSION = None
    import warnings
    warnings.warn(f"CarryTiger import failed: {e}")

try:
    from .modules.green_dragon import GreenDragonRisesFromWater
    _MODULES_AVAILABLE['green_dragon'] = True
except ImportError as e:
    import warnings
    warnings.warn(f"GreenDragon import failed: {e}")

try:
    from .modules.bend_bow import BendTheBowShootTheTiger
    _MODULES_AVAILABLE['bend_bow'] = True
except ImportError as e:
    import warnings
    warnings.warn(f"BendTheBow import failed: {e}")

try:
    from .modules.grasp_tail import grasp_birds_tail, GraspBirdsTailConfig
    _MODULES_AVAILABLE['grasp_tail'] = True
except ImportError:
    pass  # Optional module


@dataclass
class SunburstResult:
    """
    Result from SunBURST evidence calculation.
    
    Attributes:
        log_evidence: Estimated log Bayesian evidence (log Z)
        log_evidence_std: Uncertainty estimate on log Z
        n_peaks: Number of modes found
        peaks: (n_peaks, dim) array of peak locations
        hessians: List of (dim, dim) Hessian matrices at peaks (if available)
        log_evidence_per_peak: (n_peaks,) contribution from each peak
        wall_time: Total computation time in seconds
        module_times: Dict with per-stage timing breakdown
        n_likelihood_calls: Total number of likelihood evaluations
        config: Configuration used for this run
    """
    log_evidence: float
    log_evidence_std: float = 0.0
    n_peaks: int = 0
    peaks: Optional[np.ndarray] = None
    hessians: Optional[List[np.ndarray]] = None
    log_evidence_per_peak: Optional[np.ndarray] = None
    wall_time: float = 0.0
    module_times: Dict[str, float] = field(default_factory=dict)
    n_likelihood_calls: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Additional diagnostic info
    L_peaks: Optional[np.ndarray] = None
    widths: Optional[np.ndarray] = None
    diag_H: Optional[np.ndarray] = None


def compute_evidence(
    log_likelihood: Callable,
    bounds: List,
    n_oscillations: int = 1,
    fast: bool = True,
    return_peaks: bool = True,
    verbose: bool = False,
    seed: Optional[int] = None,
    use_gpu: Optional[bool] = None,
) -> SunburstResult:
    """
    Compute Bayesian evidence for a likelihood function.
    
    This is the main entry point for SunBURST evidence calculation.
    
    Parameters
    ----------
    log_likelihood : Callable
        Log-likelihood function. Must accept (N, D) array and return (N,) array
        of log-likelihood values. For single-point evaluation, will be wrapped
        to handle batching automatically.
    bounds : List[Tuple[float, float]]
        List of (lower, upper) bounds for each parameter dimension.
    n_oscillations : int, default=1
        ChiSao oscillation cycles for mode detection.
        1 = fast (for well-separated modes)
        3 = conservative (for complex multimodal)
    fast : bool, default=True
        Use fast Hessian estimation in GreenDragon.
        True = 20 random samples
        False = 2×D axis-aligned samples
    return_peaks : bool, default=True
        Include peak locations and Hessians in result.
    verbose : bool, default=False
        Print progress information.
    seed : int, optional
        Random seed for reproducibility.
    use_gpu : bool, optional
        Force GPU (True), CPU (False), or auto-detect (None).
    
    Returns
    -------
    SunburstResult
        Dataclass containing log_evidence, uncertainty, peaks, timing, etc.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sunburst import compute_evidence
    >>> 
    >>> def log_likelihood(x):
    ...     x = np.atleast_2d(x)
    ...     return -0.5 * np.sum(x**2, axis=1)
    >>> 
    >>> bounds = [(-10, 10)] * 64
    >>> result = compute_evidence(log_likelihood, bounds)
    >>> print(f"log Z = {result.log_evidence:.4f}")
    """
    # Check module availability
    missing = [name for name, avail in _MODULES_AVAILABLE.items() 
               if not avail and name != 'grasp_tail']
    if missing:
        raise ImportError(f"Missing required modules: {missing}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        if gpu_available():
            import cupy as cp
            cp.random.seed(seed)
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = gpu_available()
    
    xp = get_array_module(use_gpu)
    
    # Convert bounds to array
    bounds_arr = np.array(bounds, dtype=np.float64)
    D = bounds_arr.shape[0]
    
    if use_gpu:
        bounds_arr = to_gpu(bounds_arr)
    
    # Track timing
    t_total_start = time.time()
    module_times = {}
    n_calls = 0
    
    # Wrap likelihood to track calls
    def tracked_log_L(x):
        nonlocal n_calls
        x = xp.atleast_2d(x)
        n_calls += x.shape[0]
        return log_likelihood(x)
    
    if verbose:
        print("=" * 70)
        print(f"SunBURST Evidence Calculation: D={D}")
        print(f"GPU: {'ENABLED' if use_gpu else 'DISABLED'}")
        print("=" * 70)
    
    # ==========================================================================
    # Stage 1: CarryTiger (Mode Detection)
    # ==========================================================================
    
    if verbose:
        print("\n[Stage 1: CarryTiger - Mode Detection]")
    
    t0 = time.time()
    
    tiger = CarryTigerToMountain(
        func=tracked_log_L,
        bounds=bounds_arr,
        use_gpu=use_gpu,
        n_oscillations=n_oscillations,
    )
    
    peaks, L_peaks, widths, ray_bank, chisao_bank = tiger.detect_modes(
        verbose=verbose,
        return_bank=True
    )
    
    module_times['carry_tiger'] = time.time() - t0
    n_peaks = len(peaks) if peaks is not None else 0
    
    if verbose:
        print(f"  → Found {n_peaks} peaks in {module_times['carry_tiger']:.2f}s")
    
    if n_peaks == 0:
        return SunburstResult(
            log_evidence=np.nan,
            log_evidence_std=np.nan,
            n_peaks=0,
            wall_time=time.time() - t_total_start,
            module_times=module_times,
            n_likelihood_calls=n_calls,
            config={'n_oscillations': n_oscillations, 'fast': fast, 'dim': D},
        )
    
    # ==========================================================================
    # Stage 2: GreenDragon (Peak Refinement)
    # ==========================================================================
    
    if verbose:
        print("\n[Stage 2: GreenDragon - Peak Refinement]")
    
    t0 = time.time()
    
    dragon = GreenDragonRisesFromWater(
        func=tracked_log_L,
        bounds=bounds_arr,
        use_gpu=use_gpu,
        fast=fast,
    )
    
    dragon_result = dragon.refine(
        peaks=peaks,
        widths=widths,
        return_bank=True
    )
    
    refined_peaks = dragon_result['peaks']
    refined_L = dragon_result['L_peaks']
    diag_H = dragon_result['diag_H']
    trajectory_bank = dragon_result.get('trajectory_bank', None)
    
    module_times['green_dragon'] = time.time() - t0
    
    if verbose:
        print(f"  → Refined {len(refined_peaks)} peaks in {module_times['green_dragon']:.2f}s")
    
    # ==========================================================================
    # Stage 3: BendTheBow (Evidence Calculation)
    # ==========================================================================
    
    if verbose:
        print("\n[Stage 3: BendTheBow - Evidence Calculation]")
    
    t0 = time.time()
    
    bow = BendTheBowShootTheTiger(
        log_L_func=tracked_log_L,
        bounds=bounds_arr,
        use_gpu=use_gpu,
        verbose=verbose,
    )
    
    evidence_result = bow.compute_evidence(
        peaks=to_cpu(refined_peaks),
        logL_peaks=to_cpu(refined_L),
        diag_H=to_cpu(diag_H),
        ray_bank=ray_bank,
        trajectory_bank=trajectory_bank,
    )
    
    module_times['bend_bow'] = time.time() - t0
    
    log_Z = evidence_result.get('log_evidence', np.nan)
    log_Z_err = evidence_result.get('evidence_error', 0.0)
    
    # ==========================================================================
    # Apply prior volume correction
    # ==========================================================================
    # BendTheBow computes the integral of L(x) over parameter space
    # For Bayesian evidence with uniform prior, we need to divide by prior volume:
    # Z = integral(L(x) * p(x)) where p(x) = 1/V for uniform prior
    # log Z = log(integral(L(x))) - log(V)
    
    bounds_cpu = to_cpu(bounds_arr)
    log_prior_volume = np.sum(np.log(bounds_cpu[:, 1] - bounds_cpu[:, 0]))
    log_Z = log_Z - log_prior_volume
    
    if verbose:
        print(f"  → Prior volume correction: -{log_prior_volume:.4f}")
        err_str = f" ± {log_Z_err:.4f}" if log_Z_err else ""
        print(f"  → log Z = {log_Z:.4f}{err_str} in {module_times['bend_bow']:.2f}s")
    
    # ==========================================================================
    # Finalize
    # ==========================================================================
    
    wall_time = time.time() - t_total_start
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Pipeline Complete: {wall_time:.2f}s")
        print("=" * 70)
    
    # Prepare Hessians list if requested
    hessians = None
    if return_peaks and diag_H is not None:
        diag_H_cpu = to_cpu(diag_H)
        hessians = [np.diag(diag_H_cpu[i]) for i in range(len(diag_H_cpu))]
    
    return SunburstResult(
        log_evidence=float(log_Z),
        log_evidence_std=float(log_Z_err) if log_Z_err else 0.0,
        n_peaks=len(refined_peaks),
        peaks=to_cpu(refined_peaks) if return_peaks else None,
        hessians=hessians,
        log_evidence_per_peak=evidence_result.get('log_evidence_per_peak', None),
        wall_time=wall_time,
        module_times=module_times,
        n_likelihood_calls=n_calls,
        config={
            'n_oscillations': n_oscillations,
            'fast': fast,
            'dim': D,
            'use_gpu': use_gpu,
            'seed': seed,
        },
        L_peaks=to_cpu(refined_L),
        widths=to_cpu(widths) if widths is not None else None,
        diag_H=to_cpu(diag_H),
    )


def test(dim: int = 64, verbose: bool = True) -> SunburstResult:
    """
    Run a quick test on a standard Gaussian to verify installation.
    
    Parameters
    ----------
    dim : int, default=64
        Dimensionality of test problem.
    verbose : bool, default=True
        Print progress and results.
    
    Returns
    -------
    SunburstResult
        Result from the test run.
    
    Examples
    --------
    >>> import sunburst
    >>> result = sunburst.test(dim=64)
    """
    if verbose:
        print(f"Running SunBURST test on {dim}D Gaussian...")
    
    # Define standard Gaussian likelihood
    xp = get_array_module()
    
    def log_likelihood(x):
        x = xp.atleast_2d(x)
        return -0.5 * xp.sum(x**2, axis=1)
    
    # Bounds: [-10, 10] in each dimension
    bounds = [(-10.0, 10.0)] * dim
    
    # True log evidence for unit Gaussian with uniform prior on [-10, 10]^D
    # Z = (2π)^(D/2) / (20)^D
    true_log_Z = 0.5 * dim * np.log(2 * np.pi) - dim * np.log(20)
    
    # Run evidence calculation
    result = compute_evidence(
        log_likelihood,
        bounds,
        n_oscillations=1,
        fast=True,
        verbose=verbose,
    )
    
    # Report results
    error = abs(result.log_evidence - true_log_Z)
    error_pct = 100 * error / abs(true_log_Z) if true_log_Z != 0 else 0
    
    if verbose:
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Dimension: {dim}")
        print(f"log Z (computed): {result.log_evidence:.4f}")
        print(f"log Z (true):     {true_log_Z:.4f}")
        print(f"Error: {error:.4f} ({error_pct:.2f}%)")
        print(f"Peaks found: {result.n_peaks}")
        print(f"Time: {result.wall_time:.2f}s")
        print(f"Likelihood calls: {result.n_likelihood_calls}")
        
        status = "PASS" if error_pct < 1.0 else "FAIL"
        print(f"\nStatus: {status}")
        print("=" * 70)
    
    return result
