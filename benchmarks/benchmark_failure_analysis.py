#!/usr/bin/env python3
"""
SunBURST Failure Analysis: Push to Breaking Point

Tests extreme noise levels to identify failure modes:
- When does peak detection fail?
- When does evidence calculation break?
- When do spurious peaks appear?
- What's the critical noise threshold?
"""

import numpy as np
from scipy.stats import norm
import time
import sys
import os

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sunburst import compute_evidence
from sunburst.utils import assess_noise


def analyze_failure_modes(
    noise_levels=None,
    D=6,
    bounds_range=(-5.0, 5.0),
    n_trials=10,
    use_gpu=None,
    threshold_multiplier=50,
    verbose=True
):
    """
    Comprehensive failure analysis across extreme noise levels.
    
    Args:
        noise_levels: List of noise amplitudes (including extreme values)
        D: Dimension
        bounds_range: Parameter bounds
        n_trials: Repetitions per noise level
        use_gpu: Use GPU (None = auto-detect)
        threshold_multiplier: Multiplier for assess_noise recommendations
        verbose: Print progress
    
    Returns:
        Dictionary with detailed diagnostics
    """
    # Auto-detect GPU
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
        if verbose and GPU_AVAILABLE:
            print("GPU detected - using GPU acceleration")
    
    if noise_levels is None:
        # Push to extreme noise: 1e-2 (1%) to 1.0 (100%)
        noise_levels = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
    
    bounds = [(bounds_range[0], bounds_range[1])] * D
    bounds_arr = np.array(bounds)
    
    # Analytical evidence
    cdf_val = norm.cdf(bounds_range[1]) - norm.cdf(bounds_range[0])
    analytical_log_Z_raw = D * (0.5 * np.log(2*np.pi) + np.log(cdf_val))
    prior_volume = (bounds_range[1] - bounds_range[0])**D
    analytical_log_Z = analytical_log_Z_raw - np.log(prior_volume)
    
    if verbose:
        print("="*70)
        print("SunBURST Failure Analysis - Push to Breaking Point")
        print("="*70)
        print(f"Dimension: {D}")
        print(f"Bounds: {bounds_range}")
        print(f"Analytical log(Z): {analytical_log_Z:.4f}")
        print(f"Noise levels: {noise_levels}")
        print(f"Trials per level: {n_trials}")
        print(f"Threshold multiplier: {threshold_multiplier}×")
        print(f"GPU: {use_gpu}")
        print("="*70)
    
    results = {
        'noise_levels': noise_levels,
        'analytical_log_Z': analytical_log_Z,
        'D': D,
        'trials': []
    }
    
    for noise_level in noise_levels:
        if verbose:
            print(f"\n{'='*70}")
            print(f"NOISE LEVEL: {noise_level:.2e} ({noise_level*100:.1f}%)")
            print(f"{'='*70}")
        
        trial_results = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Trial {trial+1}/{n_trials} ---")
            
            # Generate random noise basis
            np.random.seed(42 + trial)
            n_basis = 50
            basis_freqs = np.random.randn(n_basis, D)
            basis_phases = np.random.randn(n_basis) * 2 * np.pi
            basis_coeffs = np.random.randn(n_basis)
            
            # GPU-aware vectorized noise
            def get_noise_vectorized(x, use_gpu_flag=False):
                if not use_gpu_flag or not GPU_AVAILABLE:
                    x = np.atleast_2d(x)
                    phases = np.dot(x, basis_freqs.T) + basis_phases[None, :]
                    contributions = basis_coeffs[None, :] * np.sin(phases)
                    noise = np.sum(contributions, axis=1) * noise_level / np.sqrt(n_basis)
                else:
                    x = cp.atleast_2d(x)
                    basis_freqs_gpu = cp.asarray(basis_freqs)
                    basis_phases_gpu = cp.asarray(basis_phases)
                    basis_coeffs_gpu = cp.asarray(basis_coeffs)
                    phases = cp.dot(x, basis_freqs_gpu.T) + basis_phases_gpu[None, :]
                    contributions = basis_coeffs_gpu[None, :] * cp.sin(phases)
                    noise = cp.sum(contributions, axis=1) * noise_level / cp.sqrt(n_basis)
                return noise
            
            def noisy_gaussian(x):
                """GPU-aware likelihood"""
                input_is_gpu = GPU_AVAILABLE and isinstance(x, cp.ndarray)
                
                if input_is_gpu:
                    x_gpu = cp.atleast_2d(x)
                    clean = -0.5 * cp.sum(x_gpu**2, axis=1)
                    noise = get_noise_vectorized(x_gpu, use_gpu_flag=True)
                    return clean + noise
                else:
                    x_np = np.atleast_2d(x)
                    clean = -0.5 * np.sum(x_np**2, axis=1)
                    noise = get_noise_vectorized(x_np, use_gpu_flag=False)
                    return clean + noise
            
            # Assess noise (might fail at extreme noise)
            test_point = np.ones(D) * 0.5
            try:
                noise_stats = assess_noise(
                    noisy_gaussian,
                    test_point,
                    bounds_arr,
                    verbose=False
                )
                
                stick_tol = noise_stats['stick_tolerance_recommended'] * threshold_multiplier
                grad_tol = noise_stats['grad_threshold_recommended'] * threshold_multiplier
                saddle_tol = noise_stats['saddle_threshold_recommended'] * threshold_multiplier
                
                assess_success = True
                measured_grad_noise = noise_stats['gradient_l2_variation']
                measured_hess_noise = noise_stats['hessian_mean_noise']
            except Exception as e:
                # assess_noise failed - use fallback thresholds
                if verbose:
                    print(f"  WARNING: assess_noise failed: {e}")
                assess_success = False
                stick_tol = noise_level * 10
                grad_tol = noise_level * 10
                saddle_tol = noise_level * 10
                measured_grad_noise = np.nan
                measured_hess_noise = np.nan
            
            if verbose:
                print(f"  Thresholds ({threshold_multiplier}×):")
                print(f"    stick_tolerance  = {stick_tol:.2e}")
                print(f"    grad_threshold   = {grad_tol:.2e}")
                print(f"    saddle_threshold = {saddle_tol:.2e}")
            
            # Compute evidence - catch failures
            t_start = time.time()
            try:
                result = compute_evidence(
                    noisy_gaussian,
                    bounds,
                    n_oscillations=2,
                    verbose=False,
                    use_gpu=use_gpu,
                    stick_tolerance=stick_tol,
                    grad_threshold=grad_tol,
                    saddle_threshold=saddle_tol,
                )
                computation_failed = False
                failure_reason = None
            except Exception as e:
                if verbose:
                    print(f"  ERROR: compute_evidence failed: {e}")
                computation_failed = True
                failure_reason = str(e)
                # Create dummy result
                class DummyResult:
                    log_evidence = np.nan
                    n_peaks = 0
                    peaks = []
                result = DummyResult()
            
            t_elapsed = time.time() - t_start
            
            # Analyze results
            if not computation_failed and result.n_peaks > 0:
                # Evidence error
                log_Z_error = abs(result.log_evidence - analytical_log_Z)
                rel_error_pct = 100 * log_Z_error / abs(analytical_log_Z)
                
                # Peak accuracy (distance from origin)
                peak_distances = [np.linalg.norm(peak) for peak in result.peaks]
                min_peak_dist = min(peak_distances)
                
                # Check for spurious peaks
                # True peak should be at origin - any peak >1 sigma away is spurious
                expected_peak_uncertainty = np.sqrt(noise_level)
                spurious_peaks = sum(1 for d in peak_distances if d > 3*expected_peak_uncertainty)
                
                success_category = "SUCCESS"
                if result.n_peaks > 1:
                    success_category = "SPURIOUS_PEAKS"
                elif rel_error_pct > 50:
                    success_category = "LARGE_ERROR"
                elif min_peak_dist > 10*expected_peak_uncertainty:
                    success_category = "WRONG_PEAK"
            else:
                log_Z_error = np.nan
                rel_error_pct = np.nan
                min_peak_dist = np.nan
                spurious_peaks = 0
                
                if computation_failed:
                    success_category = "COMPUTATION_FAILED"
                else:
                    success_category = "NO_PEAKS_FOUND"
            
            # Store detailed diagnostics
            trial_result = {
                'noise_level': noise_level,
                'trial': trial,
                'success': success_category == "SUCCESS",
                'success_category': success_category,
                'failure_reason': failure_reason,
                
                # Evidence metrics
                'log_Z': result.log_evidence,
                'log_Z_error': log_Z_error,
                'rel_error_pct': rel_error_pct,
                
                # Peak metrics
                'n_peaks': result.n_peaks,
                'n_spurious_peaks': spurious_peaks,
                'min_peak_distance': min_peak_dist,
                'all_peak_distances': peak_distances if not computation_failed and result.n_peaks > 0 else [],
                
                # Noise assessment
                'assess_noise_success': assess_success,
                'measured_grad_noise': measured_grad_noise,
                'measured_hess_noise': measured_hess_noise,
                
                # Thresholds used
                'stick_tolerance': stick_tol,
                'grad_threshold': grad_tol,
                'saddle_threshold': saddle_tol,
                
                # Performance
                'time': t_elapsed,
            }
            trial_results.append(trial_result)
            
            if verbose:
                print(f"  Result: {success_category}")
                if not computation_failed:
                    print(f"  log(Z):        {result.log_evidence:.4f}")
                    if not np.isnan(log_Z_error):
                        print(f"  Error:         {log_Z_error:.4f} ({rel_error_pct:.2f}%)")
                    print(f"  Peaks found:   {result.n_peaks} ({spurious_peaks} spurious)")
                    if result.n_peaks > 0:
                        print(f"  Best peak dist: {min_peak_dist:.6f}")
                print(f"  Time:          {t_elapsed:.2f}s")
        
        # Aggregate statistics
        success_counts = {}
        for category in ["SUCCESS", "SPURIOUS_PEAKS", "LARGE_ERROR", "WRONG_PEAK", "NO_PEAKS_FOUND", "COMPUTATION_FAILED"]:
            success_counts[category] = sum(1 for t in trial_results if t['success_category'] == category)
        
        # Get stats for successful trials only
        successful_trials = [t for t in trial_results if t['success']]
        if successful_trials:
            log_Z_errors = [t['log_Z_error'] for t in successful_trials]
            rel_errors = [t['rel_error_pct'] for t in successful_trials]
            peak_dists = [t['min_peak_distance'] for t in successful_trials]
            
            mean_log_Z_error = np.mean(log_Z_errors)
            std_log_Z_error = np.std(log_Z_errors)
            mean_rel_error = np.mean(rel_errors)
            std_rel_error = np.std(rel_errors)
            mean_peak_dist = np.mean(peak_dists)
            std_peak_dist = np.std(peak_dists)
        else:
            mean_log_Z_error = np.nan
            std_log_Z_error = np.nan
            mean_rel_error = np.nan
            std_rel_error = np.nan
            mean_peak_dist = np.nan
            std_peak_dist = np.nan
        
        noise_result = {
            'noise_level': noise_level,
            'trials': trial_results,
            'n_trials': n_trials,
            'success_counts': success_counts,
            'success_rate': success_counts['SUCCESS'] / n_trials,
            'mean_log_Z_error': mean_log_Z_error,
            'std_log_Z_error': std_log_Z_error,
            'mean_rel_error_pct': mean_rel_error,
            'std_rel_error_pct': std_rel_error,
            'mean_peak_distance': mean_peak_dist,
            'std_peak_distance': std_peak_dist,
        }
        results['trials'].append(noise_result)
        
        if verbose:
            print(f"\nNoise level {noise_level:.2e} summary:")
            print(f"  Success rate:  {noise_result['success_rate']*100:.0f}%")
            print(f"  Failure modes:")
            for category, count in success_counts.items():
                if count > 0:
                    print(f"    {category}: {count}/{n_trials}")
            if not np.isnan(mean_log_Z_error):
                print(f"  Mean error (successful):  {mean_log_Z_error:.4f} ± {std_log_Z_error:.4f}")
                print(f"  Mean rel err (successful): {mean_rel_error:.2f}% ± {std_rel_error:.2f}%")
    
    return results


def print_failure_summary(results):
    """Print comprehensive failure analysis summary"""
    print("\n" + "="*80)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Analytical log(Z): {results['analytical_log_Z']:.4f}")
    print()
    
    # Header
    print(f"{'Noise':>10} {'Success':>10} {'Spurious':>10} {'LargeErr':>10} {'WrongPk':>10} {'NoPeaks':>10} {'Failed':>10}")
    print("-"*80)
    
    for noise_result in results['trials']:
        noise = noise_result['noise_level']
        counts = noise_result['success_counts']
        
        print(f"{noise:>10.2e} "
              f"{counts['SUCCESS']:>10d} "
              f"{counts['SPURIOUS_PEAKS']:>10d} "
              f"{counts['LARGE_ERROR']:>10d} "
              f"{counts['WRONG_PEAK']:>10d} "
              f"{counts['NO_PEAKS_FOUND']:>10d} "
              f"{counts['COMPUTATION_FAILED']:>10d}")
    
    print("="*80)
    
    # Find critical thresholds
    print("\nCRITICAL THRESHOLDS:")
    for threshold in [0.99, 0.95, 0.90, 0.50]:
        for i, noise_result in enumerate(results['trials']):
            if noise_result['success_rate'] < threshold:
                noise = noise_result['noise_level']
                print(f"  {threshold*100:.0f}% success rate lost at noise = {noise:.2e} ({noise*100:.1f}%)")
                break
    
    # Most common failure mode at high noise
    high_noise_trials = results['trials'][-1] if results['trials'] else None
    if high_noise_trials:
        print(f"\nAt highest noise ({high_noise_trials['noise_level']:.2e}):")
        print(f"  Most common outcome: ", end="")
        counts = high_noise_trials['success_counts']
        max_category = max(counts, key=counts.get)
        print(f"{max_category} ({counts[max_category]}/{high_noise_trials['n_trials']})")


def save_results(results, filename='failure_analysis_results.npz'):
    """Save results to file"""
    np.savez(filename, **results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SunBURST failure analysis - push to breaking point')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       help='Noise levels to test (default: 1e-2 to 1.0)')
    parser.add_argument('--dimension', type=int, default=6,
                       help='Problem dimension (default: 6)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Trials per noise level (default: 10)')
    parser.add_argument('--threshold-mult', type=int, default=50,
                       help='Threshold multiplier for assess_noise (default: 50)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--output', type=str, default='failure_analysis_results.npz',
                       help='Output file for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Run failure analysis
    results = analyze_failure_modes(
        noise_levels=args.noise_levels,
        D=args.dimension,
        n_trials=args.trials,
        threshold_multiplier=args.threshold_mult,
        use_gpu=False if args.no_gpu else None,
        verbose=not args.quiet
    )
    
    # Print summary
    print_failure_summary(results)
    
    # Save results
    save_results(results, args.output)
    
    print("\nFailure analysis complete!")
    print("Use plot_failure_analysis.py to visualize results")
