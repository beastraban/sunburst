#!/usr/bin/env python3
"""
Extreme Dimension Test: D=256

Tests the absolute limits of SunBURST at extreme dimensionality.
The prior volume is (10)^256 ≈ 10^256 - astronomically huge!

This will test:
- Can SunBURST scale to hundreds of dimensions?
- Does auto-tuning still work?
- How does noise tolerance change?
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


def extreme_dimension_test(
    D=256,
    noise_levels=None,
    bounds_range=(-5.0, 5.0),
    n_trials=5,
    use_gpu=None,
    threshold_multiplier=50,
    verbose=True
):
    """
    Test SunBURST at extreme dimensionality.
    
    Args:
        D: Dimension (default: 256)
        noise_levels: List of noise amplitudes
        bounds_range: Parameter bounds
        n_trials: Repetitions per noise level
        use_gpu: Use GPU (None = auto-detect, HIGHLY recommended for D=256!)
        threshold_multiplier: Multiplier for assess_noise
        verbose: Print progress
    """
    # Auto-detect GPU
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
        if verbose:
            if GPU_AVAILABLE:
                print("GPU detected - CRITICAL for D=256!")
            else:
                print("WARNING: No GPU detected - D=256 will be VERY slow on CPU!")
    
    if noise_levels is None:
        # Test a range from low to moderate noise
        noise_levels = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    
    bounds = [(bounds_range[0], bounds_range[1])] * D
    bounds_arr = np.array(bounds)
    
    # Prior volume at D=256 is HUGE!
    prior_volume = (bounds_range[1] - bounds_range[0])**D
    log_prior_volume = D * np.log(bounds_range[1] - bounds_range[0])
    
    # Analytical evidence for Gaussian
    cdf_val = norm.cdf(bounds_range[1]) - norm.cdf(bounds_range[0])
    analytical_log_Z_raw = D * (0.5 * np.log(2*np.pi) + np.log(cdf_val))
    analytical_log_Z = analytical_log_Z_raw - log_prior_volume
    
    if verbose:
        print("="*70)
        print(f"EXTREME DIMENSION TEST: D={D}")
        print("="*70)
        print(f"Dimension: {D}")
        print(f"log(Prior volume): {log_prior_volume:.1f} (volume ≈ 10^{log_prior_volume/np.log(10):.0f})")
        print(f"Analytical log(Z): {analytical_log_Z:.4f}")
        print(f"Bounds: {bounds_range}")
        print(f"Noise levels: {noise_levels}")
        print(f"Trials per level: {n_trials}")
        print(f"GPU: {use_gpu}")
        if not use_gpu:
            print("WARNING: CPU mode at D=256 will be extremely slow!")
        print("="*70)
    
    results = {
        'D': D,
        'noise_levels': noise_levels,
        'analytical_log_Z': analytical_log_Z,
        'log_prior_volume': log_prior_volume,
        'trials': []
    }
    
    for noise_level in noise_levels:
        if verbose:
            print(f"\n{'='*70}")
            print(f"NOISE LEVEL: {noise_level:.2e} ({noise_level*100:.2f}%)")
            print(f"{'='*70}")
        
        trial_results = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Trial {trial+1}/{n_trials} ---")
            
            # Generate random noise basis
            np.random.seed(42 + trial)
            n_basis = 50  # Keep basis count manageable
            basis_freqs = np.random.randn(n_basis, D)
            basis_phases = np.random.randn(n_basis) * 2 * np.pi
            basis_coeffs = np.random.randn(n_basis)
            
            # GPU-aware noise
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
            
            def noisy_gaussian_d256(x):
                """256D Gaussian with noise"""
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
            
            # Assess noise (might take a while at D=256!)
            test_point = np.ones(D) * 0.5
            
            if verbose:
                print("  Assessing noise...", end=" ", flush=True)
            
            t_assess_start = time.time()
            try:
                noise_stats = assess_noise(
                    noisy_gaussian_d256,
                    test_point,
                    bounds_arr,
                    verbose=False
                )
                
                stick_tol = noise_stats['stick_tolerance_recommended'] * threshold_multiplier
                grad_tol = noise_stats['grad_threshold_recommended'] * threshold_multiplier
                saddle_tol = noise_stats['saddle_threshold_recommended'] * threshold_multiplier
                assess_success = True
                t_assess = time.time() - t_assess_start
                
                if verbose:
                    print(f"done ({t_assess:.1f}s)")
            except Exception as e:
                t_assess = time.time() - t_assess_start
                if verbose:
                    print(f"FAILED ({t_assess:.1f}s): {e}")
                assess_success = False
                stick_tol = noise_level * 10
                grad_tol = noise_level * 10
                saddle_tol = noise_level * 10
            
            if verbose:
                print(f"  Thresholds ({threshold_multiplier}×):")
                print(f"    stick_tolerance  = {stick_tol:.2e}")
                print(f"    grad_threshold   = {grad_tol:.2e}")
                print("  Computing evidence...", flush=True)
            
            # Compute evidence
            t_start = time.time()
            try:
                result = compute_evidence(
                    noisy_gaussian_d256,
                    bounds,
                    n_oscillations=2,
                    verbose=False,
                    use_gpu=use_gpu,
                    stick_tolerance=stick_tol,
                    grad_threshold=grad_tol,
                    saddle_threshold=saddle_tol,
                )
                
                log_Z_error = abs(result.log_evidence - analytical_log_Z)
                rel_error_pct = 100 * log_Z_error / abs(analytical_log_Z)
                
                if result.n_peaks > 0:
                    peak_dist = np.linalg.norm(result.peaks[0])
                    success = True
                else:
                    peak_dist = np.nan
                    success = False
                
                computation_failed = False
                failure_reason = None
                
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                computation_failed = True
                failure_reason = str(e)
                log_Z_error = np.nan
                rel_error_pct = np.nan
                peak_dist = np.nan
                success = False
                
                class DummyResult:
                    log_evidence = np.nan
                    n_peaks = 0
                result = DummyResult()
            
            t_elapsed = time.time() - t_start
            
            # Store results
            trial_result = {
                'noise_level': noise_level,
                'trial': trial,
                'success': success,
                'log_Z': result.log_evidence,
                'log_Z_error': log_Z_error,
                'rel_error_pct': rel_error_pct,
                'n_peaks': result.n_peaks,
                'peak_distance': peak_dist,
                'assess_success': assess_success,
                'assess_time': t_assess,
                'computation_failed': computation_failed,
                'failure_reason': failure_reason,
                'time': t_elapsed,
            }
            trial_results.append(trial_result)
            
            if verbose:
                if not computation_failed:
                    print(f"  log(Z):        {result.log_evidence:.4f}")
                    print(f"  Error:         {log_Z_error:.4f} ({rel_error_pct:.2f}%)")
                    print(f"  Peaks found:   {result.n_peaks}")
                    if not np.isnan(peak_dist):
                        print(f"  Peak dist:     {peak_dist:.6f}")
                    print(f"  Time:          {t_elapsed:.1f}s")
                else:
                    print(f"  FAILED after {t_elapsed:.1f}s")
        
        # Aggregate statistics
        successful = [t for t in trial_results if t['success']]
        success_rate = len(successful) / n_trials
        
        if successful:
            mean_error = np.mean([t['log_Z_error'] for t in successful])
            std_error = np.std([t['log_Z_error'] for t in successful])
            mean_rel_error = np.mean([t['rel_error_pct'] for t in successful])
            std_rel_error = np.std([t['rel_error_pct'] for t in successful])
            mean_peak_dist = np.mean([t['peak_distance'] for t in successful])
            mean_time = np.mean([t['time'] for t in successful])
        else:
            mean_error = np.nan
            std_error = np.nan
            mean_rel_error = np.nan
            std_rel_error = np.nan
            mean_peak_dist = np.nan
            mean_time = np.nan
        
        noise_result = {
            'noise_level': noise_level,
            'trials': trial_results,
            'n_trials': n_trials,
            'success_rate': success_rate,
            'mean_log_Z_error': mean_error,
            'std_log_Z_error': std_error,
            'mean_rel_error_pct': mean_rel_error,
            'std_rel_error_pct': std_rel_error,
            'mean_peak_distance': mean_peak_dist,
            'mean_time': mean_time,
        }
        results['trials'].append(noise_result)
        
        if verbose:
            print(f"\nNoise level {noise_level:.2e} summary:")
            print(f"  Success rate:  {success_rate*100:.0f}%")
            if not np.isnan(mean_rel_error):
                print(f"  Mean error:    {mean_error:.4f} ± {std_error:.4f}")
                print(f"  Mean rel err:  {mean_rel_error:.2f}% ± {std_rel_error:.2f}%")
                print(f"  Mean time:     {mean_time:.1f}s")
    
    return results


def print_d256_summary(results):
    """Print D=256 summary"""
    print("\n" + "="*80)
    print(f"D={results['D']} EXTREME DIMENSION SUMMARY")
    print("="*80)
    print(f"log(Prior volume): {results['log_prior_volume']:.1f}")
    print(f"Analytical log(Z): {results['analytical_log_Z']:.4f}")
    print()
    
    print(f"{'Noise':>12} {'Success':>10} {'Rel Error':>15} {'Peak Dist':>15} {'Time (s)':>10}")
    print("-"*80)
    
    for noise_result in results['trials']:
        noise = noise_result['noise_level']
        success = noise_result['success_rate'] * 100
        rel_err = noise_result['mean_rel_error_pct']
        std_err = noise_result['std_rel_error_pct']
        peak_dist = noise_result['mean_peak_distance']
        time_s = noise_result['mean_time']
        
        if not np.isnan(rel_err):
            print(f"{noise:>12.2e} {success:>9.0f}% {rel_err:>7.2f}±{std_err:<5.2f}% "
                  f"{peak_dist:>14.4f} {time_s:>10.1f}")
        else:
            print(f"{noise:>12.2e} {success:>9.0f}% {'FAILED':>15} {'FAILED':>15} {'--':>10}")
    
    print("="*80)


def save_results(results, filename='extreme_dimension_results.npz'):
    """Save results to file"""
    np.savez(filename, **results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extreme dimension test - D=256')
    parser.add_argument('--dimension', type=int, default=256,
                       help='Dimension to test (default: 256)')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       help='Noise levels (default: 1e-4 1e-3 1e-2 5e-2 1e-1)')
    parser.add_argument('--trials', type=int, default=5,
                       help='Trials per noise level (default: 5)')
    parser.add_argument('--threshold-mult', type=int, default=50,
                       help='Threshold multiplier (default: 50)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU (NOT recommended for D=256!)')
    parser.add_argument('--output', type=str, default='extreme_dimension_results.npz',
                       help='Output file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    if args.dimension > 100 and args.no_gpu:
        print("WARNING: Running D>100 without GPU will be EXTREMELY slow!")
        print("Press Ctrl+C within 5 seconds to abort...")
        import time
        time.sleep(5)
    
    results = extreme_dimension_test(
        D=args.dimension,
        noise_levels=args.noise_levels,
        n_trials=args.trials,
        threshold_multiplier=args.threshold_mult,
        use_gpu=False if args.no_gpu else None,
        verbose=not args.quiet
    )
    
    print_d256_summary(results)
    save_results(results, args.output)
    
    print(f"\nD={results['D']} extreme dimension test complete!")
