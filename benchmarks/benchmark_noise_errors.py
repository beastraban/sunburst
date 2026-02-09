#!/usr/bin/env python3
"""
Benchmark: SunBURST Error vs Noise Amplitude

Tests how evidence calculation accuracy degrades with increasing noise levels.
Validates that assess_noise() auto-tuning works across different noise regimes.
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


def run_benchmark(
    noise_levels=None,
    D=6,
    bounds_range=(-5.0, 5.0),
    n_trials=3,
    use_gpu=None,  # None = auto-detect
    verbose=True
):
    """
    Benchmark SunBURST error vs noise amplitude.
    
    Args:
        noise_levels: List of noise amplitudes to test
        D: Dimension
        bounds_range: Parameter bounds
        n_trials: Repetitions per noise level (different random noise realizations)
        use_gpu: Use GPU acceleration (None = auto-detect from GPU_AVAILABLE)
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    # Auto-detect GPU if not specified (SunBURST is GPU-native)
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
        if verbose and GPU_AVAILABLE:
            print("GPU detected - using GPU acceleration (SunBURST is GPU-native)")
    
    if noise_levels is None:
        # Test noise from 1e-6 (nearly clean) to 1e-2 (very noisy)
        noise_levels = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    
    bounds = [(bounds_range[0], bounds_range[1])] * D
    bounds_arr = np.array(bounds)
    
    # Analytical evidence (Bayesian, normalized by prior volume)
    cdf_val = norm.cdf(bounds_range[1]) - norm.cdf(bounds_range[0])
    analytical_log_Z_raw = D * (0.5 * np.log(2*np.pi) + np.log(cdf_val))
    prior_volume = (bounds_range[1] - bounds_range[0])**D
    analytical_log_Z = analytical_log_Z_raw - np.log(prior_volume)
    
    if verbose:
        print("="*70)
        print("SunBURST Noise Benchmark")
        print("="*70)
        print(f"Dimension: {D}")
        print(f"Bounds: {bounds_range}")
        print(f"Analytical log(Z): {analytical_log_Z:.4f}")
        print(f"Noise levels: {noise_levels}")
        print(f"Trials per level: {n_trials}")
        print(f"GPU: {use_gpu}")
        print("="*70)
    
    results = {
        'noise_levels': noise_levels,
        'analytical_log_Z': analytical_log_Z,
        'trials': []
    }
    
    for noise_level in noise_levels:
        if verbose:
            print(f"\n{'='*70}")
            print(f"NOISE LEVEL: {noise_level:.2e}")
            print(f"{'='*70}")
        
        trial_results = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Trial {trial+1}/{n_trials} ---")
            
            # Generate random noise basis for this trial
            np.random.seed(42 + trial)  # Different seed per trial
            n_basis = 50
            basis_freqs = np.random.randn(n_basis, D)
            basis_phases = np.random.randn(n_basis) * 2 * np.pi
            basis_coeffs = np.random.randn(n_basis)
            
            # Vectorized noise function (GPU-aware)
            def get_noise_vectorized(x, use_gpu=False):
                """Vectorized noise with GPU support"""
                if not use_gpu or not GPU_AVAILABLE:
                    xp = np
                    x = np.atleast_2d(x)
                    basis_freqs_gpu = basis_freqs
                    basis_phases_gpu = basis_phases
                    basis_coeffs_gpu = basis_coeffs
                else:
                    xp = cp
                    x = cp.atleast_2d(x)
                    basis_freqs_gpu = cp.asarray(basis_freqs)
                    basis_phases_gpu = cp.asarray(basis_phases)
                    basis_coeffs_gpu = cp.asarray(basis_coeffs)
                
                # Vectorized: (n_samples, n_basis)
                phases = xp.dot(x, basis_freqs_gpu.T) + basis_phases_gpu[None, :]
                contributions = basis_coeffs_gpu[None, :] * xp.sin(phases)
                noise = xp.sum(contributions, axis=1) * noise_level / xp.sqrt(n_basis)
                
                return noise
            
            def noisy_gaussian(x):
                """
                GPU-aware likelihood - automatically detects input type.
                Returns same type as input (NumPy in → NumPy out, CuPy in → CuPy out).
                """
                # Detect if input is already on GPU
                input_is_gpu = GPU_AVAILABLE and isinstance(x, cp.ndarray)
                
                # Get NumPy version for shape handling
                if input_is_gpu:
                    x_np = cp.asnumpy(cp.atleast_2d(x))
                else:
                    x_np = np.atleast_2d(x)
                
                # Compute on appropriate backend
                if input_is_gpu:
                    # GPU path: input was CuPy, return CuPy
                    x_gpu = cp.asarray(x_np, dtype=cp.float64)
                    clean_gpu = -0.5 * cp.sum(x_gpu**2, axis=1)
                    noise_gpu = get_noise_vectorized(x_gpu, use_gpu=True)
                    return clean_gpu + noise_gpu
                else:
                    # CPU path: input was NumPy, return NumPy
                    clean = -0.5 * np.sum(x_np**2, axis=1)
                    noise = get_noise_vectorized(x_np, use_gpu=False)
                    return clean + noise
            
            # Assess noise
            test_point = np.ones(D) * 0.5
            noise_stats = assess_noise(
                noisy_gaussian,
                test_point,
                bounds_arr,
                verbose=False
            )
            
            # Use conservative thresholds
            stick_tol = noise_stats['stick_tolerance_recommended'] * 50
            grad_tol = noise_stats['grad_threshold_recommended'] * 50
            saddle_tol = noise_stats['saddle_threshold_recommended'] * 50
            
            if verbose:
                print(f"  Auto-tuned thresholds (50×):")
                print(f"    stick_tolerance  = {stick_tol:.2e}")
                print(f"    grad_threshold   = {grad_tol:.2e}")
                print(f"    saddle_threshold = {saddle_tol:.2e}")
            
            # Compute evidence
            t_start = time.time()
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
            t_elapsed = time.time() - t_start
            
            # Calculate errors
            log_Z_error = abs(result.log_evidence - analytical_log_Z)
            rel_error_pct = 100 * log_Z_error / abs(analytical_log_Z)
            
            # Peak accuracy
            if result.n_peaks > 0:
                peak_distance = np.linalg.norm(result.peaks[0])
            else:
                peak_distance = np.nan
            
            trial_result = {
                'noise_level': noise_level,
                'trial': trial,
                'log_Z': result.log_evidence,
                'log_Z_error': log_Z_error,
                'rel_error_pct': rel_error_pct,
                'n_peaks': result.n_peaks,
                'peak_distance': peak_distance,
                'time': t_elapsed,
                'stick_tol': stick_tol,
                'grad_tol': grad_tol,
            }
            trial_results.append(trial_result)
            
            if verbose:
                print(f"  log(Z):        {result.log_evidence:.4f}")
                print(f"  Error:         {log_Z_error:.4f} ({rel_error_pct:.2f}%)")
                print(f"  Peaks found:   {result.n_peaks}")
                print(f"  Peak dist:     {peak_distance:.6f}")
                print(f"  Time:          {t_elapsed:.2f}s")
        
        # Aggregate statistics for this noise level
        log_Z_errors = [t['log_Z_error'] for t in trial_results]
        rel_errors = [t['rel_error_pct'] for t in trial_results]
        peak_dists = [t['peak_distance'] for t in trial_results if not np.isnan(t['peak_distance'])]
        
        noise_result = {
            'noise_level': noise_level,
            'trials': trial_results,
            'mean_log_Z_error': np.mean(log_Z_errors),
            'std_log_Z_error': np.std(log_Z_errors),
            'mean_rel_error_pct': np.mean(rel_errors),
            'std_rel_error_pct': np.std(rel_errors),
            'mean_peak_distance': np.mean(peak_dists) if peak_dists else np.nan,
            'std_peak_distance': np.std(peak_dists) if peak_dists else np.nan,
            'success_rate': len(peak_dists) / n_trials,
        }
        results['trials'].append(noise_result)
        
        if verbose:
            print(f"\nNoise level {noise_level:.2e} summary:")
            print(f"  Mean error:    {noise_result['mean_log_Z_error']:.4f} ± {noise_result['std_log_Z_error']:.4f}")
            print(f"  Mean rel err:  {noise_result['mean_rel_error_pct']:.2f}% ± {noise_result['std_rel_error_pct']:.2f}%")
            print(f"  Peak dist:     {noise_result['mean_peak_distance']:.6f} ± {noise_result['std_peak_distance']:.6f}")
            print(f"  Success rate:  {noise_result['success_rate']*100:.0f}%")
    
    return results


def print_summary_table(results):
    """Print summary table of results"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Analytical log(Z): {results['analytical_log_Z']:.4f}")
    print()
    print(f"{'Noise Level':>12} {'Mean Error':>12} {'Rel Error':>12} {'Peak Dist':>12} {'Success':>10}")
    print("-"*80)
    
    for noise_result in results['trials']:
        noise_level = noise_result['noise_level']
        mean_err = noise_result['mean_log_Z_error']
        std_err = noise_result['std_log_Z_error']
        mean_rel = noise_result['mean_rel_error_pct']
        std_rel = noise_result['std_rel_error_pct']
        mean_dist = noise_result['mean_peak_distance']
        std_dist = noise_result['std_peak_distance']
        success = noise_result['success_rate'] * 100
        
        print(f"{noise_level:>12.2e} {mean_err:>6.4f}±{std_err:<4.4f} {mean_rel:>6.2f}±{std_rel:<4.2f}% {mean_dist:>6.4f}±{std_dist:<4.4f} {success:>9.0f}%")
    
    print("="*80)


def save_results(results, filename='noise_benchmark_results.npz'):
    """Save results to file"""
    np.savez(filename, **results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark SunBURST error vs noise amplitude')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       help='Noise levels to test (default: 1e-6 to 1e-2)')
    parser.add_argument('--dimension', type=int, default=6,
                       help='Problem dimension (default: 6)')
    parser.add_argument('--trials', type=int, default=3,
                       help='Trials per noise level (default: 3)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration (GPU used by default if available)')
    parser.add_argument('--output', type=str, default='noise_benchmark_results.npz',
                       help='Output file for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Run benchmark (use_gpu=None means auto-detect, unless --no-gpu specified)
    results = run_benchmark(
        noise_levels=args.noise_levels,
        D=args.dimension,
        n_trials=args.trials,
        use_gpu=False if args.no_gpu else None,  # None = auto-detect
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    save_results(results, args.output)
    
    print("\nBenchmark complete!")
