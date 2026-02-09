#!/usr/bin/env python3
"""
SunBURST Dimension Scaling Analysis

Tests how noise tolerance degrades with increasing dimensionality.
The curse of dimensionality: at fixed noise, errors should grow with D.
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


def dimension_scaling_analysis(
    dimensions=None,
    noise_levels=None,
    bounds_range=(-5.0, 5.0),
    n_trials=5,
    use_gpu=None,
    threshold_multiplier=50,
    verbose=True
):
    """
    Test how errors scale with dimension at fixed noise levels.
    
    Args:
        dimensions: List of dimensions to test
        noise_levels: List of noise amplitudes
        bounds_range: Parameter bounds
        n_trials: Repetitions per (D, noise) combination
        use_gpu: Use GPU (None = auto-detect)
        threshold_multiplier: Multiplier for assess_noise recommendations
        verbose: Print progress
    
    Returns:
        Dictionary with scaling results
    """
    # Auto-detect GPU
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
        if verbose and GPU_AVAILABLE:
            print("GPU detected - using GPU acceleration")
    
    if dimensions is None:
        dimensions = [4, 6, 8, 10, 12, 16]
    
    if noise_levels is None:
        noise_levels = [1e-4, 1e-3, 1e-2, 5e-2]
    
    if verbose:
        print("="*70)
        print("SunBURST Dimension Scaling Analysis")
        print("="*70)
        print(f"Dimensions: {dimensions}")
        print(f"Noise levels: {noise_levels}")
        print(f"Trials per (D, noise): {n_trials}")
        print(f"Threshold multiplier: {threshold_multiplier}×")
        print(f"GPU: {use_gpu}")
        print("="*70)
    
    results = {
        'dimensions': dimensions,
        'noise_levels': noise_levels,
        'data': []
    }
    
    for D in dimensions:
        if verbose:
            print(f"\n{'='*70}")
            print(f"DIMENSION: {D}")
            print(f"{'='*70}")
        
        for noise_level in noise_levels:
            if verbose:
                print(f"\n  Noise: {noise_level:.2e}", end=" ")
            
            bounds = [(bounds_range[0], bounds_range[1])] * D
            bounds_arr = np.array(bounds)
            
            # Analytical evidence
            cdf_val = norm.cdf(bounds_range[1]) - norm.cdf(bounds_range[0])
            analytical_log_Z_raw = D * (0.5 * np.log(2*np.pi) + np.log(cdf_val))
            prior_volume = (bounds_range[1] - bounds_range[0])**D
            analytical_log_Z = analytical_log_Z_raw - np.log(prior_volume)
            
            trial_results = []
            
            for trial in range(n_trials):
                # Generate random noise basis
                np.random.seed(42 + trial + D*1000 + int(noise_level*1e6))
                n_basis = 50
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
                
                def noisy_gaussian(x):
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
                
                # Assess noise
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
                except:
                    stick_tol = noise_level * 10
                    grad_tol = noise_level * 10
                    saddle_tol = noise_level * 10
                
                # Compute evidence
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
                    
                    log_Z_error = abs(result.log_evidence - analytical_log_Z)
                    rel_error_pct = 100 * log_Z_error / abs(analytical_log_Z)
                    
                    if result.n_peaks > 0:
                        peak_dist = np.linalg.norm(result.peaks[0])
                        success = True
                    else:
                        peak_dist = np.nan
                        success = False
                    
                except Exception as e:
                    log_Z_error = np.nan
                    rel_error_pct = np.nan
                    peak_dist = np.nan
                    success = False
                
                t_elapsed = time.time() - t_start
                
                trial_results.append({
                    'success': success,
                    'log_Z_error': log_Z_error,
                    'rel_error_pct': rel_error_pct,
                    'peak_distance': peak_dist,
                    'time': t_elapsed,
                })
            
            # Aggregate
            success_rate = sum(t['success'] for t in trial_results) / n_trials
            
            successful = [t for t in trial_results if t['success']]
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
            
            results['data'].append({
                'D': D,
                'noise_level': noise_level,
                'analytical_log_Z': analytical_log_Z,
                'success_rate': success_rate,
                'mean_log_Z_error': mean_error,
                'std_log_Z_error': std_error,
                'mean_rel_error_pct': mean_rel_error,
                'std_rel_error_pct': std_rel_error,
                'mean_peak_distance': mean_peak_dist,
                'mean_time': mean_time,
            })
            
            if verbose:
                print(f"→ {success_rate*100:.0f}% success, ", end="")
                if not np.isnan(mean_rel_error):
                    print(f"{mean_rel_error:.2f}% error, {mean_time:.1f}s")
                else:
                    print("FAILED")
    
    return results


def print_scaling_summary(results):
    """Print dimension scaling summary"""
    print("\n" + "="*80)
    print("DIMENSION SCALING SUMMARY")
    print("="*80)
    
    for noise_level in results['noise_levels']:
        print(f"\nNoise = {noise_level:.2e}")
        print(f"{'Dim':>5} {'Success':>10} {'Rel Error':>15} {'Peak Dist':>15} {'Time (s)':>10}")
        print("-"*80)
        
        for data in results['data']:
            if data['noise_level'] == noise_level:
                D = data['D']
                success = data['success_rate'] * 100
                rel_err = data['mean_rel_error_pct']
                std_err = data['std_rel_error_pct']
                peak_dist = data['mean_peak_distance']
                time_s = data['mean_time']
                
                if not np.isnan(rel_err):
                    print(f"{D:>5} {success:>9.0f}% {rel_err:>7.2f}±{std_err:<5.2f}% "
                          f"{peak_dist:>14.4f} {time_s:>10.1f}")
                else:
                    print(f"{D:>5} {success:>9.0f}% {'FAILED':>15} {'FAILED':>15} {'--':>10}")
    
    print("="*80)


def save_results(results, filename='dimension_scaling_results.npz'):
    """Save results to file"""
    np.savez(filename, **results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SunBURST dimension scaling analysis')
    parser.add_argument('--dimensions', nargs='+', type=int,
                       help='Dimensions to test (default: 4 6 8 10 12 16)')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       help='Noise levels (default: 1e-4 1e-3 1e-2 5e-2)')
    parser.add_argument('--trials', type=int, default=5,
                       help='Trials per (D, noise) (default: 5)')
    parser.add_argument('--threshold-mult', type=int, default=50,
                       help='Threshold multiplier (default: 50)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU')
    parser.add_argument('--output', type=str, default='dimension_scaling_results.npz',
                       help='Output file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    results = dimension_scaling_analysis(
        dimensions=args.dimensions,
        noise_levels=args.noise_levels,
        n_trials=args.trials,
        threshold_multiplier=args.threshold_mult,
        use_gpu=False if args.no_gpu else None,
        verbose=not args.quiet
    )
    
    print_scaling_summary(results)
    save_results(results, args.output)
    
    print("\nDimension scaling analysis complete!")
