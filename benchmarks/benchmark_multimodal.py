#!/usr/bin/env python3
"""
Multi-Modal Failure Analysis: Can Noise Create or Destroy Modes?

Tests two-peak Gaussian likelihood:
- Can noise split one peak into two?
- Can noise merge two peaks into one?
- Can noise create spurious extra peaks?
"""

import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
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


def multimodal_analysis(
    noise_levels=None,
    D=6,
    peak_separation=4.0,  # Distance between peaks
    bounds_range=(-10.0, 10.0),
    n_trials=10,
    use_gpu=None,
    threshold_multiplier=50,
    verbose=True
):
    """
    Test multi-modal detection under noise.
    
    Two Gaussians: one at [+d/2, 0, 0, ...], one at [-d/2, 0, 0, ...]
    
    Args:
        noise_levels: List of noise amplitudes
        D: Dimension
        peak_separation: Distance between peaks (along first axis)
        bounds_range: Parameter bounds
        n_trials: Repetitions per noise level
        use_gpu: Use GPU (None = auto-detect)
        threshold_multiplier: Multiplier for assess_noise
        verbose: Print progress
    """
    # Auto-detect GPU
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
        if verbose and GPU_AVAILABLE:
            print("GPU detected - using GPU acceleration")
    
    if noise_levels is None:
        # Test moderate to extreme noise
        noise_levels = [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
    
    bounds = [(bounds_range[0], bounds_range[1])] * D
    bounds_arr = np.array(bounds)
    
    # True peak locations
    peak1 = np.zeros(D)
    peak1[0] = peak_separation / 2
    peak2 = np.zeros(D)
    peak2[0] = -peak_separation / 2
    
    if verbose:
        print("="*70)
        print("Multi-Modal Failure Analysis")
        print("="*70)
        print(f"Dimension: {D}")
        print(f"Peak separation: {peak_separation}")
        print(f"Peak 1 at: [{peak_separation/2:.1f}, 0, 0, ...]")
        print(f"Peak 2 at: [{-peak_separation/2:.1f}, 0, 0, ...]")
        print(f"Bounds: {bounds_range}")
        print(f"Noise levels: {noise_levels}")
        print(f"Trials per level: {n_trials}")
        print(f"GPU: {use_gpu}")
        print("="*70)
    
    results = {
        'noise_levels': noise_levels,
        'D': D,
        'peak_separation': peak_separation,
        'peak1': peak1,
        'peak2': peak2,
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
            
            def two_peak_gaussian(x):
                """Two Gaussians with noise"""
                input_is_gpu = GPU_AVAILABLE and isinstance(x, cp.ndarray)
                
                if input_is_gpu:
                    x_gpu = cp.atleast_2d(x)
                    peak1_gpu = cp.asarray(peak1)
                    peak2_gpu = cp.asarray(peak2)
                    
                    # log(exp(L1) + exp(L2)) = logsumexp([L1, L2])
                    L1 = -0.5 * cp.sum((x_gpu - peak1_gpu)**2, axis=1)
                    L2 = -0.5 * cp.sum((x_gpu - peak2_gpu)**2, axis=1)
                    
                    # Stack and logsumexp
                    stacked = cp.stack([L1, L2], axis=0)
                    clean = cp.logaddexp(L1, L2)  # Numerically stable
                    
                    noise = get_noise_vectorized(x_gpu, use_gpu_flag=True)
                    return clean + noise
                else:
                    x_np = np.atleast_2d(x)
                    
                    L1 = -0.5 * np.sum((x_np - peak1)**2, axis=1)
                    L2 = -0.5 * np.sum((x_np - peak2)**2, axis=1)
                    
                    clean = np.logaddexp(L1, L2)  # Numerically stable
                    
                    noise = get_noise_vectorized(x_np, use_gpu_flag=False)
                    return clean + noise
            
            # Assess noise
            test_point = np.ones(D) * 0.5
            try:
                noise_stats = assess_noise(
                    two_peak_gaussian,
                    test_point,
                    bounds_arr,
                    verbose=False
                )
                
                stick_tol = noise_stats['stick_tolerance_recommended'] * threshold_multiplier
                grad_tol = noise_stats['grad_threshold_recommended'] * threshold_multiplier
                saddle_tol = noise_stats['saddle_threshold_recommended'] * threshold_multiplier
                assess_success = True
            except Exception as e:
                if verbose:
                    print(f"  WARNING: assess_noise failed: {e}")
                assess_success = False
                stick_tol = noise_level * 10
                grad_tol = noise_level * 10
                saddle_tol = noise_level * 10
            
            if verbose:
                print(f"  Thresholds ({threshold_multiplier}×):")
                print(f"    stick_tolerance  = {stick_tol:.2e}")
                print(f"    grad_threshold   = {grad_tol:.2e}")
            
            # Compute evidence
            t_start = time.time()
            try:
                result = compute_evidence(
                    two_peak_gaussian,
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
                class DummyResult:
                    n_peaks = 0
                    peaks = []
                result = DummyResult()
            
            t_elapsed = time.time() - t_start
            
            # Analyze peak detection
            if not computation_failed and result.n_peaks > 0:
                # Find which peaks are near the true peaks
                found_peaks = result.peaks
                
                # Distance to each true peak
                dists_to_peak1 = [np.linalg.norm(p - peak1) for p in found_peaks]
                dists_to_peak2 = [np.linalg.norm(p - peak2) for p in found_peaks]
                
                # A peak is "correct" if within 3*sqrt(noise) of a true peak
                tolerance = 3 * np.sqrt(noise_level)
                
                found_peak1 = any(d < tolerance for d in dists_to_peak1)
                found_peak2 = any(d < tolerance for d in dists_to_peak2)
                
                # Spurious peaks = peaks not near either true peak
                spurious = 0
                for p in found_peaks:
                    d1 = np.linalg.norm(p - peak1)
                    d2 = np.linalg.norm(p - peak2)
                    if d1 > tolerance and d2 > tolerance:
                        spurious += 1
                
                # Classify outcome
                if found_peak1 and found_peak2 and spurious == 0:
                    outcome = "BOTH_PEAKS_CORRECT"
                elif found_peak1 and found_peak2 and spurious > 0:
                    outcome = "BOTH_PEAKS_PLUS_SPURIOUS"
                elif (found_peak1 or found_peak2) and not (found_peak1 and found_peak2):
                    outcome = "ONE_PEAK_MISSING"
                elif spurious > 0 and not found_peak1 and not found_peak2:
                    outcome = "ONLY_SPURIOUS"
                else:
                    outcome = "NO_PEAKS_NEAR_TRUE"
                
                min_dist_1 = min(dists_to_peak1) if dists_to_peak1 else np.inf
                min_dist_2 = min(dists_to_peak2) if dists_to_peak2 else np.inf
                
            else:
                found_peak1 = False
                found_peak2 = False
                spurious = 0
                min_dist_1 = np.inf
                min_dist_2 = np.inf
                
                if computation_failed:
                    outcome = "COMPUTATION_FAILED"
                else:
                    outcome = "NO_PEAKS_FOUND"
            
            # Store results
            trial_result = {
                'noise_level': noise_level,
                'trial': trial,
                'outcome': outcome,
                'n_peaks_found': result.n_peaks,
                'found_peak1': found_peak1,
                'found_peak2': found_peak2,
                'n_spurious': spurious,
                'min_dist_to_peak1': min_dist_1,
                'min_dist_to_peak2': min_dist_2,
                'assess_success': assess_success,
                'computation_failed': computation_failed,
                'failure_reason': failure_reason,
                'time': t_elapsed,
            }
            trial_results.append(trial_result)
            
            if verbose:
                print(f"  Result: {outcome}")
                print(f"  Peaks found: {result.n_peaks} (spurious: {spurious})")
                if found_peak1 or found_peak2:
                    print(f"  Dist to peak1: {min_dist_1:.4f}, peak2: {min_dist_2:.4f}")
                print(f"  Time: {t_elapsed:.2f}s")
        
        # Aggregate statistics
        outcome_counts = {}
        for outcome_type in ["BOTH_PEAKS_CORRECT", "BOTH_PEAKS_PLUS_SPURIOUS", 
                            "ONE_PEAK_MISSING", "ONLY_SPURIOUS", "NO_PEAKS_NEAR_TRUE",
                            "NO_PEAKS_FOUND", "COMPUTATION_FAILED"]:
            outcome_counts[outcome_type] = sum(1 for t in trial_results if t['outcome'] == outcome_type)
        
        # Success = found both peaks correctly
        success_rate = outcome_counts["BOTH_PEAKS_CORRECT"] / n_trials
        
        noise_result = {
            'noise_level': noise_level,
            'trials': trial_results,
            'n_trials': n_trials,
            'outcome_counts': outcome_counts,
            'success_rate': success_rate,
        }
        results['trials'].append(noise_result)
        
        if verbose:
            print(f"\nNoise level {noise_level:.2e} summary:")
            print(f"  Success rate (both correct): {success_rate*100:.0f}%")
            print(f"  Outcomes:")
            for outcome_type, count in outcome_counts.items():
                if count > 0:
                    print(f"    {outcome_type}: {count}/{n_trials}")
    
    return results


def print_multimodal_summary(results):
    """Print multi-modal analysis summary"""
    print("\n" + "="*90)
    print("MULTI-MODAL ANALYSIS SUMMARY")
    print("="*90)
    print(f"Peak separation: {results['peak_separation']}")
    print()
    
    # Header
    print(f"{'Noise':>10} {'BothOK':>8} {'Both+Spur':>10} {'OneMiss':>8} {'OnlySpur':>10} {'NoNear':>8} {'NoFind':>8} {'Failed':>8}")
    print("-"*90)
    
    for noise_result in results['trials']:
        noise = noise_result['noise_level']
        c = noise_result['outcome_counts']
        
        print(f"{noise:>10.2e} "
              f"{c['BOTH_PEAKS_CORRECT']:>8d} "
              f"{c['BOTH_PEAKS_PLUS_SPURIOUS']:>10d} "
              f"{c['ONE_PEAK_MISSING']:>8d} "
              f"{c['ONLY_SPURIOUS']:>10d} "
              f"{c['NO_PEAKS_NEAR_TRUE']:>8d} "
              f"{c['NO_PEAKS_FOUND']:>8d} "
              f"{c['COMPUTATION_FAILED']:>8d}")
    
    print("="*90)


def save_results(results, filename='multimodal_analysis_results.npz'):
    """Save results to file"""
    np.savez(filename, **results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-modal failure analysis')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       help='Noise levels to test')
    parser.add_argument('--dimension', type=int, default=6,
                       help='Problem dimension (default: 6)')
    parser.add_argument('--peak-separation', type=float, default=4.0,
                       help='Distance between peaks (default: 4.0)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Trials per noise level (default: 10)')
    parser.add_argument('--threshold-mult', type=int, default=50,
                       help='Threshold multiplier (default: 50)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU')
    parser.add_argument('--output', type=str, default='multimodal_analysis_results.npz',
                       help='Output file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    results = multimodal_analysis(
        noise_levels=args.noise_levels,
        D=args.dimension,
        peak_separation=args.peak_separation,
        n_trials=args.trials,
        threshold_multiplier=args.threshold_mult,
        use_gpu=False if args.no_gpu else None,
        verbose=not args.quiet
    )
    
    print_multimodal_summary(results)
    save_results(results, args.output)
    
    print("\nMulti-modal analysis complete!")
