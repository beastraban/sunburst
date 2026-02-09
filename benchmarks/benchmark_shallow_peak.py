#!/usr/bin/env python3
"""
Shallow Peak Analysis: When Noise >> Signal

Tests weak peaks where noise amplitude exceeds signal strength.
This should finally break SunBURST!

Peak height = A * exp(-0.5 * ||x||^2)
Where A << noise_level
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


def shallow_peak_analysis(
    peak_amplitudes=None,
    noise_level=0.1,  # Fixed noise level
    D=6,
    bounds_range=(-5.0, 5.0),
    n_trials=10,
    use_gpu=None,
    threshold_multiplier=50,
    verbose=True
):
    """
    Test weak peaks where signal < noise.
    
    Args:
        peak_amplitudes: List of peak heights A (where L_max = log(A))
        noise_level: Fixed noise amplitude
        D: Dimension
        bounds_range: Parameter bounds
        n_trials: Repetitions per amplitude
        use_gpu: Use GPU (None = auto-detect)
        threshold_multiplier: Multiplier for assess_noise
        verbose: Print progress
    """
    # Auto-detect GPU
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
        if verbose and GPU_AVAILABLE:
            print("GPU detected - using GPU acceleration")
    
    if peak_amplitudes is None:
        # Test from strong peak (A=1.0) to very weak (A=0.001)
        # noise_level = 0.1, so:
        # A=1.0: signal/noise = 10
        # A=0.1: signal/noise = 1
        # A=0.01: signal/noise = 0.1 (noise dominates!)
        # A=0.001: signal/noise = 0.01 (buried in noise)
        peak_amplitudes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    bounds = [(bounds_range[0], bounds_range[1])] * D
    bounds_arr = np.array(bounds)
    
    if verbose:
        print("="*70)
        print("Shallow Peak Analysis - Noise >> Signal")
        print("="*70)
        print(f"Dimension: {D}")
        print(f"Fixed noise level: {noise_level:.3f}")
        print(f"Peak amplitudes: {peak_amplitudes}")
        print(f"Signal/Noise ratios: {[A/noise_level for A in peak_amplitudes]}")
        print(f"Bounds: {bounds_range}")
        print(f"Trials per amplitude: {n_trials}")
        print(f"GPU: {use_gpu}")
        print("="*70)
    
    results = {
        'peak_amplitudes': peak_amplitudes,
        'noise_level': noise_level,
        'D': D,
        'trials': []
    }
    
    for amplitude in peak_amplitudes:
        snr = amplitude / noise_level  # Signal-to-noise ratio
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PEAK AMPLITUDE: {amplitude:.4f} (SNR = {snr:.2f})")
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
            
            def shallow_gaussian(x):
                """Weak Gaussian peak with fixed noise"""
                input_is_gpu = GPU_AVAILABLE and isinstance(x, cp.ndarray)
                
                if input_is_gpu:
                    x_gpu = cp.atleast_2d(x)
                    # L = log(A * exp(-0.5*r^2)) = log(A) - 0.5*r^2
                    clean = cp.log(amplitude) - 0.5 * cp.sum(x_gpu**2, axis=1)
                    noise = get_noise_vectorized(x_gpu, use_gpu_flag=True)
                    return clean + noise
                else:
                    x_np = np.atleast_2d(x)
                    clean = np.log(amplitude) - 0.5 * np.sum(x_np**2, axis=1)
                    noise = get_noise_vectorized(x_np, use_gpu_flag=False)
                    return clean + noise
            
            # Assess noise
            test_point = np.ones(D) * 0.5
            try:
                noise_stats = assess_noise(
                    shallow_gaussian,
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
                    shallow_gaussian,
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
            
            # Analyze results
            if not computation_failed and result.n_peaks > 0:
                # Peak should be at origin
                peak_distances = [np.linalg.norm(peak) for peak in result.peaks]
                min_peak_dist = min(peak_distances)
                
                # Check for spurious peaks
                tolerance = 3 * np.sqrt(noise_level)
                spurious = sum(1 for d in peak_distances if d > tolerance)
                
                if min_peak_dist < tolerance and spurious == 0:
                    outcome = "SUCCESS"
                elif min_peak_dist < tolerance and spurious > 0:
                    outcome = "PEAK_PLUS_SPURIOUS"
                elif min_peak_dist > tolerance:
                    outcome = "WRONG_PEAK"
                else:
                    outcome = "ONLY_SPURIOUS"
            else:
                min_peak_dist = np.nan
                spurious = 0
                
                if computation_failed:
                    outcome = "COMPUTATION_FAILED"
                else:
                    outcome = "NO_PEAKS_FOUND"
            
            # Store results
            trial_result = {
                'amplitude': amplitude,
                'snr': snr,
                'trial': trial,
                'outcome': outcome,
                'n_peaks_found': result.n_peaks,
                'min_peak_distance': min_peak_dist,
                'n_spurious': spurious,
                'assess_success': assess_success,
                'computation_failed': computation_failed,
                'failure_reason': failure_reason,
                'time': t_elapsed,
            }
            trial_results.append(trial_result)
            
            if verbose:
                print(f"  Result: {outcome}")
                print(f"  Peaks found: {result.n_peaks}")
                if not np.isnan(min_peak_dist):
                    print(f"  Peak distance: {min_peak_dist:.4f}")
                print(f"  Time: {t_elapsed:.2f}s")
        
        # Aggregate statistics
        outcome_counts = {}
        for outcome_type in ["SUCCESS", "PEAK_PLUS_SPURIOUS", "WRONG_PEAK", 
                            "ONLY_SPURIOUS", "NO_PEAKS_FOUND", "COMPUTATION_FAILED"]:
            outcome_counts[outcome_type] = sum(1 for t in trial_results if t['outcome'] == outcome_type)
        
        success_rate = outcome_counts["SUCCESS"] / n_trials
        
        amplitude_result = {
            'amplitude': amplitude,
            'snr': snr,
            'trials': trial_results,
            'n_trials': n_trials,
            'outcome_counts': outcome_counts,
            'success_rate': success_rate,
        }
        results['trials'].append(amplitude_result)
        
        if verbose:
            print(f"\nAmplitude {amplitude:.4f} (SNR={snr:.2f}) summary:")
            print(f"  Success rate: {success_rate*100:.0f}%")
            print(f"  Outcomes:")
            for outcome_type, count in outcome_counts.items():
                if count > 0:
                    print(f"    {outcome_type}: {count}/{n_trials}")
    
    return results


def print_shallow_summary(results):
    """Print shallow peak summary"""
    print("\n" + "="*80)
    print("SHALLOW PEAK ANALYSIS SUMMARY")
    print("="*80)
    print(f"Fixed noise level: {results['noise_level']:.3f}")
    print()
    
    print(f"{'Amplitude':>10} {'SNR':>8} {'Success':>10} {'Pk+Spur':>10} {'WrongPk':>10} {'NoFind':>10} {'Failed':>10}")
    print("-"*80)
    
    for amp_result in results['trials']:
        amp = amp_result['amplitude']
        snr = amp_result['snr']
        c = amp_result['outcome_counts']
        
        print(f"{amp:>10.4f} {snr:>8.2f} "
              f"{c['SUCCESS']:>10d} "
              f"{c['PEAK_PLUS_SPURIOUS']:>10d} "
              f"{c['WRONG_PEAK']:>10d} "
              f"{c['NO_PEAKS_FOUND']:>10d} "
              f"{c['COMPUTATION_FAILED']:>10d}")
    
    print("="*80)
    
    # Find critical SNR
    print("\nCRITICAL THRESHOLDS:")
    for threshold in [0.99, 0.95, 0.90, 0.50]:
        for amp_result in results['trials']:
            if amp_result['success_rate'] < threshold:
                snr = amp_result['snr']
                amp = amp_result['amplitude']
                print(f"  {threshold*100:.0f}% success lost at SNR = {snr:.2f} (amplitude = {amp:.4f})")
                break


def save_results(results, filename='shallow_peak_results.npz'):
    """Save results to file"""
    np.savez(filename, **results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Shallow peak analysis - weak signals')
    parser.add_argument('--amplitudes', nargs='+', type=float,
                       help='Peak amplitudes to test')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Fixed noise level (default: 0.1)')
    parser.add_argument('--dimension', type=int, default=6,
                       help='Problem dimension (default: 6)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Trials per amplitude (default: 10)')
    parser.add_argument('--threshold-mult', type=int, default=50,
                       help='Threshold multiplier (default: 50)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU')
    parser.add_argument('--output', type=str, default='shallow_peak_results.npz',
                       help='Output file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    results = shallow_peak_analysis(
        peak_amplitudes=args.amplitudes,
        noise_level=args.noise_level,
        D=args.dimension,
        n_trials=args.trials,
        threshold_multiplier=args.threshold_mult,
        use_gpu=False if args.no_gpu else None,
        verbose=not args.quiet
    )
    
    print_shallow_summary(results)
    save_results(results, args.output)
    
    print("\nShallow peak analysis complete!")
