#!/usr/bin/env python3
"""
Minimal Example: SunBURST with Noisy Likelihood + Auto-Tuning
+ Posterior Analysis: 68%/95% CL + Corner Plots

This example shows the basic pattern for using SunBURST with noisy functions:
1. Define a noisy likelihood function
2. Use assess_noise() to automatically determine thresholds
3. Compute evidence with recommended thresholds
4. Extract 68%/95% credible intervals from Laplace approximation
5. Generate publication-quality corner plot

Run:
  python minimal_noisy_example.py
"""
import sys
sys.modules["cupy"] = None  # Disable GPU (force CPU-only mode)

import numpy as np
from scipy.stats import norm
from sunburst import compute_evidence
from sunburst.utils import assess_noise
from sunburst.utils.sunburst_posterior import PosteriorAnalysis


def main():
    print("="*70)
    print("SunBURST: Noisy Gaussian — Evidence + Posterior Analysis")
    print("="*70)
    
    # ==========================================================================
    # 1. Define a Noisy Likelihood Function
    # ==========================================================================
    D = 6
    noise_level = 1e-4
    
    np.random.seed(42)
    n_basis = 50
    basis_freqs = np.random.randn(n_basis, D)
    basis_phases = np.random.randn(n_basis) * 2 * np.pi
    basis_coeffs = np.random.randn(n_basis)
    
    def noisy_gaussian(x):
        """6D Gaussian likelihood with fixed noise realization."""
        x = np.atleast_2d(x)
        clean = -0.5 * np.sum(x**2, axis=1)
        phases = np.dot(x, basis_freqs.T) + basis_phases[None, :]
        noise = np.sum(basis_coeffs[None, :] * np.sin(phases), axis=1) \
                * noise_level / np.sqrt(n_basis)
        return clean + noise
    
    bounds = [(-5.0, 5.0)] * D
    bounds_arr = np.array(bounds)
    test_point = np.ones(D) * 0.5
    param_names = [f'θ_{i+1}' for i in range(D)]
    
    print(f"\n  Dimension: {D}")
    print(f"  True peak: origin")
    print(f"  Noise level: {noise_level}")
    
    # ==========================================================================
    # 2. Assess Noise + Auto-Tune
    # ==========================================================================
    print("\n[Auto-tuning noise thresholds...]")
    
    noise_stats = assess_noise(
        noisy_gaussian, test_point, bounds_arr,
        n_samples=20, eps_range=(1e-6, 1e-3), verbose=True
    )
    
    stick_tol = noise_stats['stick_tolerance_recommended'] * 50
    grad_tol = noise_stats['grad_threshold_recommended'] * 50
    saddle_tol = noise_stats['saddle_threshold_recommended'] * 50
    
    print(f"\n  Using 50× recommended thresholds (conservative)")
    
    # ==========================================================================
    # 3. Compute Evidence
    # ==========================================================================
    print("\n[Computing evidence...]")
    
    result = compute_evidence(
        noisy_gaussian, bounds,
        n_oscillations=2, verbose=True, use_gpu=False,
        stick_tolerance=stick_tol,
        grad_threshold=grad_tol,
        saddle_threshold=saddle_tol,
    )
    
    # ==========================================================================
    # 4. Posterior Analysis: 68% and 95% Credible Intervals
    # ==========================================================================
    print("\n[Computing posterior constraints...]")
    
    pa = PosteriorAnalysis(
        result, bounds,
        param_names=param_names,
        log_likelihood=noisy_gaussian,  # FD fallback
    )
    
    pa.print_credible_intervals(peak_idx=0)
    
    # ==========================================================================
    # 5. Corner Plot
    # ==========================================================================
    print("[Generating corner plot...]")
    
    true_peak = np.zeros(D)
    fig = pa.corner_plot(
        filename='noisy_corner.png',
        peak_idx=0,
        truths=true_peak,
        n_samples=80_000,
        color='#1f77b4',
        truth_color='#d62728',
        dpi=150,
    )
    
    # ==========================================================================
    # 6. Results Summary
    # ==========================================================================
    cdf_val = norm.cdf(5) - norm.cdf(-5)
    analytical_log_Z = D * (0.5 * np.log(2*np.pi) + np.log(cdf_val)) - np.log(10**D)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"  log(Evidence):      {result.log_evidence:.4f}")
    print(f"  Analytical log(Z):  {analytical_log_Z:.4f}")
    print(f"  Absolute error:     {abs(result.log_evidence - analytical_log_Z):.4f}")
    print(f"  Relative error:     {100*abs(result.log_evidence - analytical_log_Z)/abs(analytical_log_Z):.2f}%")
    print(f"  Peaks found:        {result.n_peaks}")
    print(f"  Wall time:          {result.wall_time:.2f}s")
    print(f"  Corner plot:        noisy_corner.png")
    print("="*70)
    
    # ==========================================================================
    # 7. Comparison: Default Thresholds
    # ==========================================================================
    print("\n[Comparison: default thresholds...]")
    
    result_default = compute_evidence(
        noisy_gaussian, bounds,
        n_oscillations=2, verbose=False, use_gpu=False,
    )
    
    print(f"  Auto-tuned (50×):  {result.n_peaks} peaks, log(Z) = {result.log_evidence:.4f}")
    print(f"  Defaults:          {result_default.n_peaks} peaks, log(Z) = {result_default.log_evidence:.4f}")
    print("="*70)
    
    return result, pa


if __name__ == "__main__":
    result, pa = main()
