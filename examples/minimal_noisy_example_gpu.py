#!/usr/bin/env python3
"""
GPU-Enabled Example: SunBURST with Noisy Likelihood + Auto-Tuning
+ Posterior Analysis: 68%/95% CL + Corner Plots

Requirements:
- pip install sunburst cupy-cuda12x scipy matplotlib

Run:
  python minimal_noisy_example_gpu.py
"""
import numpy as np
from scipy.stats import norm

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected — GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found — falling back to CPU")

from sunburst import compute_evidence
from sunburst.utils import assess_noise
from sunburst.utils.sunburst_posterior import PosteriorAnalysis


def main():
    print("="*70)
    print("SunBURST GPU: Noisy Gaussian — Evidence + Posterior Analysis")
    print("="*70)
    
    # ==========================================================================
    # 1. Define GPU-Accelerated Noisy Likelihood
    # ==========================================================================
    D = 6
    noise_level = 1e-4
    
    np.random.seed(42)
    n_basis = 50
    basis_freqs = np.random.randn(n_basis, D)
    basis_phases = np.random.randn(n_basis) * 2 * np.pi
    basis_coeffs = np.random.randn(n_basis)
    
    # Pre-transfer to GPU if available
    if GPU_AVAILABLE:
        basis_freqs_gpu = cp.asarray(basis_freqs)
        basis_phases_gpu = cp.asarray(basis_phases)
        basis_coeffs_gpu = cp.asarray(basis_coeffs)
    
    def noisy_gaussian_numpy(x):
        """NumPy version for assess_noise and posterior analysis."""
        x = np.atleast_2d(x)
        clean = -0.5 * np.sum(x**2, axis=1)
        phases = np.dot(x, basis_freqs.T) + basis_phases[None, :]
        noise = np.sum(basis_coeffs[None, :] * np.sin(phases), axis=1) \
                * noise_level / np.sqrt(n_basis)
        return clean + noise
    
    def noisy_gaussian_gpu(x):
        """GPU-aware: CuPy in → CuPy out, NumPy in → NumPy out."""
        if GPU_AVAILABLE and isinstance(x, cp.ndarray):
            x = cp.atleast_2d(x)
            clean = -0.5 * cp.sum(x**2, axis=1)
            phases = cp.dot(x, basis_freqs_gpu.T) + basis_phases_gpu[None, :]
            noise = cp.sum(basis_coeffs_gpu[None, :] * cp.sin(phases), axis=1) \
                    * noise_level / cp.sqrt(float(n_basis))
            return clean + noise
        else:
            return noisy_gaussian_numpy(np.asarray(x))
    
    bounds = [(-5.0, 5.0)] * D
    bounds_arr = np.array(bounds)
    test_point = np.ones(D) * 0.5
    param_names = [f'θ_{i+1}' for i in range(D)]
    
    print(f"\n  Dimension: {D},  GPU: {GPU_AVAILABLE}")
    
    # ==========================================================================
    # 2. Auto-Tune + Compute Evidence
    # ==========================================================================
    print("\n[Auto-tuning...]")
    noise_stats = assess_noise(
        noisy_gaussian_gpu, test_point, bounds_arr,
        n_samples=20, eps_range=(1e-6, 1e-3), verbose=True
    )
    
    multiplier = 10  # 10× sufficient when assessing away from peak
    stick_tol = noise_stats['stick_tolerance_recommended'] * multiplier
    grad_tol = noise_stats['grad_threshold_recommended'] * multiplier
    saddle_tol = noise_stats['saddle_threshold_recommended'] * multiplier
    
    print(f"\n[Computing evidence (GPU={GPU_AVAILABLE})...]")
    result = compute_evidence(
        noisy_gaussian_gpu, bounds,
        n_oscillations=2, verbose=True, use_gpu=GPU_AVAILABLE,
        stick_tolerance=stick_tol,
        grad_threshold=grad_tol,
        saddle_threshold=saddle_tol,
    )
    
    # ==========================================================================
    # 3. Posterior Analysis (CPU — plotting doesn't need GPU)
    # ==========================================================================
    print("\n[Posterior analysis...]")
    
    pa = PosteriorAnalysis(
        result, bounds,
        param_names=param_names,
        log_likelihood=noisy_gaussian_numpy,  # FD fallback (NumPy)
    )
    
    pa.print_credible_intervals(peak_idx=0)
    
    # ==========================================================================
    # 4. Corner Plot
    # ==========================================================================
    print("[Generating corner plot...]")
    
    true_peak = np.zeros(D)
    fig = pa.corner_plot(
        filename='noisy_corner_gpu.png',
        peak_idx=0,
        truths=true_peak,
        n_samples=80_000,
        dpi=150,
    )
    
    # ==========================================================================
    # 5. Summary
    # ==========================================================================
    cdf_val = norm.cdf(5) - norm.cdf(-5)
    analytical_log_Z = D * (0.5 * np.log(2*np.pi) + np.log(cdf_val)) - np.log(10**D)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"  log(Evidence):      {result.log_evidence:.4f}")
    print(f"  Analytical log(Z):  {analytical_log_Z:.4f}")
    print(f"  Error:              {abs(result.log_evidence - analytical_log_Z):.4f}")
    print(f"  Peaks:              {result.n_peaks}")
    print(f"  Wall time:          {result.wall_time:.2f}s")
    print(f"  GPU:                {'✓' if GPU_AVAILABLE else '✗'}")
    print(f"  Corner plot:        noisy_corner_gpu.png")
    print("="*70)
    
    return result, pa


if __name__ == "__main__":
    result, pa = main()
