#!/usr/bin/env python3
"""
CosmoPower-JAX + SunBURST with Full GPU Acceleration
+ Posterior Analysis: 68%/95% CL + Corner Plots

IMPORTANT — NOISE MODEL
========================
This example uses a SIMPLIFIED likelihood for demonstration purposes.
It is NOT a production Planck analysis.  Key simplifications:

  1. Diagonal covariance: each multipole ℓ is treated as independent with
     σ_obs(ℓ) = 1% × |Cℓ| + floor.  A real Planck analysis uses the full
     pixel/harmonic covariance, beam transfer functions, and foreground
     marginalization (via the `clik` likelihood code).

  2. Emulator approximation noise: CosmoPower's neural network predictions
     differ from exact Boltzmann codes (CAMB/CLASS) at the ~0.1-1% level.
     The CosmoPower papers validate that this is sub-dominant relative to
     real experimental noise, so they do NOT propagate it into the error
     budget.  However, in our simplified setup with small per-ℓ errors and
     many multipoles, the emulator error is NOT negligible — omitting it
     leads to artificially tight constraints.  We therefore inflate the
     error budget:  σ²_total = σ²_obs + σ²_emu,  with σ_emu ≈ 0.5% × |Cℓ|.

  3. Prior bounds: set to ±8σ (Planck-like widths) to avoid truncating the
     posterior.  Clamped to the CosmoPower training domain.

The resulting constraints are qualitatively correct (right degeneracy
directions, reasonable σ) but should not be compared quantitatively to
published Planck results.  For that, use the actual Planck likelihood.

Requirements:
  pip install sunburst jax[cuda12] cosmopower-jax cupy-cuda12x scipy matplotlib

Run:
  python cosmoJaxBurst_gpu.py
"""
import os
os.environ['JAX_PLATFORMS'] = 'gpu,cpu'

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from sunburst import compute_evidence
from sunburst.utils import assess_noise
from sunburst.utils.sunburst_posterior import PosteriorAnalysis


def main():
    print("="*70)
    print("SunBURST × CosmoPower-JAX: Full GPU + Posterior Analysis")
    print("="*70)
    print()
    print("  ⚠  SIMPLIFIED LIKELIHOOD — not a production Planck analysis.")
    print("     Diagonal covariance + emulator noise inflation (~0.5%).")
    print("     See docstring for details.")

    jax_devices = jax.devices()
    jax_gpu = any('gpu' in str(d).lower() for d in jax_devices)
    print(f"\n  JAX GPU: {jax_gpu}  |  CuPy: {CUPY_AVAILABLE}")

    # ==========================================================================
    # 1. Load Emulator + Create Likelihood
    # ==========================================================================
    print("\n[1. Loading CosmoPower-JAX emulator...]")
    emu = CPJ(probe="cmb_tt")

    theta0 = np.array([0.0224, 0.119, 0.67, 0.054, 0.965, 3.044], dtype=np.float64)
    param_names = ['ω_b', 'ω_cdm', 'h', 'τ', 'n_s', 'ln10¹⁰A_s']
    param_latex = [r'$\omega_b$', r'$\omega_{cdm}$', r'$h$',
                   r'$\tau$', r'$n_s$', r'$\ln(10^{10}A_s)$']
    D = theta0.size

    def predict_one(theta_1d):
        y = emu.predict(theta_1d[None, :])
        y = jnp.asarray(y)
        if y.ndim == 2:
            y = y[0]
        return y

    predict_batch = jax.jit(jax.vmap(predict_one))

    Cl0 = np.asarray(predict_one(jnp.asarray(theta0)))
    L = int(Cl0.shape[0])

    rng = np.random.default_rng(0)
    frac = 1e-2
    abs_floor = 1e-5 * float(np.max(np.abs(Cl0)))
    sigma_obs = frac * np.abs(Cl0) + abs_floor

    # Emulator approximation noise (~0.5% typical for CosmoPower)
    emu_frac = 5e-3
    sigma_emu = emu_frac * np.abs(Cl0) + abs_floor
    sigma_total = np.sqrt(sigma_obs**2 + sigma_emu**2)

    Cl_obs = Cl0 + rng.normal(0.0, sigma_obs, size=L)

    Cl_obs_j = jnp.asarray(Cl_obs, dtype=jnp.float64)
    inv_var_j = jnp.asarray(1.0 / (sigma_total ** 2), dtype=jnp.float64)
    LOG_L_FLOOR = -1e8

    @jax.jit
    def loglike_jax(theta_batch):
        Cl_pred = predict_batch(theta_batch).astype(jnp.float64)
        resid = Cl_pred - Cl_obs_j[None, :]
        chi2 = jnp.sum(resid * resid * inv_var_j[None, :], axis=1)
        ll = -0.5 * chi2
        ll = jnp.where(jnp.isfinite(ll), ll, LOG_L_FLOOR)
        return jnp.maximum(ll, LOG_L_FLOOR)

    LL_REF = float(np.array(
        loglike_jax(jnp.asarray(theta0[None, :], dtype=jnp.float64)),
        dtype=np.float64
    )[0])

    def log_likelihood(x):
        """GPU pipeline: returns CuPy when GPU mode active."""
        x_np = np.asarray(x, dtype=np.float64)
        ll = loglike_jax(jnp.asarray(x_np, dtype=jnp.float64))
        ll_np = np.array(ll, dtype=np.float64, copy=True)
        ll_np = ll_np - LL_REF
        ll_np[~np.isfinite(ll_np)] = LOG_L_FLOOR
        ll_np = np.maximum(ll_np, LOG_L_FLOOR)
        if CUPY_AVAILABLE:
            return cp.asarray(ll_np)
        return ll_np

    def log_likelihood_numpy(x):
        """NumPy-only version for assess_noise and posterior analysis."""
        x_np = np.asarray(x, dtype=np.float64)
        ll = loglike_jax(jnp.asarray(x_np, dtype=jnp.float64))
        ll_np = np.array(ll, dtype=np.float64, copy=True)
        ll_np = ll_np - LL_REF
        ll_np[~np.isfinite(ll_np)] = LOG_L_FLOOR
        ll_np = np.maximum(ll_np, LOG_L_FLOOR)
        return ll_np

    print(f"  {L} multipoles, LL_REF = {LL_REF:.2f}")

    # Bounds: ±8σ using Planck-like posterior widths
    sigma_prior = np.array([0.00015, 0.0012, 0.005, 0.007, 0.004, 0.015], dtype=np.float64)
    n_sigma = 8
    widths = n_sigma * sigma_prior
    bounds = [(float(t - w), float(t + w)) for t, w in zip(theta0, widths)]
    bounds_arr = np.array(bounds, dtype=np.float64)

    # ==========================================================================
    # 2. Auto-Tune + Compute Evidence
    # ==========================================================================
    print("\n[2. Auto-tuning...]")
    noise_stats = assess_noise(
        log_likelihood_numpy, theta0, bounds_arr,
        n_samples=20, eps_range=(1e-6, 1e-3), verbose=True
    )

    multiplier = 50
    stick_tol = noise_stats['stick_tolerance_recommended'] * multiplier
    grad_tol = noise_stats['grad_threshold_recommended'] * multiplier
    saddle_tol = noise_stats['saddle_threshold_recommended'] * multiplier
    fd_eps = noise_stats['fd_eps_recommended']

    print(f"\n[3. Computing evidence (GPU={CUPY_AVAILABLE})...]")
    result = compute_evidence(
        log_likelihood, bounds,
        fast=False, n_oscillations=3, verbose=True, use_gpu=CUPY_AVAILABLE,
        stick_tolerance=stick_tol, grad_threshold=grad_tol,
        saddle_threshold=saddle_tol, fd_eps=fd_eps,
    )

    # ==========================================================================
    # 3. Posterior Analysis (CPU — plotting doesn't need GPU)
    # ==========================================================================
    print("\n[4. Posterior analysis...]")

    pa = PosteriorAnalysis(
        result, bounds,
        param_names=param_latex,
        log_likelihood=log_likelihood_numpy,  # FD fallback (NumPy)
        fd_eps=5e-3,  # ~1% of posterior σ
    )

    pa.print_credible_intervals(peak_idx=0)

    # ==========================================================================
    # 4. Corner Plot
    # ==========================================================================
    print("\n[5. Corner plot...]")

    fig = pa.corner_plot(
        filename='cosmo_corner_gpu.png',
        peak_idx=0,
        truths=theta0,
        n_samples=100_000,
        color='#1f77b4',
        truth_color='#d62728',
        dpi=200,
    )

    # ==========================================================================
    # 5. Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"  log(Evidence):     {result.log_evidence:.4f}")
    print(f"  Uncertainty:       {result.log_evidence_std:.4f}")
    print(f"  Peaks found:       {result.n_peaks}")
    print(f"  Wall time:         {result.wall_time:.2f}s")
    print(f"  GPU pipeline:      JAX={'✓' if jax_gpu else '✗'}  CuPy={'✓' if CUPY_AVAILABLE else '✗'}")
    print(f"  Corner plot:       cosmo_corner_gpu.png")

    if result.n_peaks > 0:
        print("\n  Peak locations:")
        for i, (peak, L_peak) in enumerate(zip(result.peaks, result.L_peaks)):
            print(f"    Peak {i+1}:")
            for j, (name, val, true) in enumerate(zip(param_names, peak, theta0)):
                diff = val - true
                pct = 100 * diff / true if true != 0 else 0
                print(f"      {name:12s} = {val:.6f}  (Δ={diff:+.6f}, {pct:+.2f}%)")
            print(f"      log(L) = {L_peak:.4f}")

    print("="*70)

    # Save LaTeX table
    with open('cosmo_constraints_gpu.tex', 'w') as f:
        f.write(pa.latex_table())
    print(f"  LaTeX table:       cosmo_constraints_gpu.tex")

    return result, pa


if __name__ == "__main__":
    result, pa = main()
