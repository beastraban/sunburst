#!/usr/bin/env python3
"""
CosmoPower-JAX + SunBURST Evidence Calculation with Auto-Tuning
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

This example demonstrates:
  1. Using assess_noise() to automatically determine appropriate thresholds
  2. Computing Bayesian evidence for cosmological parameters
  3. Extracting 68%/95% credible intervals from Laplace approximation
  4. Generating publication-quality corner plots

Requirements:
  pip install sunburst jax jaxlib cosmopower-jax scipy matplotlib

Run:
  python cosmoJaxBurst_example.py
"""
import sys
sys.modules["cupy"] = None  # Disable CuPy, use CPU

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

from sunburst import compute_evidence
from sunburst.utils import assess_noise
from sunburst.utils.sunburst_posterior import PosteriorAnalysis


def main():
    print("="*70)
    print("SunBURST × CosmoPower-JAX: Evidence + Posterior Analysis")
    print("="*70)
    print()
    print("  ⚠  SIMPLIFIED LIKELIHOOD — not a production Planck analysis.")
    print("     Diagonal covariance + emulator noise inflation (~0.5%).")
    print("     See docstring for details.")

    # ==========================================================================
    # 1. Load CosmoPower Emulator
    # ==========================================================================
    print("\n[1. Loading CosmoPower-JAX emulator...]")
    emu = CPJ(probe="cmb_tt")

    theta0 = np.array([0.0224, 0.119, 0.67, 0.054, 0.965, 3.044], dtype=np.float64)
    param_names = ['ω_b', 'ω_cdm', 'h', 'τ', 'n_s', 'ln10¹⁰A_s']
    param_latex = [r'$\omega_b$', r'$\omega_{cdm}$', r'$h$',
                   r'$\tau$', r'$n_s$', r'$\ln(10^{10}A_s)$']
    D = theta0.size

    print(f"  Parameters ({D}): {param_names}")
    print(f"  Fiducial: {theta0}")

    # ==========================================================================
    # 2. Create Likelihood Function
    # ==========================================================================
    print("\n[2. Creating likelihood function...]")

    def predict_one(theta_1d):
        y = emu.predict(theta_1d[None, :])
        y = jnp.asarray(y)
        if y.ndim == 2:
            y = y[0]
        return y

    predict_batch = jax.jit(jax.vmap(predict_one))

    Cl0 = np.asarray(predict_one(jnp.asarray(theta0)))
    L = int(Cl0.shape[0])

    # --- Observational noise ---
    # Planck-like: ~1% at low ℓ, rising at high ℓ due to beam/noise
    rng = np.random.default_rng(0)
    frac = 1e-2  # 1% relative error baseline
    abs_floor = 1e-5 * float(np.max(np.abs(Cl0)))
    sigma_obs = frac * np.abs(Cl0) + abs_floor

    # --- Emulator prediction noise ---
    # Measure by evaluating the same point multiple times and checking scatter
    # (neural network emulators are deterministic per-call, but have systematic
    # approximation error that varies across parameter space)
    # Estimate emulator error as fractional error on the prediction
    emu_frac = 5e-3  # ~0.5% emulator approximation error (typical for CosmoPower)
    sigma_emu = emu_frac * np.abs(Cl0) + abs_floor

    # --- Total error budget ---
    # σ²_total = σ²_obs + σ²_emu  (emulator noise broadens the posterior)
    sigma_total = np.sqrt(sigma_obs**2 + sigma_emu**2)

    Cl_obs = Cl0 + rng.normal(0.0, sigma_obs, size=L)

    print(f"  Mock data: {L} multipoles")
    print(f"  Observational noise: ~{frac*100:.0f}% relative")
    print(f"  Emulator noise:      ~{emu_frac*100:.1f}% relative")
    print(f"  Total σ inflation:   ~{np.mean(sigma_total/sigma_obs):.2f}×")

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
        """Log-likelihood for SunBURST (shifted, NumPy)."""
        x_np = np.asarray(x, dtype=np.float64)
        ll = loglike_jax(jnp.asarray(x_np, dtype=jnp.float64))
        ll_np = np.array(ll, dtype=np.float64, copy=True)
        ll_np = ll_np - LL_REF
        ll_np[~np.isfinite(ll_np)] = LOG_L_FLOOR
        ll_np = np.maximum(ll_np, LOG_L_FLOOR)
        return ll_np

    print(f"  Baseline shift: LL_REF = {LL_REF:.2f}")

    # Parameter bounds — must be wide enough that posterior isn't truncated!
    # Rule of thumb: ±5σ_posterior minimum.  These are Planck-like σ estimates.
    #                  ω_b     ω_cdm    h       τ       n_s     ln(10^10 A_s)
    sigma_prior = np.array([
        0.00015,  # ω_b:     Planck σ ~ 0.00015
        0.0012,   # ω_cdm:   Planck σ ~ 0.0012
        0.005,    # h:       Planck σ ~ 0.005
        0.007,    # τ:       Planck σ ~ 0.007
        0.004,    # n_s:     Planck σ ~ 0.004
        0.015,    # ln(A_s): Planck σ ~ 0.015
    ], dtype=np.float64)

    n_sigma = 8  # ±8σ to be safe (prior should not truncate posterior)
    widths = n_sigma * sigma_prior
    bounds = [(float(t - w), float(t + w)) for t, w in zip(theta0, widths)]

    # Clamp to physically valid / emulator-trained domain
    emu_domain = [
        (0.005,  0.040),   # ω_b
        (0.001,  0.990),   # ω_cdm
        (0.200,  1.000),   # h
        (0.010,  0.800),   # τ  (must be > 0)
        (0.840,  1.100),   # n_s
        (1.610,  5.000),   # ln(10^10 A_s)
    ]
    bounds = [
        (max(lo, emu_lo), min(hi, emu_hi))
        for (lo, hi), (emu_lo, emu_hi) in zip(bounds, emu_domain)
    ]
    bounds_arr = np.array(bounds, dtype=np.float64)

    print(f"  Bounds: ±{n_sigma}σ around fiducial (clamped to emulator domain)")
    for name, (lo, hi), t in zip(param_names, bounds, theta0):
        n_sig = (hi - lo) / (2 * sigma_prior[param_names.index(name)])
        print(f"    {name:12s}: [{lo:.5f}, {hi:.5f}]  ({n_sig:.1f}σ)")

    # ==========================================================================
    # 3. Auto-Tune Noise Thresholds
    # ==========================================================================
    print("\n[3. Auto-tuning noise thresholds...]")

    noise_stats = assess_noise(
        log_likelihood, theta0, bounds_arr,
        n_samples=20, eps_range=(1e-6, 1e-3), verbose=True
    )

    stick_tol = noise_stats['stick_tolerance_recommended'] * 50
    grad_tol = noise_stats['grad_threshold_recommended'] * 50
    saddle_tol = noise_stats['saddle_threshold_recommended'] * 50
    fd_eps = noise_stats['fd_eps_recommended']

    print(f"\n  Using 50× recommended thresholds (conservative)")

    # ==========================================================================
    # 4. Compute Evidence
    # ==========================================================================
    print("\n[4. Computing Bayesian evidence...]")

    result = compute_evidence(
        log_likelihood, bounds,
        fast=False, n_oscillations=3, verbose=True, use_gpu=False,
        stick_tolerance=stick_tol, grad_threshold=grad_tol,
        saddle_threshold=saddle_tol, fd_eps=fd_eps,
    )

    # ==========================================================================
    # 5. Posterior Analysis: 68% and 95% Credible Intervals
    # ==========================================================================
    print("\n[5. Posterior constraints...]")

    # FD fallback likelihood (only used if pipeline didn't expose Hessian)
    def log_likelihood_np(x):
        """NumPy-compatible wrapper for FD Hessian fallback."""
        x_np = np.asarray(x, dtype=np.float64)
        ll = loglike_jax(jnp.asarray(x_np, dtype=jnp.float64))
        ll_np = np.array(ll, dtype=np.float64, copy=True)
        ll_np[~np.isfinite(ll_np)] = LOG_L_FLOOR
        ll_np = np.maximum(ll_np, LOG_L_FLOOR)
        return ll_np

    pa = PosteriorAnalysis(
        result, bounds,
        param_names=param_latex,
        log_likelihood=log_likelihood_np,  # FD fallback only
        fd_eps=5e-3,  # ~1% of posterior σ (not prior width)
    )

    pa.print_credible_intervals(peak_idx=0)
    pa.print_diagnostics(peak_idx=0)

    # ==========================================================================
    # 6. Corner Plot
    # ==========================================================================
    print("\n[6. Corner plot...]")

    fig = pa.corner_plot(
        filename='cosmo_corner.png',
        peak_idx=0,
        truths=theta0,
        n_samples=100_000,
        color='#1f77b4',
        truth_color='#d62728',
        dpi=200,
    )

    # ==========================================================================
    # 7. Results Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"  log(Evidence):     {result.log_evidence:.4f}")
    print(f"  Uncertainty:       {result.log_evidence_std:.4f}")
    print(f"  Number of peaks:   {result.n_peaks}")
    print(f"  Wall time:         {result.wall_time:.2f}s")
    print(f"  Likelihood calls:  {result.n_likelihood_calls}")
    print(f"  Corner plot:       cosmo_corner.png")

    if result.n_peaks > 0:
        print("\n  Peak locations:")
        for i, (peak, L_peak) in enumerate(zip(result.peaks, result.L_peaks)):
            print(f"    Peak {i+1}:")
            for j, (name, val, true) in enumerate(zip(param_names, peak, theta0)):
                diff = val - true
                pct = 100 * diff / true if true != 0 else 0
                print(f"      {name:12s} = {val:.6f}  (true: {true:.6f}, Δ={diff:+.6f}, {pct:+.2f}%)")
            print(f"      log(L) = {L_peak:.4f}")

    print("="*70)

    # Save LaTeX table
    with open('cosmo_constraints.tex', 'w') as f:
        f.write(pa.latex_table())
    print(f"  LaTeX table:       cosmo_constraints.tex")

    return result, pa


if __name__ == "__main__":
    result, pa = main()
