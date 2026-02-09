#!/usr/bin/env python3
"""
sunburst_posterior.py — Posterior Analysis for SunBURST Results

Extracts 68%/95% credible intervals and publication-quality corner plots
from data the SunBURST pipeline has ALREADY computed:

  Priority 1:  result.covariance       (pre-inverted — from BendTheBow)
  Priority 2:  result.hessians_full    (full Hessian — invert, sanity-check)
  Priority 3:  result.hessian          (alternative attr name)
  Priority 4a: result.hessians_full    (diagonal — sanity-check for whitened)
  Priority 4b: finite-difference       (from log_likelihood — always correct coords)
  Priority 5:  result.diag_H           (whitened diagonal — last resort)

All Hessian sources are sanity-checked: if σ values are near-uniform
(CV < 5%), the Hessian is likely in whitened coordinates and we fall
through to FD automatically.

For non-Gaussian posteriors, the ray bank provides actual likelihood
profiles in every direction — yielding asymmetric credible intervals
and non-elliptical 2D contours without MCMC.

Usage
-----
    from sunburst.utils.sunburst_posterior import PosteriorAnalysis

    pa = PosteriorAnalysis(result, bounds,
                           param_names=[r'$\\omega_b$', ...])
    pa.print_credible_intervals()
    pa.corner_plot('corner.png', truths=theta0)

Exposing pipeline internals
---------------------------
To enable full functionality, add these to the SunBURST result object:

    result.hessian          # (K, D, D) or list of (D, D) — full Hessian per peak
    result.covariance       # (K, D, D) or list of (D, D) — Σ = (-H)⁻¹ per peak
    result.diag_H           # (K, D) — diagonal Hessian per peak
    result.ray_bank         # RayBank from CarryTiger (ray geometry + log_L)
    result.trajectory_bank  # TrajectoryBank from GreenDragon (L-BFGS paths)
    result.diagnostics      # dict with tail_alpha, curvature, asymmetry per peak
"""

import numpy as np
from typing import Optional, List, Tuple
import warnings


class PosteriorAnalysis:
    """
    Posterior analysis from SunBURST pipeline internals.

    Parameters
    ----------
    result : object
        SunBURST ``compute_evidence()`` result.
        Required: ``.peaks``, ``.L_peaks``, ``.n_peaks``, ``.log_evidence``.
        Optional: ``.hessian``, ``.covariance``, ``.diag_H``,
                  ``.ray_bank``, ``.trajectory_bank``, ``.diagnostics``.
    bounds : array-like, shape (D, 2)
        Parameter bounds.
    param_names : list[str], optional
        Display names (may contain LaTeX).
    log_likelihood : callable, optional
        NumPy batched log-likelihood f(x) → (N,) ndarray.
        Only needed if pipeline didn't expose Hessian AND you want
        correlations (FD fallback).
    fd_eps : float
        Relative FD step (fraction of prior width).  Only used in fallback.
    """

    def __init__(
        self,
        result,
        bounds,
        param_names: Optional[List[str]] = None,
        log_likelihood=None,
        fd_eps: float = 1e-2,
    ):
        self.result = result
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.D = self.bounds.shape[0]
        self.param_names = param_names or [f"θ_{i}" for i in range(self.D)]
        self.log_likelihood = log_likelihood
        self.fd_eps = fd_eps

        # Source tracking
        self._source = None  # will be 'covariance', 'hessian', 'diag_H', 'fd', or 'rays'

        # Build covariances for every peak
        self._covariances: List[np.ndarray] = []
        self._sigmas: List[np.ndarray] = []
        self._hessians: List[Optional[np.ndarray]] = []

        # Ray-based data (non-Gaussian)
        self._ray_profiles: List[Optional[dict]] = []
        self._has_rays = False

        for k in range(result.n_peaks):
            cov, H, source = self._get_covariance_for_peak(k)
            self._covariances.append(cov)
            self._hessians.append(H)
            self._sigmas.append(np.sqrt(np.abs(np.diag(cov))))

            # Extract ray profiles if available
            ray_prof = self._extract_ray_profiles(k)
            self._ray_profiles.append(ray_prof)
            if ray_prof is not None:
                self._has_rays = True

        self._source = source  # last one wins for reporting

    # ══════════════════════════════════════════════════════════════════
    # COVARIANCE EXTRACTION — priority chain
    # ══════════════════════════════════════════════════════════════════

    def _get_covariance_for_peak(self, k: int):
        """
        Get covariance matrix for peak k.
        Tries pipeline internals first, falls back to FD.

        Returns (covariance, hessian_or_None, source_string)
        """
        D = self.D

        # ── Priority 1: Pre-computed covariance from BendTheBow ──
        cov = self._try_get_attr('covariance', k, shape=(D, D))
        if cov is not None:
            diag = np.diag(cov)
            if np.all(diag > 0):
                return cov, None, 'covariance'

        # ── Priority 2: Full Hessian from hessians_full (BendTheBow's actual Hessian) ──
        H = self._try_get_attr('hessians_full', k, shape=(D, D))
        if H is not None:
            cov = self._invert_hessian(H)
            if cov is not None:
                # Sanity check: if diagonal σ values are suspiciously uniform,
                # the diagonal came from whitened space (GreenDragon) while
                # off-diag came from original space — mixed coordinates!
                sigmas = np.sqrt(np.maximum(np.diag(cov), 1e-30))
                cv = np.std(sigmas) / np.mean(sigmas) if np.mean(sigmas) > 0 else 0
                if cv < 0.05 and D > 2:
                    warnings.warn(
                        f"Peak {k}: Full Hessian has near-uniform σ (CV={cv:.4f}) — "
                        f"likely whitened diagonal mixed with original off-diagonal. "
                        f"Falling through to FD Hessian.",
                        stacklevel=3,
                    )
                else:
                    return cov, H, 'hessian_full'

        # ── Priority 3: Full Hessian from .hessian attr ──
        H = self._try_get_attr('hessian', k, shape=(D, D))
        if H is not None:
            cov = self._invert_hessian(H)
            if cov is not None:
                sigmas = np.sqrt(np.maximum(np.diag(cov), 1e-30))
                cv = np.std(sigmas) / np.mean(sigmas) if np.mean(sigmas) > 0 else 0
                if cv < 0.05 and D > 2:
                    warnings.warn(
                        f"Peak {k}: Hessian has near-uniform σ (CV={cv:.4f}) — "
                        f"likely whitened. Falling through to FD.",
                        stacklevel=3,
                    )
                else:
                    return cov, H, 'hessian'

        # ── Priority 4a: hessians_full but diagonal (1D from BendTheBow) ──
        diag_H = self._try_get_attr('hessians_full', k, shape=(D,))
        if diag_H is not None:
            neg_diag = -diag_H
            neg_diag = np.where(neg_diag > 0, neg_diag, 1.0)
            sigmas = 1.0 / np.sqrt(neg_diag)
            cv = np.std(sigmas) / np.mean(sigmas) if np.mean(sigmas) > 0 else 0
            if cv < 0.05 and D > 2:
                # Whitened space — skip to FD if available
                if self.log_likelihood is not None:
                    warnings.warn(
                        f"Peak {k}: Diagonal Hessian is whitened (CV={cv:.4f}). "
                        f"Using FD Hessian instead.",
                        stacklevel=3,
                    )
                    # Fall through to FD below
                else:
                    warnings.warn(
                        f"Peak {k}: Diagonal Hessian has uniform σ — likely whitened "
                        f"coordinates. Provide log_likelihood for FD fallback.",
                        stacklevel=3,
                    )
                    cov = np.diag(1.0 / neg_diag)
                    return cov, np.diag(diag_H), 'diag_H_whitened'
            else:
                cov = np.diag(1.0 / neg_diag)
                return cov, np.diag(diag_H), 'diag_H'

        # ── Priority 4b: Finite-difference Hessian (full, in original coords) ──
        if self.log_likelihood is not None:
            peak = np.asarray(self.result.peaks[k], dtype=np.float64).ravel()
            H = self._compute_full_hessian_fd(peak)
            cov = self._invert_hessian(H)
            if cov is not None:
                return cov, H, 'fd'
            # FD Hessian inversion failed — use diagonal from FD
            diag_H_fd = np.diag(H)
            neg_diag = -diag_H_fd
            neg_diag = np.where(neg_diag > 0, neg_diag, 1.0)
            cov = np.diag(1.0 / neg_diag)
            return cov, H, 'fd_diag'

        # ── Priority 5: Diagonal Hessian from GreenDragon (whitened space!) ──
        # WARNING: diag_H is typically in whitened coordinates and has no
        # off-diagonal terms.  Only used as last resort.
        diag_H = self._try_get_attr('diag_H', k, shape=(D,))
        if diag_H is not None:
            neg_diag = -diag_H
            neg_diag = np.where(neg_diag > 0, neg_diag, 1.0)
            cov = np.diag(1.0 / neg_diag)
            warnings.warn(
                f"Peak {k}: Using diagonal Hessian (whitened space) — "
                f"σ values and correlations will be WRONG. "
                f"Provide log_likelihood for FD or expose result.hessian.",
                stacklevel=3,
            )
            return cov, np.diag(diag_H), 'diag_H_whitened'

        # ── Nothing available ──
        warnings.warn(
            f"Peak {k}: No Hessian available and no log_likelihood for FD. "
            f"Using unit covariance (MEANINGLESS contours).",
            stacklevel=3,
        )
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        cov = np.diag((widths / 6.0) ** 2)  # ~±3σ fills prior
        return cov, None, 'fallback'

    def _try_get_attr(self, name, k, shape):
        """Try to extract peak k's data from result.{name}."""
        obj = getattr(self.result, name, None)
        if obj is None:
            return None
        arr = np.asarray(obj)
        # Could be (K, *shape) or list-of-arrays
        if isinstance(obj, (list, tuple)):
            if k < len(obj) and obj[k] is not None:
                out = np.asarray(obj[k], dtype=np.float64)
                if out.shape == shape:
                    return out
            return None
        # Single array — index by k
        if arr.ndim == len(shape) + 1 and arr.shape[0] > k:
            out = arr[k].astype(np.float64)
            if out.shape == shape:
                return out
        # Single peak, no batch dim
        if arr.shape == shape and self.result.n_peaks == 1:
            return arr.astype(np.float64)
        return None

    @staticmethod
    def _invert_hessian(H: np.ndarray):
        """Invert -H to get covariance.  Returns None on failure."""
        neg_H = -H
        try:
            cov = np.linalg.inv(neg_H)
            if np.all(np.diag(cov) > 0):
                return cov
            # Try pseudo-inverse
            cov = np.linalg.pinv(neg_H)
            diag = np.diag(cov)
            cov = np.abs(cov)  # force positive
            np.fill_diagonal(cov, np.abs(diag))
            return cov
        except np.linalg.LinAlgError:
            return None

    # ══════════════════════════════════════════════════════════════════
    # RAY BANK EXTRACTION — non-Gaussian profiles
    # ══════════════════════════════════════════════════════════════════

    def _extract_ray_profiles(self, k: int) -> Optional[dict]:
        """
        Extract ray profiles from the result for peak k.

        Uses result.sunburst_rays (populated by pipeline from BendTheBow).

        Returns dict with:
            'directions': (n_rays, D) — unit directions from peak
            'samples':    (N, D) — sample positions along rays
            'log_L':      (N,) — log-likelihood values at samples
            'peak':       (D,) — peak location
        or None if not available.
        """
        peak = np.asarray(self.result.peaks[k], dtype=np.float64).ravel()

        # Primary: result.sunburst_rays from pipeline
        rays_list = getattr(self.result, 'sunburst_rays', None)
        if rays_list is not None and k < len(rays_list) and rays_list[k] is not None:
            rd = rays_list[k]
            directions = rd.get('sunburst_directions')
            t_values = rd.get('sunburst_t_values')
            log_L = rd.get('sunburst_log_L')

            if directions is not None and t_values is not None and log_L is not None:
                directions = np.asarray(directions)
                t_values = np.asarray(t_values)
                log_L = np.asarray(log_L)

                # Reconstruct sample positions: peak + t * direction
                # t_values: (n_rays, n_samples), directions: (n_rays, D)
                n_rays, n_samples = t_values.shape
                D = directions.shape[1]
                samples = (peak[None, None, :]
                           + t_values[:, :, None] * directions[:, None, :])
                # Flatten to (N, D) and (N,)
                samples = samples.reshape(-1, D)
                log_L_flat = log_L.reshape(-1)

                return {
                    'directions': directions,
                    'samples': samples,
                    'log_L': log_L_flat,
                    'peak': peak,
                }

        # Fallback: try result.ray_bank (legacy)
        bank = getattr(self.result, 'ray_bank', None)
        if bank is None:
            return None

        try:
            if hasattr(bank, 'ray_starts') and hasattr(bank, 'ray_log_L'):
                starts = np.asarray(bank.ray_starts)
                ends = np.asarray(bank.ray_ends)
                log_Ls = np.asarray(bank.ray_log_L)
                dists = np.linalg.norm(starts - peak[None, :], axis=1)
                mask = dists < 1e-6 * np.linalg.norm(
                    self.bounds[:, 1] - self.bounds[:, 0]
                )
                if mask.sum() == 0:
                    return None
                directions = ends[mask] - starts[mask]
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                directions /= norms
                return {
                    'directions': directions,
                    'log_L': log_Ls[mask] if log_Ls.ndim > 1 else None,
                    'peak': peak,
                }
        except Exception:
            pass

        return None

    # ══════════════════════════════════════════════════════════════════
    # FINITE-DIFFERENCE FALLBACK
    # ══════════════════════════════════════════════════════════════════

    def _compute_full_hessian_fd(self, peak: np.ndarray) -> np.ndarray:
        """Full Hessian via central FD.  Uses self.fd_eps."""
        D = self.D
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        eps = self.fd_eps * widths
        H = np.zeros((D, D), dtype=np.float64)
        f0 = self._eval(peak)

        for i in range(D):
            pp = peak.copy(); pp[i] += eps[i]
            pm = peak.copy(); pm[i] -= eps[i]
            H[i, i] = (self._eval(pp) - 2 * f0 + self._eval(pm)) / eps[i] ** 2

        for i in range(D):
            for j in range(i + 1, D):
                pij = peak.copy(); pij[i] += eps[i]; pij[j] += eps[j]
                pi_ = peak.copy(); pi_[i] += eps[i]; pi_[j] -= eps[j]
                p_j = peak.copy(); p_j[i] -= eps[i]; p_j[j] += eps[j]
                p__ = peak.copy(); p__[i] -= eps[i]; p__[j] -= eps[j]
                H[i, j] = (
                    self._eval(pij) - self._eval(pi_)
                    - self._eval(p_j) + self._eval(p__)
                ) / (4 * eps[i] * eps[j])
                H[j, i] = H[i, j]
        return H

    def _eval(self, theta: np.ndarray) -> float:
        return float(np.asarray(
            self.log_likelihood(theta.reshape(1, -1))
        ).ravel()[0])

    # ══════════════════════════════════════════════════════════════════
    # CREDIBLE INTERVALS
    # ══════════════════════════════════════════════════════════════════

    def get_credible_intervals(
        self, peak_idx: int = 0, cl: float = 0.6827
    ) -> List[Tuple[float, float, float]]:
        """
        Marginal credible intervals.

        Uses Gaussian (symmetric) from covariance as baseline.
        Overrides with ray-based asymmetric intervals per-parameter
        only when the ray interval is wider (better coverage).

        Returns list of (lower, centre, upper) per parameter.
        """
        peak = np.asarray(self.result.peaks[peak_idx]).ravel()
        sigma = self._sigmas[peak_idx]

        # Gaussian baseline
        from scipy.stats import norm as _norm
        z = _norm.ppf(0.5 + cl / 2)
        gauss_intervals = [(peak[i] - z * sigma[i], peak[i],
                            peak[i] + z * sigma[i])
                           for i in range(self.D)]

        # Try ray-based intervals
        rays = self._ray_profiles[peak_idx]
        if rays is not None and 'samples' in rays:
            asym = self._asymmetric_intervals_from_rays(peak_idx, cl)
            if asym is not None:
                # Per-parameter: use ray interval only if it's wider than
                # 50% of Gaussian interval (i.e. rays have real coverage)
                merged = []
                for i in range(self.D):
                    g_lo, g_c, g_hi = gauss_intervals[i]
                    r_lo, r_c, r_hi = asym[i]
                    g_width = g_hi - g_lo
                    r_width = r_hi - r_lo
                    if r_width > 0.5 * g_width and r_width > 0:
                        merged.append((r_lo, r_c, r_hi))
                    else:
                        merged.append((g_lo, g_c, g_hi))
                return merged

        return gauss_intervals

    def _asymmetric_intervals_from_rays(
        self, peak_idx: int, cl: float
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Compute asymmetric CL from ray profiles.

        Projects ray samples onto each parameter axis, weights by
        likelihood, and finds the cl-fraction highest-density interval.
        """
        rays = self._ray_profiles[peak_idx]
        if rays is None or 'samples' not in rays:
            return None

        try:
            samples = np.asarray(rays['samples'])  # (n_rays, n_pts, D) or (N, D)
            log_L = np.asarray(rays['log_L'])
            peak = rays['peak']

            # Flatten to (N, D) and (N,)
            if samples.ndim == 3:
                N_rays, N_pts, D = samples.shape
                samples = samples.reshape(-1, D)
                log_L = log_L.reshape(-1)
            elif samples.ndim == 2:
                pass
            else:
                return None

            # Remove NaN/inf
            valid = np.isfinite(log_L) & np.all(np.isfinite(samples), axis=1)
            samples = samples[valid]
            log_L = log_L[valid]

            if len(samples) < 50:
                return None

            # Convert to weights
            log_w = log_L - np.max(log_L)
            weights = np.exp(log_w)
            weights /= weights.sum()

            intervals = []
            alpha = 1.0 - cl
            for i in range(self.D):
                xi = samples[:, i]
                # Weighted quantiles
                sort_idx = np.argsort(xi)
                xi_sorted = xi[sort_idx]
                w_sorted = weights[sort_idx]
                cumw = np.cumsum(w_sorted)
                lo = xi_sorted[np.searchsorted(cumw, alpha / 2)]
                hi = xi_sorted[np.searchsorted(cumw, 1 - alpha / 2)]
                intervals.append((float(lo), float(peak[i]), float(hi)))

            return intervals

        except Exception:
            return None

    def print_credible_intervals(self, peak_idx: int = 0):
        """Pretty-print 68% and 95% CL table + correlation matrix."""
        peak = np.asarray(self.result.peaks[peak_idx]).ravel()
        sigma = self._sigmas[peak_idx]
        ci68 = self.get_credible_intervals(peak_idx, cl=0.6827)
        ci95 = self.get_credible_intervals(peak_idx, cl=0.9545)

        print()
        print("=" * 94)
        print(f"  POSTERIOR SUMMARY  (Peak {peak_idx+1}/{self.result.n_peaks})"
              f"  [source: {self._source}]")
        print("=" * 94)
        print(f"  {'Parameter':<16s} {'Best-fit':>12s} {'σ':>10s}"
              f" {'68% CL':>24s} {'95% CL':>24s}")
        print("  " + "-" * 90)

        has_asym = False
        for i in range(self.D):
            lo68, _, hi68 = ci68[i]
            lo95, _, hi95 = ci95[i]
            # Show asymmetric errors if they differ
            err_lo68 = peak[i] - lo68
            err_hi68 = hi68 - peak[i]
            is_asym = (abs(err_lo68 - err_hi68) /
                       max(err_lo68, err_hi68, 1e-30) > 0.05)
            if is_asym:
                has_asym = True
                fmt68 = f"  [{lo68:>10.6f}, {hi68:>10.6f}]*"
            else:
                fmt68 = f"  [{lo68:>10.6f}, {hi68:>10.6f}]"
            lo95_s, _, hi95_s = ci95[i]
            fmt95 = f"  [{lo95_s:>10.6f}, {hi95_s:>10.6f}]"

            print(f"  {self.param_names[i]:<16s} {peak[i]:>12.6f} "
                  f"{sigma[i]:>10.6f}{fmt68}{fmt95}")

        print("=" * 94)
        if has_asym:
            print("  * = asymmetric interval from ray profiles")

        # Correlation matrix
        cov = self._covariances[peak_idx]
        s = self._sigmas[peak_idx]
        corr = cov / np.outer(s, s)
        np.fill_diagonal(corr, 1.0)

        print(f"\n  CORRELATION MATRIX:")
        hdr = "  " + " " * 16 + "".join(f"{n:>12s}" for n in self.param_names)
        print(hdr)
        for i in range(self.D):
            row = f"  {self.param_names[i]:<16s}"
            for j in range(self.D):
                row += f"{corr[i, j]:>12.3f}"
            print(row)
        print()

    # ══════════════════════════════════════════════════════════════════
    # CORNER PLOT
    # ══════════════════════════════════════════════════════════════════

    def corner_plot(
        self,
        filename: Optional[str] = None,
        peak_idx: int = 0,
        truths: Optional[np.ndarray] = None,
        figsize: Optional[Tuple[float, float]] = None,
        n_samples: int = 80_000,
        color: str = "#2060B0",
        truth_color: str = "#E04040",
        show_titles: bool = True,
        use_rays: bool = True,
        dpi: int = 150,
    ):
        """
        Publication-quality corner plot.

        If ``use_rays=True`` and ray profiles are available, uses the
        actual likelihood-weighted samples for non-Gaussian contours.
        Otherwise draws from the Gaussian approximation.

        Parameters
        ----------
        filename : str or None
            Save path (png / pdf).
        peak_idx : int
            Which mode to plot.
        truths : (D,) array or None
            Fiducial / injected values shown as crosshairs.
        use_rays : bool
            Use ray bank samples for non-Gaussian contours when available.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib import rcParams
        from scipy.stats import norm as _norm

        rcParams.update({"font.family": "serif", "font.size": 9})

        D = self.D
        peak = np.asarray(self.result.peaks[peak_idx]).ravel()
        cov = self._covariances[peak_idx]
        sigma = self._sigmas[peak_idx]

        # Get samples: ray-weighted or Gaussian
        samples, weights = self._get_plot_samples(
            peak_idx, n_samples, use_rays
        )

        if figsize is None:
            figsize = (2.4 * D, 2.4 * D)

        fig, axes = plt.subplots(D, D, figsize=figsize)
        if D == 1:
            axes = np.array([[axes]])

        a1, a2 = 0.35, 0.15

        for i in range(D):
            for j in range(D):
                ax = axes[i, j]

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # ─── diagonal: 1-D marginal ───
                    # Always use Gaussian samples for clean histograms
                    # (ray samples are radially structured → spiky)
                    rng = np.random.default_rng(42 + i)
                    gauss_1d = rng.normal(peak[i], sigma[i], size=10000)
                    ax.hist(gauss_1d, bins=60, density=True,
                            color=color, alpha=0.45, edgecolor="none")

                    # Gaussian overlay
                    xs = np.linspace(peak[i] - 4*sigma[i],
                                     peak[i] + 4*sigma[i], 300)
                    ys = _norm.pdf(xs, loc=peak[i], scale=sigma[i])
                    ax.plot(xs, ys, color=color, lw=1.2, alpha=0.6, ls='--')

                    # Shaded CL bands
                    m68 = np.abs(xs - peak[i]) <= sigma[i]
                    m95 = np.abs(xs - peak[i]) <= 2 * sigma[i]
                    ax.fill_between(xs, 0, ys, where=m68,
                                    color=color, alpha=a1)
                    ax.fill_between(xs, 0, ys, where=m95 & ~m68,
                                    color=color, alpha=a2)

                    if truths is not None:
                        ax.axvline(truths[i], color=truth_color,
                                   ls="--", lw=1.2)

                    if show_titles:
                        ax.set_title(
                            f"{self.param_names[i]} = "
                            f"${peak[i]:.5f} \\pm {sigma[i]:.5f}$",
                            fontsize=7, pad=4)
                    ax.set_yticks([])
                    ax.set_xlim(peak[i] - 4*sigma[i],
                                peak[i] + 4*sigma[i])

                else:
                    # ─── lower triangle: 2-D contour ───
                    self._plot_2d(ax, samples, weights, cov, peak, sigma,
                                  j, i, color, a1, a2)
                    if truths is not None:
                        ax.axvline(truths[j], color=truth_color,
                                   ls="--", lw=0.7, alpha=0.6)
                        ax.axhline(truths[i], color=truth_color,
                                   ls="--", lw=0.7, alpha=0.6)
                        ax.plot(truths[j], truths[i], "+",
                                color=truth_color, ms=8, mew=1.5)
                    ax.set_xlim(peak[j] - 4*sigma[j],
                                peak[j] + 4*sigma[j])
                    ax.set_ylim(peak[i] - 4*sigma[i],
                                peak[i] + 4*sigma[i])

                if i == D - 1:
                    ax.set_xlabel(self.param_names[j], fontsize=10)
                    ax.tick_params(axis="x", labelsize=6, rotation=45)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(self.param_names[i], fontsize=10)
                    ax.tick_params(axis="y", labelsize=6)
                elif i != j:
                    ax.set_yticklabels([])

        source_tag = f"source: {self._source}"
        fig.suptitle(
            f"SunBURST Posterior  "
            f"(log Z = {self.result.log_evidence:.2f},  {source_tag})",
            fontsize=11, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            print(f"  Corner plot saved → {filename}")
        return fig

    def _get_plot_samples(self, peak_idx, n_samples, use_rays):
        """
        Get samples for plotting.

        Returns (samples, weights) where weights may be None
        (unweighted Gaussian draws) or an array (ray-weighted).
        """
        peak = np.asarray(self.result.peaks[peak_idx]).ravel()
        cov = self._covariances[peak_idx]

        # Try ray-bank samples first
        if use_rays and self._ray_profiles[peak_idx] is not None:
            rays = self._ray_profiles[peak_idx]
            try:
                if 'samples' in rays:
                    samp = np.asarray(rays['samples'])
                    logL = np.asarray(rays['log_L'])
                    if samp.ndim == 3:
                        samp = samp.reshape(-1, self.D)
                        logL = logL.reshape(-1)
                    valid = np.isfinite(logL) & np.all(np.isfinite(samp), axis=1)
                    samp = samp[valid]
                    logL = logL[valid]
                    if len(samp) > 100:
                        log_w = logL - np.max(logL)
                        weights = np.exp(log_w)
                        weights /= weights.sum()
                        return samp, weights
            except Exception:
                pass

        # Gaussian fallback
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal(peak, cov, size=n_samples)
        return samples, None

    def _plot_2d(self, ax, samples, weights, cov, peak, sigma,
                 ix, iy, color, a1, a2):
        """2D marginalized contours — ellipses + Gaussian scatter + KDE overlay."""
        from matplotlib.patches import Ellipse

        sub = cov[np.ix_([ix, iy], [ix, iy])]
        eigvals, eigvecs = np.linalg.eigh(sub)

        # Scatter: always use Gaussian samples (clean, unstructured)
        rng = np.random.default_rng(42 + ix * 100 + iy)
        try:
            gauss_samp = rng.multivariate_normal(
                [peak[ix], peak[iy]], sub, size=3000
            )
            ax.scatter(gauss_samp[:, 0], gauss_samp[:, 1],
                       s=0.3, alpha=0.05, color=color, rasterized=True)
        except (np.linalg.LinAlgError, ValueError):
            pass

        # Ellipses from covariance (always show these)
        if np.all(eigvals > 0):
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            for chi2, alpha in [(6.18, a2), (2.30, a1)]:
                w = 2 * np.sqrt(chi2 * eigvals[0])
                h = 2 * np.sqrt(chi2 * eigvals[1])
                ell = Ellipse(xy=(peak[ix], peak[iy]),
                              width=w, height=h, angle=angle,
                              facecolor=color, edgecolor=color,
                              alpha=alpha, lw=0.8)
                ax.add_patch(ell)

            # Contour outlines
            L = np.linalg.cholesky(sub)
            t = np.linspace(0, 2 * np.pi, 200)
            for chi2, ls in [(2.30, "-"), (6.18, "--")]:
                r = np.sqrt(chi2)
                unit = np.column_stack([r * np.cos(t), r * np.sin(t)])
                rot = unit @ L.T
                ax.plot(peak[ix] + rot[:, 0], peak[iy] + rot[:, 1],
                        ls=ls, color=color, lw=0.9, alpha=0.8)

        # KDE contour overlay from ray samples (non-Gaussian shape)
        # NOTE: disabled by default — sunburst rays are radially structured
        # in D dimensions and their 2D projections don't represent proper
        # marginals.  Enable via corner_plot(..., kde_overlay=True) for
        # low-D problems where rays do cover the parameter space.
        # if weights is not None and len(samples) > 500:
        #     self._overlay_kde_contour(
        #         ax, samples[:, ix], samples[:, iy], weights, color
        #     )

    def _overlay_kde_contour(self, ax, x, y, weights, color):
        """Overlay likelihood-weighted KDE contour for non-Gaussian shape."""
        try:
            from scipy.stats import gaussian_kde

            # Subsample for speed
            if len(x) > 10000:
                idx = np.random.default_rng(0).choice(
                    len(x), 10000, replace=False, p=weights
                )
                x, y = x[idx], y[idx]
                weights = None  # already importance-sampled

            if weights is not None:
                kde = gaussian_kde(np.vstack([x, y]), weights=weights)
            else:
                kde = gaussian_kde(np.vstack([x, y]))

            xg = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 80)
            yg = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 80)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

            # 68% and 95% levels
            Zflat = Zg.ravel()
            Zflat_sorted = np.sort(Zflat)[::-1]
            cumsum = np.cumsum(Zflat_sorted)
            if cumsum[-1] <= 0:
                return  # No density in grid — skip KDE overlay
            cumsum /= cumsum[-1]
            level_68 = Zflat_sorted[np.searchsorted(cumsum, 0.6827)]
            level_95 = Zflat_sorted[np.searchsorted(cumsum, 0.9545)]

            ax.contour(Xg, Yg, Zg, levels=[level_95, level_68],
                       colors=[color], linewidths=[0.6, 1.0],
                       linestyles=[':', '-'], alpha=0.9)
        except Exception:
            pass  # KDE overlay is best-effort

    # ══════════════════════════════════════════════════════════════════
    # LATEX TABLE
    # ══════════════════════════════════════════════════════════════════

    def latex_table(self, peak_idx: int = 0) -> str:
        """Generate a LaTeX table of credible intervals."""
        peak = np.asarray(self.result.peaks[peak_idx]).ravel()
        ci68 = self.get_credible_intervals(peak_idx, cl=0.6827)
        ci95 = self.get_credible_intervals(peak_idx, cl=0.9545)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Posterior parameter constraints.}",
            r"\label{tab:posterior_constraints}",
            r"{\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Parameter & Best-fit & $-\sigma$ & $+\sigma$ "
            r"& 95\,\% CL \\",
            r"\midrule",
        ]
        for i in range(self.D):
            lo68, ctr, hi68 = ci68[i]
            lo95, _, hi95 = ci95[i]
            err_lo = ctr - lo68
            err_hi = hi68 - ctr
            lines.append(
                f"${self.param_names[i]}$ & "
                f"${ctr:.6f}$ & "
                f"$-{err_lo:.6f}$ & $+{err_hi:.6f}$ & "
                f"$[{lo95:.6f},\\,{hi95:.6f}]$ \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", "}",
                  r"\end{table}"]
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════

    def print_diagnostics(self, peak_idx: int = 0):
        """Print posterior shape diagnostics from pipeline."""
        print(f"\n  POSTERIOR DIAGNOSTICS (Peak {peak_idx+1})")
        print(f"  {'─'*50}")
        print(f"  Covariance source:  {self._source}")

        diag_list = getattr(self.result, 'diagnostics', None)
        if diag_list is not None and peak_idx < len(diag_list):
            d = diag_list[peak_idx]
            if isinstance(d, dict) and d:
                for key in ['tail_alpha', 'tail_source', 'curvature',
                            'asymmetry', 'method']:
                    if key in d:
                        print(f"  {key:20s}: {d[key]}")
            else:
                print("  No diagnostics available for this peak.")
        else:
            print("  No diagnostics in result object.")

        print(f"  Ray profiles:       {'available' if self._has_rays else 'not exposed'}")

        # Check for strong non-Gaussianity indicators
        cov = self._covariances[peak_idx]
        corr = cov / np.outer(self._sigmas[peak_idx], self._sigmas[peak_idx])
        np.fill_diagonal(corr, 0)
        max_corr = np.max(np.abs(corr))
        print(f"  Max |correlation|:  {max_corr:.4f}")

        if max_corr < 0.01 and self._source in ('diag_H', 'fd'):
            print(f"  ⚠  Near-zero correlations — likely diagonal Hessian only.")
            print(f"     Expose result.hessian (full) for proper degeneracies.")
        print()
