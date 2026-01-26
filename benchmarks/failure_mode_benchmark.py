#!/usr/bin/env python3
"""
SunBURST Failure Mode Benchmark
================================
Test functions where Laplace approximation is expected to fail or degrade,
but analytical evidence is computable for verification.

Tests SunBURST's robustness on challenging distributions:
- Heavy tails (Student-t)
- Multimodality (mixture of Gaussians)
- Non-Gaussian shapes (banana/twisted)
- Extreme anisotropy (cigar)

Usage:
    cd benchmarks
    pip install -e ..  # Install sunburst package
    python failure_mode_benchmark.py
    python failure_mode_benchmark.py --dims 2,4,8,16
    python failure_mode_benchmark.py --functions student_t,mixture

Author: Ira Wolfson
Date: January 2026
"""

import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Optional
from scipy import special
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SUNBURST IMPORT
# =============================================================================

try:
    from sunburst import compute_evidence, gpu_available, __version__
    HAS_SUNBURST = True
except ImportError:
    HAS_SUNBURST = False
    __version__ = "not installed"

# =============================================================================
# GPU SETUP
# =============================================================================

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    GPU_AVAILABLE = False


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class FailureModeResult:
    function: str
    dim: int
    failure_mode: str
    log_evidence: float = np.nan
    log_evidence_true: float = np.nan
    error_percent: float = np.nan
    wall_time: float = 0.0
    n_peaks: int = 0
    status: str = "pending"
    notes: str = ""


# =============================================================================
# FAILURE MODE TEST FUNCTIONS
# =============================================================================

class StudentT:
    """
    Multivariate Student-t distribution.
    Heavy tails -> Laplace (Gaussian assumption) underestimates tail mass.
    """
    def __init__(self, dim: int, nu: float = 3.0, bound: float = 20.0):
        self.dim = dim
        self.nu = nu
        self.bound = bound
        self.bounds = [(-bound, bound)] * dim
        self.failure_mode = "heavy_tails"
        self.n_calls = 0
    
    def log_likelihood(self, x):
        x = np.atleast_2d(x)
        self.n_calls += len(x)
        # Unnormalized: (1 + ||x||^2/nu)^(-(nu+d)/2)
        r2 = np.sum(x**2, axis=1)
        return -(self.nu + self.dim) / 2 * np.log(1 + r2 / self.nu)
    
    def log_evidence_true(self):
        """Analytical evidence for Student-t in hypercube."""
        # This requires numerical integration for general case
        # For low dimensions, use simple approximation
        if self.dim <= 2:
            from scipy import integrate
            if self.dim == 1:
                result, _ = integrate.quad(
                    lambda x: np.exp(self.log_likelihood(np.array([[x]]))[0]),
                    -self.bound, self.bound
                )
                return np.log(result) - np.log(2 * self.bound)
        return np.nan  # Numerical integration needed


class MixtureOfGaussians:
    """
    Well-separated mixture of Gaussians.
    Tests multimodal detection capability.
    """
    def __init__(self, dim: int, n_modes: int = 4, separation: float = 5.0,
                 width: float = 1.0, bound: float = 15.0):
        self.dim = dim
        self.n_modes = n_modes
        self.width = width
        self.bound = bound
        self.failure_mode = "multimodal"
        self.n_calls = 0
        
        # Place modes in a ring pattern
        self.centers = np.zeros((n_modes, dim))
        for i in range(n_modes):
            angle = 2 * np.pi * i / n_modes
            self.centers[i, 0] = separation * np.cos(angle)
            if dim > 1:
                self.centers[i, 1] = separation * np.sin(angle)
        
        self.bounds = [(-bound, bound)] * dim
    
    def log_likelihood(self, x):
        x = np.atleast_2d(x)
        self.n_calls += len(x)
        
        log_probs = []
        for center in self.centers:
            diff = x - center
            log_prob = -0.5 * np.sum(diff**2, axis=1) / self.width**2
            log_probs.append(log_prob)
        
        log_probs = np.array(log_probs)
        max_log = np.max(log_probs, axis=0)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0)) - np.log(self.n_modes)
    
    def log_evidence_true(self):
        """Analytical evidence for mixture."""
        log_single = 0.5 * self.dim * np.log(2 * np.pi) + self.dim * np.log(self.width)
        log_prior_volume = self.dim * np.log(2 * self.bound)
        return log_single - log_prior_volume


class BananaDistribution:
    """
    Twisted/Banana distribution (Haario et al. 2001).
    Curved likelihood surface challenges Laplace quadratic assumption.
    """
    def __init__(self, dim: int, curvature: float = 0.1, bound: float = 20.0):
        self.dim = dim
        self.curvature = curvature
        self.bound = bound
        self.failure_mode = "non_gaussian_shape"
        self.n_calls = 0
        self.bounds = [(-bound, bound)] * dim
    
    def log_likelihood(self, x):
        x = np.atleast_2d(x)
        self.n_calls += len(x)
        
        # Transform: y1 = x1, y2 = x2 - b*x1^2 + 100b, yi = xi for i>2
        y = x.copy()
        if self.dim >= 2:
            y[:, 1] = x[:, 1] - self.curvature * (x[:, 0]**2 - 100)
        
        return -0.5 * np.sum(y**2, axis=1)
    
    def log_evidence_true(self):
        """Banana has same evidence as Gaussian (volume-preserving transform)."""
        log_gaussian = 0.5 * self.dim * np.log(2 * np.pi)
        log_prior_volume = self.dim * np.log(2 * self.bound)
        return log_gaussian - log_prior_volume


class ExtremeCigar:
    """
    Extremely anisotropic Gaussian (aspect ratio 1000:1).
    Tests handling of ill-conditioned Hessians.
    """
    def __init__(self, dim: int, aspect_ratio: float = 1000.0, bound: float = 10.0):
        self.dim = dim
        self.aspect_ratio = aspect_ratio
        self.bound = bound
        self.failure_mode = "extreme_anisotropy"
        self.n_calls = 0
        
        self.widths = np.ones(dim)
        self.widths[0] = 1.0 / aspect_ratio
        self.bounds = [(-bound, bound)] * dim
    
    def log_likelihood(self, x):
        x = np.atleast_2d(x)
        self.n_calls += len(x)
        return -0.5 * np.sum((x / self.widths)**2, axis=1)
    
    def log_evidence_true(self):
        log_gaussian = 0.5 * self.dim * np.log(2 * np.pi) + np.sum(np.log(self.widths))
        log_prior_volume = self.dim * np.log(2 * self.bound)
        return log_gaussian - log_prior_volume


class DonutDistribution:
    """
    Ring/Donut distribution in 2D (extends to shell in higher D).
    Zero density at center challenges mode-finding.
    """
    def __init__(self, dim: int, radius: float = 5.0, width: float = 1.0, bound: float = 15.0):
        self.dim = dim
        self.radius = radius
        self.width = width
        self.bound = bound
        self.failure_mode = "ring_shape"
        self.n_calls = 0
        self.bounds = [(-bound, bound)] * dim
    
    def log_likelihood(self, x):
        x = np.atleast_2d(x)
        self.n_calls += len(x)
        r = np.sqrt(np.sum(x**2, axis=1))
        return -0.5 * ((r - self.radius) / self.width)**2
    
    def log_evidence_true(self):
        # Approximate: volume of shell * Gaussian in radial direction
        # This is complex for arbitrary D, return NaN for numerical
        return np.nan


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_failure_mode_test(test_func, n_oscillations: int = 3,
                          fast: bool = True) -> FailureModeResult:
    """Run SunBURST on a failure mode test function."""
    import time
    
    result = FailureModeResult(
        function=test_func.__class__.__name__,
        dim=test_func.dim,
        failure_mode=test_func.failure_mode
    )
    
    if not HAS_SUNBURST:
        result.status = "skipped"
        result.notes = "sunburst not installed"
        return result
    
    try:
        test_func.n_calls = 0
        
        start = time.perf_counter()
        
        sb_result = compute_evidence(
            test_func.log_likelihood,
            test_func.bounds,
            n_oscillations=n_oscillations,
            fast=fast,
            verbose=False,
            use_gpu=GPU_AVAILABLE
        )
        
        result.wall_time = time.perf_counter() - start
        result.log_evidence = sb_result.log_evidence
        result.log_evidence_true = test_func.log_evidence_true()
        result.n_peaks = sb_result.n_peaks
        
        if np.isfinite(result.log_evidence_true) and abs(result.log_evidence_true) > 1e-10:
            result.error_percent = 100 * abs(result.log_evidence - result.log_evidence_true) / abs(result.log_evidence_true)
        
        result.status = "completed"
        
    except Exception as e:
        result.status = "error"
        result.notes = str(e)
    
    return result


def run_benchmark(dims: List[int], functions: List[str],
                  n_oscillations: int = 3, fast: bool = True) -> List[FailureModeResult]:
    """Run full failure mode benchmark."""
    
    func_map = {
        'student_t': StudentT,
        'mixture': MixtureOfGaussians,
        'banana': BananaDistribution,
        'cigar': ExtremeCigar,
        'donut': DonutDistribution,
    }
    
    results = []
    
    for func_name in functions:
        if func_name not in func_map:
            print(f"Unknown function: {func_name}")
            continue
        
        FuncClass = func_map[func_name]
        
        for dim in dims:
            print(f"Testing {func_name} ({dim}D)... ", end="", flush=True)
            
            test_func = FuncClass(dim)
            result = run_failure_mode_test(test_func, n_oscillations, fast)
            results.append(result)
            
            if result.status == "completed":
                if np.isfinite(result.error_percent):
                    print(f"✓ {result.wall_time:.2f}s | err={result.error_percent:.2f}% | peaks={result.n_peaks}")
                else:
                    print(f"✓ {result.wall_time:.2f}s | log Z={result.log_evidence:.4f} | peaks={result.n_peaks}")
            else:
                print(f"✗ {result.status}: {result.notes}")
    
    return results


def save_results(results: List[FailureModeResult], output_dir: Path):
    """Save benchmark results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = output_dir / f"failure_mode_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("function,dim,failure_mode,log_evidence,log_evidence_true,error_percent,wall_time,n_peaks,status,notes\n")
        for r in results:
            f.write(f"{r.function},{r.dim},{r.failure_mode},{r.log_evidence:.6f},"
                    f"{r.log_evidence_true:.6f},{r.error_percent:.4f},{r.wall_time:.4f},"
                    f"{r.n_peaks},{r.status},{r.notes}\n")
    
    print(f"\nResults saved to {csv_path}")


def print_summary(results: List[FailureModeResult]):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("FAILURE MODE BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Function':<20} {'Dim':>5} {'Mode':<20} {'Error %':>10} {'Status':<10}")
    print("-" * 70)
    
    for r in results:
        err_str = f"{r.error_percent:.2f}" if np.isfinite(r.error_percent) else "N/A"
        print(f"{r.function:<20} {r.dim:>5} {r.failure_mode:<20} {err_str:>10} {r.status:<10}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SunBURST Failure Mode Benchmark")
    parser.add_argument('--dims', type=str, default="2,4,8",
                        help='Comma-separated dimensions')
    parser.add_argument('--functions', type=str, default="student_t,mixture,banana,cigar",
                        help='Comma-separated test functions')
    parser.add_argument('--n-oscillations', type=int, default=3,
                        help='n_oscillations (default: 3 for robustness)')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Use fast Hessian')
    parser.add_argument('--output-dir', type=str, default='./failure_mode_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    dims = [int(d) for d in args.dims.split(',')]
    functions = args.functions.split(',')
    
    # Print header
    print("=" * 60)
    print("SUNBURST FAILURE MODE BENCHMARK")
    print("=" * 60)
    print(f"SunBURST version: {__version__}")
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"Dimensions: {dims}")
    print(f"Functions: {functions}")
    print(f"n_oscillations: {args.n_oscillations}")
    print("=" * 60 + "\n")
    
    if not HAS_SUNBURST:
        print("ERROR: sunburst package not installed!")
        print("Run: pip install -e ..")
        return
    
    # Run benchmark
    results = run_benchmark(dims, functions, args.n_oscillations, args.fast)
    
    # Save and summarize
    save_results(results, Path(args.output_dir))
    print_summary(results)


if __name__ == "__main__":
    main()
