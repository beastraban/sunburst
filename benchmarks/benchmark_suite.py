#!/usr/bin/env python3
"""
SunBURST Benchmark Suite v2.0
=============================
Head-to-head comparison against nested sampling baselines.

Compatible with the sunburst PyPI package.

USAGE:
  cd benchmarks
  pip install -e ..          # Install sunburst package
  pip install dynesty ultranest  # Install competitors
  python benchmark_suite.py --quick
  python benchmark_suite.py --full
  python benchmark_suite.py --dims 2,8,32,64

Author: Ira Wolfson
Date: January 2026
"""

import numpy as np
import time
import csv
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SUNBURST IMPORT
# ============================================================================

try:
    from sunburst import compute_evidence, gpu_available, gpu_info, __version__ as SUNBURST_VERSION
    HAS_SUNBURST = True
except ImportError:
    HAS_SUNBURST = False
    SUNBURST_VERSION = "not installed"

# Try GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# ============================================================================
# COMPETITOR IMPORTS
# ============================================================================

try:
    import dynesty
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

try:
    import ultranest
    HAS_ULTRANEST = True
except ImportError:
    HAS_ULTRANEST = False

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEOUT_SECONDS = 600  # 10 minutes per run
DEFAULT_DIMS = [2, 4, 8, 16, 32, 64, 128, 256]
QUICK_DIMS = [2, 8, 32]

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

class TestFunction:
    """Base class for test functions with analytical evidence."""
    
    def __init__(self, dim: int, bounds: List[tuple]):
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_calls = 0
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Evaluate log-likelihood. x is (N, dim) array."""
        raise NotImplementedError
    
    def log_evidence_true(self) -> float:
        """Return true log evidence (analytical)."""
        raise NotImplementedError
    
    def reset_calls(self):
        self.n_calls = 0
    
    def __call__(self, x):
        """Wrapper that counts calls."""
        x = np.atleast_2d(x)
        self.n_calls += len(x)
        return self.log_likelihood(x)


class Gaussian(TestFunction):
    """Isotropic Gaussian - the simplest test case."""
    
    def __init__(self, dim: int, width: float = 1.0, bound: float = 10.0):
        bounds = [(-bound, bound)] * dim
        super().__init__(dim, bounds)
        self.width = width
        self.bound = bound
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2, axis=1) / self.width**2
    
    def log_evidence_true(self) -> float:
        # Z = (sqrt(2*pi)*width)^dim / (2*bound)^dim
        log_gaussian_norm = 0.5 * self.dim * np.log(2 * np.pi) + self.dim * np.log(self.width)
        log_prior_volume = self.dim * np.log(2 * self.bound)
        return log_gaussian_norm - log_prior_volume


class CorrelatedGaussian(TestFunction):
    """Gaussian with off-diagonal covariance."""
    
    def __init__(self, dim: int, correlation: float = 0.9, bound: float = 10.0):
        bounds = [(-bound, bound)] * dim
        super().__init__(dim, bounds)
        self.bound = bound
        
        # Build covariance matrix with correlation
        self.cov = np.eye(dim)
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    self.cov[i, j] = correlation ** abs(i - j)
        
        self.cov_inv = np.linalg.inv(self.cov)
        self.log_det_cov = np.linalg.slogdet(self.cov)[1]
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        # -0.5 * x @ cov_inv @ x.T for each row
        return -0.5 * np.sum(x @ self.cov_inv * x, axis=1)
    
    def log_evidence_true(self) -> float:
        log_gaussian_norm = 0.5 * self.dim * np.log(2 * np.pi) + 0.5 * self.log_det_cov
        log_prior_volume = self.dim * np.log(2 * self.bound)
        return log_gaussian_norm - log_prior_volume


class CigarGaussian(TestFunction):
    """Axis-aligned anisotropic Gaussian (cigar shape)."""
    
    def __init__(self, dim: int, aspect_ratio: float = 100.0, bound: float = 10.0):
        bounds = [(-bound, bound)] * dim
        super().__init__(dim, bounds)
        self.bound = bound
        
        # First dimension narrow, rest wide
        self.widths = np.ones(dim)
        self.widths[0] = 1.0 / aspect_ratio
        self.log_det_cov = 2 * np.sum(np.log(self.widths))
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -0.5 * np.sum((x / self.widths)**2, axis=1)
    
    def log_evidence_true(self) -> float:
        log_gaussian_norm = 0.5 * self.dim * np.log(2 * np.pi) + 0.5 * self.log_det_cov
        log_prior_volume = self.dim * np.log(2 * self.bound)
        return log_gaussian_norm - log_prior_volume


class MixtureOfGaussians(TestFunction):
    """Multimodal: mixture of well-separated Gaussians."""
    
    def __init__(self, dim: int, n_modes: int = 4, separation: float = 5.0, 
                 width: float = 1.0, bound: float = 15.0):
        bounds = [(-bound, bound)] * dim
        super().__init__(dim, bounds)
        self.n_modes = n_modes
        self.width = width
        self.bound = bound
        
        # Place modes in a grid pattern
        np.random.seed(42)  # Reproducible
        self.centers = np.zeros((n_modes, dim))
        for i in range(n_modes):
            angle = 2 * np.pi * i / n_modes
            self.centers[i, 0] = separation * np.cos(angle)
            if dim > 1:
                self.centers[i, 1] = separation * np.sin(angle)
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        # Sum over modes using logsumexp
        log_probs = []
        for center in self.centers:
            diff = x - center
            log_prob = -0.5 * np.sum(diff**2, axis=1) / self.width**2
            log_probs.append(log_prob)
        log_probs = np.array(log_probs)  # (n_modes, N)
        # logsumexp - log(n_modes) for equal weights
        max_log = np.max(log_probs, axis=0)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0)) - np.log(self.n_modes)
    
    def log_evidence_true(self) -> float:
        # Each mode contributes equally, sum of n_modes identical Gaussians
        log_single = 0.5 * self.dim * np.log(2 * np.pi) + self.dim * np.log(self.width)
        log_prior_volume = self.dim * np.log(2 * self.bound)
        # n_modes peaks, but we divide by n_modes in likelihood, so it cancels
        return log_single - log_prior_volume


class Rosenbrock(TestFunction):
    """Rosenbrock function - curved valley."""
    
    def __init__(self, dim: int, bound: float = 5.0, a: float = 1.0, b: float = 100.0):
        bounds = [(-bound, bound)] * dim
        super().__init__(dim, bounds)
        self.bound = bound
        self.a = a
        self.b = b
        self._log_Z = None  # Cache
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        # Rosenbrock: sum of (a-x_i)^2 + b*(x_{i+1} - x_i^2)^2
        result = np.zeros(len(x))
        for i in range(self.dim - 1):
            result += (self.a - x[:, i])**2 + self.b * (x[:, i+1] - x[:, i]**2)**2
        return -result  # Negative because we minimize Rosenbrock
    
    def log_evidence_true(self) -> float:
        # No analytical solution - return NaN
        return np.nan


# ============================================================================
# BENCHMARK RESULT
# ============================================================================

@dataclass
class BenchmarkResult:
    method: str
    function: str
    dim: int
    log_evidence: float = np.nan
    log_evidence_true: float = np.nan
    error_percent: float = np.nan
    wall_time: float = np.nan
    n_calls: int = 0
    status: str = "pending"
    error_message: str = ""


# ============================================================================
# RUNNERS
# ============================================================================

def run_sunburst(test_func: TestFunction, n_oscillations: int = 1, 
                 fast: bool = True, use_gpu: bool = True) -> BenchmarkResult:
    """Run SunBURST on test function."""
    result = BenchmarkResult(
        method="sunburst",
        function=test_func.__class__.__name__,
        dim=test_func.dim
    )
    
    if not HAS_SUNBURST:
        result.status = "skipped"
        result.error_message = "sunburst not installed"
        return result
    
    try:
        test_func.reset_calls()
        
        # Create wrapper that counts calls
        def log_L(x):
            return test_func(x)
        
        start = time.perf_counter()
        
        sb_result = compute_evidence(
            log_L,
            test_func.bounds.tolist(),
            n_oscillations=n_oscillations,
            fast=fast,
            verbose=False,
            use_gpu=use_gpu if HAS_CUPY else False
        )
        
        wall_time = time.perf_counter() - start
        
        result.log_evidence = sb_result.log_evidence
        result.log_evidence_true = test_func.log_evidence_true()
        result.wall_time = wall_time
        result.n_calls = test_func.n_calls
        
        if np.isfinite(result.log_evidence_true) and result.log_evidence_true != 0:
            result.error_percent = 100 * abs(result.log_evidence - result.log_evidence_true) / abs(result.log_evidence_true)
        else:
            result.error_percent = np.nan
        
        result.status = "completed"
        
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
    
    return result


def run_dynesty(test_func: TestFunction, nlive: int = 500, 
                maxcall: int = 1_000_000) -> BenchmarkResult:
    """Run dynesty on test function."""
    result = BenchmarkResult(
        method="dynesty",
        function=test_func.__class__.__name__,
        dim=test_func.dim
    )
    
    if not HAS_DYNESTY:
        result.status = "skipped"
        result.error_message = "dynesty not installed"
        return result
    
    try:
        test_func.reset_calls()
        
        def prior_transform(u):
            # Transform from [0,1] to bounds
            return test_func.bounds[:, 0] + u * (test_func.bounds[:, 1] - test_func.bounds[:, 0])
        
        def log_L(x):
            return float(test_func(x.reshape(1, -1))[0])
        
        start = time.perf_counter()
        
        sampler = dynesty.DynamicNestedSampler(
            log_L, prior_transform, test_func.dim
        )
        sampler.run_nested(maxcall=maxcall, nlive_init=nlive)
        
        wall_time = time.perf_counter() - start
        
        results = sampler.results
        result.log_evidence = results.logz[-1]
        result.log_evidence_true = test_func.log_evidence_true()
        result.wall_time = wall_time
        result.n_calls = test_func.n_calls
        
        if np.isfinite(result.log_evidence_true) and result.log_evidence_true != 0:
            result.error_percent = 100 * abs(result.log_evidence - result.log_evidence_true) / abs(result.log_evidence_true)
        else:
            result.error_percent = np.nan
        
        result.status = "completed"
        
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
    
    return result


def run_ultranest(test_func: TestFunction, min_num_live_points: int = 400,
                  max_ncalls: int = 1_000_000) -> BenchmarkResult:
    """Run UltraNest on test function."""
    result = BenchmarkResult(
        method="ultranest",
        function=test_func.__class__.__name__,
        dim=test_func.dim
    )
    
    if not HAS_ULTRANEST:
        result.status = "skipped"
        result.error_message = "ultranest not installed"
        return result
    
    try:
        test_func.reset_calls()
        
        def prior_transform(u):
            return test_func.bounds[:, 0] + u * (test_func.bounds[:, 1] - test_func.bounds[:, 0])
        
        def log_L(x):
            return float(test_func(x.reshape(1, -1))[0])
        
        param_names = [f"x{i}" for i in range(test_func.dim)]
        
        start = time.perf_counter()
        
        sampler = ultranest.ReactiveNestedSampler(
            param_names, log_L, prior_transform,
            log_dir=None, resume='overwrite'
        )
        results = sampler.run(
            min_num_live_points=min_num_live_points,
            max_ncalls=max_ncalls
        )
        
        wall_time = time.perf_counter() - start
        
        result.log_evidence = results['logz']
        result.log_evidence_true = test_func.log_evidence_true()
        result.wall_time = wall_time
        result.n_calls = test_func.n_calls
        
        if np.isfinite(result.log_evidence_true) and result.log_evidence_true != 0:
            result.error_percent = 100 * abs(result.log_evidence - result.log_evidence_true) / abs(result.log_evidence_true)
        else:
            result.error_percent = np.nan
        
        result.status = "completed"
        
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
    
    return result


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(dims: List[int], methods: List[str], 
                  functions: List[str], output_dir: Path) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    
    results = []
    
    # Map function names to classes
    func_map = {
        'gaussian': Gaussian,
        'correlated': CorrelatedGaussian,
        'cigar': CigarGaussian,
        'mixture': MixtureOfGaussians,
        'rosenbrock': Rosenbrock,
    }
    
    # Map method names to runners
    method_map = {
        'sunburst': run_sunburst,
        'dynesty': run_dynesty,
        'ultranest': run_ultranest,
    }
    
    total_runs = len(dims) * len(methods) * len(functions)
    run_idx = 0
    
    for func_name in functions:
        if func_name not in func_map:
            print(f"Unknown function: {func_name}")
            continue
        
        for dim in dims:
            test_func = func_map[func_name](dim)
            
            for method_name in methods:
                if method_name not in method_map:
                    print(f"Unknown method: {method_name}")
                    continue
                
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {method_name} | {func_name} | {dim}D ... ", end="", flush=True)
                
                runner = method_map[method_name]
                result = runner(test_func)
                results.append(result)
                
                if result.status == "completed":
                    print(f"✓ {result.wall_time:.2f}s | log Z = {result.log_evidence:.4f} | err = {result.error_percent:.2f}%")
                elif result.status == "skipped":
                    print(f"⊘ skipped ({result.error_message})")
                else:
                    print(f"✗ error: {result.error_message[:50]}")
    
    return results


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save results to CSV."""
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_results_{timestamp}.csv"
    
    fieldnames = ['method', 'function', 'dim', 'log_evidence', 'log_evidence_true',
                  'error_percent', 'wall_time', 'n_calls', 'status', 'error_message']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'method': r.method,
                'function': r.function,
                'dim': r.dim,
                'log_evidence': r.log_evidence,
                'log_evidence_true': r.log_evidence_true,
                'error_percent': r.error_percent,
                'wall_time': r.wall_time,
                'n_calls': r.n_calls,
                'status': r.status,
                'error_message': r.error_message,
            })
    
    print(f"\nResults saved to: {csv_path}")
    return csv_path


def print_summary(results: List[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by function and dim
    from collections import defaultdict
    grouped = defaultdict(dict)
    
    for r in results:
        if r.status == "completed":
            key = (r.function, r.dim)
            grouped[key][r.method] = r
    
    # Print table
    print(f"\n{'Function':<15} {'Dim':>5} | {'SunBURST':>12} | {'dynesty':>12} | {'UltraNest':>12} | {'Speedup':>10}")
    print("-" * 80)
    
    for (func, dim), methods in sorted(grouped.items()):
        sb = methods.get('sunburst')
        dy = methods.get('dynesty')
        un = methods.get('ultranest')
        
        sb_time = f"{sb.wall_time:.2f}s" if sb else "—"
        dy_time = f"{dy.wall_time:.2f}s" if dy else "—"
        un_time = f"{un.wall_time:.2f}s" if un else "—"
        
        # Calculate speedup vs dynesty
        if sb and dy and dy.wall_time > 0:
            speedup = f"{dy.wall_time / sb.wall_time:.1f}×"
        else:
            speedup = "—"
        
        print(f"{func:<15} {dim:>5} | {sb_time:>12} | {dy_time:>12} | {un_time:>12} | {speedup:>10}")


def main():
    parser = argparse.ArgumentParser(description="SunBURST Benchmark Suite")
    parser.add_argument('--quick', action='store_true', help="Quick test (2D-32D)")
    parser.add_argument('--full', action='store_true', help="Full suite (2D-256D)")
    parser.add_argument('--dims', type=str, default=None, help="Comma-separated dimensions")
    parser.add_argument('--methods', type=str, default="sunburst,dynesty,ultranest",
                        help="Comma-separated methods")
    parser.add_argument('--functions', type=str, default="gaussian,correlated,mixture",
                        help="Comma-separated test functions")
    parser.add_argument('--output', type=str, default="./results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("SunBURST Benchmark Suite v2.0")
    print("="*60)
    print(f"SunBURST version: {SUNBURST_VERSION}")
    print(f"GPU available: {gpu_available() if HAS_SUNBURST else 'N/A'}")
    print(f"dynesty available: {HAS_DYNESTY}")
    print(f"ultranest available: {HAS_ULTRANEST}")
    print("="*60 + "\n")
    
    # Determine dimensions
    if args.dims:
        dims = [int(d) for d in args.dims.split(',')]
    elif args.quick:
        dims = QUICK_DIMS
    elif args.full:
        dims = DEFAULT_DIMS
    else:
        dims = QUICK_DIMS  # Default to quick
    
    methods = args.methods.split(',')
    functions = args.functions.split(',')
    output_dir = Path(args.output)
    
    print(f"Dimensions: {dims}")
    print(f"Methods: {methods}")
    print(f"Functions: {functions}")
    print()
    
    # Run benchmarks
    results = run_benchmark(dims, methods, functions, output_dir)
    
    # Save and summarize
    save_results(results, output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
