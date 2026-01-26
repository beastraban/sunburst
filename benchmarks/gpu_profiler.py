#!/usr/bin/env python3
"""
SunBURST GPU Performance Profiler v3.0
======================================
GPU profiling for accurate timing measurements using the sunburst package.

Usage:
    cd benchmarks
    pip install -e ..  # Install sunburst package
    python gpu_profiler.py
    python gpu_profiler.py --dims 32,64,128,256,512,1024
    python gpu_profiler.py --runs 4
    python gpu_profiler.py --output-dir my_results

Output (in output directory):
    dashboard.png      - Visual dashboard
    raw_results.csv    - Per-run raw data
    summary.csv        - Per-dimension aggregated data
    session.log        - Human-readable log

Author: Ira Wolfson
Date: January 2026
"""

import numpy as np
import time
import gc
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# =============================================================================
# SUNBURST IMPORT
# =============================================================================

try:
    from sunburst import compute_evidence, gpu_available, gpu_info, __version__
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
# CONFIGURATION
# =============================================================================

DEFAULT_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_RUNS = 4
DEFAULT_BOUND = 10.0

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ProfileResult:
    """Result from a single profiling run."""
    dimension: int
    run_index: int
    wall_time: float = 0.0
    log_evidence: float = np.nan
    log_evidence_true: float = np.nan
    error_percent: float = np.nan
    n_peaks: int = 0
    n_calls: int = 0
    memory_mb: float = 0.0
    success: bool = True
    error_msg: str = ""


# =============================================================================
# TEST FUNCTION
# =============================================================================

def create_gaussian_likelihood(dim: int, bound: float = DEFAULT_BOUND):
    """Create a simple Gaussian log-likelihood for profiling."""
    call_count = [0]  # Mutable container for counting
    
    if GPU_AVAILABLE:
        def log_L(x):
            call_count[0] += len(x)
            x = cp.atleast_2d(x)
            return -0.5 * cp.sum(x**2, axis=1)
    else:
        def log_L(x):
            call_count[0] += len(x)
            x = np.atleast_2d(x)
            return -0.5 * np.sum(x**2, axis=1)
    
    bounds = [(-bound, bound)] * dim
    
    # True evidence: (sqrt(2*pi))^d / (2*bound)^d
    log_Z_true = 0.5 * dim * np.log(2 * np.pi) - dim * np.log(2 * bound)
    
    return log_L, bounds, log_Z_true, call_count


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def purge_memory():
    """Aggressively clear memory."""
    gc.collect()
    gc.collect()
    gc.collect()
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if not GPU_AVAILABLE:
        return 0.0
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    return pool.used_bytes() / (1024**2)


# =============================================================================
# PROFILING
# =============================================================================

def profile_dimension(dim: int, n_runs: int = DEFAULT_RUNS,
                      n_oscillations: int = 1, fast: bool = True,
                      verbose: bool = True) -> List[ProfileResult]:
    """Profile SunBURST at a specific dimension."""
    results = []
    
    for run_idx in range(n_runs):
        result = ProfileResult(dimension=dim, run_index=run_idx)
        
        try:
            # Create test function
            log_L, bounds, log_Z_true, call_count = create_gaussian_likelihood(dim)
            result.log_evidence_true = log_Z_true
            
            # Purge memory before run
            purge_memory()
            
            # Time the run
            start = time.perf_counter()
            
            sb_result = compute_evidence(
                log_L, bounds,
                n_oscillations=n_oscillations,
                fast=fast,
                verbose=False,
                use_gpu=GPU_AVAILABLE
            )
            
            result.wall_time = time.perf_counter() - start
            result.log_evidence = sb_result.log_evidence
            result.n_peaks = sb_result.n_peaks
            result.n_calls = call_count[0]
            result.memory_mb = get_gpu_memory_mb()
            
            # Calculate error
            if abs(log_Z_true) > 1e-10:
                result.error_percent = 100 * abs(result.log_evidence - log_Z_true) / abs(log_Z_true)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_msg = str(e)
        
        results.append(result)
        
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"  Run {run_idx+1}/{n_runs}: {status} {result.wall_time:.3f}s | "
                  f"log Z = {result.log_evidence:.4f} | err = {result.error_percent:.2f}%")
    
    return results


def run_profiler(dims: List[int], n_runs: int = DEFAULT_RUNS,
                 n_oscillations: int = 1, fast: bool = True) -> List[ProfileResult]:
    """Run profiler across all dimensions."""
    all_results = []
    
    # Warmup run
    print("Warming up GPU...")
    purge_memory()
    log_L, bounds, _, _ = create_gaussian_likelihood(dims[0])
    try:
        compute_evidence(log_L, bounds, n_oscillations=1, fast=True, verbose=False)
    except:
        pass
    purge_memory()
    print()
    
    for dim in dims:
        print(f"Profiling {dim}D ({n_runs} runs):")
        results = profile_dimension(dim, n_runs, n_oscillations, fast)
        all_results.extend(results)
        
        # Summary for this dimension
        times = [r.wall_time for r in results if r.success]
        if times:
            print(f"  → Mean: {np.mean(times):.3f}s ± {np.std(times):.3f}s\n")
    
    return all_results


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(results: List[ProfileResult], output_dir: Path,
                 config: Dict[str, Any]):
    """Save profiling results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Raw results CSV
    csv_path = output_dir / f"raw_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("dimension,run,wall_time,log_evidence,log_evidence_true,error_percent,n_peaks,n_calls,memory_mb,success\n")
        for r in results:
            f.write(f"{r.dimension},{r.run_index},{r.wall_time:.6f},{r.log_evidence:.6f},"
                    f"{r.log_evidence_true:.6f},{r.error_percent:.4f},{r.n_peaks},{r.n_calls},"
                    f"{r.memory_mb:.2f},{r.success}\n")
    
    # Summary CSV
    summary_path = output_dir / f"summary_{timestamp}.csv"
    with open(summary_path, 'w') as f:
        f.write("dimension,mean_time,std_time,mean_error,n_runs\n")
        dims = sorted(set(r.dimension for r in results))
        for dim in dims:
            dim_results = [r for r in results if r.dimension == dim and r.success]
            if dim_results:
                times = [r.wall_time for r in dim_results]
                errors = [r.error_percent for r in dim_results]
                f.write(f"{dim},{np.mean(times):.6f},{np.std(times):.6f},"
                        f"{np.mean(errors):.4f},{len(dim_results)}\n")
    
    # Session log
    log_path = output_dir / f"session_{timestamp}.log"
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SUNBURST GPU PROFILER SESSION LOG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"SunBURST version: {__version__}\n")
        f.write(f"GPU available: {GPU_AVAILABLE}\n\n")
        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("Results:\n")
        dims = sorted(set(r.dimension for r in results))
        for dim in dims:
            dim_results = [r for r in results if r.dimension == dim and r.success]
            if dim_results:
                times = [r.wall_time for r in dim_results]
                f.write(f"  {dim}D: {np.mean(times):.3f}s ± {np.std(times):.3f}s\n")
    
    print(f"\nResults saved to {output_dir}/")
    return csv_path, summary_path, log_path


def create_dashboard(results: List[ProfileResult], output_dir: Path):
    """Create visual dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dims = sorted(set(r.dimension for r in results))
    mean_times = []
    std_times = []
    mean_errors = []
    
    for dim in dims:
        dim_results = [r for r in results if r.dimension == dim and r.success]
        if dim_results:
            times = [r.wall_time for r in dim_results]
            errors = [r.error_percent for r in dim_results]
            mean_times.append(np.mean(times))
            std_times.append(np.std(times))
            mean_errors.append(np.mean(errors))
        else:
            mean_times.append(np.nan)
            std_times.append(np.nan)
            mean_errors.append(np.nan)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time vs Dimension
    ax = axes[0]
    ax.errorbar(dims, mean_times, yerr=std_times, fmt='o-', capsize=3, color='blue')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Wall Time (s)')
    ax.set_title('SunBURST Scaling')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Error vs Dimension
    ax = axes[1]
    ax.plot(dims, mean_errors, 'o-', color='red')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Evidence Error (%)')
    ax.set_title('Accuracy vs Dimension')
    ax.set_xscale('log', base=2)
    ax.axhline(y=1, color='green', linestyle='--', label='1% error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    dashboard_path = output_dir / "dashboard.png"
    plt.savefig(dashboard_path, dpi=150)
    plt.close()
    
    print(f"Dashboard saved to {dashboard_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SunBURST GPU Profiler")
    parser.add_argument('--dims', type=str, default=None,
                        help='Comma-separated dimensions (default: 2,4,8,...,1024)')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help=f'Runs per dimension (default: {DEFAULT_RUNS})')
    parser.add_argument('--n-oscillations', type=int, default=1,
                        help='n_oscillations parameter (default: 1)')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Use fast Hessian estimation (default: True)')
    parser.add_argument('--output-dir', type=str, default='./profiler_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Parse dimensions
    if args.dims:
        dims = [int(d) for d in args.dims.split(',')]
    else:
        dims = DEFAULT_DIMS
    
    # Print header
    print("=" * 60)
    print("SUNBURST GPU PROFILER v3.0")
    print("=" * 60)
    print(f"SunBURST version: {__version__}")
    print(f"GPU available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"GPU info: {gpu_info() if HAS_SUNBURST else 'N/A'}")
    print(f"Dimensions: {dims}")
    print(f"Runs per dimension: {args.runs}")
    print(f"n_oscillations: {args.n_oscillations}")
    print(f"fast: {args.fast}")
    print("=" * 60 + "\n")
    
    if not HAS_SUNBURST:
        print("ERROR: sunburst package not installed!")
        print("Run: pip install -e ..")
        sys.exit(1)
    
    # Run profiler
    config = {
        'dims': dims,
        'runs': args.runs,
        'n_oscillations': args.n_oscillations,
        'fast': args.fast,
    }
    
    results = run_profiler(dims, args.runs, args.n_oscillations, args.fast)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, config)
    create_dashboard(results, output_dir)


if __name__ == "__main__":
    main()
