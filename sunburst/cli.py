"""
SunBURST Command-Line Interface

Usage:
    sunburst --test gaussian --dim 64
    sunburst --test mixture --dim 32 --modes 4
    sunburst --help
"""

import argparse
import sys
import numpy as np


def make_gaussian(dim: int, center: float = 0.0, sigma: float = 1.0):
    """Create a Gaussian likelihood for testing."""
    def log_likelihood(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(((x - center) / sigma) ** 2, axis=1)
    
    # Bounds: [-10, 10] in each dimension
    bounds = [(-10.0, 10.0)] * dim
    
    # True log evidence
    # Z = (2πσ²)^(D/2) / (20)^D
    true_log_Z = 0.5 * dim * np.log(2 * np.pi * sigma**2) - dim * np.log(20)
    
    return log_likelihood, bounds, true_log_Z


def make_mixture(dim: int, n_modes: int = 4, separation: float = 5.0):
    """Create a Gaussian mixture likelihood for testing."""
    # Place modes in a symmetric pattern
    centers = []
    for i in range(n_modes):
        center = np.zeros(dim)
        # Distribute modes along first few dimensions
        if i < dim:
            center[i % dim] = separation * (1 if (i // dim) % 2 == 0 else -1)
        centers.append(center)
    centers = np.array(centers)
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        # Log-sum-exp over modes
        log_probs = []
        for center in centers:
            log_probs.append(-0.5 * np.sum((x - center) ** 2, axis=1))
        log_probs = np.array(log_probs)
        max_log = np.max(log_probs, axis=0)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0))
    
    # Bounds: [-15, 15] to accommodate separated modes
    bounds = [(-15.0, 15.0)] * dim
    
    # Approximate true log evidence (each mode contributes ~equal)
    single_mode_log_Z = 0.5 * dim * np.log(2 * np.pi) - dim * np.log(30)
    true_log_Z = single_mode_log_Z + np.log(n_modes)
    
    return log_likelihood, bounds, true_log_Z


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SunBURST: GPU-accelerated Bayesian evidence calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sunburst --test gaussian --dim 64
  sunburst --test mixture --dim 32 --modes 4
  sunburst --version
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit"
    )
    
    parser.add_argument(
        "--test", "-t",
        choices=["gaussian", "mixture"],
        help="Run built-in test function"
    )
    
    parser.add_argument(
        "--dim", "-d",
        type=int,
        default=64,
        help="Dimensionality for test (default: 64)"
    )
    
    parser.add_argument(
        "--modes", "-m",
        type=int,
        default=4,
        help="Number of modes for mixture test (default: 4)"
    )
    
    parser.add_argument(
        "--n-osc",
        type=int,
        default=1,
        help="ChiSao oscillations (default: 1)"
    )
    
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable fast Hessian estimation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show GPU information and exit"
    )
    
    args = parser.parse_args()
    
    # Handle version
    if args.version:
        from sunburst import __version__
        print(f"sunburst {__version__}")
        return 0
    
    # Handle GPU info
    if args.gpu_info:
        from sunburst import gpu_available, gpu_info
        if gpu_available():
            info = gpu_info()
            print("GPU Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("No GPU available")
        return 0
    
    # Handle test
    if args.test:
        from sunburst import compute_evidence
        
        print("=" * 70)
        print(f"SunBURST Test: {args.test} (D={args.dim})")
        print("=" * 70)
        
        if args.test == "gaussian":
            log_L, bounds, true_log_Z = make_gaussian(args.dim)
        elif args.test == "mixture":
            log_L, bounds, true_log_Z = make_mixture(args.dim, args.modes)
            print(f"Mixture with {args.modes} modes")
        
        result = compute_evidence(
            log_L,
            bounds,
            n_oscillations=args.n_osc,
            fast=not args.no_fast,
            verbose=args.verbose,
        )
        
        error = abs(result.log_evidence - true_log_Z)
        error_pct = 100 * error / abs(true_log_Z) if true_log_Z != 0 else 0
        
        print("\n" + "-" * 70)
        print("RESULTS")
        print("-" * 70)
        print(f"log Z (computed): {result.log_evidence:.4f}")
        print(f"log Z (true):     {true_log_Z:.4f}")
        print(f"Error: {error:.4f} ({error_pct:.2f}%)")
        print(f"Peaks found: {result.n_peaks}")
        print(f"Time: {result.wall_time:.2f}s")
        print(f"Likelihood calls: {result.n_likelihood_calls}")
        
        status = "PASS" if error_pct < 5.0 else "FAIL"
        print(f"\nStatus: {status}")
        print("=" * 70)
        
        return 0 if status == "PASS" else 1
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
