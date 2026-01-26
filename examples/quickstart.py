#!/usr/bin/env python3
"""
SunBURST Quick Start Example

Demonstrates basic usage of SunBURST for Bayesian evidence calculation.
"""

import numpy as np
from sunburst import compute_evidence, gpu_available

# Check GPU availability
print(f"GPU available: {gpu_available()}")

# Define a simple Gaussian likelihood
def log_likelihood(x):
    """
    Log-likelihood for a unit Gaussian.
    
    Must handle batched inputs: (N, D) → (N,)
    """
    x = np.atleast_2d(x)
    return -0.5 * np.sum(x**2, axis=1)

# Define parameter bounds
dim = 64
bounds = [(-10, 10)] * dim

# Compute evidence
print(f"\nComputing evidence for {dim}D Gaussian...")
result = compute_evidence(
    log_likelihood,
    bounds,
    n_oscillations=1,  # Fast mode
    verbose=True,
)

# True log evidence for comparison
true_log_Z = 0.5 * dim * np.log(2 * np.pi) - dim * np.log(20)

# Report results
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"log Z (computed): {result.log_evidence:.4f}")
print(f"log Z (true):     {true_log_Z:.4f}")
print(f"Error: {abs(result.log_evidence - true_log_Z):.4f}")
print(f"Peaks found: {result.n_peaks}")
print(f"Time: {result.wall_time:.2f}s")
print(f"Likelihood calls: {result.n_likelihood_calls}")
