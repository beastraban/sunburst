#!/usr/bin/env python3
"""
Example: Using SunBURST with the Dashboard
==========================================

This shows how to integrate the dashboard callbacks into your SunBURST runs.

Usage:
    1. Start the dashboard server (in a separate terminal):
       python server.py
    
    2. Run this script:
       python example_with_dashboard.py
    
    3. Watch the dashboard update in real-time in your browser!
"""

import numpy as np
import time

# Import the dashboard client
from server import DashboardClient

# Simulated SunBURST pipeline for demonstration
# In real usage, you'd import your actual pipeline

def simulate_sunburst_run(dim: int, dashboard: DashboardClient):
    """
    Simulate a SunBURST run with dashboard updates.
    Replace this with your actual pipeline calls.
    """
    
    bounds = np.array([[-10, 10]] * dim)
    
    # 1. Pipeline starts
    dashboard.on_pipeline_start(
        dim=dim,
        bounds=bounds,
        config={"n_oscillations": 3, "greendragon_fast": True}
    )
    
    # 2. PreCheck
    dashboard.on_module_start("PreCheck")
    time.sleep(0.1)
    dashboard.on_module_end("PreCheck", time_seconds=0.1)
    
    # 3. CarryTiger - Mode Detection
    dashboard.on_module_start("CarryTiger")
    
    # Simulate ray casting with updates
    total_rays = 100
    ray_samples = []
    for i in range(total_rays):
        # Generate some fake samples
        sample = np.random.randn(dim) * 3
        ray_samples.append(sample[:2].tolist())  # First 2 dims for viz
        
        if i % 10 == 0:
            dashboard.on_rays_update(
                rays_cast=i,
                total_rays=total_rays,
                ray_samples=ray_samples
            )
            dashboard.on_progress(percent=(i / total_rays) * 100)
        
        time.sleep(0.01)
    
    # Found some peaks
    n_peaks = np.random.randint(1, 4)
    peaks = np.random.randn(n_peaks, dim) * 0.5
    likelihoods = -0.5 * np.sum(peaks**2, axis=1)
    
    dashboard.on_peaks_found(peaks, likelihoods)
    dashboard.on_module_end("CarryTiger", time_seconds=1.2, peaks_found=n_peaks)
    
    # 4. GreenDragon - Peak Refinement
    dashboard.on_module_start("GreenDragon")
    
    for i in range(20):
        dashboard.on_progress(percent=(i / 20) * 100, label="Refining peaks...")
        time.sleep(0.02)
    
    # Refined peaks (slightly different)
    refined_peaks = peaks + np.random.randn(*peaks.shape) * 0.01
    dashboard.on_peaks_found(refined_peaks, likelihoods)
    dashboard.on_module_end("GreenDragon", time_seconds=0.4)
    
    # 5. BendTheBow - Evidence Calculation
    dashboard.on_module_start("BendTheBow")
    
    for i in range(10):
        dashboard.on_progress(percent=(i / 10) * 100, label="Computing evidence...")
        time.sleep(0.05)
    
    # Compute "evidence" (just for demo)
    log_evidence = -0.5 * dim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(20))  # Gaussian evidence
    
    dashboard.on_module_end("BendTheBow", time_seconds=0.5)
    
    # 6. Final result
    total_time = 0.1 + 1.2 + 0.4 + 0.5
    dashboard.on_result(
        log_evidence=log_evidence,
        n_peaks=n_peaks,
        time_seconds=total_time,
        dim=dim
    )
    
    dashboard.on_pipeline_end(success=True)
    
    return {
        "log_evidence": log_evidence,
        "n_peaks": n_peaks,
        "time": total_time
    }


def main():
    """Run several test problems with dashboard."""
    
    print("="*60)
    print("  SunBURST Dashboard Demo")
    print("="*60)
    print()
    print("Make sure the dashboard server is running:")
    print("  python server.py")
    print()
    print("Then open http://localhost:8080 in your browser")
    print()
    print("Press Enter to start demo runs...")
    input()
    
    # Create dashboard client
    dashboard = DashboardClient()
    
    # Run a few test cases
    test_dims = [8, 16, 32, 64]
    
    for dim in test_dims:
        print(f"\n[Demo] Running {dim}D Gaussian test...")
        result = simulate_sunburst_run(dim, dashboard)
        print(f"[Demo] Result: log Z = {result['log_evidence']:.2f}, "
              f"{result['n_peaks']} peaks, {result['time']:.2f}s")
        
        time.sleep(1)  # Pause between runs
    
    print("\n" + "="*60)
    print("  Demo complete! Check the dashboard for results.")
    print("="*60)


if __name__ == "__main__":
    main()
