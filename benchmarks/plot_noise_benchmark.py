#!/usr/bin/env python3
"""
Plot benchmark results: Error vs Noise Amplitude
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_benchmark_results(results_file='noise_benchmark_results.npz', output_file='noise_benchmark.png'):
    """
    Plot benchmark results showing how errors scale with noise.
    
    Args:
        results_file: NPZ file with benchmark results
        output_file: Output image file
    """
    # Load results
    data = np.load(results_file, allow_pickle=True)
    trials = data['trials'].item()  # List of dicts
    analytical_log_Z = float(data['analytical_log_Z'])
    
    # Extract data
    noise_levels = []
    mean_log_Z_errors = []
    std_log_Z_errors = []
    mean_rel_errors = []
    std_rel_errors = []
    mean_peak_dists = []
    std_peak_dists = []
    success_rates = []
    
    for noise_result in trials:
        noise_levels.append(noise_result['noise_level'])
        mean_log_Z_errors.append(noise_result['mean_log_Z_error'])
        std_log_Z_errors.append(noise_result['std_log_Z_error'])
        mean_rel_errors.append(noise_result['mean_rel_error_pct'])
        std_rel_errors.append(noise_result['std_rel_error_pct'])
        mean_peak_dists.append(noise_result['mean_peak_distance'])
        std_peak_dists.append(noise_result['std_peak_distance'])
        success_rates.append(noise_result['success_rate'] * 100)
    
    noise_levels = np.array(noise_levels)
    mean_log_Z_errors = np.array(mean_log_Z_errors)
    std_log_Z_errors = np.array(std_log_Z_errors)
    mean_rel_errors = np.array(mean_rel_errors)
    std_rel_errors = np.array(std_rel_errors)
    mean_peak_dists = np.array(mean_peak_dists)
    std_peak_dists = np.array(std_peak_dists)
    success_rates = np.array(success_rates)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SunBURST Performance vs Noise Amplitude', fontsize=14, fontweight='bold')
    
    # Plot 1: Absolute evidence error
    ax = axes[0, 0]
    ax.errorbar(noise_levels, mean_log_Z_errors, yerr=std_log_Z_errors,
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('|log(Z) - log(Z_true)|', fontsize=11)
    ax.set_title('Evidence Error (Absolute)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference line: error ~ noise
    x_ref = np.array([noise_levels.min(), noise_levels.max()])
    y_ref = x_ref * 10  # Rough scaling
    ax.plot(x_ref, y_ref, '--', color='gray', alpha=0.5, label='~10× noise')
    ax.legend()
    
    # Plot 2: Relative evidence error
    ax = axes[0, 1]
    ax.errorbar(noise_levels, mean_rel_errors, yerr=std_rel_errors,
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
    ax.set_xscale('log')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('Evidence Error (Relative)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='1% error')
    ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5% error')
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% error')
    ax.legend()
    
    # Plot 3: Peak finding accuracy
    ax = axes[1, 0]
    ax.errorbar(noise_levels, mean_peak_dists, yerr=std_peak_dists,
                marker='^', capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Peak Distance from True', fontsize=11)
    ax.set_title('Peak Finding Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference line: distance ~ sqrt(noise)
    y_ref = np.sqrt(x_ref) * 0.01
    ax.plot(x_ref, y_ref, '--', color='gray', alpha=0.5, label='~√noise')
    ax.legend()
    
    # Plot 4: Success rate
    ax = axes[1, 1]
    ax.plot(noise_levels, success_rates, marker='D', linewidth=2, markersize=8, color='purple')
    ax.set_xscale('log')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Peak Detection Success Rate', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color='green', linestyle='--', alpha=0.5, label='100% success')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot noise benchmark results')
    parser.add_argument('--input', type=str, default='noise_benchmark_results.npz',
                       help='Input NPZ file with results')
    parser.add_argument('--output', type=str, default='noise_benchmark.png',
                       help='Output image file')
    
    args = parser.parse_args()
    
    plot_benchmark_results(args.input, args.output)
