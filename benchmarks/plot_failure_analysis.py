#!/usr/bin/env python3
"""
Plot SunBURST Failure Analysis Results

Visualizes what breaks and when across extreme noise levels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_failure_analysis(results_file='failure_analysis_results.npz', output_file='failure_analysis.png'):
    """
    Plot comprehensive failure analysis.
    
    Shows:
    1. Success rate vs noise (with failure mode breakdown)
    2. Evidence error for successful trials
    3. Peak finding accuracy
    4. Failure mode timeline (stacked area chart)
    """
    # Load results
    data = np.load(results_file, allow_pickle=True)
    trials = data['trials'].item()
    analytical_log_Z = float(data['analytical_log_Z'])
    
    # Extract data
    noise_levels = []
    success_rates = []
    
    # Failure mode counts
    spurious_counts = []
    large_error_counts = []
    wrong_peak_counts = []
    no_peaks_counts = []
    comp_failed_counts = []
    
    # Success metrics
    mean_errors = []
    std_errors = []
    mean_rel_errors = []
    std_rel_errors = []
    mean_peak_dists = []
    std_peak_dists = []
    
    for noise_result in trials:
        noise_levels.append(noise_result['noise_level'])
        success_rates.append(noise_result['success_rate'] * 100)
        
        counts = noise_result['success_counts']
        n_trials = noise_result['n_trials']
        spurious_counts.append(counts['SPURIOUS_PEAKS'] / n_trials * 100)
        large_error_counts.append(counts['LARGE_ERROR'] / n_trials * 100)
        wrong_peak_counts.append(counts['WRONG_PEAK'] / n_trials * 100)
        no_peaks_counts.append(counts['NO_PEAKS_FOUND'] / n_trials * 100)
        comp_failed_counts.append(counts['COMPUTATION_FAILED'] / n_trials * 100)
        
        mean_errors.append(noise_result['mean_log_Z_error'])
        std_errors.append(noise_result['std_log_Z_error'])
        mean_rel_errors.append(noise_result['mean_rel_error_pct'])
        std_rel_errors.append(noise_result['std_rel_error_pct'])
        mean_peak_dists.append(noise_result['mean_peak_distance'])
        std_peak_dists.append(noise_result['std_peak_distance'])
    
    noise_levels = np.array(noise_levels)
    success_rates = np.array(success_rates)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Stacked failure modes
    ax1 = fig.add_subplot(gs[0, :])
    
    # Stack from bottom: comp_failed, no_peaks, wrong_peak, large_error, spurious, success
    ax1.fill_between(noise_levels, 0, comp_failed_counts, 
                     label='Computation Failed', color='black', alpha=0.8)
    
    bottom = np.array(comp_failed_counts)
    ax1.fill_between(noise_levels, bottom, bottom + no_peaks_counts,
                     label='No Peaks Found', color='red', alpha=0.7)
    
    bottom += no_peaks_counts
    ax1.fill_between(noise_levels, bottom, bottom + wrong_peak_counts,
                     label='Wrong Peak', color='orange', alpha=0.7)
    
    bottom += wrong_peak_counts
    ax1.fill_between(noise_levels, bottom, bottom + large_error_counts,
                     label='Large Error (>50%)', color='yellow', alpha=0.7)
    
    bottom += large_error_counts
    ax1.fill_between(noise_levels, bottom, bottom + spurious_counts,
                     label='Spurious Peaks', color='cyan', alpha=0.7)
    
    bottom += spurious_counts
    ax1.fill_between(noise_levels, bottom, 100,
                     label='Success', color='green', alpha=0.7)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Noise Level', fontsize=11)
    ax1.set_ylabel('Percentage of Trials (%)', fontsize=11)
    ax1.set_title('Failure Mode Distribution vs Noise', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    
    # Add vertical line at critical threshold (95% success)
    critical_idx = np.where(success_rates < 95)[0]
    if len(critical_idx) > 0:
        critical_noise = noise_levels[critical_idx[0]]
        ax1.axvline(critical_noise, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax1.text(critical_noise, 50, f'95% threshold\n{critical_noise:.2e}', 
                rotation=90, va='bottom', ha='right', fontsize=9, color='red')
    
    # Plot 2: Success rate
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(noise_levels, success_rates, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xscale('log')
    ax2.set_xlabel('Noise Level', fontsize=11)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('Success Rate vs Noise', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(100, color='green', linestyle='--', alpha=0.3, label='100%')
    ax2.axhline(95, color='orange', linestyle='--', alpha=0.3, label='95%')
    ax2.axhline(90, color='red', linestyle='--', alpha=0.3, label='90%')
    ax2.axhline(50, color='darkred', linestyle='--', alpha=0.3, label='50%')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Evidence error (successful trials only)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Filter out NaNs
    valid_idx = ~np.isnan(mean_rel_errors)
    if np.any(valid_idx):
        ax3.errorbar(noise_levels[valid_idx], mean_rel_errors[valid_idx], 
                    yerr=std_rel_errors[valid_idx],
                    marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Noise Level', fontsize=11)
        ax3.set_ylabel('Relative Error (%) [successful trials]', fontsize=11)
        ax3.set_title('Evidence Error (Successful Trials Only)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Reference lines
        for err in [1, 10, 50]:
            ax3.axhline(err, color='gray', linestyle='--', alpha=0.3)
            ax3.text(noise_levels[0]*0.9, err, f'{err}%', fontsize=8, va='bottom')
    
    # Plot 4: Peak distance (successful trials only)
    ax4 = fig.add_subplot(gs[2, 0])
    
    valid_idx = ~np.isnan(mean_peak_dists)
    if np.any(valid_idx):
        ax4.errorbar(noise_levels[valid_idx], mean_peak_dists[valid_idx],
                    yerr=std_peak_dists[valid_idx],
                    marker='^', capsize=5, capthick=2, linewidth=2, markersize=8, color='purple')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xlabel('Noise Level', fontsize=11)
        ax4.set_ylabel('Peak Distance from True [successful]', fontsize=11)
        ax4.set_title('Peak Finding Accuracy', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Reference: sqrt(noise)
        x_ref = np.array([noise_levels.min(), noise_levels.max()])
        y_ref = np.sqrt(x_ref) * 0.01
        ax4.plot(x_ref, y_ref, '--', color='gray', alpha=0.5, label='~√noise')
        ax4.legend()
    
    # Plot 5: Number of peaks found (histogram for highest noise)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Get all trials from highest noise level
    highest_noise = trials[-1]
    peak_counts = [t['n_peaks'] for t in highest_noise['trials']]
    
    bins = np.arange(0, max(peak_counts)+2) - 0.5
    ax5.hist(peak_counts, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(1, color='green', linestyle='--', linewidth=2, label='Expected: 1 peak')
    ax5.set_xlabel('Number of Peaks Found', fontsize=11)
    ax5.set_ylabel('Number of Trials', fontsize=11)
    ax5.set_title(f'Peak Count Distribution at Noise = {highest_noise["noise_level"]:.2e}', 
                 fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('SunBURST Failure Analysis - Extreme Noise', fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot failure analysis results')
    parser.add_argument('--input', type=str, default='failure_analysis_results.npz',
                       help='Input NPZ file with results')
    parser.add_argument('--output', type=str, default='failure_analysis.png',
                       help='Output image file')
    
    args = parser.parse_args()
    
    plot_failure_analysis(args.input, args.output)
