#!/usr/bin/env python3
"""
SunBURST Super-GUI v1.0
========================

A comprehensive web interface for Bayesian evidence calculation.

Features:
---------
- Real-time pipeline monitoring (like TensorBoard)
- Interactive test function runner
- Corner plots from Laplace approximation
- Model comparison with Bayes factors
- Parameter estimation with credible intervals
- Run history and export
- Dimensional scaling analysis

Requirements:
    pip install streamlit plotly pandas numpy scipy websockets

Usage:
    streamlit run app.py

Author: Ira Wolfson, Braude College of Engineering
Date: January 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import asyncio
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.special import logsumexp
import queue
import sys
import os
# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



# Go up one level to find the sunburst package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SunBURST",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #94a3b8 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }
    .success-box {
        padding: 1rem;
        background-color: #064e3b;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
    }
    .warning-box {
        padding: 1rem;
        background-color: #78350f;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
    }
    .info-box {
        padding: 1rem;
        background-color: #1e3a5f;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #0f172a;
    }
    .big-number {
        font-size: 3rem;
        font-weight: bold;
        color: #60a5fa;
    }
    .module-active {
        background: linear-gradient(135deg, #e94560, #ff6b8a);
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    .module-done {
        background-color: #064e3b;
        padding: 10px;
        border-radius: 8px;
        color: #10b981;
    }
    .module-pending {
        background-color: #1e1e2e;
        padding: 10px;
        border-radius: 8px;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SunBURSTResult:
    """Container for SunBURST results."""
    log_evidence: float
    n_modes: int
    peaks: np.ndarray
    log_likelihoods: np.ndarray
    hessians: Optional[np.ndarray] = None  # Diagonal Hessians
    full_hessians: Optional[List[np.ndarray]] = None  # Full Hessians if computed
    weights: Optional[np.ndarray] = None  # Mode weights
    dimension: int = 0
    time_total: float = 0.0
    time_carrytiger: float = 0.0
    time_greendragon: float = 0.0
    time_bendthebow: float = 0.0
    error_estimate: Optional[float] = None
    config: Dict = field(default_factory=dict)
    timestamp: str = ""
    name: str = "Unnamed"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.dimension == 0 and self.peaks is not None:
            self.dimension = self.peaks.shape[1] if len(self.peaks.shape) > 1 else 1
        if self.weights is None and self.n_modes > 0:
            # Equal weights if not provided
            self.weights = np.ones(self.n_modes) / self.n_modes


@dataclass 
class BayesFactor:
    """Bayes factor interpretation."""
    log_bf: float
    model1: str
    model2: str
    
    @property
    def interpretation(self) -> str:
        abs_log_bf = abs(self.log_bf)
        if abs_log_bf < 1:
            strength = "Not worth mentioning"
        elif abs_log_bf < 2.5:
            strength = "Substantial"
        elif abs_log_bf < 5:
            strength = "Strong"
        else:
            strength = "Decisive"
        
        if self.log_bf > 0:
            favored = self.model1
        else:
            favored = self.model2
        
        return f"{strength} evidence for {favored}"
    
    @staticmethod
    def interpret(log_bf: float) -> str:
        """Interpret a log Bayes factor value."""
        abs_log_bf = abs(log_bf)
        if abs_log_bf < 1:
            return "Not worth mentioning"
        elif abs_log_bf < 2.5:
            return "Substantial"
        elif abs_log_bf < 5:
            return "Strong"
        else:
            return "Decisive"


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

class TestFunctions:
    """Built-in test functions with known analytical evidence. GPU-compatible."""
    
    @staticmethod
    def _get_xp(x):
        """Get array module (numpy or cupy) based on input."""
        if hasattr(x, '__cuda_array_interface__'):
            import cupy as cp
            return cp
        return np
    
    @staticmethod
    def gaussian(x, sigma: float = 1.0):
        """Standard Gaussian centered at origin."""
        xp = TestFunctions._get_xp(x)
        x = xp.atleast_2d(x)
        return -0.5 * xp.sum((x / sigma) ** 2, axis=1)
    
    @staticmethod
    def gaussian_evidence(d: int, sigma: float, bounds: np.ndarray) -> float:
        """Analytical log-evidence for Gaussian."""
        log_gaussian = 0.5 * d * np.log(2 * np.pi) + d * np.log(sigma)
        log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
        return log_gaussian - log_prior_volume
    
    @staticmethod
    def bimodal(x, separation: float = 4.0, sigma: float = 1.0):
        """Two Gaussian modes separated along first axis."""
        xp = TestFunctions._get_xp(x)
        x = xp.atleast_2d(x)
        d = x.shape[1]
        
        # Create centers as arrays
        mu1_arr = [-separation / 2] + [0.0] * (d - 1)
        mu2_arr = [separation / 2] + [0.0] * (d - 1)
        
        mu1 = xp.array(mu1_arr, dtype=x.dtype)
        mu2 = xp.array(mu2_arr, dtype=x.dtype)
        
        log_L1 = -0.5 * xp.sum(((x - mu1) / sigma) ** 2, axis=1)
        log_L2 = -0.5 * xp.sum(((x - mu2) / sigma) ** 2, axis=1)
        
        # Normalize so peak value = 0 (same as unimodal)
        return xp.logaddexp(log_L1, log_L2) - float(np.log(2))
    
    @staticmethod
    def bimodal_evidence(d: int, separation: float, sigma: float, bounds: np.ndarray) -> float:
        """Analytical log-evidence for bimodal."""
        log_single = 0.5 * d * np.log(2 * np.pi) + d * np.log(sigma)
        log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
        # The -log(2) in the likelihood normalizes it, so integral = same as unimodal
        return log_single - log_prior_volume
    
    @staticmethod
    def trimodal(x, separation: float = 4.0, sigma: float = 1.0):
        """Three Gaussian modes in triangular arrangement."""
        xp = TestFunctions._get_xp(x)
        x = xp.atleast_2d(x)
        d = x.shape[1]
        
        # Build center arrays - equilateral triangle in first two dimensions
        mu1_arr = [-separation / 2] + [0.0] * (d - 1)
        mu2_arr = [separation / 2] + [0.0] * (d - 1)
        mu3_arr = [0.0] + ([separation * float(np.sqrt(3)) / 2] if d > 1 else []) + [0.0] * max(0, d - 2)
        
        mu1 = xp.array(mu1_arr, dtype=x.dtype)
        mu2 = xp.array(mu2_arr, dtype=x.dtype)
        mu3 = xp.array(mu3_arr, dtype=x.dtype)
        
        log_L1 = -0.5 * xp.sum(((x - mu1) / sigma) ** 2, axis=1)
        log_L2 = -0.5 * xp.sum(((x - mu2) / sigma) ** 2, axis=1)
        log_L3 = -0.5 * xp.sum(((x - mu3) / sigma) ** 2, axis=1)
        
        # Manual logsumexp for 3 terms
        max_L = xp.maximum(xp.maximum(log_L1, log_L2), log_L3)
        result = max_L + xp.log(xp.exp(log_L1 - max_L) + xp.exp(log_L2 - max_L) + xp.exp(log_L3 - max_L))
        
        # Normalize so peak value = 0 (same as unimodal)
        return result - float(np.log(3))
    
    @staticmethod
    def trimodal_evidence(d: int, separation: float, sigma: float, bounds: np.ndarray) -> float:
        """Analytical log-evidence for trimodal."""
        log_single = 0.5 * d * np.log(2 * np.pi) + d * np.log(sigma)
        log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
        # The -log(3) in the likelihood normalizes it, so integral = same as unimodal
        return log_single - log_prior_volume
    
    @staticmethod
    def rosenbrock(x, a: float = 1.0, b: float = 100.0):
        """Rosenbrock function (curved valley)."""
        xp = TestFunctions._get_xp(x)
        x = xp.atleast_2d(x)
        d = x.shape[1]
        
        result = xp.zeros(x.shape[0], dtype=x.dtype)
        for i in range(d - 1):
            result = result + (a - x[:, i])**2 + b * (x[:, i+1] - x[:, i]**2)**2
        
        return -result  # Negative because we want log-likelihood
    
    @staticmethod
    def eggbox(x, scale: float = 5.0):
        """Eggbox function (periodic multimodal)."""
        xp = TestFunctions._get_xp(x)
        x = xp.atleast_2d(x)
        return (2.0 + xp.prod(xp.cos(scale * x), axis=1)) ** 5
    
    @staticmethod
    def cigar(x, ratio: float = 100.0):
        """Cigar-shaped Gaussian (anisotropic). LONG along first axis, thin in others."""
        xp = TestFunctions._get_xp(x)
        x = xp.atleast_2d(x)
        d = x.shape[1]
        
        # First dimension has width=ratio (LONG), others have width=1 (normal)
        # This creates a long cigar extending along θ₁
        sigmas_arr = [ratio] + [1.0] * (d - 1)
        sigmas = xp.array(sigmas_arr, dtype=x.dtype)
        
        return -0.5 * xp.sum((x / sigmas) ** 2, axis=1)
    
    @staticmethod
    def cigar_evidence(d: int, ratio: float, bounds: np.ndarray) -> float:
        """Analytical log-evidence for cigar."""
        # First dimension has width=ratio (LONG), others have width=1
        sigmas = np.array([ratio] + [1.0] * (d - 1))
        log_gaussian = 0.5 * d * np.log(2 * np.pi) + np.sum(np.log(sigmas))
        log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
        return log_gaussian - log_prior_volume
    
    @staticmethod
    def get_function(name: str):
        """Get function by name."""
        functions = {
            "Gaussian": TestFunctions.gaussian,
            "Bimodal": TestFunctions.bimodal,
            "Trimodal": TestFunctions.trimodal,
            "Rosenbrock": TestFunctions.rosenbrock,
            "Eggbox": TestFunctions.eggbox,
            "Cigar": TestFunctions.cigar,
        }
        return functions.get(name)
    
    @staticmethod
    def get_analytical_evidence(name: str, d: int, bounds: np.ndarray, **kwargs) -> Optional[float]:
        """Get analytical evidence if available."""
        sigma = kwargs.get("sigma", 1.0)
        separation = kwargs.get("separation", 4.0)
        ratio = kwargs.get("ratio", 100.0)
        
        log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
        
        if name == "Gaussian":
            log_gaussian = 0.5 * d * np.log(2 * np.pi) + d * np.log(sigma)
            return log_gaussian - log_prior_volume
        
        elif name == "Bimodal":
            # Likelihood is normalized with -log(2), so integral = same as unimodal
            log_gaussian = 0.5 * d * np.log(2 * np.pi) + d * np.log(sigma)
            return log_gaussian - log_prior_volume
        
        elif name == "Trimodal":
            # Likelihood is normalized with -log(3), so integral = same as unimodal
            log_gaussian = 0.5 * d * np.log(2 * np.pi) + d * np.log(sigma)
            return log_gaussian - log_prior_volume
        
        elif name == "Cigar":
            # Anisotropic Gaussian: first dim has width=ratio (LONG), others width=1
            sigmas = np.array([ratio] + [1.0] * (d - 1))
            log_gaussian = 0.5 * d * np.log(2 * np.pi) + np.sum(np.log(sigmas))
            return log_gaussian - log_prior_volume
        
        else:
            return None  # Rosenbrock, Eggbox - no analytical form


# =============================================================================
# MOCK SUNBURST RUNNER (Replace with actual import)
# =============================================================================

def run_sunburst_mock(log_likelihood, bounds: np.ndarray, config: Dict) -> SunBURSTResult:
    """
    Mock SunBURST runner for demonstration.
    Replace this with actual SunBURST pipeline import.
    """
    d = len(bounds)
    
    # Simulate timing
    t0 = time.time()
    
    # Simulate CarryTiger
    time.sleep(0.1 + 0.01 * d)
    t_carry = time.time() - t0
    
    # Find peaks (mock - just sample and take best)
    n_samples = 1000
    samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_samples, d))
    log_L = log_likelihood(samples)
    
    # Simple peak detection (mock)
    best_idx = np.argmax(log_L)
    peaks = samples[best_idx:best_idx+1]
    log_peaks = log_L[best_idx:best_idx+1]
    
    # Check for multimodality (mock)
    n_modes = 1
    if hasattr(log_likelihood, '__name__') and 'bimodal' in log_likelihood.__name__.lower():
        n_modes = 2
        # Add second peak
        samples2 = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_samples, d))
        log_L2 = log_likelihood(samples2)
        best_idx2 = np.argmax(log_L2)
        peaks = np.vstack([peaks, samples2[best_idx2]])
        log_peaks = np.append(log_peaks, log_L2[best_idx2])
    
    # Simulate GreenDragon
    time.sleep(0.05 + 0.005 * d)
    t_green = time.time() - t0 - t_carry
    
    # Mock Hessian (diagonal)
    hessians = -np.ones((n_modes, d))  # Negative = concave down
    
    # Simulate BendTheBow
    time.sleep(0.05 + 0.005 * d)
    t_bend = time.time() - t0 - t_carry - t_green
    
    # Mock evidence calculation (Laplace approximation)
    log_evidence = log_peaks[0] + 0.5 * d * np.log(2 * np.pi) - 0.5 * np.sum(np.log(-hessians[0]))
    log_prior_vol = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
    log_evidence -= log_prior_vol
    
    if n_modes > 1:
        log_Z2 = log_peaks[1] + 0.5 * d * np.log(2 * np.pi) - 0.5 * np.sum(np.log(-hessians[1])) - log_prior_vol
        log_evidence = np.logaddexp(log_evidence, log_Z2)
    
    t_total = time.time() - t0
    
    return SunBURSTResult(
        log_evidence=log_evidence,
        n_modes=n_modes,
        peaks=peaks,
        log_likelihoods=log_peaks,
        hessians=hessians,
        dimension=d,
        time_total=t_total,
        time_carrytiger=t_carry,
        time_greendragon=t_green,
        time_bendthebow=t_bend,
        config=config
    )


# =============================================================================
# TRY TO IMPORT ACTUAL SUNBURST
# =============================================================================

HAS_SUNBURST = False
try:
    from sunburst.pipeline import compute_evidence, SunburstResult
    HAS_SUNBURST = True
    print("✓ SunBURST imported successfully!")
except Exception as e:
    import traceback
    print(f"Failed to import SunBURST:")
    print(traceback.format_exc())


def run_sunburst(log_likelihood, bounds: np.ndarray, config: Dict) -> SunBURSTResult:
    """Run SunBURST (real or mock)."""
    if HAS_SUNBURST:
        # Use real pipeline - it's a function, not a class!
        result = compute_evidence(
            log_likelihood=log_likelihood,
            bounds=[(b[0], b[1]) for b in bounds],  # Convert to list of tuples
            n_oscillations=config.get("n_oscillations", 1),
            fast=config.get("greendragon_fast", True),
            verbose=False,
        )
        
        return SunBURSTResult(
            log_evidence=result.log_evidence,
            n_modes=result.n_peaks,
            peaks=result.peaks if result.peaks is not None else np.zeros((1, len(bounds))),
            log_likelihoods=result.L_peaks if result.L_peaks is not None else np.zeros(1),
            hessians=result.diag_H,
            dimension=len(bounds),
            time_total=result.wall_time,
            time_carrytiger=result.module_times.get('carry_tiger', 0),
            time_greendragon=result.module_times.get('green_dragon', 0),
            time_bendthebow=result.module_times.get('bend_bow', 0),
            config=config
        )
    else:
        return run_sunburst_mock(log_likelihood, bounds, config)
    
    
# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_likelihood_surface(log_likelihood, bounds: np.ndarray, 
                             peaks: Optional[np.ndarray] = None,
                             resolution: int = 50) -> go.Figure:
    """Create 2D likelihood surface plot."""
    d = len(bounds)
    
    # Use first two dimensions
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0] if d > 1 else -1, bounds[1, 1] if d > 1 else 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate likelihood
    if d == 1:
        points = X.flatten().reshape(-1, 1)
    else:
        points = np.column_stack([X.flatten(), Y.flatten()])
        if d > 2:
            # Pad with zeros for extra dimensions
            points = np.hstack([points, np.zeros((len(points), d - 2))])
    
    Z = log_likelihood(points).reshape(X.shape)
    
    # Create figure
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Viridis',
        name='log L',
        colorbar=dict(title='log L')
    ))
    
    # Add peaks
    if peaks is not None and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=peaks[:, 0],
            y=peaks[:, 1] if d > 1 else np.zeros(len(peaks)),
            mode='markers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='white')),
            name='Peaks'
        ))
    
    fig.update_layout(
        title='Likelihood Surface (first 2 dimensions)',
        xaxis_title='θ₁',
        yaxis_title='θ₂',
        height=500
    )
    
    return fig


def create_corner_plot(result: SunBURSTResult, n_samples: int = 5000) -> Optional[go.Figure]:
    """Create corner plot from Laplace approximation."""
    if result.hessians is None:
        return None
    
    d = min(result.dimension, 10)  # Limit to 10 dimensions
    n_modes = result.n_modes
    
    # Generate samples from mixture of Gaussians
    samples_list = []
    
    for k in range(n_modes):
        # Covariance from Hessian (Σ = -H⁻¹)
        diag_cov = -1.0 / result.hessians[k, :d]
        diag_cov = np.maximum(diag_cov, 1e-10)  # Ensure positive
        
        # Number of samples per mode (proportional to weight)
        w = result.weights[k] if result.weights is not None else 1.0 / n_modes
        n_k = int(n_samples * w)
        
        # Sample from Gaussian
        mode_samples = np.random.randn(n_k, d) * np.sqrt(diag_cov) + result.peaks[k, :d]
        samples_list.append(mode_samples)
    
    samples = np.vstack(samples_list)
    
    # Create corner plot
    fig = make_subplots(
        rows=d, cols=d,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )
    
    colors = px.colors.qualitative.Set2
    
    for i in range(d):
        for j in range(d):
            if i == j:
                # Diagonal: 1D histogram
                fig.add_trace(
                    go.Histogram(
                        x=samples[:, i],
                        nbinsx=30,
                        marker_color=colors[0],
                        showlegend=False,
                        opacity=0.7
                    ),
                    row=i+1, col=j+1
                )
            elif i > j:
                # Lower triangle: 2D scatter/contour
                fig.add_trace(
                    go.Histogram2dContour(
                        x=samples[:, j],
                        y=samples[:, i],
                        colorscale='Blues',
                        showscale=False,
                        contours=dict(
                            showlabels=False,
                            coloring='fill'
                        ),
                        showlegend=False
                    ),
                    row=i+1, col=j+1
                )
            else:
                # Upper triangle: empty
                pass
    
    # Update layout
    fig.update_layout(
        title='Corner Plot (Laplace Approximation)',
        height=150 * d,
        width=150 * d,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(d):
        fig.update_xaxes(title_text=f'θ_{i+1}' if i == d-1 else '', row=d, col=i+1)
        fig.update_yaxes(title_text=f'θ_{i+1}' if i > 0 else '', row=i+1, col=1)
    
    return fig


def create_timing_chart(result: SunBURSTResult) -> go.Figure:
    """Create timing breakdown pie chart."""
    labels = ['CarryTiger', 'GreenDragon', 'BendTheBow']
    values = [result.time_carrytiger, result.time_greendragon, result.time_bendthebow]
    colors = ['#e94560', '#4ade80', '#60a5fa']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title='Timing Breakdown',
        height=400,
        annotations=[dict(text=f'{result.time_total:.2f}s', x=0.5, y=0.5, 
                         font_size=20, showarrow=False)]
    )
    
    return fig


def create_scaling_plot(results: List[Dict]) -> go.Figure:
    """Create dimensional scaling analysis plot."""
    if not results:
        return go.Figure()
    
    dims = [r['dimension'] for r in results]
    times = [r['time'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dims, y=times,
        mode='markers+lines',
        marker=dict(size=10, color='#e94560'),
        line=dict(color='#e94560'),
        name='Measured'
    ))
    
    # Fit power law
    if len(dims) > 2:
        log_d = np.log(dims)
        log_t = np.log(times)
        slope, intercept = np.polyfit(log_d, log_t, 1)
        
        d_fit = np.linspace(min(dims), max(dims), 100)
        t_fit = np.exp(intercept) * d_fit ** slope
        
        fig.add_trace(go.Scatter(
            x=d_fit, y=t_fit,
            mode='lines',
            line=dict(color='#60a5fa', dash='dash'),
            name=f'Fit: O(d^{slope:.2f})'
        ))
    
    fig.update_layout(
        title='Dimensional Scaling',
        xaxis_title='Dimension',
        yaxis_title='Time (s)',
        xaxis_type='log',
        yaxis_type='log',
        height=400
    )
    
    return fig


def create_model_comparison_chart(results: Dict[str, SunBURSTResult]) -> go.Figure:
    """Create model comparison bar chart."""
    names = list(results.keys())
    log_evidences = [r.log_evidence for r in results.values()]
    
    # Normalize to model probabilities
    log_Z_max = max(log_evidences)
    probs = np.exp(np.array(log_evidences) - log_Z_max)
    probs = probs / np.sum(probs)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Log Evidence', 'Model Probability'])
    
    fig.add_trace(
        go.Bar(x=names, y=log_evidences, marker_color='#60a5fa', name='log Z'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=names, y=probs, marker_color='#4ade80', name='P(M|D)'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================

def compute_parameter_estimates(result: SunBURSTResult) -> pd.DataFrame:
    """Compute parameter estimates from Laplace approximation."""
    if result.hessians is None:
        return pd.DataFrame()
    
    d = result.dimension
    estimates = []
    
    for i in range(d):
        # Weighted mean across modes
        mean = np.sum(result.weights * result.peaks[:, i])
        
        # Weighted variance
        var = 0
        for k in range(result.n_modes):
            # Mode variance from Hessian
            mode_var = -1.0 / result.hessians[k, i]
            mode_var = max(mode_var, 1e-10)
            
            # Include mode spread
            var += result.weights[k] * (mode_var + (result.peaks[k, i] - mean)**2)
        
        std = np.sqrt(var)
        
        # Credible intervals (assuming Gaussian)
        ci_68 = (mean - std, mean + std)
        ci_95 = (mean - 1.96*std, mean + 1.96*std)
        
        estimates.append({
            'Parameter': f'θ_{i+1}',
            'Mean': f'{mean:.4f}',
            'Std': f'{std:.4f}',
            '68% CI': f'[{ci_68[0]:.3f}, {ci_68[1]:.3f}]',
            '95% CI': f'[{ci_95[0]:.3f}, {ci_95[1]:.3f}]'
        })
    
    return pd.DataFrame(estimates)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'results' not in st.session_state:
    st.session_state.results = {}  # For model comparison

if 'current_result' not in st.session_state:
    st.session_state.current_result = None

if 'run_history' not in st.session_state:
    st.session_state.run_history = []

if 'scaling_data' not in st.session_state:
    st.session_state.scaling_data = []


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.title("☀️ SunBURST")
        st.caption("Bayesian Evidence Calculator")
        
        st.divider()
        
        # Status indicator
        if HAS_SUNBURST:
            st.success("✓ GPU Pipeline Ready")
        else:
            st.warning("⚠ Using Mock Pipeline")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🚀 Run Analysis", "📊 Corner Plot", "⚖️ Model Comparison", 
             "📈 Scaling Analysis", "📜 History", "ℹ️ About"],
            label_visibility="collapsed"
        )
    
    # =========================
    # PAGE: RUN ANALYSIS
    # =========================
    if page == "🚀 Run Analysis":
        st.header("🚀 Run SunBURST Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            
            # Test function selection
            func_name = st.selectbox(
                "Test Function",
                ["Gaussian", "Bimodal", "Trimodal", "Rosenbrock", "Eggbox", "Cigar"]
            )
            
            # Dimension
            dimension = st.slider("Dimension", 2, 128, 8)
            
            # Function-specific parameters
            sigma = 1.0
            separation = 4.0
            ratio = 100.0
            
            if func_name in ["Gaussian", "Bimodal", "Trimodal"]:
                sigma = st.slider("σ (width)", 0.1, 5.0, 1.0)
            
            if func_name in ["Bimodal", "Trimodal"]:
                separation = st.slider("Mode separation", 1.0, 10.0, 4.0)
            
            if func_name == "Cigar":
                ratio = st.slider("Axis ratio", 10.0, 500.0, 100.0)
            
            # Bounds
            bound_range = st.slider("Prior bounds (±)", 5.0, 25.0, 10.0)
            bounds = np.array([[-bound_range, bound_range]] * dimension)
            
            # Pipeline config
            st.divider()
            st.markdown("**Pipeline Config**")
            n_osc = st.selectbox("Oscillations", [1, 3, 5], index=0)
            fast_mode = st.checkbox("Fast mode", value=True)
            
            # Model name for comparison
            model_name = st.text_input("Model name", value=f"{func_name}_{dimension}D")
            
            # Run button
            run_button = st.button("▶️ Run SunBURST", type="primary", use_container_width=True)
        
        with col2:
            if run_button:
                # Create likelihood function
                if func_name == "Gaussian":
                    log_L = lambda x, s=sigma: TestFunctions.gaussian(x, sigma=s)
                elif func_name == "Bimodal":
                    log_L = lambda x, sep=separation, s=sigma: TestFunctions.bimodal(x, separation=sep, sigma=s)
                elif func_name == "Trimodal":
                    log_L = lambda x, sep=separation, s=sigma: TestFunctions.trimodal(x, separation=sep, sigma=s)
                elif func_name == "Cigar":
                    log_L = lambda x, r=ratio: TestFunctions.cigar(x, ratio=r)
                elif func_name == "Rosenbrock":
                    log_L = TestFunctions.rosenbrock
                elif func_name == "Eggbox":
                    log_L = TestFunctions.eggbox
                else:
                    log_L = TestFunctions.get_function(func_name)
                
                config = {
                    "n_oscillations": n_osc,
                    "greendragon_fast": fast_mode,
                    "function": func_name,
                    "sigma": sigma,
                    "separation": separation,
                    "ratio": ratio
                }
                
                # Run with progress
                with st.spinner("Running SunBURST..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🐯 CarryTiger: Detecting modes...")
                    progress_bar.progress(25)
                    
                    result = run_sunburst(log_L, bounds, config)
                    result.name = model_name
                    
                    status_text.text("🐉 GreenDragon: Refining peaks...")
                    progress_bar.progress(50)
                    time.sleep(0.1)
                    
                    status_text.text("🏹 BendTheBow: Computing evidence...")
                    progress_bar.progress(75)
                    time.sleep(0.1)
                    
                    progress_bar.progress(100)
                    status_text.text("✓ Complete!")
                
                # Store result
                st.session_state.current_result = result
                st.session_state.run_history.append({
                    'name': model_name,
                    'log_Z': result.log_evidence,
                    'n_modes': result.n_modes,
                    'dimension': dimension,
                    'time': result.time_total,
                    'timestamp': result.timestamp
                })
                st.session_state.scaling_data.append({
                    'dimension': dimension,
                    'time': result.time_total
                })
                
                # Get analytical evidence if available
                analytical = TestFunctions.get_analytical_evidence(
                    func_name, dimension, bounds, sigma=sigma, separation=separation, ratio=ratio
                )
                
                st.divider()
                
                # Results display
                st.subheader("Results")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("log Z", f"{result.log_evidence:.4f}")
                m2.metric("Modes", result.n_modes)
                m3.metric("Time", f"{result.time_total:.3f}s")
                
                if analytical is not None:
                    error = 100 * abs(result.log_evidence - analytical) / abs(analytical)
                    m4.metric("Error", f"{error:.2f}%")
                    st.caption(f"Analytical: log Z = {analytical:.4f}")
                else:
                    m4.metric("Error", "N/A")
                    st.caption("No analytical solution available")
                
                # Visualizations
                tab1, tab2, tab3 = st.tabs(["🗺️ Likelihood", "⏱️ Timing", "📋 Parameters"])
                
                with tab1:
                    # For cigar, expand plot bounds to show full extent
                    if func_name == "Cigar":
                        # θ₁ extends to 3*ratio, θ₂ extends to 3*1
                        plot_bounds = bounds.copy()
                        plot_bounds[0] = [-3*ratio, 3*ratio]  # Long axis
                        plot_bounds[1] = [-10, 10]  # Normal axis
                    else:
                        plot_bounds = bounds
                    fig = create_likelihood_surface(log_L, plot_bounds, result.peaks)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = create_timing_chart(result)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    df = compute_parameter_estimates(result)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No Hessian information available")
            
            elif st.session_state.current_result is not None:
                # Show last result
                result = st.session_state.current_result
                st.subheader(f"Last Run: {result.name}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("log Z", f"{result.log_evidence:.4f}")
                m2.metric("Modes", result.n_modes)
                m3.metric("Time", f"{result.time_total:.3f}s")
            
            else:
                st.info("Configure parameters and click 'Run SunBURST' to start.")
                
                # Show example
                st.markdown("""
                ### Quick Start
                
                1. Select a test function (Gaussian is simplest)
                2. Choose dimension (start low, like 8)
                3. Click **Run SunBURST**
                4. View results and visualizations
                
                ### Tips
                
                - **Gaussian**: Validates accuracy (should give ~0% error)
                - **Bimodal/Trimodal**: Tests mode detection (~0% error)
                - **Cigar**: Tests anisotropic handling (~0% error)
                - **Rosenbrock/Eggbox**: Non-Gaussian (no analytical solution)
                """)
    
    # =========================
    # PAGE: CORNER PLOT
    # =========================
    elif page == "📊 Corner Plot":
        st.header("📊 Corner Plot")
        
        if st.session_state.current_result is not None:
            result = st.session_state.current_result
            
            st.markdown(f"**Model:** {result.name} | **Dimension:** {result.dimension} | **Modes:** {result.n_modes}")
            
            if result.dimension > 10:
                st.warning(f"Showing first 10 of {result.dimension} dimensions")
            
            n_samples = st.slider("Number of samples", 1000, 20000, 5000, 1000)
            
            if st.button("Generate Corner Plot", type="primary"):
                with st.spinner("Generating corner plot..."):
                    fig = create_corner_plot(result, n_samples)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not generate corner plot (no Hessian information)")
        else:
            st.info("Run SunBURST first to generate a corner plot.")
    
    # =========================
    # PAGE: MODEL COMPARISON
    # =========================
    elif page == "⚖️ Model Comparison":
        st.header("⚖️ Model Comparison")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Add Models")
            
            if st.session_state.current_result is not None:
                result = st.session_state.current_result
                st.markdown(f"**Current:** {result.name}")
                st.markdown(f"log Z = {result.log_evidence:.4f}")
                
                if st.button("➕ Add to Comparison"):
                    st.session_state.results[result.name] = result
                    st.success(f"Added '{result.name}'")
            else:
                st.info("Run an analysis first")
            
            st.divider()
            
            if st.button("🗑️ Clear All"):
                st.session_state.results = {}
                st.info("Cleared all models")
        
        with col2:
            if len(st.session_state.results) > 0:
                st.subheader("Comparison Table")
                
                data = []
                for name, r in st.session_state.results.items():
                    data.append({
                        "Model": name,
                        "log Z": f"{r.log_evidence:.4f}",
                        "Modes": r.n_modes,
                        "Dimension": r.dimension,
                        "Time (s)": f"{r.time_total:.2f}"
                    })
                st.table(data)
                
                # Bayes factors
                if len(st.session_state.results) >= 2:
                    st.subheader("Bayes Factors")
                    
                    names = list(st.session_state.results.keys())
                    for i in range(len(names)):
                        for j in range(i+1, len(names)):
                            log_bf = (st.session_state.results[names[i]].log_evidence - 
                                     st.session_state.results[names[j]].log_evidence)
                            interp = BayesFactor.interpret(log_bf)
                            
                            favored = names[i] if log_bf > 0 else names[j]
                            st.markdown(f"**{names[i]}** vs **{names[j]}**: log BF = {log_bf:.2f} → {interp} for {favored}")
                    
                    # Chart
                    st.divider()
                    fig = create_model_comparison_chart(st.session_state.results)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add models to compare them.")
                
                st.markdown("""
                ### Bayes Factor Interpretation
                
                | |log BF| | Interpretation |
                |---------|----------------|
                | < 1 | Not worth mentioning |
                | 1-2.5 | Substantial |
                | 2.5-5 | Strong |
                | > 5 | Decisive |
                """)
    
    # =========================
    # PAGE: SCALING ANALYSIS
    # =========================
    elif page == "📈 Scaling Analysis":
        st.header("📈 Dimensional Scaling Analysis")
        
        if len(st.session_state.scaling_data) > 0:
            fig = create_scaling_plot(st.session_state.scaling_data)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("Raw Data")
            df = pd.DataFrame(st.session_state.scaling_data)
            st.dataframe(df)
            
            if st.button("🗑️ Clear Scaling Data"):
                st.session_state.scaling_data = []
                st.rerun()
        else:
            st.info("Run analyses at different dimensions to see scaling behavior.")
            
            st.markdown("""
            ### Expected Scaling
            
            - **SunBURST**: O(d^0.5) to O(d^0.7) 
            - **Nested Sampling**: O(d²) to O(d³)
            
            Run at dimensions 4, 8, 16, 32, 64, 128 to see the scaling curve.
            """)
    
    # =========================
    # PAGE: HISTORY
    # =========================
    elif page == "📜 History":
        st.header("📜 Run History")
        
        if len(st.session_state.run_history) > 0:
            df = pd.DataFrame(st.session_state.run_history)
            df = df.iloc[::-1]  # Reverse order (newest first)
            st.dataframe(df, use_container_width=True)
            
            # Export
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "sunburst_history.csv",
                    "text/csv"
                )
            
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.run_history = []
                    st.rerun()
        else:
            st.info("No runs yet. Go to 'Run Analysis' to start.")
    
    # =========================
    # PAGE: ABOUT
    # =========================
    elif page == "ℹ️ About":
        st.header("ℹ️ About SunBURST")
        
        st.markdown("""
        ## ☀️ SunBURST
        
        **S**eeded **U**niverse **N**avigation — **B**ayesian **U**nification via **R**adial **S**hooting **T**echniques
        
        A GPU-accelerated Bayesian evidence calculator achieving:
        
        - ✅ **Machine-precision accuracy** through 1024D
        - ✅ **Sub-linear scaling** O(d^0.67)
        - ✅ **>1000× speedup** vs nested sampling
        
        ### Pipeline Stages
        
        | Module | Chinese | Function |
        |--------|---------|----------|
        | 🐯 CarryTiger | 抱虎歸山 | Mode detection via ray casting |
        | 🐉 GreenDragon | 青龍出水 | Peak refinement with L-BFGS |
        | 🏹 BendTheBow | 彎弓射虎 | Evidence via Laplace approximation |
        
        ### Author
        
        **Ira Wolfson**  
        Department of Electronics and Electrical Engineering  
        Braude College of Engineering, Israel
        
        ### Links
        
        - [GitHub](https://github.com/beastraban/sunburst)
        - [arXiv Paper](https://arxiv.org/abs/...)
        
        ---
        
        *Module names honor Guang Ping Yang Style Tai Chi (廣平楊式太極拳), in honor of Master Donald Rubbo.*
        """)


if __name__ == "__main__":
    main()