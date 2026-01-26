"""
Tests for SunBURST on Gaussian posteriors.

These tests verify basic functionality and accuracy on standard Gaussians.
"""

import pytest
import numpy as np


def test_import():
    """Test that sunburst imports successfully."""
    import sunburst
    assert hasattr(sunburst, 'compute_evidence')
    assert hasattr(sunburst, 'test')
    assert hasattr(sunburst, '__version__')


def test_version():
    """Test version string format."""
    from sunburst import __version__
    parts = __version__.split('.')
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_gpu_detection():
    """Test GPU detection utilities."""
    from sunburst import gpu_available, gpu_info
    
    available = gpu_available()
    assert isinstance(available, bool)
    
    info = gpu_info()
    assert isinstance(info, dict)
    assert 'available' in info


def test_2d_gaussian():
    """Test evidence calculation on 2D Gaussian."""
    from sunburst import compute_evidence
    
    dim = 2
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2, axis=1)
    
    bounds = [(-10.0, 10.0)] * dim
    true_log_Z = 0.5 * dim * np.log(2 * np.pi) - dim * np.log(20)
    
    result = compute_evidence(
        log_likelihood,
        bounds,
        n_oscillations=1,
        verbose=False,
    )
    
    assert result.n_peaks >= 1
    assert not np.isnan(result.log_evidence)
    
    error_pct = 100 * abs(result.log_evidence - true_log_Z) / abs(true_log_Z)
    assert error_pct < 10.0, f"Error too large: {error_pct:.2f}%"


def test_8d_gaussian():
    """Test evidence calculation on 8D Gaussian."""
    from sunburst import compute_evidence
    
    dim = 8
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2, axis=1)
    
    bounds = [(-10.0, 10.0)] * dim
    true_log_Z = 0.5 * dim * np.log(2 * np.pi) - dim * np.log(20)
    
    result = compute_evidence(
        log_likelihood,
        bounds,
        n_oscillations=1,
        verbose=False,
    )
    
    assert result.n_peaks >= 1
    assert not np.isnan(result.log_evidence)
    
    error_pct = 100 * abs(result.log_evidence - true_log_Z) / abs(true_log_Z)
    assert error_pct < 5.0, f"Error too large: {error_pct:.2f}%"


def test_result_attributes():
    """Test that SunburstResult has expected attributes."""
    from sunburst import compute_evidence
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2, axis=1)
    
    bounds = [(-10.0, 10.0)] * 4
    
    result = compute_evidence(log_likelihood, bounds, verbose=False)
    
    # Check required attributes
    assert hasattr(result, 'log_evidence')
    assert hasattr(result, 'log_evidence_std')
    assert hasattr(result, 'n_peaks')
    assert hasattr(result, 'peaks')
    assert hasattr(result, 'wall_time')
    assert hasattr(result, 'module_times')
    assert hasattr(result, 'n_likelihood_calls')
    assert hasattr(result, 'config')
    
    # Check types
    assert isinstance(result.log_evidence, float)
    assert isinstance(result.n_peaks, int)
    assert isinstance(result.wall_time, float)
    assert isinstance(result.module_times, dict)
    assert isinstance(result.n_likelihood_calls, int)
    
    # Check peaks shape if present
    if result.peaks is not None:
        assert result.peaks.shape[0] == result.n_peaks
        assert result.peaks.shape[1] == 4


def test_builtin_test():
    """Test the built-in test function."""
    from sunburst import test
    
    result = test(dim=4, verbose=False)
    
    assert result.n_peaks >= 1
    assert not np.isnan(result.log_evidence)
    assert result.wall_time > 0


def test_reproducibility():
    """Test that seed parameter gives reproducible results."""
    from sunburst import compute_evidence
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2, axis=1)
    
    bounds = [(-10.0, 10.0)] * 4
    
    result1 = compute_evidence(log_likelihood, bounds, seed=42, verbose=False)
    result2 = compute_evidence(log_likelihood, bounds, seed=42, verbose=False)
    
    # Should get same result with same seed
    assert abs(result1.log_evidence - result2.log_evidence) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
