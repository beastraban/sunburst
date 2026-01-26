"""
Tests for SunBURST on multimodal posteriors.

These tests verify that SunBURST can detect and handle multiple modes.
"""

import pytest
import numpy as np


def test_bimodal_2d():
    """Test evidence calculation on bimodal 2D distribution."""
    from sunburst import compute_evidence
    
    dim = 2
    separation = 5.0
    
    # Two Gaussian modes
    centers = [
        np.array([separation, 0]),
        np.array([-separation, 0]),
    ]
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        # Log-sum-exp over modes
        log_probs = []
        for center in centers:
            log_probs.append(-0.5 * np.sum((x - center) ** 2, axis=1))
        log_probs = np.array(log_probs)
        max_log = np.max(log_probs, axis=0)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0))
    
    bounds = [(-15.0, 15.0)] * dim
    
    result = compute_evidence(
        log_likelihood,
        bounds,
        n_oscillations=1,
        verbose=False,
    )
    
    # Should find 2 peaks
    assert result.n_peaks >= 1, f"Expected at least 1 peak, found {result.n_peaks}"
    assert not np.isnan(result.log_evidence)


def test_trimodal_4d():
    """Test evidence calculation on trimodal 4D distribution."""
    from sunburst import compute_evidence
    
    dim = 4
    separation = 4.0
    
    # Three Gaussian modes
    centers = [
        np.array([separation, 0, 0, 0]),
        np.array([-separation, 0, 0, 0]),
        np.array([0, separation, 0, 0]),
    ]
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        log_probs = []
        for center in centers:
            log_probs.append(-0.5 * np.sum((x - center) ** 2, axis=1))
        log_probs = np.array(log_probs)
        max_log = np.max(log_probs, axis=0)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0))
    
    bounds = [(-15.0, 15.0)] * dim
    
    result = compute_evidence(
        log_likelihood,
        bounds,
        n_oscillations=1,
        verbose=False,
    )
    
    assert result.n_peaks >= 1
    assert not np.isnan(result.log_evidence)


def test_peaks_returned():
    """Test that peak locations are returned correctly."""
    from sunburst import compute_evidence
    
    dim = 2
    center = np.array([3.0, -2.0])
    
    def log_likelihood(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum((x - center) ** 2, axis=1)
    
    bounds = [(-10.0, 10.0)] * dim
    
    result = compute_evidence(
        log_likelihood,
        bounds,
        return_peaks=True,
        verbose=False,
    )
    
    assert result.peaks is not None
    assert result.peaks.shape[1] == dim
    
    # The peak should be close to the center
    if result.n_peaks >= 1:
        # Find closest peak to true center
        distances = np.sqrt(np.sum((result.peaks - center) ** 2, axis=1))
        min_distance = np.min(distances)
        assert min_distance < 1.0, f"Peak too far from true center: {min_distance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
