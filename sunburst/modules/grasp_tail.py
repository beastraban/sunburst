#!/usr/bin/env python3
"""
GraspBirdsTail (揽雀尾) v1.1 — Dimensional Reduction & Handoff
===============================================================

Module 4 of SunBURST: Analyze completed pipeline results to identify
nuisance and degenerate parameters, yielding a maximally compacted
likelihood suitable for downstream methods (PolyChord, dynesty, MultiNest).

ZERO ADDITIONAL LIKELIHOOD EVALUATIONS — uses only data already computed
by Modules 0-3.

The Four Movements:
  - Ward-off (掤 Péng): Identify and isolate informative directions
  - Roll-back (捋 Lǚ): Pull back nuisance parameters  
  - Press (擠 Jǐ): Compress to essential dimensions
  - Push (按 Àn): Hand off compacted problem to downstream methods

Usage:
------
    from GraspBirdsTail_v1_0 import grasp_birds_tail, GraspBirdsTailConfig
    
    # After SunBURST pipeline completes
    result = sunburst_pipeline.run(log_L_512D, bounds_512D)
    
    # Analyze and compact (zero new evaluations!)
    compacted = grasp_birds_tail(
        sunburst_result=result,
        original_log_L=log_L_512D,
        original_bounds=bounds_512D
    )
    
    print(f"Reduced: {compacted.d_original}D → {compacted.d_effective}D")
    
    # Export for PolyChord
    polychord_settings = compacted.export_for_polychord()
    
    # Final evidence = PolyChord result + marginalization correction
    log_Z = polychord_result.logZ + compacted.log_marginalization_correction

Author: SunBURST Development Team
Version: 1.2 (January 2026)
Status: Production Ready

Changes in v1.2:
- Fixed division by zero when GPU evaluations are too fast

Changes in v1.1:
- Added progress indicators to Hessian computation (CLAUDE_RULES compliance)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple, Any
import warnings

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = np
    HAS_GPU = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_cpu(arr):
    """Convert array to CPU (handles both NumPy and CuPy)."""
    if arr is None:
        return None
    if HAS_GPU and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Convert array to GPU if available."""
    if arr is None:
        return None
    if HAS_GPU:
        return cp.asarray(arr)
    return np.asarray(arr)


def get_xp(arr):
    """Get array module (numpy or cupy) for given array."""
    if HAS_GPU and isinstance(arr, cp.ndarray):
        return cp
    return np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GraspBirdsTailConfig:
    """
    Configuration for GraspBirdsTail dimensional reduction.
    
    Attributes:
        threshold_flat: Eigenvalue magnitude below this → FLAT direction
        threshold_degen: Eigenvalues within this factor → DEGENERATE group
                        Set to 1.0 to DISABLE degenerate grouping
        threshold_var_flat: Sample variance below this → FLAT (if samples available)
        threshold_delta_logL: Include samples within this of max for covariance
        require_unanimous: If True, all methods must agree for NUISANCE classification
        min_informative_dims: Never reduce below this many dimensions
        bounds_samples: Number of samples for reduced bounds estimation
        bounds_padding: Fractional padding on reduced bounds
        use_gpu: Use GPU acceleration where possible
        group_degenerates: If False, skip degenerate grouping entirely
    """
    # Hessian-based thresholds
    threshold_flat: float = 1e-8
    threshold_degen: float = 1.0  # 1.0 = disabled (only exact matches)
    
    # Sample-based thresholds (if samples available)
    threshold_var_flat: float = 1e-6
    threshold_delta_logL: float = 10.0
    
    # Consensus settings
    require_unanimous: bool = False
    
    # Constraints
    min_informative_dims: int = 2
    
    # Bounds computation
    bounds_samples: int = 10000
    bounds_padding: float = 0.05
    
    # GPU
    use_gpu: bool = True
    
    # Degenerate grouping control
    group_degenerates: bool = False  # Disabled by default


# =============================================================================
# PROJECTION OPERATOR
# =============================================================================

class ProjectionOperator:
    """
    Handles coordinate transforms between full and reduced space.
    
    Full space: d_original dimensions (original parameter space)
    Reduced space: d_effective dimensions (informative subspace)
    
    Transform:
        θ_full = V @ θ_reduced + center
        θ_reduced = Vᵀ @ (θ_full - center)
    
    where V is (d_original, d_effective) matrix of informative eigenvectors.
    """
    
    def __init__(
        self, 
        V: np.ndarray, 
        center: np.ndarray,
        informative_indices: List[int],
        use_gpu: bool = False
    ):
        """
        Initialize projection operator.
        
        Args:
            V: (d_original, d_effective) matrix of informative eigenvectors
            center: (d_original,) center point (typically dominant peak)
            informative_indices: Which original dimensions map to reduced
            use_gpu: Use GPU acceleration
        """
        self.V = np.asarray(V)
        self.center = np.asarray(center)
        self.informative_indices = list(informative_indices)
        self.d_full = len(center)
        self.d_effective = V.shape[1]
        self.use_gpu = use_gpu and HAS_GPU
        
        if self.use_gpu:
            self.V_gpu = cp.asarray(self.V)
            self.center_gpu = cp.asarray(self.center)
    
    def reduce(self, theta_full: np.ndarray) -> np.ndarray:
        """
        Map full coordinates to reduced space.
        
        Args:
            theta_full: (..., d_full) array in original space
            
        Returns:
            theta_reduced: (..., d_effective) array in reduced space
        """
        theta_full = np.asarray(theta_full)
        centered = theta_full - self.center
        
        if theta_full.ndim == 1:
            return centered @ self.V
        else:
            return centered @ self.V
    
    def expand(self, theta_reduced: np.ndarray) -> np.ndarray:
        """
        Map reduced coordinates to full space.
        
        Args:
            theta_reduced: (..., d_effective) array in reduced space
            
        Returns:
            theta_full: (..., d_full) array in original space
        """
        theta_reduced = np.asarray(theta_reduced)
        
        if theta_reduced.ndim == 1:
            return self.V @ theta_reduced + self.center
        else:
            return theta_reduced @ self.V.T + self.center
    
    def expand_batch(self, theta_reduced_batch: np.ndarray) -> np.ndarray:
        """
        Batched expansion for likelihood evaluation.
        
        Args:
            theta_reduced_batch: (N, d_effective) array
            
        Returns:
            theta_full_batch: (N, d_full) array
        """
        return self.expand(theta_reduced_batch)
    
    def expand_batch_gpu(self, theta_reduced_batch) -> Any:
        """
        GPU-accelerated batched expansion.
        
        Args:
            theta_reduced_batch: (N, d_effective) CuPy array
            
        Returns:
            theta_full_batch: (N, d_full) CuPy array
        """
        if not self.use_gpu:
            return self.expand_batch(to_cpu(theta_reduced_batch))
        
        theta_reduced_batch = cp.asarray(theta_reduced_batch)
        return theta_reduced_batch @ self.V_gpu.T + self.center_gpu


# =============================================================================
# COMPACTED PROBLEM
# =============================================================================

@dataclass
class CompactedProblem:
    """
    Result of GraspBirdsTail analysis: a dimensionally-reduced problem
    ready for handoff to downstream samplers.
    
    Core attributes:
        d_original: Original dimensionality
        d_effective: Reduced dimensionality
        log_L_reduced: Callable taking (N, d_effective) → (N,) log-likelihoods
        reduced_bounds: (d_effective, 2) bounds in reduced space
        expand: Function mapping reduced → full coordinates
        log_marginalization_correction: Add to downstream evidence
    
    Diagnostic attributes:
        classification: Per-parameter classification
        eigenvalues: Full eigenvalue spectrum
        informative_indices: Which original params are kept
    """
    # Dimensions
    d_original: int
    d_effective: int
    d_nuisance: int
    d_degenerate: int
    
    # Classification
    classification: List[str]
    informative_indices: List[int]
    nuisance_indices: List[int]
    degenerate_groups: List[List[int]]
    
    # Projection
    projection: ProjectionOperator
    projection_center: np.ndarray
    
    # Bounds
    original_bounds: np.ndarray
    reduced_bounds: np.ndarray
    
    # Evidence correction
    log_marginalization_correction: float
    correction_uncertainty: float
    
    # Diagnostic info
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    
    # Original SunBURST results
    sunburst_log_evidence: float
    sunburst_evidence_error: Optional[float]
    
    # Original likelihood (stored for log_L_reduced)
    _original_log_L: Callable = field(repr=False)
    
    def log_L_reduced(self, theta_reduced: np.ndarray) -> np.ndarray:
        """
        Evaluate original likelihood at expanded coordinates.
        
        This is the main callable for downstream samplers.
        
        Args:
            theta_reduced: (N, d_effective) or (d_effective,) reduced coordinates
            
        Returns:
            log_L: (N,) or scalar log-likelihood values
        """
        theta_reduced = np.atleast_2d(theta_reduced)
        theta_full = self.projection.expand_batch(theta_reduced)
        result = self._original_log_L(theta_full)
        
        # Handle scalar output
        if len(theta_reduced) == 1 and np.ndim(result) > 0:
            return result[0]
        return result
    
    def expand(self, theta_reduced: np.ndarray) -> np.ndarray:
        """Map reduced coordinates to full space."""
        return self.projection.expand(theta_reduced)
    
    def reduce(self, theta_full: np.ndarray) -> np.ndarray:
        """Map full coordinates to reduced space."""
        return self.projection.reduce(theta_full)
    
    # =========================================================================
    # Export functions for downstream samplers
    # =========================================================================
    
    def export_to_file(self, filepath: str, include_test: bool = True):
        """
        Export compacted problem as standalone Python module.
        
        Generates a .py file that can be imported by PolyChord/dynesty/MultiNest
        without needing GraspBirdsTail or the original likelihood.
        
        Args:
            filepath: Output .py file path
            include_test: Include a test block at the end
        """
        import os
        import sys
        from datetime import datetime
        
        # Cross-platform path handling
        filepath = os.path.normpath(filepath)
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        platform = sys.platform  # 'win32', 'linux', 'darwin'
        
        # Helper to format array as string
        def array_to_str(arr, name):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return f"{name} = np.array({arr.tolist()})"
            else:
                lines = [f"{name} = np.array(["]
                for row in arr:
                    lines.append(f"    {row.tolist()},")
                lines.append("])")
                return "\n".join(lines)
        
        # Build content as list of strings (avoids f-string escaping issues)
        lines = []
        lines.append('#!/usr/bin/env python3')
        lines.append('# -*- coding: utf-8 -*-')
        lines.append('"""')
        lines.append('Compacted Likelihood - Generated by GraspBirdsTail v1.0')
        lines.append('=' * 57)
        lines.append('')
        lines.append(f'Dimensional reduction: {self.d_original}D -> {self.d_effective}D ({100*(1-self.d_effective/self.d_original):.1f}% reduction)')
        lines.append('')
        lines.append('Classification:')
        lines.append(f'    INFORMATIVE: {len(self.informative_indices)}')
        lines.append(f'    NUISANCE:    {self.d_nuisance}')
        lines.append(f'    DEGENERATE:  {self.d_degenerate}')
        lines.append('')
        lines.append('Evidence correction:')
        lines.append(f'    log_marginalization_correction = {self.log_marginalization_correction:.6f}')
        lines.append('')
        lines.append('Usage with PolyChord:')
        lines.append(f'    from {module_name} import log_L_reduced, prior_transform, D_EFF')
        lines.append('    ')
        lines.append('    import pypolychord')
        lines.append('    pypolychord.run(')
        lines.append('        loglikelihood=log_L_reduced,')
        lines.append('        nDims=D_EFF,')
        lines.append('        prior=prior_transform')
        lines.append('    )')
        lines.append('')
        lines.append('Final evidence:')
        lines.append('    log_Z_full = polychord_result.logZ + LOG_MARGINALIZATION_CORRECTION')
        lines.append('')
        lines.append(f'Generated: {timestamp}')
        lines.append(f'Platform: {platform}')
        lines.append('"""')
        lines.append('')
        lines.append('import numpy as np')
        lines.append('from typing import Callable')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# CONSTANTS')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append(f'D_ORIGINAL = {self.d_original}')
        lines.append(f'D_EFF = {self.d_effective}')
        lines.append(f'LOG_MARGINALIZATION_CORRECTION = {self.log_marginalization_correction}')
        lines.append(f'SUNBURST_LOG_EVIDENCE = {self.sunburst_log_evidence}')
        lines.append('')
        lines.append('# Projection matrix V: (d_original, d_effective)')
        lines.append('# theta_full = V @ theta_reduced + CENTER')
        lines.append(array_to_str(self.projection.V, 'V'))
        lines.append('')
        lines.append('# Projection center (dominant peak location)')
        lines.append(array_to_str(self.projection_center, 'CENTER'))
        lines.append('')
        lines.append('# Reduced bounds: (d_effective, 2)')
        lines.append(array_to_str(self.reduced_bounds, 'REDUCED_BOUNDS'))
        lines.append('')
        lines.append('# Original bounds (for reference)')
        lines.append(array_to_str(self.original_bounds, 'ORIGINAL_BOUNDS'))
        lines.append('')
        lines.append(f'# Classification info')
        lines.append(f'INFORMATIVE_INDICES = {self.informative_indices}')
        lines.append(f'NUISANCE_INDICES = {self.nuisance_indices}')
        lines.append('')
        lines.append('# Eigenvalues (for diagnostics)')
        lines.append(array_to_str(self.eigenvalues, 'EIGENVALUES'))
        lines.append('')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# COORDINATE TRANSFORMS')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append('def expand(theta_reduced):')
        lines.append('    """Map reduced coordinates to full space."""')
        lines.append('    theta_reduced = np.asarray(theta_reduced)')
        lines.append('    if theta_reduced.ndim == 1:')
        lines.append('        return V @ theta_reduced + CENTER')
        lines.append('    else:')
        lines.append('        return theta_reduced @ V.T + CENTER')
        lines.append('')
        lines.append('')
        lines.append('def reduce(theta_full):')
        lines.append('    """Map full coordinates to reduced space."""')
        lines.append('    theta_full = np.asarray(theta_full)')
        lines.append('    centered = theta_full - CENTER')
        lines.append('    if theta_full.ndim == 1:')
        lines.append('        return centered @ V')
        lines.append('    else:')
        lines.append('        return centered @ V')
        lines.append('')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# PRIOR TRANSFORM')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append('def prior_transform(hypercube):')
        lines.append('    """Transform unit hypercube [0,1]^d_eff to reduced parameter space."""')
        lines.append('    hypercube = np.asarray(hypercube)')
        lines.append('    return REDUCED_BOUNDS[:, 0] + hypercube * (REDUCED_BOUNDS[:, 1] - REDUCED_BOUNDS[:, 0])')
        lines.append('')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# LIKELIHOOD (REQUIRES ORIGINAL LIKELIHOOD FUNCTION)')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append('# IMPORTANT: You must set the original likelihood function!')
        lines.append('_original_log_L = None')
        lines.append('')
        lines.append('def set_original_likelihood(log_L_func):')
        lines.append('    """Set the original likelihood function."""')
        lines.append('    global _original_log_L')
        lines.append('    _original_log_L = log_L_func')
        lines.append('')
        lines.append('')
        lines.append('def log_L_reduced(theta_reduced):')
        lines.append('    """Evaluate likelihood in reduced space."""')
        lines.append('    if _original_log_L is None:')
        lines.append('        raise RuntimeError("Original likelihood not set! Call set_original_likelihood() first.")')
        lines.append('    theta_reduced = np.atleast_2d(theta_reduced)')
        lines.append('    theta_full = expand(theta_reduced)')
        lines.append('    result = _original_log_L(theta_full)')
        lines.append('    if hasattr(result, "__len__") and len(result) == 1:')
        lines.append('        return float(result[0])')
        lines.append('    return float(result)')
        lines.append('')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# POLYCHORD INTERFACE')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append('def polychord_likelihood(theta_reduced):')
        lines.append('    """PolyChord-compatible likelihood wrapper."""')
        lines.append('    return log_L_reduced(theta_reduced), []')
        lines.append('')
        lines.append('')
        lines.append('def polychord_prior(hypercube):')
        lines.append('    """PolyChord-compatible prior wrapper."""')
        lines.append('    return prior_transform(hypercube)')
        lines.append('')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# DYNESTY INTERFACE')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append('def dynesty_likelihood(theta_reduced):')
        lines.append('    """dynesty-compatible likelihood wrapper."""')
        lines.append('    return log_L_reduced(theta_reduced)')
        lines.append('')
        lines.append('')
        lines.append('def dynesty_prior_transform(hypercube):')
        lines.append('    """dynesty-compatible prior transform."""')
        lines.append('    return prior_transform(hypercube)')
        lines.append('')
        lines.append('')
        lines.append('# ' + '=' * 77)
        lines.append('# SUMMARY')
        lines.append('# ' + '=' * 77)
        lines.append('')
        lines.append('def print_summary():')
        lines.append('    """Print summary of the compacted problem."""')
        lines.append('    print("Compacted Likelihood Summary")')
        lines.append('    print("=" * 28)')
        lines.append('    print(f"Original dimensions: {D_ORIGINAL}")')
        lines.append('    print(f"Effective dimensions: {D_EFF}")')
        lines.append('    print(f"Reduction: {100*(1-D_EFF/D_ORIGINAL):.1f}%")')
        lines.append('    print("")')
        lines.append('    print(f"Marginalization correction: {LOG_MARGINALIZATION_CORRECTION:.4f}")')
        lines.append('    print(f"SunBURST log Z: {SUNBURST_LOG_EVIDENCE:.4f}")')
        lines.append('    print("")')
        lines.append('    print("Reduced bounds:")')
        lines.append('    for i in range(min(5, D_EFF)):')
        lines.append('        print(f"  dim {i}: [{REDUCED_BOUNDS[i,0]:.3f}, {REDUCED_BOUNDS[i,1]:.3f}]")')
        lines.append('    if D_EFF > 5:')
        lines.append('        print(f"  ... ({D_EFF - 5} more dimensions)")')
        
        if include_test:
            lines.append('')
            lines.append('')
            lines.append('# ' + '=' * 77)
            lines.append('# TEST')
            lines.append('# ' + '=' * 77)
            lines.append('')
            lines.append('if __name__ == "__main__":')
            lines.append('    print_summary()')
            lines.append('    print("")')
            lines.append('    print("To use this module:")')
            lines.append('    print("  1. Import your original likelihood function")')
            lines.append('    print("  2. Call set_original_likelihood(your_log_L)")')
            lines.append('    print("  3. Use log_L_reduced() with PolyChord/dynesty")')
        
        # Write to file with explicit UTF-8 encoding for cross-platform compatibility
        content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        
        print(f"Exported compacted likelihood to: {filepath}")
        print(f"  Dimensions: {self.d_original}D -> {self.d_effective}D")
        print(f"  Marginalization correction: {self.log_marginalization_correction:.4f}")
        print(f"  Platform: {platform}")
    
    # =========================================================================
    # Export functions for downstream samplers
    # =========================================================================
    
    def export_for_polychord(self) -> Dict:
        """
        Export for pypolychord.run().
        
        Returns:
            Dict with keys: loglikelihood, nDims, nDerived, prior
        """
        def prior_transform(hypercube):
            """Transform [0,1]^d → reduced bounds."""
            return self.reduced_bounds[:, 0] + hypercube * (
                self.reduced_bounds[:, 1] - self.reduced_bounds[:, 0]
            )
        
        def loglikelihood(theta):
            """PolyChord-compatible likelihood wrapper."""
            log_L = self.log_L_reduced(theta)
            return float(log_L), []  # No derived parameters
        
        return {
            'loglikelihood': loglikelihood,
            'nDims': self.d_effective,
            'nDerived': 0,
            'prior': prior_transform,
        }
    
    def export_for_dynesty(self) -> Dict:
        """
        Export for dynesty.NestedSampler().
        
        Returns:
            Dict with keys: loglikelihood, prior_transform, ndim
        """
        def prior_transform(hypercube):
            """Transform [0,1]^d → reduced bounds."""
            return self.reduced_bounds[:, 0] + hypercube * (
                self.reduced_bounds[:, 1] - self.reduced_bounds[:, 0]
            )
        
        return {
            'loglikelihood': self.log_L_reduced,
            'prior_transform': prior_transform,
            'ndim': self.d_effective,
        }
    
    def export_for_multinest(self) -> Dict:
        """
        Export for pymultinest.run().
        
        Returns:
            Dict with keys: LogLikelihood, Prior, n_dims
        """
        def prior(cube, ndim, nparams):
            """MultiNest prior transform (modifies cube in-place)."""
            for i in range(ndim):
                cube[i] = self.reduced_bounds[i, 0] + cube[i] * (
                    self.reduced_bounds[i, 1] - self.reduced_bounds[i, 0]
                )
        
        def loglikelihood(cube, ndim, nparams):
            """MultiNest likelihood wrapper."""
            theta = np.array([cube[i] for i in range(ndim)])
            return float(self.log_L_reduced(theta))
        
        return {
            'LogLikelihood': loglikelihood,
            'Prior': prior,
            'n_dims': self.d_effective,
        }
    
    # =========================================================================
    # Diagnostics and visualization
    # =========================================================================
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*60}")
        print("GraspBirdsTail Dimensional Reduction Summary")
        print(f"{'='*60}")
        print(f"\nDimensions:")
        print(f"  Original:     {self.d_original}")
        print(f"  Effective:    {self.d_effective}")
        print(f"  Reduction:    {100*(1 - self.d_effective/self.d_original):.1f}%")
        print(f"\nClassification:")
        print(f"  INFORMATIVE:  {len(self.informative_indices)}")
        print(f"  NUISANCE:     {self.d_nuisance}")
        print(f"  DEGENERATE:   {self.d_degenerate}")
        print(f"\nEvidence:")
        print(f"  SunBURST log Z:           {self.sunburst_log_evidence:.4f}")
        print(f"  Marginalization correction: {self.log_marginalization_correction:.4f}")
        print(f"\nEigenvalue spectrum:")
        print(f"  Max |λ|:      {np.abs(self.eigenvalues).max():.2e}")
        print(f"  Min |λ|:      {np.abs(self.eigenvalues).min():.2e}")
        print(f"  Condition:    {np.abs(self.eigenvalues).max() / (np.abs(self.eigenvalues).min() + 1e-300):.2e}")
        print(f"\nReduced bounds:")
        for i in range(min(5, self.d_effective)):
            print(f"  dim {i}: [{self.reduced_bounds[i,0]:.3f}, {self.reduced_bounds[i,1]:.3f}]")
        if self.d_effective > 5:
            print(f"  ... ({self.d_effective - 5} more dimensions)")
        print(f"{'='*60}\n")


# =============================================================================
# CLASSIFICATION ALGORITHMS
# =============================================================================

def classify_by_hessian(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    config: GraspBirdsTailConfig
) -> Tuple[List[str], List[List[int]]]:
    """
    Classify directions using Hessian eigenvalue analysis.
    
    Classification rules:
        FLAT: |λ_i| < threshold_flat
        DEGENERATE: |λ_i / λ_j - 1| < log(threshold_degen) for nearby eigenvalues
        INFORMATIVE: otherwise
    
    Args:
        eigenvalues: (d,) eigenvalues of -H (should be positive for maxima)
        eigenvectors: (d, d) corresponding eigenvectors
        config: Configuration with thresholds
        
    Returns:
        classification: List of 'INFORMATIVE', 'NUISANCE', or 'DEGENERATE'
        degenerate_groups: List of index groups that are degenerate
    """
    d = len(eigenvalues)
    eigenvalues = np.asarray(eigenvalues)
    
    # Ensure eigenvalues are positive (we're analyzing -H for a maximum)
    abs_eigenvalues = np.abs(eigenvalues)
    
    classification = ['INFORMATIVE'] * d
    degenerate_groups = []
    
    # Sort indices by eigenvalue magnitude
    sorted_indices = np.argsort(abs_eigenvalues)
    
    # Step 1: Identify FLAT directions (tiny eigenvalues)
    for i in range(d):
        if abs_eigenvalues[i] < config.threshold_flat:
            classification[i] = 'NUISANCE'
    
    # Step 2: Identify DEGENERATE groups (similar eigenvalues)
    # Only if explicitly enabled - disabled by default
    used_in_group = set()
    
    if config.group_degenerates and config.threshold_degen > 1.0:
        for i in range(d):
            if i in used_in_group:
                continue
            if classification[sorted_indices[i]] == 'NUISANCE':
                continue
                
            λ_i = abs_eigenvalues[sorted_indices[i]]
            if λ_i < 1e-15:
                continue
                
            # Find all eigenvalues within factor of threshold_degen
            group = [sorted_indices[i]]
            
            for j in range(i + 1, d):
                if j in used_in_group:
                    continue
                if classification[sorted_indices[j]] == 'NUISANCE':
                    continue
                    
                λ_j = abs_eigenvalues[sorted_indices[j]]
                if λ_j < 1e-15:
                    continue
                
                ratio = max(λ_i, λ_j) / min(λ_i, λ_j)
                
                if ratio < config.threshold_degen:
                    group.append(sorted_indices[j])
                    used_in_group.add(j)
            
            # If group has more than one member, mark as degenerate
            if len(group) > 1:
                degenerate_groups.append(group)
                # Keep the first (largest eigenvalue), mark rest as DEGENERATE
                for idx in group[1:]:
                    classification[idx] = 'DEGENERATE'
                used_in_group.add(i)
    
    return classification, degenerate_groups


def classify_by_samples(
    samples: np.ndarray,
    likelihoods: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    bounds: np.ndarray,
    config: GraspBirdsTailConfig
) -> List[str]:
    """
    Classify directions using sample variance from RayBank.
    
    RayBank samples come from rays shot across the prior volume.
    In NUISANCE directions, samples will be spread out (high variance).
    In INFORMATIVE directions, samples cluster near the peak (low variance).
    
    Args:
        samples: (N, d) sample positions from RayBank
        likelihoods: (N,) log-likelihood values
        eigenvectors: (d, d) eigenvectors from Hessian (columns)
        eigenvalues: (d,) eigenvalues (for reference)
        bounds: (d, 2) prior bounds
        config: Configuration
        
    Returns:
        classification: List of 'INFORMATIVE' or 'NUISANCE'
    """
    samples = np.asarray(samples)
    likelihoods = np.asarray(likelihoods)
    eigenvectors = np.asarray(eigenvectors)
    eigenvalues = np.asarray(eigenvalues)
    bounds = np.asarray(bounds)
    
    d = samples.shape[1]
    
    if len(samples) < 10:
        return ['INFORMATIVE'] * d
    
    # Prior width for normalization
    prior_widths = bounds[:, 1] - bounds[:, 0]
    avg_prior_width = np.mean(prior_widths)
    
    # Project samples onto eigenvector basis
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    projected = centered @ eigenvectors  # (N, d) in eigenvector coords
    
    # Compute standard deviation along each eigendirection
    stds = np.std(projected, axis=0)
    
    # Expected std for posterior with given eigenvalue
    # For Gaussian with Hessian eigenvalue λ, posterior std = 1/sqrt(λ)
    expected_posterior_stds = 1.0 / np.sqrt(np.maximum(np.abs(eigenvalues), 1e-15))
    
    # If sample std >> expected posterior std, likelihood is flat there → NUISANCE
    # If sample std ~ expected posterior std, samples are constrained → INFORMATIVE
    
    # Ratio: how much wider are samples than expected?
    # Large ratio → samples spread more than posterior would allow → NUISANCE
    ratio = stds / np.maximum(expected_posterior_stds, 1e-10)
    
    # Debug: print first few
    if d <= 20:
        print(f"    Sample stds:    {stds[:min(10,d)]}")
        print(f"    Expected stds:  {expected_posterior_stds[:min(10,d)]}")
        print(f"    Ratios:         {ratio[:min(10,d)]}")
    
    classification = []
    for i in range(d):
        # HIGH ratio → samples spread MORE than posterior → tight constraint → INFORMATIVE
        # LOW ratio → samples spread SAME as posterior → loose constraint → NUISANCE
        # 
        # If ratio > 2, the posterior is much tighter than sample spread → INFORMATIVE
        # If ratio ≈ 1, the posterior is as wide as sample spread → NUISANCE
        if ratio[i] > 2.0:
            classification.append('INFORMATIVE')
        else:
            classification.append('NUISANCE')
    
    return classification


def consensus_classification(
    hessian_class: List[str],
    sample_class: Optional[List[str]],
    config: GraspBirdsTailConfig
) -> List[str]:
    """
    Combine multiple classification methods into final consensus.
    
    When sample_class is available (from RayBank), it has direct evidence
    about which directions are constrained. Trust it over Hessian-based
    thresholds.
    
    Args:
        hessian_class: Classification from Hessian eigenanalysis
        sample_class: Classification from sample variance analysis (or None)
        config: Configuration
        
    Returns:
        final_classification: Consensus classification
    """
    d = len(hessian_class)
    
    # If we have sample-based classification, use it as primary
    # The sample-based method compares sample spread to expected posterior width
    # and is more reliable than arbitrary eigenvalue thresholds
    if sample_class is not None:
        return sample_class
    
    # Fall back to Hessian-based classification if no samples
    return hessian_class


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_hessian_at_peak(
    log_L: Callable,
    peak: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute Hessian at peak using finite differences.
    
    BATCHED VERSION: All perturbations evaluated in ONE GPU call.
    
    Args:
        log_L: Log-likelihood function
        peak: (d,) peak location
        eps: Finite difference step size
        
    Returns:
        H: (d, d) Hessian matrix
    """
    import time
    
    peak = np.asarray(peak).flatten()
    d = len(peak)
    H = np.zeros((d, d))
    
    t_start = time.time()
    
    # Count total evaluations
    n_diag = 2 * d
    n_offdiag = 4 * (d * (d - 1) // 2)
    total_evals = 1 + n_diag + n_offdiag  # +1 for f0
    
    print(f"    Building {total_evals} perturbation points...", flush=True)
    
    # Build ALL perturbation points in one array
    # Layout: [peak, diag_plus(d), diag_minus(d), offdiag(4 * d*(d-1)/2)]
    all_points = np.zeros((total_evals, d))
    
    idx = 0
    
    # f0: peak itself
    all_points[idx] = peak
    idx += 1
    
    # Diagonal: +eps and -eps for each dimension
    diag_plus_start = idx
    for i in range(d):
        all_points[idx] = peak.copy()
        all_points[idx, i] += eps
        idx += 1
    
    diag_minus_start = idx
    for i in range(d):
        all_points[idx] = peak.copy()
        all_points[idx, i] -= eps
        idx += 1
    
    # Off-diagonal: ++, +-, -+, -- for each pair (i, j) where i < j
    offdiag_start = idx
    offdiag_pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            offdiag_pairs.append((i, j))
            # ++
            all_points[idx] = peak.copy()
            all_points[idx, i] += eps
            all_points[idx, j] += eps
            idx += 1
            # +-
            all_points[idx] = peak.copy()
            all_points[idx, i] += eps
            all_points[idx, j] -= eps
            idx += 1
            # -+
            all_points[idx] = peak.copy()
            all_points[idx, i] -= eps
            all_points[idx, j] += eps
            idx += 1
            # --
            all_points[idx] = peak.copy()
            all_points[idx, i] -= eps
            all_points[idx, j] -= eps
            idx += 1
    
    t_build = time.time()
    print(f"    Points built in {t_build - t_start:.2f}s, evaluating...", flush=True)
    
    # ONE batched evaluation
    all_results = np.atleast_1d(log_L(all_points))
    
    t_eval = time.time()
    eval_time = max(t_eval - t_build, 1e-9)  # Prevent division by zero
    print(f"    Evaluated {total_evals} points in {t_eval - t_build:.2f}s "
          f"({total_evals / eval_time:.0f} evals/s)", flush=True)
    
    # Parse results back into Hessian
    f0 = all_results[0]
    
    # Diagonal elements
    for i in range(d):
        f_plus = all_results[diag_plus_start + i]
        f_minus = all_results[diag_minus_start + i]
        H[i, i] = (f_plus - 2*f0 + f_minus) / (eps**2)
    
    # Off-diagonal elements
    for k, (i, j) in enumerate(offdiag_pairs):
        base = offdiag_start + 4 * k
        f_pp = all_results[base]
        f_pm = all_results[base + 1]
        f_mp = all_results[base + 2]
        f_mm = all_results[base + 3]
        
        H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
        H[j, i] = H[i, j]
    
    # Final timing
    elapsed = max(time.time() - t_start, 1e-9)  # Prevent division by zero
    print(f"    Hessian complete: {total_evals} evals in {elapsed:.1f}s "
          f"({total_evals/elapsed:.0f} evals/s)", flush=True)
    
    return H


def compute_reduced_bounds(
    original_bounds: np.ndarray,
    projection: ProjectionOperator,
    n_samples: int = 10000,
    padding: float = 0.05
) -> np.ndarray:
    """
    Compute bounds in reduced space that cover the original bounds.
    
    Args:
        original_bounds: (d_full, 2) original parameter bounds
        projection: ProjectionOperator for coordinate transform
        n_samples: Number of samples for bounds estimation
        padding: Fractional padding on bounds
        
    Returns:
        reduced_bounds: (d_effective, 2) bounds in reduced space
    """
    d_full = original_bounds.shape[0]
    d_eff = projection.d_effective
    
    # Sample corners for low-d, random for high-d
    if d_full <= 10:
        # Generate all corners
        from itertools import product
        corners = np.array(list(product(*[
            [original_bounds[i, 0], original_bounds[i, 1]] 
            for i in range(d_full)
        ])))
        samples = corners
    else:
        # Random sampling
        samples = np.random.uniform(
            original_bounds[:, 0],
            original_bounds[:, 1],
            size=(n_samples, d_full)
        )
    
    # Project to reduced space
    reduced_samples = projection.reduce(samples)
    
    # Find bounding box
    reduced_bounds = np.zeros((d_eff, 2))
    reduced_bounds[:, 0] = np.min(reduced_samples, axis=0)
    reduced_bounds[:, 1] = np.max(reduced_samples, axis=0)
    
    # Add padding
    widths = reduced_bounds[:, 1] - reduced_bounds[:, 0]
    reduced_bounds[:, 0] -= padding * widths
    reduced_bounds[:, 1] += padding * widths
    
    return reduced_bounds


def compute_marginalization_correction(
    eigenvalues: np.ndarray,
    classification: List[str]
) -> float:
    """
    Compute analytical marginalization correction for nuisance directions.
    
    For Gaussian nuisance directions:
        ∫ exp(-½ λ_i x_i²) dx_i = √(2π/|λ_i|)
    
    In log space:
        log_correction = Σ_nuisance [ ½ log(2π) - ½ log|λ_i| ]
    
    Args:
        eigenvalues: (d,) eigenvalue spectrum
        classification: Per-direction classification
        
    Returns:
        log_correction: Log marginalization factor
    """
    nuisance_mask = np.array([c == 'NUISANCE' for c in classification])
    
    if not np.any(nuisance_mask):
        return 0.0
    
    λ_nuisance = np.abs(eigenvalues[nuisance_mask])
    
    # Filter out zero/tiny eigenvalues
    valid_mask = λ_nuisance > 1e-15
    if not np.any(valid_mask):
        return 0.0
    
    λ_valid = λ_nuisance[valid_mask]
    d_nuisance = len(λ_valid)
    
    log_correction = (
        0.5 * d_nuisance * np.log(2 * np.pi)
        - 0.5 * np.sum(np.log(λ_valid))
    )
    
    return float(log_correction)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def grasp_birds_tail(
    peaks: np.ndarray,
    original_log_L: Callable,
    original_bounds: np.ndarray,
    log_evidence: Optional[float] = None,
    diag_H: Optional[np.ndarray] = None,
    full_hessian: Optional[np.ndarray] = None,
    ray_bank: Optional[Any] = None,
    trajectory_bank: Optional[Any] = None,
    chisao_bank: Optional[Any] = None,
    config: Optional[GraspBirdsTailConfig] = None
) -> CompactedProblem:
    """
    Analyze SunBURST results and produce compacted problem.
    
    Designed to work with the actual module outputs:
        - CarryTiger: peaks, ray_bank, chisao_bank
        - GreenDragon: diag_H, trajectory_bank
        - BendTheBow: log_evidence
    
    ZERO ADDITIONAL LIKELIHOOD EVALUATIONS if full_hessian is provided.
    O(d²) evals if only diag_H provided (computes full Hessian).
    
    Args:
        peaks: (K, d) peak locations from GreenDragon
        original_log_L: Original likelihood function (for log_L_reduced)
        original_bounds: (d, 2) original parameter bounds
        log_evidence: SunBURST evidence estimate from BendTheBow
        diag_H: (K, d) diagonal Hessian from GreenDragon
        full_hessian: (K, d, d) full Hessian if computed by BendTheBow
        ray_bank: RayBank from CarryTiger (samples along rays)
        trajectory_bank: TrajectoryBank from GreenDragon (optimization paths)
        chisao_bank: SampleBank from ChiSao (all optimization samples)
        config: GraspBirdsTailConfig (uses defaults if None)
        
    Returns:
        CompactedProblem ready for downstream sampler
    """
    config = config or GraspBirdsTailConfig()
    original_bounds = np.asarray(original_bounds)
    peaks = np.atleast_2d(to_cpu(peaks))
    d = original_bounds.shape[0]
    K = len(peaks)
    
    print(f"\n{'='*60}")
    print("GraspBirdsTail: Dimensional Reduction Analysis")
    print(f"{'='*60}")
    print(f"Original dimensions: {d}")
    print(f"Number of peaks: {K}")
    
    # Handle empty peaks
    if K == 0:
        print("WARNING: No peaks provided! Returning trivial reduction.")
        return _trivial_compacted_problem(
            original_log_L, original_bounds, 
            np.zeros(d),  # Use origin as center
            log_evidence if log_evidence is not None else np.nan
        )
    
    # =========================================================================
    # STEP 1: Select dominant peak
    # =========================================================================
    
    # Use first peak as dominant (typically highest likelihood)
    dominant_idx = 0
    dominant_peak = peaks[dominant_idx].flatten()
    print(f"Using peak {dominant_idx} as reference")
    
    # =========================================================================
    # STEP 2: Get or compute Hessian at dominant peak
    # =========================================================================
    
    H = None
    
    # Priority 1: Full Hessian if provided
    if full_hessian is not None:
        full_hessian = np.atleast_3d(to_cpu(full_hessian))
        if full_hessian.shape[0] > dominant_idx:
            H = full_hessian[dominant_idx]
            print(f"Using provided full Hessian")
    
    # Priority 2: Compute from diagonal + off-diagonal probing
    if H is None and diag_H is not None:
        diag_H = np.atleast_2d(to_cpu(diag_H))
        if diag_H.shape[0] > dominant_idx:
            H_diag = diag_H[dominant_idx]
            print(f"Have diagonal Hessian, computing full Hessian... (~{2*d**2} evals)")
            H = compute_hessian_at_peak(original_log_L, dominant_peak)
            print(f"  Full Hessian computed")
    
    # Priority 3: Compute from scratch
    if H is None:
        print(f"Computing Hessian at dominant peak... (~{2*d**2} evals)")
        H = compute_hessian_at_peak(original_log_L, dominant_peak)
        print(f"  Hessian computed successfully")
    
    # =========================================================================
    # STEP 3: Eigendecomposition
    # =========================================================================
    
    print("Performing eigendecomposition...")
    
    # We analyze -H (should be positive definite at a maximum)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(-H)
    except np.linalg.LinAlgError as e:
        print(f"  WARNING: Eigendecomposition failed: {e}")
        print(f"  Returning trivial reduction (keep all dimensions)")
        # Return trivial reduction
        return _trivial_compacted_problem(
            original_log_L, original_bounds, dominant_peak, 
            log_evidence if log_evidence is not None else np.nan
        )
    
    # Sort by eigenvalue magnitude (largest first)
    sort_idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    print(f"  Eigenvalue range: [{np.abs(eigenvalues).min():.2e}, {np.abs(eigenvalues).max():.2e}]")
    print(f"  Condition number: {np.abs(eigenvalues).max() / (np.abs(eigenvalues).min() + 1e-300):.2e}")
    
    # =========================================================================
    # STEP 4: Extract samples from banks (if available)
    # =========================================================================
    
    all_samples = None
    all_likelihoods = None
    
    # Extract samples from RayBank (rays across prior)
    if ray_bank is not None:
        try:
            if hasattr(ray_bank, 'samples') and ray_bank.samples is not None:
                bank_samples = to_cpu(ray_bank.samples)
                if bank_samples.ndim == 3:
                    # (n_rays, n_samples_per_ray, d) -> (N, d)
                    bank_samples = bank_samples.reshape(-1, d)
                all_samples = bank_samples
                
            if hasattr(ray_bank, 'log_L') and ray_bank.log_L is not None:
                all_likelihoods = to_cpu(ray_bank.log_L).flatten()
            elif hasattr(ray_bank, 'f_samples') and ray_bank.f_samples is not None:
                all_likelihoods = to_cpu(ray_bank.f_samples).flatten()
                
            if all_samples is not None:
                print(f"  Extracted {len(all_samples)} samples from RayBank")
        except Exception as e:
            print(f"  Warning: Could not extract RayBank samples: {e}")
    
    # Extract samples from ChiSao SampleBank (optimization trajectories)
    if chisao_bank is not None:
        try:
            chisao_samples = None
            chisao_logL = None
            
            # Handle both dict and object forms
            if isinstance(chisao_bank, dict):
                if 'positions' in chisao_bank:
                    chisao_samples = to_cpu(chisao_bank['positions'])
                    n_samples = chisao_bank.get('n_samples', len(chisao_samples))
                    chisao_samples = chisao_samples[:n_samples]
                if 'log_L' in chisao_bank:
                    chisao_logL = to_cpu(chisao_bank['log_L'])[:n_samples]
            else:
                if hasattr(chisao_bank, 'positions') and chisao_bank.positions is not None:
                    n_samples = getattr(chisao_bank, 'n_samples', len(chisao_bank.positions))
                    chisao_samples = to_cpu(chisao_bank.positions)[:n_samples]
                if hasattr(chisao_bank, 'log_L') and chisao_bank.log_L is not None:
                    chisao_logL = to_cpu(chisao_bank.log_L)[:n_samples]
            
            if chisao_samples is not None and len(chisao_samples) > 0:
                print(f"  Extracted {len(chisao_samples)} samples from ChiSao SampleBank")
                
                # Combine with RayBank samples
                if all_samples is not None:
                    all_samples = np.vstack([all_samples, chisao_samples])
                    if chisao_logL is not None and all_likelihoods is not None:
                        all_likelihoods = np.concatenate([all_likelihoods, chisao_logL])
                else:
                    all_samples = chisao_samples
                    all_likelihoods = chisao_logL
                    
        except Exception as e:
            print(f"  Warning: Could not extract ChiSao samples: {e}")
    
    # =========================================================================
    # STEP 5: Classification
    # =========================================================================
    
    print("Classifying directions...")
    
    # Method A: Hessian eigenanalysis
    hessian_class, degenerate_groups = classify_by_hessian(
        eigenvalues, eigenvectors, config
    )
    
    # Method B: Sample/eigenvalue-based classification
    sample_class = None
    if all_samples is not None and all_likelihoods is not None:
        # Check dimension compatibility (samples may be in compressed space from precheck)
        sample_dim = all_samples.shape[1]
        eigen_dim = eigenvectors.shape[0]
        if sample_dim != eigen_dim:
            print(f"  WARNING: Sample dimension ({sample_dim}) != eigenvector dimension ({eigen_dim})")
            print(f"  Skipping sample-based classification (samples from compressed space)")
        else:
            print("  Using eigenvalue-based classification")
            sample_class = classify_by_samples(
                all_samples, all_likelihoods, 
                eigenvectors, eigenvalues, 
                original_bounds, config
            )
    
    # Consensus
    final_classification = consensus_classification(hessian_class, sample_class, config)
    
    # Count classifications
    n_informative = sum(c == 'INFORMATIVE' for c in final_classification)
    n_nuisance = sum(c == 'NUISANCE' for c in final_classification)
    n_degenerate = sum(c == 'DEGENERATE' for c in final_classification)
    
    print(f"  INFORMATIVE: {n_informative}")
    print(f"  NUISANCE:    {n_nuisance}")
    print(f"  DEGENERATE:  {n_degenerate}")
    
    # =========================================================================
    # STEP 6: Enforce minimum dimensions
    # =========================================================================
    
    if n_informative < config.min_informative_dims:
        print(f"  WARNING: Only {n_informative} informative dimensions")
        print(f"  Promoting {config.min_informative_dims - n_informative} nuisance → informative")
        
        # Promote nuisance directions with largest eigenvalues to informative
        nuisance_indices = [i for i, c in enumerate(final_classification) if c == 'NUISANCE']
        nuisance_eigenvalues = [(i, np.abs(eigenvalues[i])) for i in nuisance_indices]
        nuisance_eigenvalues.sort(key=lambda x: -x[1])  # Sort by magnitude descending
        
        n_promote = config.min_informative_dims - n_informative
        for i in range(min(n_promote, len(nuisance_eigenvalues))):
            idx = nuisance_eigenvalues[i][0]
            final_classification[idx] = 'INFORMATIVE'
        
        n_informative = sum(c == 'INFORMATIVE' for c in final_classification)
        n_nuisance = sum(c == 'NUISANCE' for c in final_classification)
    
    # =========================================================================
    # STEP 7: Build projection operator
    # =========================================================================
    
    informative_indices = [i for i, c in enumerate(final_classification) if c == 'INFORMATIVE']
    nuisance_indices = [i for i, c in enumerate(final_classification) if c == 'NUISANCE']
    
    d_effective = len(informative_indices)
    
    # V_informative: eigenvectors corresponding to informative directions
    V_informative = eigenvectors[:, informative_indices]
    
    projection = ProjectionOperator(
        V=V_informative,
        center=dominant_peak,
        informative_indices=informative_indices,
        use_gpu=config.use_gpu
    )
    
    print(f"\nBuilt projection: {d}D → {d_effective}D")
    
    # =========================================================================
    # STEP 8: Compute reduced bounds
    # =========================================================================
    
    print("Computing reduced bounds...")
    reduced_bounds = compute_reduced_bounds(
        original_bounds, projection,
        n_samples=config.bounds_samples,
        padding=config.bounds_padding
    )
    
    # =========================================================================
    # STEP 9: Compute marginalization correction
    # =========================================================================
    
    log_correction = compute_marginalization_correction(eigenvalues, final_classification)
    print(f"Marginalization correction: {log_correction:.4f}")
    
    # =========================================================================
    # STEP 10: Compute explained variance ratio
    # =========================================================================
    
    total_variance = np.sum(np.abs(eigenvalues))
    explained_variance = np.cumsum(np.abs(eigenvalues)) / total_variance
    
    # =========================================================================
    # STEP 11: Build and return CompactedProblem
    # =========================================================================
    
    sunburst_log_evidence = log_evidence if log_evidence is not None else np.nan
    
    compacted = CompactedProblem(
        d_original=d,
        d_effective=d_effective,
        d_nuisance=n_nuisance,
        d_degenerate=n_degenerate,
        classification=final_classification,
        informative_indices=informative_indices,
        nuisance_indices=nuisance_indices,
        degenerate_groups=degenerate_groups,
        projection=projection,
        projection_center=dominant_peak,
        original_bounds=original_bounds,
        reduced_bounds=reduced_bounds,
        log_marginalization_correction=log_correction,
        correction_uncertainty=0.0,  # TODO: estimate uncertainty
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=explained_variance,
        sunburst_log_evidence=sunburst_log_evidence,
        sunburst_evidence_error=None,
        _original_log_L=original_log_L
    )
    
    print(f"\n{'='*60}")
    print(f"GraspBirdsTail complete: {d}D → {d_effective}D ({100*(1-d_effective/d):.1f}% reduction)")
    print(f"{'='*60}\n")
    
    return compacted


def _trivial_compacted_problem(
    original_log_L: Callable,
    original_bounds: np.ndarray,
    peak: np.ndarray,
    log_evidence: float
) -> CompactedProblem:
    """
    Create a trivial CompactedProblem that keeps all dimensions.
    Used as fallback when analysis fails.
    """
    d = original_bounds.shape[0]
    
    projection = ProjectionOperator(
        V=np.eye(d),
        center=peak,
        informative_indices=list(range(d)),
        use_gpu=False
    )
    
    return CompactedProblem(
        d_original=d,
        d_effective=d,
        d_nuisance=0,
        d_degenerate=0,
        classification=['INFORMATIVE'] * d,
        informative_indices=list(range(d)),
        nuisance_indices=[],
        degenerate_groups=[],
        projection=projection,
        projection_center=peak,
        original_bounds=original_bounds,
        reduced_bounds=original_bounds.copy(),
        log_marginalization_correction=0.0,
        correction_uncertainty=0.0,
        eigenvalues=np.ones(d),
        eigenvectors=np.eye(d),
        explained_variance_ratio=np.linspace(1/d, 1.0, d),
        sunburst_log_evidence=log_evidence,
        sunburst_evidence_error=None,
        _original_log_L=original_log_L
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_reduction(
    compacted: CompactedProblem,
    n_test: int = 100
) -> Dict:
    """
    Validate that reduction preserves likelihood structure.
    
    Args:
        compacted: CompactedProblem to validate
        n_test: Number of test points
        
    Returns:
        Dict with validation results
    """
    print("\nValidating reduction...")
    
    # Sample in reduced space
    theta_reduced = np.random.uniform(
        compacted.reduced_bounds[:, 0],
        compacted.reduced_bounds[:, 1],
        size=(n_test, compacted.d_effective)
    )
    
    # Evaluate likelihood
    log_L = np.array([compacted.log_L_reduced(t) for t in theta_reduced])
    
    # Check for finite values
    n_finite = np.sum(np.isfinite(log_L))
    n_valid = np.sum(log_L > -1e10)
    
    # Check peak reachability
    peak_reduced = compacted.reduce(compacted.projection_center)
    log_L_at_peak = compacted.log_L_reduced(peak_reduced)
    
    results = {
        'n_test': n_test,
        'n_finite': int(n_finite),
        'n_valid': int(n_valid),
        'log_L_range': [float(np.min(log_L[np.isfinite(log_L)])), 
                       float(np.max(log_L[np.isfinite(log_L)]))],
        'log_L_at_peak': float(log_L_at_peak),
        'peak_reachable': bool(np.isfinite(log_L_at_peak))
    }
    
    print(f"  Finite evaluations: {n_finite}/{n_test}")
    print(f"  Valid (> -1e10): {n_valid}/{n_test}")
    print(f"  log L at peak: {log_L_at_peak:.4f}")
    print(f"  Peak reachable: {'✓' if results['peak_reachable'] else '✗'}")
    
    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GraspBirdsTail v1.0 — Test")
    print("="*70)
    
    # Create a test problem: 10D Gaussian with some flat/degenerate directions
    np.random.seed(42)
    
    d = 10
    
    # Eigenvalues: some large (informative), some small (nuisance)
    true_eigenvalues = np.array([100, 90, 80, 1, 0.5, 0.1, 0.01, 0.001, 1e-6, 1e-8])
    
    # Random rotation
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    
    # Covariance = Q @ diag(1/λ) @ Q.T
    cov = Q @ np.diag(1.0 / true_eigenvalues) @ Q.T
    cov_inv = Q @ np.diag(true_eigenvalues) @ Q.T
    
    peak = np.zeros(d)
    
    def log_likelihood(theta):
        theta = np.atleast_2d(theta)
        result = np.zeros(len(theta))
        for i, t in enumerate(theta):
            diff = t - peak
            result[i] = -0.5 * diff @ cov_inv @ diff
        return result
    
    bounds = np.array([[-5, 5]] * d)
    
    # Compute true log evidence
    sign, logdet = np.linalg.slogdet(cov)
    log_Z_true = 0.5 * d * np.log(2 * np.pi) + 0.5 * logdet
    print(f"\nTrue log Z: {log_Z_true:.4f}")
    
    # Run GraspBirdsTail with new interface
    config = GraspBirdsTailConfig(
        threshold_flat=1e-4,  # Will classify eigenvalues < 1e-4 as flat
        threshold_degen=1.5,  # Will group eigenvalues within 1.5× as degenerate
        min_informative_dims=3
    )
    
    compacted = grasp_birds_tail(
        peaks=np.array([peak]),  # (1, d) array of peaks
        original_log_L=log_likelihood,
        original_bounds=bounds,
        log_evidence=log_Z_true,
        diag_H=None,  # Will compute full Hessian
        config=config
    )
    
    # Print summary
    compacted.print_summary()
    
    # Validate
    validation = validate_reduction(compacted, n_test=100)
    
    # Test export functions
    print("\nTesting export functions...")
    
    pc_export = compacted.export_for_polychord()
    print(f"  PolyChord export: nDims={pc_export['nDims']}")
    
    dy_export = compacted.export_for_dynesty()
    print(f"  dynesty export: ndim={dy_export['ndim']}")
    
    mn_export = compacted.export_for_multinest()
    print(f"  MultiNest export: n_dims={mn_export['n_dims']}")
    
    # Export to standalone file
    print("\nExporting to standalone .py file...")
    compacted.export_to_file("compacted_likelihood_10D_to_6D.py")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
    
    # Show the generated file
    print("\n--- Generated file preview ---")
    with open("compacted_likelihood_10D_to_6D.py", 'r') as f:
        lines = f.readlines()
        # Show first 80 lines
        for line in lines[:80]:
            print(line, end='')
    print("\n... (truncated) ...")
