"""
GPU utilities for SunBURST.

Provides automatic GPU detection and array module selection.
"""

import numpy as np
from typing import Dict, Any, Optional

# GPU detection
_GPU_AVAILABLE = False
_cp = None

try:
    import cupy as cp
    _cp = cp
    _GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    pass


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def get_array_module(use_gpu: Optional[bool] = None):
    """
    Get the appropriate array module (numpy or cupy).
    
    Args:
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)
    
    Returns:
        numpy or cupy module
    """
    if use_gpu is None:
        use_gpu = _GPU_AVAILABLE
    
    if use_gpu and _GPU_AVAILABLE:
        return _cp
    return np


def gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.
    
    Returns:
        Dictionary with GPU details, or empty dict if no GPU
    """
    if not _GPU_AVAILABLE:
        return {"available": False}
    
    device = _cp.cuda.Device()
    attrs = device.attributes
    
    # Memory info
    free_mem, total_mem = _cp.cuda.runtime.memGetInfo()
    
    # Compute capability
    major = attrs.get('ComputeCapabilityMajor', 0)
    minor = attrs.get('ComputeCapabilityMinor', 0)
    
    # CUDA version
    cuda_ver = _cp.cuda.runtime.runtimeGetVersion()
    
    # Try to get device name
    try:
        props = _cp.cuda.runtime.getDeviceProperties(0)
        name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    except Exception:
        name = "Unknown GPU"
    
    return {
        "available": True,
        "name": name,
        "memory_total_mb": int(total_mem / (1024**2)),
        "memory_free_mb": int(free_mem / (1024**2)),
        "compute_capability": f"{major}.{minor}",
        "cuda_version": f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}",
        "sm_count": attrs.get('MultiProcessorCount', 0),
    }


def to_cpu(arr):
    """Convert array to CPU (handles both NumPy and CuPy)."""
    if arr is None:
        return None
    if _GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def to_gpu(arr):
    """Convert array to GPU if available."""
    if arr is None:
        return None
    if _GPU_AVAILABLE:
        return _cp.asarray(arr)
    return np.asarray(arr)


def sync_gpu():
    """Synchronize GPU operations (wait for completion)."""
    if _GPU_AVAILABLE:
        _cp.cuda.Stream.null.synchronize()
