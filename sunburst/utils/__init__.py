"""
SunBURST utilities module.

Contains GPU utilities, optimization engines, and sample banking.
"""

from .gpu import (
    gpu_available,
    gpu_info,
    get_array_module,
    to_cpu,
    to_gpu,
    sync_gpu,
)

from .chisao import (
    SampleBank,
    sticky_hands,
    lbfgs_batch,
    deduplicate_peaks_L_infinity,
    SINGLEWHIP_AVAILABLE,
    SINGLEWHIP_VERSION,
)

from .single_whip import (
    SingleWhip,
    randcoord_line_search_batch,
)

from .assess_noise import assess_noise

__all__ = [
    # GPU utilities
    "gpu_available",
    "gpu_info", 
    "get_array_module",
    "to_cpu",
    "to_gpu",
    "sync_gpu",
    # ChiSao optimization
    "SampleBank",
    "sticky_hands",
    "lbfgs_batch",
    "deduplicate_peaks_L_infinity",
    "SINGLEWHIP_AVAILABLE",
    "SINGLEWHIP_VERSION",
    # SingleWhip toolkit
    "SingleWhip",
    "randcoord_line_search_batch",
    "assess_noise",
]
