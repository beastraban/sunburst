"""
SunBURST — GPU-accelerated Bayesian Evidence Calculation

Seeded Universe Navigation — Bayesian Unification via Radial Shooting Techniques

A GPU-accelerated Bayesian evidence calculator achieving machine-precision
accuracy through 1024 dimensions with sub-linear scaling.

Quick Start
-----------
>>> import numpy as np
>>> from sunburst import compute_evidence
>>> 
>>> def log_likelihood(x):
...     x = np.atleast_2d(x)
...     return -0.5 * np.sum(x**2, axis=1)
>>> 
>>> bounds = [(-10, 10)] * 64
>>> result = compute_evidence(log_likelihood, bounds)
>>> print(f"log Z = {result.log_evidence:.4f}")

Test Installation
-----------------
>>> import sunburst
>>> result = sunburst.test(dim=64)

Module Names
------------
All module names come from Guang Ping Yang Style Tai Chi (廣平楊式太極拳),
in honor of Master Donald Rubbo:

- CarryTiger (抱虎歸山): Mode detection
- GreenDragon (青龍出水): Peak refinement
- BendTheBow (彎弓射虎): Evidence calculation
- GraspBirdsTail (攬雀尾): Dimensional reduction
- ChiSao (黏手): Sticky hands optimization
- SingleWhip (單鞭): GPU toolkit
"""

__version__ = "1.0.2"
__author__ = "Ira Wolfson"
__email__ = "ira.wolfson@braude.ac.il"

# Main public API
from .pipeline import (
    compute_evidence,
    test,
    SunburstResult,
)

# GPU utilities
from .utils.gpu import (
    gpu_available,
    gpu_info,
    get_array_module,
)

__all__ = [
    # Main API
    "compute_evidence",
    "test", 
    "SunburstResult",
    # GPU utilities
    "gpu_available",
    "gpu_info",
    "get_array_module",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
