"""
SunBURST pipeline modules.

Each module is named after a Guang Ping Yang Style Tai Chi form,
in honor of Master Donald Rubbo.
"""

from .carry_tiger import (
    CarryTigerToMountain,
    RayBank,
    CHISAO_VERSION,
    SINGLEWHIP_VERSION,
)

from .green_dragon import (
    GreenDragonRisesFromWater,
    TrajectoryBank,
)

from .bend_bow import (
    BendTheBowShootTheTiger,
)

from .grasp_tail import (
    grasp_birds_tail,
    GraspBirdsTailConfig,
    CompactedProblem,
)

__all__ = [
    # Stage 1: Mode Detection
    "CarryTigerToMountain",
    "RayBank",
    "CHISAO_VERSION",
    "SINGLEWHIP_VERSION",
    # Stage 2: Peak Refinement
    "GreenDragonRisesFromWater",
    "TrajectoryBank",
    # Stage 3: Evidence Calculation
    "BendTheBowShootTheTiger",
    # Stage 4: Dimensional Reduction
    "grasp_birds_tail",
    "GraspBirdsTailConfig",
    "CompactedProblem",
]
