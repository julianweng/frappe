"""Optimization algorithms for opponent modeling"""

from .base import OptimizerBase
from .gradients import compute_gradient
from .pgd import ProjectedGradientDescent
from .fw import FrankWolfe
from .sfw import StochasticFrankWolfe

__all__ = [
    "OptimizerBase",
    "compute_gradient",
    "ProjectedGradientDescent",
    "FrankWolfe",
    "StochasticFrankWolfe",
]
