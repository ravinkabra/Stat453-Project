from typing import Union

# Import all optimizer parameter classes
from .base import BaseOptimizerParams
from .adam import AdamParams
from .sgd import SGDParams
from .adamw import AdamWParams
from .rmsprop import RMSpropParams
from .adagrad import AdagradParams
from .adadelta import AdadeltaParams
from .adamax import AdamaxParams
from .lbfgs import LBFGSParams
from .rprop import RpropParams

# Union type for all optimizer parameters
UnionOptimizerParams = Union[
    AdamParams,
    SGDParams,
    AdamWParams,
    RMSpropParams,
    AdagradParams,
    AdadeltaParams,
    AdamaxParams,
    LBFGSParams,
    RpropParams,
    dict,  # Allow dict for backward compatibility
]

__all__ = [
    # Base class
    "BaseOptimizerParams",
    # Optimizer parameter classes
    "AdamParams",
    "SGDParams",
    "AdamWParams",
    "RMSpropParams",
    "AdagradParams",
    "AdadeltaParams",
    "AdamaxParams",
    "LBFGSParams",
    "RpropParams",
    # Union type
    "UnionOptimizerParams",
]
