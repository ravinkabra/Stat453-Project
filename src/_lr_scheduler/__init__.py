from typing import Union

# Import all LR scheduler parameter classes
from .base import BaseLRSchedulerParams
from .step import StepLRParams
from .multistep import MultiStepLRParams
from .exponential import ExponentialLRParams
from .cosine import CosineAnnealingLRParams
from .reduce_on_plateau import ReduceLROnPlateauParams
from .cyclic import CyclicLRParams
from .onecycle import OneCycleLRParams
from .cosine_warm_restarts import CosineAnnealingWarmRestartsParams

# Union type for all LR scheduler parameters
UnionLRSchedulerParams = Union[
    BaseLRSchedulerParams,
    StepLRParams,
    MultiStepLRParams,
    ExponentialLRParams,
    CosineAnnealingLRParams,
    ReduceLROnPlateauParams,
    CyclicLRParams,
    OneCycleLRParams,
    CosineAnnealingWarmRestartsParams,
    dict,  # Allow dict for backward compatibility
]

__all__ = [
    # Base class
    "BaseLRSchedulerParams",
    # LR scheduler parameter classes
    "StepLRParams",
    "MultiStepLRParams",
    "ExponentialLRParams",
    "CosineAnnealingLRParams",
    "ReduceLROnPlateauParams",
    "CyclicLRParams",
    "OneCycleLRParams",
    "CosineAnnealingWarmRestartsParams",
    # Union type
    "UnionLRSchedulerParams",
]
