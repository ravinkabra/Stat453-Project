from typing import Union
from .base import BaseLRSchedulerParams
from .example import ExampleLRSchedulerParams

UnionLRSchedulerParams = Union[BaseLRSchedulerParams, ExampleLRSchedulerParams, dict]
