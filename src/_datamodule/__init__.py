from .base.config import BaseDataModuleConfig
from .base.datamodule import BaseDataModule

from typing import Union, Any

UnionDataModuleConfig = Union[BaseDataModuleConfig, dict]
UnionDataModule = Union[BaseDataModule, Any]
