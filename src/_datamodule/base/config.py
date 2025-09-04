from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Union, Literal, Any, Dict
from ...dataset import UnionDatasetConfig


@dataclass
class BaseDataModuleConfig:
    """数据模块配置基类"""

    _target_: Literal["src._datamodule.base.datamodule.BaseDataModule"] = Field(
        default="src._datamodule.base.datamodule.BaseDataModule",
        description="DataModule class",
    )

    train: Optional[UnionDatasetConfig] = Field(default=None)
    val: Optional[UnionDatasetConfig] = Field(default=None)
    test: Optional[UnionDatasetConfig] = Field(default=None)
    predict: Optional[UnionDatasetConfig] = Field(default=None)
