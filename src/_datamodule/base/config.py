from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Union, Literal, Any, Dict
from ...dataset import UnionDatasetConfig


@dataclass
class BaseDataModuleConfig:
    """数据模块配置基类"""

    _target_: Literal["src.datamodule.base.datamodule.BaseDataModule"] = Field(
        default="src.datamodule.base.datamodule.BaseDataModule",
        description="DataModule class",
    )

    train: Optional[UnionDatasetConfig] = Field(
        default_factory=lambda: UnionDatasetConfig(shuffle=True, drop_last=True)
    )
    val: Optional[UnionDatasetConfig] = Field(
        default_factory=lambda: UnionDatasetConfig(shuffle=False, drop_last=False)
    )
    test: Optional[UnionDatasetConfig] = Field(
        default_factory=lambda: UnionDatasetConfig(shuffle=False, drop_last=False)
    )
    predict: Optional[UnionDatasetConfig] = Field(
        default_factory=lambda: UnionDatasetConfig(shuffle=False, drop_last=False)
    )
