from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Union, Literal, Any, Dict


@dataclass
class BaseDatasetConfig:
    """数据集配置基类"""

    batch_size: int = Field(default=32, description="Batch size for the dataset")
    num_workers: int = Field(
        default=4, description="Number of workers for data loading"
    )
    shuffle: bool = Field(default=True, description="Whether to shuffle the dataset")
    pin_memory: bool = Field(
        default=True, description="Whether to pin memory for the dataset"
    )
    drop_last: bool = Field(
        default=False, description="Whether to drop the last incomplete batch"
    )
    sampler: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="Sampler for the dataset"
    )


@dataclass
class BaseDataModuleConfig:
    """数据模块配置基类"""

    train: Optional[BaseDatasetConfig] = Field(
        default_factory=lambda: BaseDatasetConfig(shuffle=True, drop_last=True)
    )
    val: Optional[BaseDatasetConfig] = Field(
        default_factory=lambda: BaseDatasetConfig(shuffle=False, drop_last=False)
    )
    test: Optional[BaseDatasetConfig] = Field(
        default_factory=lambda: BaseDatasetConfig(shuffle=False, drop_last=False)
    )
    predict: Optional[BaseDatasetConfig] = Field(
        default_factory=lambda: BaseDatasetConfig(shuffle=False, drop_last=False)
    )
