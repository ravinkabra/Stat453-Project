from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Dict, Any, List, Literal
from ..._utils.config_base import DictAccessMixin


@dataclass
class BaseDatasetConfig:
    """数据集配置基类"""

    _target_: Literal["src.dataset.base.dataset.BaseDataset"] = Field(
        default="src.dataset.base.dataset.BaseDataset", description="Dataset class"
    )
    # stage: str = Field(default="train", description="Stage (train/val/test/predict)")
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
    persistent_workers: bool = Field(
        default=False, description="Whether to use persistent workers for data loading"
    )
