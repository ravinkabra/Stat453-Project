from ..base.config import BaseDatasetConfig, dataclass, Field
from typing import Literal


@dataclass
class NewPtDatasetConfig(BaseDatasetConfig):
    """配置用于 NewPt 数据集（来源于 StreamMUSE 的新 datamodule）。"""

    _target_: Literal["src.dataset.new_pt.dataset.NewPtDataset"] = Field(
        default="src.dataset.new_pt.dataset.NewPtDataset"
    )

    file_path: str = Field(..., description=".pt 文件路径，包含伴奏（acc）数据")
    target_length: int = Field(1024, description="目标序列长度")
    split_ratio: int = Field(10, description="训练/验证划分的比例基数")
    stage: Literal["train", "val", "test", "all"] = Field(
        default="train", description="数据阶段"
    )
    sequence_shift: int = Field(0, description="序列合并时的位移长度")

    # DataLoader specific overrides are inherited from BaseDatasetConfig
