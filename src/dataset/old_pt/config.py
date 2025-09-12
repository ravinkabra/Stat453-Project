from ..base.config import BaseDatasetConfig, dataclass, Field
from typing import Literal


@dataclass
class OldPtDatasetConfig(BaseDatasetConfig):
    """
    配置用于 OldPt 数据集的参数（迁移自 legacy datamodule）。
    """

    _target_: Literal["src.dataset.old_pt.dataset.OldPtDataset"] = Field(
        default="src.dataset.old_pt.dataset.OldPtDataset"
    )

    file_path: str = Field(..., description=".pt 文件路径，包含伴奏(acc)数据")
    target_length: int = Field(default=384, description="目标序列长度")
    split_ratio: int = Field(10, description="训练/验证划分基数")
    stage: Literal["train", "val", "test", "all"] = Field(default="train", description="数据阶段")
