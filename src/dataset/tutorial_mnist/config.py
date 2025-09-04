from ..base.config import BaseDatasetConfig, dataclass, Field
from typing import Optional, List, Union, Literal


@dataclass
class MnistDatasetConfig(BaseDatasetConfig):
    """MNIST 数据集配置"""

    _target_: Literal["src.dataset.tutorial_mnist.dataset.MnistDataset"] = Field(
        default="src.dataset.tutorial_mnist.dataset.MnistDataset",
        description="目标类路径",
    )
    data_dir: str = Field(default="./data/mnist", description="数据存储目录")
    download: bool = Field(default=True, description="是否下载数据集")
    subset_size: Optional[int] = Field(
        default=None, description="子集大小，用于快速实验（None表示使用完整数据集）"
    )

    random_rotation: Optional[float] = Field(
        default=None, description="随机旋转角度（度），None表示不使用"
    )
    random_crop: Optional[List[int]] = Field(
        default=None, description="随机裁剪尺寸 [height, width]，None表示不使用"
    )
    dataset_range: Optional[Union[List[float], List[int]]] = Field(
        default=None,
        description="数据集范围, None表示使用完整数据集, [min, max]表示使用子集",
    )
