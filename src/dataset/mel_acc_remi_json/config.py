from ..base.config import BaseDatasetConfig, dataclass, Field
from typing import Literal, Optional, Tuple, Union
from typing import Any


@dataclass
class MelAccRemiJsonDatasetConfig(BaseDatasetConfig):
    """配置用于从 RE:MI JSON 文件读取旋律/伴奏对的 dataset。"""

    _target_: Literal["src.dataset.remi_json.dataset.MelAccRemiJsonDataset"] = Field(
        default="src.dataset.remi_json.dataset.MelAccRemiJsonDataset"
    )

    mel_dir: str = Field(..., description="目录，包含 melody json 文件")
    acc_dir: str = Field(..., description="目录，包含 accompaniment json 文件")
    file_pattern: str = Field("*.json", description="文件匹配模式，用于 glob")

    max_seq_len: int = Field(512, description="最大序列长度（包含 SOS/EOS/填充）")
    tokenization_type: Optional[str] = Field(
        None, description="可选的 tokenization 类型标识"
    )
    transform: Optional[Any] = Field(
        None, description="可选的 transform 函数/可调用对象（在 dataset 中会被调用）"
    )

    data_range: Tuple[Union[int, float], Union[int, float]] = Field(
        default=(0, 1),
        description="数据范围切片，支持两 int 或 两 float（float 表示比例）",
    )

    stage: Literal["train", "val", "test", "predict"] = Field(
        default="train", description="阶段"
    )
