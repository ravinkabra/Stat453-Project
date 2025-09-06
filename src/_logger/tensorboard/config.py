from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base.config import BaseLoggerParams


@dataclass
class TensorBoardLoggerParams(BaseLoggerParams):
    """
    🔧 TensorBoard Logger 参数

    所有参数直接传递给 TensorBoardLogger 构造函数
    """

    _target_: str = Field(
        default="pytorch_lightning.loggers.TensorBoardLogger",
        description="TensorBoardLogger 类路径",
    )

    # 基础参数
    save_dir: str = Field("./tb_logs", description="保存目录")
    name: str = Field("lightning_logs", description="实验名称")
    default_hp_metric: bool = Field(True, description="是否记录默认超参数指标")

    # 高级参数
    log_graph: bool = Field(False, description="是否记录计算图")
    sub_dir: Optional[str] = Field(None, description="子目录")
    filename_suffix: str = Field("", description="文件名后缀")
