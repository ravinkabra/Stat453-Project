from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional
from ..base.config import BaseLoggerParams


@dataclass
class CSVLoggerParams(BaseLoggerParams):
    """
    🔧 CSV Logger 参数

    所有参数直接传递给 CSVLogger 构造函数
    """

    _target_: str = Field(default="pytorch_lightning.loggers.CSVLogger", description="CSVLogger 类路径")

    # 基础参数
    save_dir: Optional[str] = Field(None, description="保存目录")
    name: Optional[str] = Field(None, description="实验名称")
    flush_logs_every_n_steps: int = Field(100, description="每N步刷新日志")
