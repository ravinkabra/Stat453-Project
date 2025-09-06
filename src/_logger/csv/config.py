from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base.config import BaseLoggerParams


@dataclass
class CSVLoggerParams(BaseLoggerParams):
    """
    🔧 CSV Logger 参数

    所有参数直接传递给 CSVLogger 构造函数
    """

    _target_: str = Field(
        default="pytorch_lightning.loggers.CSVLogger", description="CSVLogger 类路径"
    )

    # 基础参数
    save_dir: str = Field("./csv_logs", description="保存目录")
    name: str = Field("lightning_logs", description="实验名称")
    flush_logs_every_n_steps: int = Field(100, description="每N步刷新日志")
