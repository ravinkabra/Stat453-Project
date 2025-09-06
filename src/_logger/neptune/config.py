from typing import Optional, Any, List
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base.config import BaseLoggerParams


@dataclass
class NeptuneLoggerParams(BaseLoggerParams):
    """
    🔧 Neptune Logger 参数

    所有参数直接传递给 NeptuneLogger 构造函数
    """

    _target_: str = Field(
        default="pytorch_lightning.loggers.NeptuneLogger",
        description="NeptuneLogger 类路径",
    )

    # 基础参数
    api_key: Optional[str] = Field(None, description="Neptune API Key")
    project: Optional[str] = Field(None, description="项目名称")
    name: Optional[str] = Field(None, description="实验名称")
    run: Optional[Any] = Field(None, description="现有的 Neptune run")

    # 高级参数
    log_model_checkpoints: bool = Field(True, description="记录模型检查点")
    capture_stdout: bool = Field(True, description="捕获标准输出")
    capture_stderr: bool = Field(True, description="捕获标准错误")
    capture_hardware_metrics: bool = Field(True, description="捕获硬件指标")
    tags: Optional[List[str]] = Field(None, description="实验标签")
    source_files: Optional[List[str]] = Field(None, description="源代码文件")

    # 保存参数
    save_dir: str = Field("./neptune_logs", description="本地保存目录")
    mode: str = Field(
        "async", description="运行模式: async, sync, offline, read-only, debug"
    )
