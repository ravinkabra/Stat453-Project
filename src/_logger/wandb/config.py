from typing import Optional, Dict, Any, List, Union
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base.config import BaseLoggerParams


@dataclass
class WandbLoggerParams(BaseLoggerParams):
    """
    🔧 Weights & Biases Logger 参数

    所有参数直接传递给 WandbLogger 构造函数
    """

    _target_: str = Field(
        default="pytorch_lightning.loggers.WandbLogger",
        description="WandbLogger 类路径",
    )

    # 基础参数
    name: Optional[str] = Field(None, description="实验名称")
    project: Optional[str] = Field(None, description="项目名称")
    entity: Optional[str] = Field(None, description="团队/用户名")
    id: Optional[str] = Field(None, description="实验ID，用于恢复")

    # 目录参数
    save_dir: Optional[str] = Field(None, description="本地保存目录")
    offline: bool = Field(False, description="离线模式")

    # 高级参数
    log_model: Union[bool, str] = Field(False, description="是否记录模型")
    experiment: Optional[Any] = Field(None, description="现有的 wandb run")
    tags: Optional[List[str]] = Field(None, description="实验标签")
    group: Optional[str] = Field(None, description="实验组")
    job_type: Optional[str] = Field(None, description="任务类型")
    config: Optional[Dict[str, Any]] = Field(None, description="额外配置")

    # 同步参数
    sync_tensorboard: bool = Field(False, description="同步 TensorBoard 日志")
    monitor_gym: bool = Field(False, description="监控 gym 环境")

    # 保存参数
    save_code: bool = Field(True, description="保存代码")
    resume: Optional[str] = Field(None, description="恢复模式: allow, must, never, auto")
