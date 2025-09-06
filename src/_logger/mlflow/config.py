from typing import Optional, Dict, Any
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base.config import BaseLoggerParams


@dataclass
class MLFlowLoggerParams(BaseLoggerParams):
    """
    🔧 MLflow Logger 参数

    所有参数直接传递给 MLFlowLogger 构造函数
    """

    _target_: str = Field(
        default="pytorch_lightning.loggers.MLFlowLogger",
        description="MLFlowLogger 类路径",
    )

    # 基础参数
    experiment_name: Optional[str] = Field(None, description="实验名称")
    run_name: Optional[str] = Field(None, description="运行名称")
    tracking_uri: Optional[str] = Field(None, description="MLflow 服务器URI")
    registry_uri: Optional[str] = Field(None, description="模型注册URI")

    # 高级参数
    tags: Optional[Dict[str, Any]] = Field(None, description="运行标签")
    save_dir: str = Field("./mlflow_logs", description="本地保存目录")
    artifact_location: Optional[str] = Field(None, description="Artifact 存储位置")
    run_id: Optional[str] = Field(None, description="现有运行ID")
