"""
🔧 Logger 配置系统 - 统一管理各种日志器的配置

设计理念：
1. 区分 Config 和 Params：
   - XxxParams: 直接作为类实例化的参数（除了 _target_）
   - XxxConfig: 包含控制逻辑的配置（是否启用、选择哪个等）

2. Union 类型放在前面，提供简洁的类型名称
3. 使用 pydantic dataclass 确保类型安全
4. 支持 hydra instantiate 自动实例化

支持的 Logger：
- WandbLogger: Weights & Biases 日志器
- TensorBoardLogger: TensorBoard 日志器
- CSVLogger: CSV 格式日志器
- MLFlowLogger: MLflow 日志器
- NeptuneLogger: Neptune 日志器
"""

from typing import Optional, Dict, Any, List, Union
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..config_base import DictAccessMixin


# ============================================================================
# 第二类：Params - 直接作为实例化参数（除了 _target_）
# ============================================================================


@dataclass
class BaseLoggerParams(DictAccessMixin):
    """
    🔧 Logger 参数基类

    所有 logger 参数的基础类，除了 _target_ 外的所有字段
    都会直接传递给对应的 Logger 类构造函数
    """

    _target_: str = Field(..., description="Logger 类的完整路径")
    version: Optional[Union[int, str]] = Field(None, description="实验版本")
    prefix: str = Field("", description="日志前缀")


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
    save_dir: str = Field("./wandb_logs", description="本地保存目录")
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
    resume: Optional[str] = Field(
        None, description="恢复模式: allow, must, never, auto"
    )


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


# ============================================================================
# Union 类型定义 - 放在前面，使用简洁名称
# ============================================================================

UnionLoggerParams = Union[
    WandbLoggerParams,
    TensorBoardLoggerParams,
    CSVLoggerParams,
    MLFlowLoggerParams,
    NeptuneLoggerParams,
]


# ============================================================================
# 第一类：Config - 包含控制逻辑的配置
# ============================================================================


@dataclass
class LoggerConfig(DictAccessMixin):
    """
    🔧 Logger 配置类

    第一类配置：包含控制逻辑，决定是否启用某个 logger，
    以及该 logger 的具体参数
    """

    enabled: bool = Field(True, description="是否启用该 logger")
    params: UnionLoggerParams = Field(..., description="Logger 的具体参数")


@dataclass
class MultiLoggerConfig(DictAccessMixin):
    """
    🔧 多 Logger 配置

    管理多个 logger 的启用和配置
    """

    loggers: Dict[str, LoggerConfig] = Field(
        default_factory=dict, description="Logger 配置字典"
    )
    default_logger: Optional[str] = Field(None, description="默认 logger 名称")

    def add_logger(self, name: str, config: LoggerConfig) -> None:
        """添加 logger 配置"""
        self.loggers[name] = config

    def get_enabled_loggers(self) -> Dict[str, UnionLoggerParams]:
        """获取所有启用的 logger 参数"""
        return {
            name: config.params
            for name, config in self.loggers.items()
            if config.enabled
        }


# ============================================================================
# 预设配置和工厂函数
# ============================================================================


class LoggerPresets:
    """
    🔧 Logger 预设配置

    提供常用的 logger 配置模板
    """

    @staticmethod
    def wandb_basic(project: str, name: str = "experiment") -> WandbLoggerParams:
        """基础 WandB 参数"""
        return WandbLoggerParams(
            project=project,
            name=name,
            save_code=True,
            log_model=True,
        )

    @staticmethod
    def wandb_offline(project: str, name: str = "experiment") -> WandbLoggerParams:
        """离线 WandB 参数"""
        return WandbLoggerParams(
            project=project,
            name=name,
            offline=True,
            save_code=True,
        )

    @staticmethod
    def tensorboard_basic(save_dir: str = "./tb_logs") -> TensorBoardLoggerParams:
        """基础 TensorBoard 参数"""
        return TensorBoardLoggerParams(
            save_dir=save_dir,
            log_graph=True,
        )

    @staticmethod
    def csv_basic(save_dir: str = "./csv_logs") -> CSVLoggerParams:
        """基础 CSV 参数"""
        return CSVLoggerParams(
            save_dir=save_dir,
            flush_logs_every_n_steps=50,
        )

    @staticmethod
    def multi_logger_setup(
        project: str, experiment_name: str = "experiment", base_dir: str = "./logs"
    ) -> MultiLoggerConfig:
        """多 Logger 组合配置"""
        config = MultiLoggerConfig()

        config.add_logger(
            "wandb",
            LoggerConfig(
                enabled=True, params=LoggerPresets.wandb_basic(project, experiment_name)
            ),
        )

        config.add_logger(
            "tensorboard",
            LoggerConfig(
                enabled=True,
                params=LoggerPresets.tensorboard_basic(f"{base_dir}/tensorboard"),
            ),
        )

        config.add_logger(
            "csv",
            LoggerConfig(
                enabled=True, params=LoggerPresets.csv_basic(f"{base_dir}/csv")
            ),
        )

        config.default_logger = "wandb"
        return config


# ============================================================================
# 工厂函数
# ============================================================================


def create_logger_params(logger_type: str, **kwargs) -> UnionLoggerParams:
    """
    🚀 便捷函数：创建 logger 参数

    Args:
        logger_type: logger 类型 (wandb, tensorboard, csv, mlflow, neptune)
        **kwargs: logger 特定的配置参数

    Returns:
        对应的 logger 参数对象
    """
    logger_type = logger_type.lower()

    if logger_type == "wandb":
        return WandbLoggerParams(**kwargs)
    elif logger_type == "tensorboard":
        return TensorBoardLoggerParams(**kwargs)
    elif logger_type == "csv":
        return CSVLoggerParams(**kwargs)
    elif logger_type == "mlflow":
        return MLFlowLoggerParams(**kwargs)
    elif logger_type == "neptune":
        return NeptuneLoggerParams(**kwargs)
    else:
        raise ValueError(f"不支持的 logger 类型: {logger_type}")


def create_logger_config(
    logger_type: str, enabled: bool = True, **kwargs
) -> LoggerConfig:
    """
    🚀 便捷函数：创建 logger 配置

    Args:
        logger_type: logger 类型
        enabled: 是否启用
        **kwargs: logger 参数

    Returns:
        Logger 配置对象
    """
    params = create_logger_params(logger_type, **kwargs)
    return LoggerConfig(enabled=enabled, params=params)
