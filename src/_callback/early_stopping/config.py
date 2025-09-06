from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Union, Optional, Literal, Dict, Any


@dataclass
class EarlyStoppingParams:
    """EarlyStopping 回调配置类

    基于 PyTorch Lightning 的 EarlyStopping 回调，提供统一的配置接口。
    支持所有原生 EarlyStopping 的功能，并提供更好的类型安全和文档。

    使用示例:
        config = EarlyStoppingConfig(
            monitor="val_loss",
            patience=10,
            mode="min",
            min_delta=0.001
        )

        # 通过 instantiate 创建实际的回调
        from hydra.utils import instantiate
        callback = instantiate(config)
    """

    _target_: str = Field(
        default="pytorch_lightning.callbacks.EarlyStopping",
        description="PyTorch Lightning EarlyStopping 类的完整路径",
    )

    # 核心监控配置
    monitor: str = Field(description="监控的指标名称，如 'val_loss', 'val_acc'")
    patience: int = Field(
        default=3, description="在停止训练前，监控指标可以不改善的 epoch 数量"
    )
    mode: Literal["min", "max"] = Field(
        default="min",
        description="监控模式，'min' 表示监控值越小越好，'max' 表示越大越好",
    )

    # 停止条件配置
    min_delta: Union[int, float] = Field(
        default=0.0, description="最小改善阈值，只有当改善超过此值时才算作改善"
    )
    stopping_threshold: Optional[Union[int, float]] = Field(
        default=None, description="停止阈值，当监控指标达到此值时停止训练"
    )
    divergence_threshold: Optional[float] = Field(
        default=None, description="发散阈值，当监控指标超过此值时立即停止训练"
    )

    # 检查配置
    check_finite: bool = Field(default=True, description="是否检查监控值是否为有限值")
    check_on_train_epoch_end: Optional[bool] = Field(
        default=None, description="是否在训练 epoch 结束时进行检查"
    )

    # 日志配置
    log_rank_zero_only: bool = Field(
        default=False, description="是否只在 rank 0 上记录日志"
    )

    # 自定义参数
    custom_stop_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="传递给 EarlyStopping 的其他参数"
    )


# 预定义配置函数
def get_conservative_early_stopping_config() -> EarlyStoppingParams:
    """保守的早停配置 - 耐心值较大，适合稳定训练"""
    return EarlyStoppingParams(
        monitor="val_loss", patience=10, mode="min", min_delta=0.001
    )


def get_aggressive_early_stopping_config() -> EarlyStoppingParams:
    """激进的早停配置 - 耐心值较小，适合快速迭代"""
    return EarlyStoppingParams(
        monitor="val_loss", patience=3, mode="min", min_delta=0.01
    )


def get_accuracy_based_early_stopping_config() -> EarlyStoppingParams:
    """基于准确率的早停配置 - 适用于分类任务"""
    return EarlyStoppingParams(
        monitor="val_acc",
        patience=5,
        mode="max",
        min_delta=0.005,
        stopping_threshold=0.95,  # 当准确率达到95%时停止
    )


def get_loss_based_early_stopping_config() -> EarlyStoppingParams:
    """基于损失的早停配置 - 适用于回归任务"""
    return EarlyStoppingParams(
        monitor="val_loss",
        patience=7,
        mode="min",
        min_delta=0.001,
        divergence_threshold=10.0,  # 当损失发散到10时立即停止
    )


def get_custom_metric_early_stopping_config(
    monitor: str, patience: int = 5, mode: Literal["min", "max"] = "min"
) -> EarlyStoppingParams:
    """自定义指标的早停配置"""
    return EarlyStoppingParams(
        monitor=monitor, patience=patience, mode=mode, min_delta=0.001
    )
