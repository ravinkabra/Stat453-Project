from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Union, Optional, Literal, Dict, Any


@dataclass
class ModelCheckpointParams:
    """ModelCheckpoint 回调配置类

    基于 PyTorch Lightning 的 ModelCheckpoint 回调，提供统一的配置接口。
    支持所有原生 ModelCheckpoint 的功能，并提供更好的类型安全和文档。

    使用示例:
        config = ModelCheckpointConfig(
            dirpath="./checkpoints",
            filename="best-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min"
        )

        # 通过 instantiate 创建实际的回调
        from hydra.utils import instantiate
        callback = instantiate(config)
    """

    _target_: str = Field(
        default="pytorch_lightning.callbacks.ModelCheckpoint",
        description="PyTorch Lightning ModelCheckpoint 类的完整路径",
    )

    # 基本保存配置
    dirpath: Optional[Union[str, Any]] = Field(
        default="./checkpoints", description="检查点保存目录路径"
    )
    filename: Optional[str] = Field(
        default=None,
        description="检查点文件名模板，支持格式化字符串，如 'best-{epoch:02d}-{val_loss:.2f}'",
    )

    # 监控和保存策略
    monitor: Optional[str] = Field(
        default=None, description="监控的指标名称，如 'val_loss', 'val_acc'"
    )
    save_top_k: Union[int, Literal[-1]] = Field(
        default=1, description="保存最好的 k 个检查点，-1 表示保存所有"
    )
    mode: Literal["min", "max"] = Field(
        default="min",
        description="监控模式，'min' 表示监控值越小越好，'max' 表示越大越好",
    )
    save_last: Optional[bool] = Field(
        default=None, description="是否保存最后一个检查点"
    )

    # 保存条件和频率
    every_n_train_steps: Optional[int] = Field(
        default=None, description="每 N 个训练步骤保存一次"
    )
    every_n_epochs: Optional[int] = Field(
        default=None, description="每 N 个 epoch 保存一次"
    )
    train_time_interval: Optional[Union[str, Any]] = Field(
        default=None,
        description="训练时间间隔保存，如 '00:00:30:00' 表示每30分钟保存一次",
    )

    # 文件管理
    auto_insert_metric_name: bool = Field(
        default=True, description="是否自动在文件名中插入指标名称"
    )
    save_weights_only: bool = Field(
        default=False, description="是否只保存模型权重，不保存优化器状态"
    )

    # 高级配置
    save_on_train_epoch_end: Optional[bool] = Field(
        default=None, description="是否在训练 epoch 结束时保存"
    )
    enable_version_counter: bool = Field(default=True, description="是否启用版本计数器")

    # 自定义参数
    custom_save_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="传递给 ModelCheckpoint 的其他参数"
    )


# 预定义配置函数
def get_basic_checkpoint_config() -> ModelCheckpointParams:
    """基本的检查点配置 - 保存最好的 1 个模型"""
    return ModelCheckpointParams(
        dirpath="./checkpoints",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )


def get_comprehensive_checkpoint_config() -> ModelCheckpointParams:
    """全面的检查点配置 - 保存多个模型和定期备份"""
    return ModelCheckpointParams(
        dirpath="./checkpoints",
        filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        every_n_epochs=10,  # 每10个epoch保存一次
        auto_insert_metric_name=False,
    )


def get_time_based_checkpoint_config() -> ModelCheckpointParams:
    """基于时间的检查点配置 - 定期保存"""
    return ModelCheckpointParams(
        dirpath="./checkpoints",
        filename="checkpoint-{epoch:02d}-{step:06d}",
        save_top_k=-1,  # 保存所有
        every_n_train_steps=1000,  # 每1000步保存一次
        train_time_interval="01:00:00:00",  # 每小时保存一次
        auto_insert_metric_name=False,
    )


def get_weights_only_config() -> ModelCheckpointParams:
    """仅权重保存配置 - 只保存模型权重"""
    return ModelCheckpointParams(
        dirpath="./weights",
        filename="weights-{epoch:02d}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
        save_last=True,
    )
