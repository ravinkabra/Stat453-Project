"""
🔧 项目配置 - 统整所有组件的配置类
"""

from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Dict, Any, Optional, List, Literal

from ..._logger import UnionLoggerParams
from ..._datamodule import UnionDataModuleConfig
from ..._trainer import UnionTrainerConfig
from ..._callback import UnionCallbackConfig
from ...model import UnionModelConfig


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProjectConfig:
    """
    项目级配置类 - 统一管理整个训练项目的配置

    设计理念：
    - 作为所有组件配置的顶层容器
    - 直接配置 PyTorch Lightning 原生 Trainer
    - 提供项目级的元信息和设置
    - 支持组件间的配置协调
    """

    # 项目元信息
    name: str = Field(default="training_project", description="项目名称")
    description: str = Field(default="", description="项目描述")
    version: str = Field(default="1.0.0", description="项目版本")

    # 输出设置
    output_dir: str = Field(default="./outputs", description="输出根目录")
    experiment_name: Optional[str] = Field(
        default=None, description="实验名称，None则使用项目名称"
    )

    # 核心组件配置
    model: UnionModelConfig = Field(description="模型配置")
    datamodule: Optional[UnionDataModuleConfig] = Field(
        default=None, description="数据模块配置"
    )

    # PyTorch Lightning Trainer 配置
    trainer: Optional[UnionTrainerConfig] = Field(
        default=None, description="PyTorch Lightning Trainer 原生配置"
    )

    # 回调配置
    callbacks: Optional[dict[str, UnionCallbackConfig]] = Field(
        default=None, description="回调配置字典，key为回调名称"
    )

    # 日志器配置 - 使用类型化的 logger 配置
    loggers: Optional[Dict[str, UnionLoggerParams]] = Field(
        default=None, description="日志器配置字典，key为日志器名称，value为类型化配置"
    )

    # 运行模式
    mode: Literal["train", "validate", "test", "predict"] = Field(
        default="train", description="运行模式"
    )

    # 实验管理
    tags: List[str] = Field(default_factory=list, description="实验标签")
    notes: str = Field(default="", description="实验备注")

    # 重现性设置
    seed: Optional[int] = Field(default=None, description="随机种子")
    deterministic: bool = Field(default=False, description="是否启用确定性模式")

    def get_experiment_name(self) -> str:
        """获取实验名称"""
        return self.experiment_name or self.name


# 预定义的项目配置示例
def get_unet_project_config() -> ProjectConfig:
    """UNet 图像分割项目配置示例 - 完全使用 PyTorch Lightning 原生配置"""
    return ProjectConfig(
        name="unet_segmentation",
        description="UNet图像分割训练项目",
        output_dir="./outputs/unet_experiments",
        # 模型配置
        model={
            "_target_": "src.model.example.model.UNetModel",
            "in_channels": 3,
            "out_channels": 1,
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
        },
        # 数据模块配置
        datamodule={
            "_target_": "src.datamodule.example.ImageSegmentationDataModule",
            "data_dir": "./data/segmentation",
            "batch_size": 16,
        },
        # PyTorch Lightning Trainer 原生配置
        trainer={
            "max_epochs": 100,
            "devices": "auto",
            "accelerator": "auto",
            "precision": "16-mixed",
            "log_every_n_steps": 10,
            "val_check_interval": 0.5,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        },
        # 回调配置
        callbacks={
            "sample_saver": {
                "_target_": "src.callback.sample_saver.callback.TrainingSampleSaverCallback",
                "save_dir": "./outputs/unet_experiments/samples",
                "save_keys": {
                    "predictions": {
                        "source": "outputs",
                        "keys": ["predictions"],
                        "format": "image",
                        "extension": ".png",
                    }
                },
            },
            "model_checkpoint": {
                "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                "dirpath": "./outputs/unet_experiments/checkpoints",
                "filename": "unet-{epoch:02d}-{val_loss:.2f}",
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "save_last": True,
            },
            "early_stopping": {
                "_target_": "pytorch_lightning.callbacks.EarlyStopping",
                "monitor": "val_loss",
                "patience": 10,
                "mode": "min",
                "verbose": True,
            },
            "lr_monitor": {
                "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
                "logging_interval": "step",
            },
        },
        # 日志器配置
        logging={
            "tensorboard": {
                "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
                "save_dir": "./outputs/unet_experiments/logs",
                "name": "tensorboard",
            },
            "csv": {
                "_target_": "pytorch_lightning.loggers.CSVLogger",
                "save_dir": "./outputs/unet_experiments/logs",
                "name": "csv_logs",
            },
        },
        # 实验管理
        tags=["unet", "segmentation", "medical"],
        seed=42,
        deterministic=True,
    )


def get_llm_project_config() -> ProjectConfig:
    """大语言模型训练项目配置示例"""
    return ProjectConfig(
        name="llm_training",
        description="大语言模型训练项目",
        output_dir="./outputs/llm_experiments",
        # 模型配置
        model={
            "_target_": "src.model.example.model.LLMModel",
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 5e-5,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": 1000,
            },
        },
        # 数据模块配置
        datamodule={
            "_target_": "src.datamodule.example.TextDataModule",
            "data_dir": "./data/text",
            "batch_size": 32,
            "max_length": 512,
        },
        # PyTorch Lightning Trainer 原生配置
        trainer={
            "max_epochs": 10,
            "devices": [0, 1],  # 使用两个GPU
            "accelerator": "gpu",
            "strategy": "ddp",  # 分布式训练
            "precision": "16-mixed",
            "accumulate_grad_batches": 4,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 50,
            "val_check_interval": 1000,
        },
        # 回调配置
        callbacks={
            "model_checkpoint": {
                "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                "dirpath": "./outputs/llm_experiments/checkpoints",
                "filename": "llm-{epoch:02d}-{val_perplexity:.2f}",
                "monitor": "val_perplexity",
                "mode": "min",
                "save_top_k": 2,
                "save_last": True,
                "every_n_train_steps": 500,
            },
            "lr_monitor": {
                "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
                "logging_interval": "step",
            },
        },
        # 日志器配置
        logging={
            "wandb": {
                "_target_": "pytorch_lightning.loggers.WandbLogger",
                "project": "llm-training",
                "name": "llm-experiment",
            },
        },
        # 实验管理
        tags=["llm", "transformer", "language-model"],
        seed=42,
        deterministic=False,  # LLM训练通常不需要完全确定性
    )


# 便捷构建函数
def create_simple_config(
    model_config: Dict[str, Any],
    project_name: str = "simple_project",
    output_dir: str = "./outputs",
    max_epochs: int = 10,
    **kwargs,
) -> ProjectConfig:
    """创建简单的项目配置"""

    # 默认的 Trainer 配置
    trainer_config = {
        "max_epochs": max_epochs,
        "devices": "auto",
        "accelerator": "auto",
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }

    return ProjectConfig(
        name=project_name,
        output_dir=output_dir,
        model=model_config,
        trainer=trainer_config,
        **kwargs,
    )
