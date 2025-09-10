"""
🔧 项目统整器 - 负责整个训练项目的组织和执行

功能：
- 统一管理所有组件的配置和实例化
- 协调 model、datamodule、trainer、callbacks、loggers 等
- 提供一站式的训练启动接口
- 直接使用 PyTorch Lightning 原生 Trainer
- 支持实验管理和复现
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from hydra.utils import instantiate, get_class
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from .config import ProjectConfig
from src._trainer.base.config import BaseTrainerConfig

# 创建 logger
logger = logging.getLogger(__name__)


class ProjectManager:
    """
    🔧 项目管理器 - 统整所有组件的核心类

    设计理念：
    - 作为所有训练组件的统一入口
    - 直接使用 PyTorch Lightning 原生 Trainer
    - 通过配置管理所有参数，无需额外包装
    - 支持完整的训练生命周期管理

    职责：
    1. 根据配置实例化所有组件（model、datamodule、callbacks、loggers）
    2. 配置并创建 PyTorch Lightning Trainer
    3. 提供统一的训练、验证、测试接口
    4. 管理实验输出和日志
    5. 支持实验复现和管理
    """

    def __init__(self, config: ProjectConfig):
        """
        Args:
            config: 项目级配置对象
        """
        self.config = config

        # 核心组件
        self.model: Optional[LightningModule] = None
        self.datamodule: Optional[LightningDataModule] = None
        self.trainer: Optional[Trainer] = None

        # 训练组件
        self.callbacks: List[Callback] = []
        self.loggers: List[Logger] = []

        # 创建输出目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 实验目录
        self.save_experiment_version_dir = config.save_experiment_version_dir

    def build_model(self) -> LightningModule:
        """构建模型"""
        if self.model is None:
            model_cls = get_class(self.config.model._target_)
            self.model = model_cls(self.config.model)
            self._log_info(f"✓ 模型已构建: {type(self.model).__name__}")
        return self.model

    def build_datamodule(self) -> Optional[LightningDataModule]:
        """构建数据模块"""
        if self.config.datamodule and self.datamodule is None:
            datamodule_cls = get_class(self.config.datamodule._target_)
            self.datamodule = datamodule_cls(self.config.datamodule)
            self._log_info(f"✓ 数据模块已构建: {type(self.datamodule).__name__}")
        return self.datamodule

    def build_callbacks(self) -> List[Callback]:
        """构建回调列表"""
        if not self.callbacks and self.config.callbacks:
            for callback_name, callback_config in self.config.callbacks.items():
                try:
                    callback = instantiate(callback_config)
                except Exception as e:
                    from dataclasses import asdict

                    callback_cls = get_class(callback_config._target_)
                    callback = callback_cls(**{k: v for k, v in asdict(callback_config).items() if k != "_target_"})
                self.callbacks.append(callback)
                self._log_info(f"✓ 回调已构建: {callback_name} -> {type(callback).__name__}")
        return self.callbacks

    def build_loggers(self) -> List[Logger]:
        """构建日志器列表"""
        if not self.loggers and self.config.loggers:
            for logger_name, logger_config in self.config.loggers.items():
                logger = instantiate(logger_config)
                self.loggers.append(logger)
                self._log_info(f"✓ 日志器已构建: {logger_name} -> {type(logger).__name__}")
        return self.loggers

    def build_trainer(self) -> Trainer:
        """
        构建 PyTorch Lightning 原生 Trainer

        直接使用配置中的 trainer 参数，无需额外包装
        """
        if self.trainer is None:
            # 确保组件已构建
            callbacks = self.build_callbacks()
            loggers = self.build_loggers()

            # 获取基础训练器配置
            if isinstance(self.config.trainer, dict):
                trainer_config = self.config.trainer.copy() if self.config.trainer else {}
            elif isinstance(self.config.trainer, BaseTrainerConfig):
                from dataclasses import asdict

                trainer_config = asdict(self.config.trainer)
            # 自动添加组件
            if callbacks:
                trainer_config["callbacks"] = callbacks
            if loggers:
                trainer_config["logger"] = loggers  # if len(loggers) > 1 else loggers[0]

            # 设置默认输出目录（如果未指定）
            if "default_root_dir" not in trainer_config:
                trainer_config["default_root_dir"] = str(self.save_experiment_version_dir)
            elif trainer_config["default_root_dir"] is None:
                trainer_config["default_root_dir"] = str(self.save_experiment_version_dir)

            # 设置确定性模式
            if self.config.deterministic and "deterministic" not in trainer_config:
                trainer_config["deterministic"] = True

            # 创建原生 PyTorch Lightning Trainer
            self.trainer = Trainer(**trainer_config)
            self._log_info(f"✓ PyTorch Lightning Trainer 已构建: {len(callbacks)} 个回调, {len(loggers)} 个日志器")

        return self.trainer

    def build_all(
        self,
    ) -> tuple[LightningModule, Optional[LightningDataModule], Trainer]:
        """构建所有组件"""
        self._log_info("🔧 开始构建项目组件...")

        model = self.build_model()
        datamodule = self.build_datamodule()
        trainer = self.build_trainer()

        self._log_info("🎉 所有组件构建完成！")
        return model, datamodule, trainer

    def train(self) -> None:
        """开始训练"""
        self._log_info("🚀 开始训练...")

        model, datamodule, trainer = self.build_all()

        # 设置随机种子
        if self.config.seed is not None:
            import pytorch_lightning as pl

            pl.seed_everything(self.config.seed, workers=True)

        # 开始训练
        trainer.fit(model, datamodule=datamodule)

        self._log_info("✅ 训练完成！")

    def validate(self) -> List[Dict[str, Any]]:
        """运行验证"""
        self._log_info("🔍 开始验证...")

        model, datamodule, trainer = self.build_all()

        # 运行验证
        results = trainer.validate(model, datamodule=datamodule)

        self._log_info("✅ 验证完成！")
        return results

    def test(self) -> List[Dict[str, Any]]:
        """运行测试"""
        self._log_info("🧪 开始测试...")

        model, datamodule, trainer = self.build_all()

        # 运行测试
        results = trainer.test(model, datamodule=datamodule)

        self._log_info("✅ 测试完成！")
        return results

    def predict(self, ckpt_path: Optional[str] = None) -> List[Any]:
        """运行预测"""
        self._log_info("🔮 开始预测...")

        model, datamodule, trainer = self.build_all()

        # 运行预测
        results = trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)

        self._log_info("✅ 预测完成！")
        return results

    def run(self) -> Any:
        """根据配置的模式运行相应的任务"""
        mode = self.config.mode

        if mode == "train":
            return self.train()
        elif mode == "validate":
            return self.validate()
        elif mode == "test":
            return self.test()
        elif mode == "predict":
            return self.predict()
        else:
            raise ValueError(f"不支持的运行模式: {mode}")

    def get_summary(self) -> Dict[str, Any]:
        """获取项目摘要信息"""
        summary = {
            "project_name": self.config.name,
            "experiment_name": self.config.get_experiment_name(),
            "mode": self.config.mode,
            "save_dir": str(self.save_dir),
            "save_experiment_version_dir": str(self.save_experiment_version_dir),
            "model_type": (self.config.model.get("_target_", "Unknown") if self.config.model else None),
            "callbacks_count": (len(self.config.callbacks) if self.config.callbacks else 0),
            "loggers_count": (len(self.config.logging) if self.config.logging else 0),
            "has_datamodule": self.config.datamodule is not None,
            "seed": self.config.seed,
            "deterministic": self.config.deterministic,
            "tags": self.config.tags,
        }

        # 添加已构建组件的信息
        if self.model:
            summary["model_class"] = type(self.model).__name__
        if self.trainer:
            summary["trainer_info"] = {
                "max_epochs": getattr(self.trainer, "max_epochs", None),
                "num_devices": getattr(self.trainer, "num_devices", None),
                "accelerator": str(getattr(self.trainer, "accelerator", None)),
                "precision": str(getattr(self.trainer, "precision", None)),
            }

        return summary

    def print_summary(self) -> None:
        """打印项目摘要"""
        summary = self.get_summary()

        self._log_info("\n" + "=" * 60)
        self._log_info(f"📊 项目摘要: {summary['project_name']}")
        self._log_info("=" * 60)
        self._log_info(f"实验名称: {summary['experiment_name']}")
        self._log_info(f"运行模式: {summary['mode']}")
        self._log_info(f"输出目录: {summary['save_dir']}")
        self._log_info(f"实验目录: {summary['save_experiment_version_dir']}")
        self._log_info(f"模型类型: {summary.get('model_class', '未构建')}")
        self._log_info(f"回调数量: {summary['callbacks_count']}")
        self._log_info(f"日志器数量: {summary['loggers_count']}")
        self._log_info(f"数据模块: {'是' if summary['has_datamodule'] else '否'}")
        self._log_info(f"随机种子: {summary['seed']}")
        self._log_info(f"确定性模式: {'是' if summary['deterministic'] else '否'}")

        if summary["tags"]:
            self._log_info(f"标签: {', '.join(summary['tags'])}")

        if "trainer_info" in summary:
            trainer_info = summary["trainer_info"]
            self._log_info(
                f"训练器配置: 最大轮数={trainer_info['max_epochs']}, "
                f"设备数={trainer_info['num_devices']}, "
                f"加速器={trainer_info['accelerator']}, "
                f"精度={trainer_info['precision']}"
            )

        self._log_info("=" * 60 + "\n")

    @rank_zero_only
    def _log_info(self, message: str) -> None:
        logger.info(message)


class ProjectBuilder:
    """
    🔧 项目构建器 - 提供便捷的项目构建方法
    """

    @staticmethod
    def from_config_file(config_path: str) -> ProjectManager:
        """从配置文件构建项目"""
        # 这里可以集成 Hydra 来加载配置
        # 暂时作为占位符
        raise NotImplementedError("需要集成 Hydra 配置加载")

    @staticmethod
    def quick_setup(
        model_config: Dict[str, Any],
        save_dir: str = "./outputs",
        project_name: str = "training_project",
    ) -> ProjectManager:
        """快速设置项目"""
        config = ProjectConfig(name=project_name, save_dir=save_dir, model=model_config)
        return ProjectManager(config)


def create_training_project(
    model_config: Dict[str, Any],
    datamodule_config: Optional[Dict[str, Any]] = None,
    callbacks_config: Optional[Dict[str, Dict[str, Any]]] = None,
    trainer_config: Optional[Dict[str, Any]] = None,
    logging_config: Optional[Dict[str, Dict[str, Any]]] = None,
    save_dir: str = "./outputs",
    project_name: str = "training_project",
    mode: str = "train",
    **kwargs,
) -> ProjectManager:
    """
    🚀 便捷函数：创建完整的训练项目

    这是最常用的入口函数，提供一站式项目创建

    Args:
        model_config: 模型配置
        datamodule_config: 数据模块配置
        callbacks_config: 回调配置字典
        trainer_config: PyTorch Lightning Trainer 配置
        logging_config: 日志器配置字典
        save_dir: 输出目录
        project_name: 项目名称
        mode: 运行模式 (train/validate/test/predict)
        **kwargs: 其他项目配置参数

    Returns:
        配置好的项目管理器
    """
    config = ProjectConfig(
        name=project_name,
        save_dir=save_dir,
        model=model_config,
        datamodule=datamodule_config,
        callbacks=callbacks_config or {},
        trainer=trainer_config or {},
        logging=logging_config or {},
        mode=mode,
        **kwargs,
    )

    return ProjectManager(config)
