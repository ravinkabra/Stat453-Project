"""
🚀 简化项目架构演示

展示新的设计理念：
- ProjectManager 统一管理所有组件
- 直接使用 PyTorch Lightning 原生 Trainer
- 通过配置管理所有参数，无需额外包装
"""

from src._project.base.project import ProjectManager, create_training_project
from src._project.base.config import (
    ProjectConfig,
    get_unet_project_config,
    get_llm_project_config,
)


def demo_simple_project():
    """演示最简单的项目设置"""
    print("🎯 演示1: 最简单的项目设置")
    print("=" * 50)

    # 最简单的模型配置
    model_config = {
        "_target_": "src.model.example.model.SimpleModel",
        "input_size": 784,
        "hidden_size": 128,
        "output_size": 10,
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
    }

    # 创建项目
    project = create_training_project(
        model_config=model_config,
        project_name="simple_mnist",
        output_dir="./outputs/simple_demo",
        # 直接配置 PyTorch Lightning Trainer
        trainer_config={
            "max_epochs": 5,
            "devices": 1,
            "accelerator": "auto",
            "enable_progress_bar": True,
            "log_every_n_steps": 10,
        },
    )

    # 打印摘要
    project.print_summary()

    # 构建组件（不运行训练）
    model, datamodule, trainer = project.build_all()

    print("✅ 简单项目演示完成！\n")


def demo_unet_project():
    """演示 UNet 项目配置"""
    print("🎯 演示2: UNet 图像分割项目")
    print("=" * 50)

    # 使用预定义配置
    config = get_unet_project_config()
    project = ProjectManager(config)

    # 打印摘要
    project.print_summary()

    # 显示 trainer 配置
    print("📋 PyTorch Lightning Trainer 配置:")
    trainer_config = config.trainer or {}
    for key, value in trainer_config.items():
        print(f"  {key}: {value}")

    print("✅ UNet 项目演示完成！\n")


def demo_llm_project():
    """演示 LLM 项目配置"""
    print("🎯 演示3: 大语言模型训练项目")
    print("=" * 50)

    # 使用预定义配置
    config = get_llm_project_config()
    project = ProjectManager(config)

    # 打印摘要
    project.print_summary()

    # 显示分布式训练配置
    print("🚀 分布式训练配置:")
    trainer_config = config.trainer or {}
    print(f"  设备: {trainer_config.get('devices', 'auto')}")
    print(f"  策略: {trainer_config.get('strategy', 'auto')}")
    print(f"  梯度累积: {trainer_config.get('accumulate_grad_batches', 1)}")
    print(f"  梯度裁剪: {trainer_config.get('gradient_clip_val', 'None')}")

    print("✅ LLM 项目演示完成！\n")


def demo_custom_project():
    """演示自定义项目配置"""
    print("🎯 演示4: 自定义项目 - 完全控制所有配置")
    print("=" * 50)

    # 完全自定义的项目配置
    custom_config = ProjectConfig(
        name="custom_experiment",
        description="完全自定义的训练实验",
        output_dir="./outputs/custom_demo",
        # 模型配置
        model={
            "_target_": "src.model.example.model.CustomModel",
            "architecture": "resnet50",
            "num_classes": 1000,
            "pretrained": True,
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "lr_scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 30,
                "gamma": 0.1,
            },
        },
        # 数据模块
        datamodule={
            "_target_": "src.datamodule.example.ImageClassificationDataModule",
            "data_dir": "./data/imagenet",
            "batch_size": 256,
            "num_workers": 8,
        },
        # PyTorch Lightning Trainer - 完全自定义
        trainer={
            "max_epochs": 90,
            "devices": [0, 1, 2, 3],  # 4个GPU
            "accelerator": "gpu",
            "strategy": "ddp",
            "sync_batchnorm": True,
            "precision": "16-mixed",
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 100,
            "val_check_interval": 0.25,
            "check_val_every_n_epoch": 1,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "max_time": "12:00:00",  # 最大运行12小时
        },
        # 回调配置
        callbacks={
            "model_checkpoint": {
                "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                "dirpath": "./outputs/custom_demo/checkpoints",
                "filename": "custom-{epoch:02d}-{val_acc:.3f}",
                "monitor": "val_acc",
                "mode": "max",
                "save_top_k": 5,
                "save_last": True,
                "every_n_epochs": 5,
            },
            "early_stopping": {
                "_target_": "pytorch_lightning.callbacks.EarlyStopping",
                "monitor": "val_acc",
                "patience": 15,
                "mode": "max",
                "verbose": True,
                "strict": False,
            },
            "lr_monitor": {
                "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
                "logging_interval": "epoch",
            },
            "rich_progress": {
                "_target_": "pytorch_lightning.callbacks.RichProgressBar",
                "refresh_rate": 10,
            },
        },
        # 日志器配置
        logging={
            "tensorboard": {
                "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
                "save_dir": "./outputs/custom_demo/logs",
                "name": "tensorboard",
                "version": "v1",
            },
            "wandb": {
                "_target_": "pytorch_lightning.loggers.WandbLogger",
                "project": "custom-training",
                "name": "custom-experiment-v1",
                "tags": ["custom", "resnet50", "imagenet"],
            },
        },
        # 实验管理
        tags=["custom", "resnet", "classification", "distributed"],
        notes="自定义分布式训练实验，使用4个GPU",
        seed=42,
        deterministic=True,
        mode="train",
    )

    # 创建项目管理器
    project = ProjectManager(custom_config)

    # 打印详细摘要
    project.print_summary()

    # 展示配置的灵活性
    print("🔧 配置亮点:")
    print(f"  ✓ 多GPU分布式训练: {custom_config.trainer['devices']}")
    print(f"  ✓ 混合精度训练: {custom_config.trainer['precision']}")
    print(f"  ✓ 多种日志器: {list(custom_config.logging.keys())}")
    print(f"  ✓ 丰富的回调: {list(custom_config.callbacks.keys())}")
    print(f"  ✓ 实验标签: {custom_config.tags}")

    print("✅ 自定义项目演示完成！\n")


def compare_architectures():
    """对比新旧架构的差异"""
    print("📊 架构对比")
    print("=" * 50)

    print("🔄 旧架构 (EnhancedTrainer + ProjectManager):")
    print("  ❌ 需要维护自定义 EnhancedTrainer 包装")
    print("  ❌ 配置分散在多个地方")
    print("  ❌ 额外的抽象层增加复杂性")
    print("  ❌ 与 PyTorch Lightning 原生用法不一致")

    print("\n✨ 新架构 (统一的 ProjectManager):")
    print("  ✅ 直接使用 PyTorch Lightning 原生 Trainer")
    print("  ✅ 配置集中在 ProjectConfig 中")
    print("  ✅ 更简洁的代码结构")
    print("  ✅ 完全兼容 PyTorch Lightning 生态")
    print("  ✅ 更容易维护和扩展")
    print("  ✅ 支持所有 PyTorch Lightning 原生功能")

    print("\n🎯 核心改进:")
    print("  1. 移除了 EnhancedTrainer 包装层")
    print("  2. 将所有功能合并到 ProjectManager")
    print("  3. 直接配置 PyTorch Lightning Trainer")
    print("  4. 提供更灵活的配置方式")
    print("  5. 保持框架的简洁性和可维护性")


if __name__ == "__main__":
    print("🚀 简化项目架构演示")
    print("=" * 60)
    print("新设计理念：ProjectManager + 原生 PyTorch Lightning Trainer")
    print("=" * 60)

    # 运行演示
    demo_simple_project()
    demo_unet_project()
    demo_llm_project()
    demo_custom_project()
    compare_architectures()

    print("🎉 所有演示完成！")
    print("\n💡 总结:")
    print("  新架构更加简洁和实用，直接使用 PyTorch Lightning 原生功能，")
    print("  通过 ProjectManager 统一管理所有组件，配置更加灵活和直观。")
