"""
🔧 Logger 系统演示

展示如何使用新的类型化 logger 配置系统：
1. 各种 logger 的配置方法
2. 与 ProjectManager 的集成
3. 多 logger 组合使用
4. Hydra instantiate 的使用
"""

from src._project import ProjectManager, ProjectConfig
from src._logger import (
    WandbLoggerConfig,
    TensorBoardLoggerConfig,
    CSVLoggerConfig,
    LoggerPresets,
    create_logger_config,
)


def demo_basic_logger_configs():
    """演示基础的 logger 配置"""
    print("🔧 基础 Logger 配置演示")
    print("=" * 50)

    # 1. 直接创建配置
    print("1. 直接创建 WandB 配置:")
    wandb_config = WandbLoggerConfig(
        project="my_awesome_project",
        name="experiment_001",
        tags=["baseline", "unet"],
        log_model=True,
        save_code=True,
    )
    print(f"   项目: {wandb_config.project}")
    print(f"   实验: {wandb_config.name}")
    print(f"   标签: {wandb_config.tags}")
    print(f"   _target_: {wandb_config._target_}")

    # 2. 使用预设配置
    print("\n2. 使用预设创建 TensorBoard 配置:")
    tb_config = LoggerPresets.tensorboard_basic("./outputs/tensorboard")
    print(f"   保存目录: {tb_config.save_dir}")
    print(f"   记录计算图: {tb_config.log_graph}")
    print(f"   _target_: {tb_config._target_}")

    # 3. 使用工厂函数
    print("\n3. 使用工厂函数创建 CSV 配置:")
    csv_config = create_logger_config(
        "csv", save_dir="./outputs/csv_logs", flush_logs_every_n_steps=25
    )
    print(f"   保存目录: {csv_config.save_dir}")
    print(f"   刷新频率: {csv_config.flush_logs_every_n_steps}")
    print(f"   _target_: {csv_config._target_}")

    print("\n" + "=" * 50 + "\n")


def demo_multi_logger_setup():
    """演示多 logger 组合配置"""
    print("🔧 多 Logger 组合配置演示")
    print("=" * 50)

    # 使用预设的多 logger 配置
    multi_loggers = LoggerPresets.multi_logger_setup(
        project="image_segmentation",
        experiment_name="unet_baseline_v1",
        base_dir="./outputs/logs",
    )

    print("配置的 Logger 列表:")
    for name, config in multi_loggers.items():
        print(f"   {name}: {type(config).__name__}")
        print(f"      _target_: {config._target_}")
        if hasattr(config, "save_dir"):
            print(f"      save_dir: {config.save_dir}")
        if hasattr(config, "project"):
            print(f"      project: {config.project}")
        print()

    print("=" * 50 + "\n")


def demo_project_integration():
    """演示与 ProjectManager 的集成"""
    print("🔧 ProjectManager 集成演示")
    print("=" * 50)

    # 创建项目配置，包含多个 logger
    project_config = ProjectConfig(
        name="segmentation_project",
        description="使用多种 logger 的图像分割项目",
        output_dir="./outputs/segmentation_demo",
        # 模型配置 (使用 _target_ 方式)
        model={
            "_target_": "src.model.example.model.UNetModel",
            "in_channels": 3,
            "out_channels": 1,
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
        },
        # PyTorch Lightning Trainer 配置
        trainer={
            "max_epochs": 10,
            "devices": 1,
            "accelerator": "cpu",  # 演示用
            "log_every_n_steps": 5,
            "enable_checkpointing": True,
        },
        # 类型化的 logger 配置
        logging={
            "wandb": WandbLoggerConfig(
                project="segmentation_demo",
                name="unet_experiment",
                tags=["demo", "multi-logger"],
                offline=True,  # 演示模式，使用离线
                log_model=False,  # 演示时不上传模型
            ),
            "tensorboard": TensorBoardLoggerConfig(
                save_dir="./outputs/segmentation_demo/tensorboard",
                log_graph=True,
                name="unet_logs",
            ),
            "csv": CSVLoggerConfig(
                save_dir="./outputs/segmentation_demo/csv",
                flush_logs_every_n_steps=10,
            ),
        },
        mode="train",
        seed=42,
        deterministic=True,
    )

    # 创建项目管理器
    project_manager = ProjectManager(project_config)

    # 打印项目摘要
    project_manager.print_summary()

    # 展示 logger 配置的类型安全性
    print("Logger 配置类型验证:")
    for name, logger_config in project_config.logging.items():
        print(f"   {name}: {type(logger_config).__name__}")
        print(f"      _target_: {logger_config._target_}")

        # 类型安全的属性访问
        if isinstance(logger_config, WandbLoggerConfig):
            print(f"      project: {logger_config.project}")
            print(f"      offline: {logger_config.offline}")
        elif isinstance(logger_config, TensorBoardLoggerConfig):
            print(f"      save_dir: {logger_config.save_dir}")
            print(f"      log_graph: {logger_config.log_graph}")
        elif isinstance(logger_config, CSVLoggerConfig):
            print(f"      save_dir: {logger_config.save_dir}")
            print(
                f"      flush_logs_every_n_steps: {logger_config.flush_logs_every_n_steps}"
            )
        print()

    print("=" * 50 + "\n")


def demo_hydra_instantiate():
    """演示 Hydra instantiate 的使用"""
    print("🔧 Hydra Instantiate 演示")
    print("=" * 50)

    from hydra.utils import instantiate

    # 演示如何使用 instantiate 创建实际的 logger 对象
    wandb_config = WandbLoggerConfig(
        project="hydra_demo",
        name="test_experiment",
        offline=True,
        save_code=False,
    )

    print("配置对象:")
    print(f"   类型: {type(wandb_config).__name__}")
    print(f"   _target_: {wandb_config._target_}")
    print(f"   project: {wandb_config.project}")

    try:
        # 使用 hydra instantiate 创建实际的 logger
        # 注意：这里可能会因为缺少依赖而失败，这是正常的
        logger = instantiate(wandb_config)
        print(f"\n✅ 成功创建 logger: {type(logger).__name__}")
    except Exception as e:
        print(f"\n⚠️  创建 logger 失败 (这是正常的，可能缺少相关依赖): {e}")
        print("   在实际使用中，确保安装了相关的依赖包 (如 wandb)")

    print("\n" + "=" * 50 + "\n")


def demo_dict_access():
    """演示配置的字典式访问"""
    print("🔧 字典式访问演示")
    print("=" * 50)

    # 创建一个 wandb 配置
    config = WandbLoggerConfig(
        project="dict_access_demo",
        name="test_experiment",
        tags=["demo", "dict-access"],
        log_model=True,
    )

    print("原始访问方式:")
    print(f"   config.project = {config.project}")
    print(f"   config.name = {config.name}")
    print(f"   config.tags = {config.tags}")

    print("\n字典式访问方式:")
    print(f"   config['project'] = {config['project']}")
    print(f"   config['name'] = {config['name']}")
    print(f"   config['tags'] = {config['tags']}")

    print("\n修改配置:")
    config["name"] = "updated_experiment"
    config["offline"] = True

    print(f"   config.name = {config.name}")
    print(f"   config.offline = {config.offline}")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    print("🚀 Logger 系统完整演示")
    print("=" * 60)
    print()

    # 运行各种演示
    demo_basic_logger_configs()
    demo_multi_logger_setup()
    demo_project_integration()
    demo_hydra_instantiate()
    demo_dict_access()

    print("🎉 演示完成！")
    print("\n总结:")
    print("✓ 类型化的 logger 配置系统")
    print("✓ 支持多种常用 logger (WandB, TensorBoard, CSV, MLflow, Neptune)")
    print("✓ 与 ProjectManager 无缝集成")
    print("✓ 支持 Hydra instantiate 自动实例化")
    print("✓ 提供字典式访问接口")
    print("✓ 丰富的预设和工厂函数")
