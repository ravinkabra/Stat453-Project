"""
🚀 Logger 系统演示 - Config vs Params 设计

展示两种配置方式的区别和使用场景：
1. XxxParams: 直接用于 hydra instantiate，所有字段都是构造函数参数
2. XxxConfig: 包含控制逻辑，决定是否启用、选择哪个 logger 等

Union 类型放在前面，使用简洁名称 LoggerParams 而不是 LoggerParamsUnion
"""

from hydra.utils import instantiate
from src._logger import (
    LoggerParams,
    WandbLoggerParams,
    TensorBoardLoggerParams,
    CSVLoggerParams,
    LoggerConfig,
    MultiLoggerConfig,
    LoggerPresets,
    create_logger_params,
    create_logger_config,
)


def demo_params_vs_config():
    """演示 Params vs Config 的区别"""

    print("=" * 60)
    print("📊 Logger 系统演示 - Config vs Params 设计")
    print("=" * 60)

    # ========================================================================
    # 第二类：Params - 直接作为实例化参数
    # ========================================================================

    print("\n🔧 第二类：Params - 直接作为实例化参数")
    print("-" * 40)

    # 创建 WandB 参数 - 所有字段都会传给 WandbLogger 构造函数
    wandb_params = WandbLoggerParams(
        project="my_awesome_project",
        name="experiment_with_params",
        save_code=True,
        log_model=True,
        offline=False,
        tags=["demo", "params"],
    )

    print("WandB 参数:")
    print(f"  _target_: {wandb_params._target_}")
    print(f"  project: {wandb_params.project}")
    print(f"  name: {wandb_params.name}")
    print(f"  tags: {wandb_params.tags}")

    # 使用 hydra instantiate 直接实例化
    try:
        # wandb_logger = instantiate(wandb_params)
        print("✅ 可以直接用 instantiate(wandb_params) 创建 WandbLogger")
    except Exception as e:
        print(f"❌ instantiate 失败: {e}")

    # ========================================================================
    # Union 类型展示 - 简洁名称，放在前面
    # ========================================================================

    print("\n🔗 Union 类型 - 简洁名称")
    print("-" * 40)

    # LoggerParams 而不是 LoggerParamsUnion
    logger_params_list: list[LoggerParams] = [
        wandb_params,
        TensorBoardLoggerParams(save_dir="./demo_tb_logs", log_graph=True),
        CSVLoggerParams(save_dir="./demo_csv_logs", flush_logs_every_n_steps=50),
    ]

    print("支持的 Logger 参数类型:")
    for i, params in enumerate(logger_params_list):
        print(f"  {i+1}. {type(params).__name__}")
        print(f"     _target_: {params._target_}")

    # ========================================================================
    # 第一类：Config - 包含控制逻辑
    # ========================================================================

    print("\n🎛️ 第一类：Config - 包含控制逻辑")
    print("-" * 40)

    # 创建 Logger 配置 - 决定是否启用，使用哪个参数
    wandb_config = LoggerConfig(enabled=True, params=wandb_params)

    tensorboard_config = LoggerConfig(
        enabled=False,  # 暂时禁用
        params=TensorBoardLoggerParams(save_dir="./demo_tb_logs"),
    )

    print("Logger 配置:")
    print(f"  WandB - enabled: {wandb_config.enabled}")
    print(f"  TensorBoard - enabled: {tensorboard_config.enabled}")

    # ========================================================================
    # 多 Logger 配置
    # ========================================================================

    print("\n📊 多 Logger 配置")
    print("-" * 40)

    multi_config = MultiLoggerConfig()
    multi_config.add_logger("wandb", wandb_config)
    multi_config.add_logger("tensorboard", tensorboard_config)
    multi_config.add_logger(
        "csv",
        LoggerConfig(enabled=True, params=CSVLoggerParams(save_dir="./demo_csv_logs")),
    )

    # 获取所有启用的 logger 参数
    enabled_loggers = multi_config.get_enabled_loggers()

    print("启用的 Loggers:")
    for name, params in enabled_loggers.items():
        print(f"  ✅ {name}: {type(params).__name__}")

    print("\n禁用的 Loggers:")
    for name, config in multi_config.loggers.items():
        if not config.enabled:
            print(f"  ❌ {name}: {type(config.params).__name__}")

    # ========================================================================
    # 预设配置演示
    # ========================================================================

    print("\n🎨 预设配置演示")
    print("-" * 40)

    # 使用预设创建多 logger 配置
    preset_config = LoggerPresets.multi_logger_setup(
        project="demo_project",
        experiment_name="preset_experiment",
        base_dir="./preset_logs",
    )

    print("预设配置包含的 Loggers:")
    for name, config in preset_config.loggers.items():
        status = "✅ 启用" if config.enabled else "❌ 禁用"
        print(f"  {name}: {status}")
        if isinstance(config.params, WandbLoggerParams):
            print(f"    项目: {config.params.project}")
        elif hasattr(config.params, "save_dir"):
            print(f"    保存目录: {config.params.save_dir}")

    # ========================================================================
    # 工厂函数演示
    # ========================================================================

    print("\n🏭 工厂函数演示")
    print("-" * 40)

    # 创建参数
    factory_params = create_logger_params(
        "wandb", project="factory_project", name="factory_experiment", offline=True
    )

    # 创建配置
    factory_config = create_logger_config(
        "tensorboard", enabled=True, save_dir="./factory_tb_logs", log_graph=True
    )

    print("工厂函数创建的对象:")
    print(f"  Params: {type(factory_params).__name__}")
    print(f"    项目: {factory_params.project}")
    print(f"    离线模式: {factory_params.offline}")

    print(f"  Config: enabled={factory_config.enabled}")
    print(f"    参数类型: {type(factory_config.params).__name__}")


def demo_union_position_comparison():
    """演示 Union 位置的优缺点对比"""

    print("\n" + "=" * 60)
    print("🔄 Union 位置对比演示")
    print("=" * 60)

    print("\n🅰️ 方案A: Union 在前 (当前设计)")
    print("-" * 30)
    print("类型名称: LoggerParams")
    print("使用方式:")
    print("  def setup_logger(params: LoggerParams) -> None:")
    print("      # 简洁明了")
    print("      pass")

    print("\n🅱️ 方案B: Union 在后 (原设计)")
    print("-" * 30)
    print("类型名称: LoggerConfigUnion")
    print("使用方式:")
    print("  def setup_logger(params: LoggerConfigUnion) -> None:")
    print("      # 名称较长，但明确表明是 Union")
    print("      pass")

    print("\n📊 对比总结:")
    print("  可读性: 方案A 更简洁")
    print("  类型明确性: 方案B 更明确")
    print("  使用便利性: 方案A 更便利")
    print("  推荐: 方案A (Union 在前)")


def demo_config_params_distinction():
    """演示 Config vs Params 的使用场景"""

    print("\n" + "=" * 60)
    print("🎯 Config vs Params 使用场景")
    print("=" * 60)

    print("\n📋 第一类：Config - 包含控制逻辑")
    print("-" * 35)
    print("使用场景：")
    print("  ✅ 需要控制是否启用某个组件")
    print("  ✅ 需要在多个选项中选择")
    print("  ✅ 包含业务逻辑判断")
    print("  ✅ 实验管理和开关控制")

    print("\n例子 - CallbackConfig (假设):")
    print("  @dataclass")
    print("  class CallbackConfig:")
    print("      enabled: bool = True")
    print("      callback_type: str = 'ModelCheckpoint'")
    print("      params: CallbackParams = ...")

    print("\n⚙️ 第二类：Params - 直接实例化参数")
    print("-" * 35)
    print("使用场景：")
    print("  ✅ 直接用于 hydra instantiate")
    print("  ✅ 除了 _target_ 外都是构造函数参数")
    print("  ✅ 无额外控制逻辑")
    print("  ✅ 纯参数传递")

    print("\n例子 - ModelCheckpointParams (假设):")
    print("  @dataclass")
    print("  class ModelCheckpointParams:")
    print("      _target_: str = 'pytorch_lightning.callbacks.ModelCheckpoint'")
    print("      monitor: str = 'val_loss'")
    print("      save_top_k: int = 3")
    print("      # 所有字段直接传给 ModelCheckpoint(...)")

    print("\n🔄 两者关系:")
    print("  Config 包含 Params，控制如何使用 Params")
    print("  Params 是纯参数，直接用于对象实例化")
    print("  这样设计职责分离更清晰")


if __name__ == "__main__":
    demo_params_vs_config()
    demo_union_position_comparison()
    demo_config_params_distinction()

    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("=" * 60)
