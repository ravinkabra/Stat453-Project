"""
展示配置类字典访问功能的使用示例
"""

from src.model.base.config import BaseModelConfig
from src.config_base import ConfigBase


def test_dict_access():
    """测试配置类的字典访问功能"""

    # 创建配置实例
    config = BaseModelConfig()

    print("=== 字典访问功能测试 ===")

    # 1. 属性访问（传统方式）
    print(f"传统属性访问: {config._target_}")

    # 2. 字典访问
    print(f"字典访问: {config['_target_']}")

    # 3. 检查键是否存在
    print(f"'_target_' 存在: {'_target_' in config}")
    print(f"'nonexistent' 存在: {'nonexistent' in config}")

    # 4. 使用 get 方法（安全访问）
    print(f"安全访问存在的键: {config.get('_target_', 'default')}")
    print(f"安全访问不存在的键: {config.get('nonexistent', 'default')}")

    # 5. 遍历所有配置
    print("\n所有配置项:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 6. 转换为字典
    config_dict = config.to_dict()
    print(f"\n转换为字典: {config_dict}")

    # 7. 修改配置（如果需要的话）
    try:
        # 注意：由于使用了 Field 默认值，直接设置可能有限制
        # 但可以通过字典方式访问和检查
        print(f"\n优化器配置类型: {type(config['optimizer'])}")
        print(f"学习率调度器配置类型: {type(config['lr_scheduler'])}")
    except Exception as e:
        print(f"访问出错: {e}")


def demonstrate_custom_config():
    """演示自定义配置类"""

    class MyCustomConfig(ConfigBase):
        model_name: str = "default_model"
        learning_rate: float = 0.001
        batch_size: int = 32
        epochs: int = 100

    config = MyCustomConfig()

    print("\n=== 自定义配置类演示 ===")

    # 字典访问
    print(f"模型名称: {config['model_name']}")

    # 修改配置
    config["learning_rate"] = 0.01
    print(f"修改后的学习率: {config.learning_rate}")

    # 批量更新
    updates = {"batch_size": 64, "epochs": 200}
    for key, value in updates.items():
        config[key] = value

    print(f"批量更新后: {config.to_dict()}")


if __name__ == "__main__":
    test_dict_access()
    demonstrate_custom_config()
