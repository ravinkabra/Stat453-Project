"""
TrainingSampleSaverCallback 新构造函数使用示例
展示如何使用 @overload 构造函数直接传入配置参数
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src._callback.sample_saver.callback import TrainingSampleSaverCallback
from src._callback.sample_saver.config import SaveKeyConfig


def example_new_constructor():
    """使用新构造函数的示例"""
    print("=== 使用新构造函数示例 ===")

    # 方法1：使用新的 @overload 构造函数，直接传入参数
    callback1 = TrainingSampleSaverCallback(
        save_dir="./direct_params_samples",
        save_keys={
            "generated_images": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "generated"],
                format="image",
                extension=".png",
                save_frequency=50,
                max_samples=8,
            ),
            "target_images": SaveKeyConfig(
                source="batch",
                keys="target",
                format="image",
                extension=".png",
            ),
        },
        default_save_frequency=100,
        default_max_samples=4,
        default_phases=["val"],
        dir_structure="epoch_step",
        auto_create_dirs=True,
        add_timestamp=True,
        add_step_info=True,
    )

    print("✅ 使用新构造函数创建成功!")
    print(f"保存目录: {callback1.config.save_dir}")
    print(f"目录结构: {callback1.config.dir_structure}")
    print(f"保存键: {list(callback1.config.save_keys.keys())}")
    print(f"默认保存频率: {callback1.config.default_save_frequency}")
    print(f"默认阶段: {callback1.config.default_phases}")


def example_mixed_usage():
    """混合使用新旧构造函数的示例"""
    print("\n=== 混合使用示例 ===")

    # 方法2：仍然可以使用传统的 config 对象方式
    from src._callback.sample_saver.config import TrainingSampleSaverCallbackConfig

    config = TrainingSampleSaverCallbackConfig(
        save_dir="./config_object_samples",
        save_keys={
            "logits": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "logits"],
                format="tensor",
                extension=".pt",
            ),
        },
        default_save_frequency=50,
        dir_structure="step",
    )

    callback2 = TrainingSampleSaverCallback(config=config)

    print("✅ 使用配置对象创建成功!")
    print(f"保存目录: {callback2.config.save_dir}")
    print(f"目录结构: {callback2.config.dir_structure}")


def example_minimal_config():
    """最小配置示例"""
    print("\n=== 最小配置示例 ===")

    # 只传入必要的参数，其他使用默认值
    callback3 = TrainingSampleSaverCallback(
        save_dir="./minimal_samples",
        save_keys={
            "simple_output": SaveKeyConfig(
                source="outputs",
                keys="output",
                format="json",
                extension=".json",
            ),
        },
    )

    print("✅ 使用最小配置创建成功!")
    print(f"保存目录: {callback3.config.save_dir}")
    print(f"默认保存频率: {callback3.config.default_save_frequency}")
    print(f"默认最大样本数: {callback3.config.default_max_samples}")
    print(f"目录结构: {callback3.config.dir_structure}")


def example_advanced_config():
    """高级配置示例"""
    print("\n=== 高级配置示例 ===")

    callback4 = TrainingSampleSaverCallback(
        save_dir="./advanced_samples",
        save_keys={
            "nested_data": SaveKeyConfig(
                source="outputs",
                keys=["model", "decoder", "layer_3", "activations"],
                format="tensor",
                extension=".pt",
                save_frequency=25,  # 这个键特殊设置
                max_samples=2,
                phases=["train", "val"],  # 这个键特殊设置
            ),
            "batch_info": SaveKeyConfig(
                source="batch",
                keys=["metadata", "sample_id"],
                format="json",
                extension=".json",
            ),
        },
        default_save_frequency=100,
        default_max_samples=5,
        default_phases=["val", "test"],
        dir_structure="epoch_step",
        use_epoch_dirs=True,
        use_step_dirs=True,
        auto_create_dirs=True,
        add_timestamp=False,  # 不添加时间戳
        add_step_info=True,
    )

    print("✅ 使用高级配置创建成功!")
    print(f"保存目录: {callback4.config.save_dir}")
    print(f"目录结构: {callback4.config.dir_structure}")
    print(f"添加时间戳: {callback4.config.add_timestamp}")
    print(f"添加步数信息: {callback4.config.add_step_info}")


if __name__ == "__main__":
    example_new_constructor()
    example_mixed_usage()
    example_minimal_config()
    example_advanced_config()

    print("\n✅ 所有新构造函数示例运行完成!")
    print("\n📋 新构造函数特性:")
    print("1. 支持直接传入配置参数，无需创建配置对象")
    print("2. 自动扣除 _target_ 字段")
    print("3. 保持向后兼容性")
    print("4. 支持所有原有功能")
    print("5. 类型提示完整，支持 IDE 自动补全")
