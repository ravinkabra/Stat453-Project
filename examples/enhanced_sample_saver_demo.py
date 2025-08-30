"""
增强版 TrainingSampleSaverCallback 演示

新功能：
1. 重命名为 TrainingSampleSaverCallback
2. 支持无限制样本数 (max_samples = -1)
3. 支持目录层级结构 (epoch/step 目录)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.callback.sample_saver.config import (
    TrainingSampleSaverCallbackConfig,
    SaveKeyConfig,
    get_unet_example_config,
    get_llm_example_config,
)
from src.callback.sample_saver.callback import TrainingSampleSaverCallback
import torch
import numpy as np
from pathlib import Path


def test_unlimited_samples():
    """测试无限制样本数功能"""
    print("=== 测试无限制样本数功能 ===")

    config = TrainingSampleSaverCallbackConfig(
        save_dir="./test_unlimited_samples",
        dir_structure="flat",
        save_keys={
            "all_samples": SaveKeyConfig(
                source="outputs",
                keys="predictions",
                format="tensor",
                extension=".pt",
                max_samples=-1,  # 无限制
                save_frequency=1,  # 每次都保存
                phases=["val"],
            ),
            "limited_samples": SaveKeyConfig(
                source="outputs",
                keys="predictions",
                format="tensor",
                extension="_limited.pt",
                max_samples=2,  # 限制2个
                save_frequency=1,
                phases=["val"],
            ),
        },
        default_max_samples=4,
        default_phases=["val"],
    )

    print(f"配置创建成功:")
    print(f"  - 保存目录: {config.save_dir}")
    print(f"  - 目录结构: {config.dir_structure}")
    print(f"  - 保存键数量: {len(config.save_keys)}")

    for key, save_config in config.save_keys.items():
        max_samples = save_config.max_samples or config.default_max_samples
        samples_text = "无限制" if max_samples == -1 else str(max_samples)
        print(f"  - {key}: 最大样本数 {samples_text}")


def test_directory_structures():
    """测试不同的目录结构"""
    print("\n=== 测试目录结构功能 ===")

    structures = ["flat", "epoch", "step", "epoch_step"]

    for structure in structures:
        config = TrainingSampleSaverCallbackConfig(
            save_dir=f"./test_dir_structure_{structure}",
            dir_structure=structure,
            save_keys={
                "test_data": SaveKeyConfig(
                    source="outputs",
                    keys="data",
                    format="json",
                    extension=".json",
                    max_samples=2,
                    save_frequency=1,
                    phases=["train", "val"],
                )
            },
            default_phases=["train", "val"],
        )

        print(f"目录结构 '{structure}' 配置:")
        print(f"  - 保存根目录: {config.save_dir}")
        print(f"  - 自动创建目录: {config.auto_create_dirs}")
        print(f"  - 添加时间戳: {config.add_timestamp}")


def test_config_examples():
    """测试预定义的配置示例"""
    print("\n=== 测试预定义配置示例 ===")

    # UNet 示例
    unet_config = get_unet_example_config()
    print(f"UNet 配置:")
    print(f"  - 保存目录: {unet_config.save_dir}")
    print(f"  - 目录结构: {unet_config.dir_structure}")
    print(f"  - 默认最大样本数: {unet_config.default_max_samples}")

    # 检查特殊配置
    for key, save_config in unet_config.save_keys.items():
        max_samples = save_config.max_samples or unet_config.default_max_samples
        samples_text = "无限制" if max_samples == -1 else str(max_samples)
        print(f"    - {key}: {samples_text} 样本, 格式: {save_config.format}")

    # LLM 示例
    llm_config = get_llm_example_config()
    print(f"\nLLM 配置:")
    print(f"  - 保存目录: {llm_config.save_dir}")
    print(f"  - 目录结构: {llm_config.dir_structure}")
    print(
        f"  - 默认最大样本数: {'无限制' if llm_config.default_max_samples == -1 else llm_config.default_max_samples}"
    )


def test_nested_keys():
    """测试嵌套键访问"""
    print("\n=== 测试嵌套键访问 ===")

    config = TrainingSampleSaverCallbackConfig(
        save_dir="./test_nested_keys",
        save_keys={
            "simple_key": SaveKeyConfig(
                source="outputs",
                keys="simple",  # 单个键
                format="tensor",
            ),
            "nested_keys": SaveKeyConfig(
                source="outputs",
                keys=["level1", "level2", "data"],  # 嵌套键列表
                format="tensor",
            ),
            "deep_nested": SaveKeyConfig(
                source="batch",
                keys=["metadata", "sample_info", "id"],  # 深度嵌套
                format="json",
            ),
        },
    )

    print("嵌套键配置:")
    for key, save_config in config.save_keys.items():
        keys_text = (
            save_config.keys
            if isinstance(save_config.keys, str)
            else " -> ".join(save_config.keys)
        )
        print(f"  - {key}: {save_config.source}[{keys_text}] -> {save_config.format}")


def test_custom_format():
    """测试自定义格式"""
    print("\n=== 测试自定义格式配置 ===")

    config = TrainingSampleSaverCallbackConfig(
        save_dir="./test_custom_format",
        save_keys={
            "attention_visualization": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "attention_weights"],
                format="custom",  # 需要模型实现 save 方法
                extension=".png",
                custom_save_kwargs={
                    "visualization_type": "heatmap",
                    "colormap": "viridis",
                    "normalize": True,
                    "save_raw": False,
                },
            ),
            "feature_maps": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "feature_maps"],
                format="custom",
                extension=".jpg",
                custom_save_kwargs={
                    "visualization_type": "feature_grid",
                    "grid_size": (8, 8),
                    "padding": 2,
                },
            ),
        },
    )

    print("自定义格式配置:")
    for key, save_config in config.save_keys.items():
        print(f"  - {key}:")
        print(f"    格式: {save_config.format}")
        print(f"    扩展名: {save_config.extension}")
        print(f"    自定义参数: {save_config.custom_save_kwargs}")


def simulate_directory_creation():
    """模拟目录创建过程"""
    print("\n=== 模拟目录创建 ===")

    # 模拟 trainer 对象
    class MockTrainer:
        def __init__(self):
            self.current_epoch = 5
            self.global_step = 250
            self.is_global_zero = True

    trainer = MockTrainer()

    config = TrainingSampleSaverCallbackConfig(
        save_dir="./test_directory_creation",
        dir_structure="epoch_step",
        auto_create_dirs=True,
    )

    callback = TrainingSampleSaverCallback(config)

    # 测试目录生成
    val_dir = callback._get_save_directory(trainer, "val")
    train_dir = callback._get_save_directory(trainer, "train")

    print(f"生成的目录结构:")
    print(f"  - 验证目录: {val_dir}")
    print(f"  - 训练目录: {train_dir}")

    # 清理测试目录
    import shutil

    if Path("./test_directory_creation").exists():
        shutil.rmtree("./test_directory_creation")
        print("  - 测试目录已清理")


def run_enhanced_demo():
    """运行增强版演示"""
    print("🚀 TrainingSampleSaverCallback 增强版功能演示\n")

    try:
        test_unlimited_samples()
        test_directory_structures()
        test_config_examples()
        test_nested_keys()
        test_custom_format()
        simulate_directory_creation()

        print("\n✅ 所有演示完成！")
        print("\n主要改进总结:")
        print("1. ✅ 类名改为 TrainingSampleSaverCallback")
        print("2. ✅ 支持无限制样本数 (max_samples = -1)")
        print("3. ✅ 支持目录层级结构 (flat/epoch/step/epoch_step)")
        print("4. ✅ 保持向后兼容性")
        print("5. ✅ 增强的配置灵活性")

    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_enhanced_demo()
