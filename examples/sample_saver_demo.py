"""
TrainingSampleSaver 使用示例
展示如何在不同类型的训练中使用样本保存回调
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.callback.sample_saver.callback import TrainingSampleSaver
from src.callback.sample_saver.config import (
    TrainingSampleSaverConfig,
    SaveKeyConfig,
    get_unet_example_config,
)


def example_unet_usage():
    """UNet 训练中的使用示例"""
    print("=== UNet 训练样本保存示例 ===")

    # 方法1：直接配置
    callback = TrainingSampleSaver(
        save_dir="./unet_training_samples",
        save_keys={
            "generated_images": {
                "source": "outputs",
                "key": "generated",
                "format": "image",
                "extension": ".png",
            },
            "target_images": {
                "source": "batch",
                "key": "target",
                "format": "image",
                "extension": ".png",
            },
            "input_images": {
                "source": "batch",
                "key": "input",
                "format": "image",
                "extension": ".png",
            },
        },
        save_frequency=50,
        max_samples=8,
        phases=["val"],
    )

    print(f"保存目录: {callback.save_dir}")
    print(f"保存键: {list(callback.save_keys.keys())}")
    print(f"保存频率: 每 {callback.save_frequency} 个batch")
    print(f"最大样本数: {callback.max_samples}")

    # 方法2：使用配置类
    config = get_unet_example_config()
    print(f"\n使用配置类: {config}")


def example_llm_usage():
    """LLM 训练中的使用示例"""
    print("\n=== LLM 训练样本保存示例 ===")

    callback = TrainingSampleSaver(
        save_dir="./llm_training_samples",
        save_keys={
            "generated_text": {
                "source": "outputs",
                "key": "decoded_text",
                "format": "text",
                "extension": ".txt",
            },
            "target_text": {
                "source": "batch",
                "key": "target_text",
                "format": "text",
                "extension": ".txt",
            },
            "attention_weights": {
                "source": "outputs",
                "key": "attention",
                "format": "tensor",
                "extension": ".pt",
            },
        },
        save_frequency=100,
        max_samples=4,
        phases=["val", "test"],
    )

    print(f"保存目录: {callback.save_dir}")
    print(f"保存键: {list(callback.save_keys.keys())}")


def example_custom_format():
    """自定义格式保存示例"""
    print("\n=== 自定义格式保存示例 ===")

    callback = TrainingSampleSaver(
        save_dir="./custom_samples",
        save_keys={
            "model_predictions": {
                "source": "outputs",
                "key": "predictions",
                "format": "json",  # JSON 格式保存
                "extension": ".json",
            },
            "feature_maps": {
                "source": "outputs",
                "key": "features",
                "format": "tensor",  # 张量格式保存
                "extension": ".pt",
            },
            "metadata": {
                "source": "batch",
                "key": "meta_info",
                "format": "json",
                "extension": ".json",
            },
        },
        save_frequency=200,
        max_samples=2,
        phases=["train", "val"],  # 训练和验证都保存
    )

    print("支持的格式: image, text, tensor, json")
    print(f"多阶段保存: {callback.phases}")


def demonstrate_config_flexibility():
    """展示配置的灵活性"""
    print("\n=== 配置灵活性演示 ===")

    # 使用 Pydantic dataclass 配置
    from src.config_base import DictAccessMixin

    class FlexibleConfig(TrainingSampleSaverConfig, DictAccessMixin):
        pass

    config = FlexibleConfig()

    # 字典访问修改配置
    config["save_dir"] = "./flexible_samples"
    config["save_frequency"] = 75

    # 动态添加保存键
    config.save_keys["new_output"] = SaveKeyConfig(
        source="outputs", key="new_data", format="text", extension=".log"
    )

    print("动态修改后的配置:")
    print(f"  保存目录: {config['save_dir']}")
    print(f"  保存频率: {config.save_frequency}")
    print(f"  保存键数量: {len(config.save_keys)}")


if __name__ == "__main__":
    example_unet_usage()
    example_llm_usage()
    example_custom_format()
    demonstrate_config_flexibility()

    print("\n✅ 所有示例运行完成!")
    print("\n📋 使用总结:")
    print("1. 通过 save_keys 配置要保存的数据")
    print("2. 支持从 outputs 和 batch 中提取数据")
    print("3. 支持多种格式: image, text, tensor, json")
    print("4. 可配置保存频率、样本数和阶段")
    print("5. 结合 config_base 支持字典访问")
