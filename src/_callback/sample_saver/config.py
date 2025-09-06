from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Dict, List, Literal, Union, Optional


@dataclass
class SaveKeyConfig:
    """单个保存键的配置

    设计思路：
    - key 改为 list，支持嵌套访问，如 ["outputs", "model_output", "logits"]
    - 添加 custom 格式，需要 model 实现 save 方法
    - 每个 key 可以有自己的保存频率和样本数限制
    """

    source: Literal["outputs", "batch"] = Field(
        description="数据来源：outputs 或 batch"
    )
    keys: Union[str, List[str]] = Field(
        description="键路径，支持嵌套访问，如 ['model_output', 'logits'] 或单个键 'logits'"
    )
    format: Literal["image", "text", "tensor", "json", "custom"] = Field(
        default="tensor", description="保存格式。custom 需要 model 实现 save 方法"
    )
    extension: str = Field(default=".pt", description="文件扩展名")

    # 每个 key 可以有自己的控制参数
    save_frequency: Optional[int] = Field(
        default=None, description="保存频率，None 则使用全局设置"
    )
    max_samples: Optional[int] = Field(
        default=None, description="最大样本数，None 则使用全局设置，-1 表示无限制"
    )
    phases: Optional[List[Literal["train", "val", "test"]]] = Field(
        default=None, description="保存阶段，None 则使用全局设置"
    )

    # custom 格式的额外参数
    custom_save_kwargs: Dict = Field(
        default_factory=dict, description="custom 格式的额外参数"
    )


@dataclass
class SampleSaverCallbackConfig:
    """SampleSaverCallback 的配置类

    设计思路：
    - 全局配置作为默认值
    - 每个 SaveKeyConfig 可以覆盖全局配置
    - 支持目录层级结构组织
    - 调用时机在训练、验证、测试的 on_train_batch_end, on_validation_batch_end, on_test_batch_end 结束后
    """

    _target_: str = Field(
        default="src._callback.sample_saver.callback.SampleSaverCallback"
    )
    save_dir: str = Field(default="./samples", description="保存根目录")
    save_keys: Dict[str, SaveKeyConfig] = Field(
        default_factory=dict, description="保存配置字典"
    )

    # 全局默认设置
    default_save_frequency: int = Field(
        default=100, description="默认保存频率（每N个batch保存一次）"
    )
    default_max_samples: int = Field(
        default=4, description="默认每次最多保存的样本数，-1表示无限制"
    )
    default_phases: List[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["val"], description="默认在哪些阶段保存"
    )

    # 目录结构设置
    use_epoch_dirs: bool = Field(default=True, description="是否按 epoch 创建子目录")
    use_step_dirs: bool = Field(default=False, description="是否按 step 创建子目录")
    dir_structure: Literal["flat", "epoch", "step", "epoch_step"] = Field(
        default="epoch",
        description="目录结构：flat(平铺), epoch(按epoch), step(按step), epoch_step(epoch/step)",
    )

    # 其他设置
    auto_create_dirs: bool = Field(default=True, description="是否自动创建保存目录")
    add_timestamp: bool = Field(default=True, description="是否在文件名中添加时间戳")
    add_step_info: bool = Field(default=True, description="是否在文件名中添加步数信息")


# 示例配置
def get_unet_example_config() -> SampleSaverCallbackConfig:
    """UNet 训练的示例配置"""
    return SampleSaverCallbackConfig(
        save_dir="./unet_samples",
        dir_structure="epoch_step",  # 使用 epoch/step 目录结构
        save_keys={
            "generated_images": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "generated"],
                format="image",
                extension=".png",
                save_frequency=50,  # 这个键特殊，50步保存一次
                max_samples=8,  # 这个键保存8个样本
            ),
            "target_images": SaveKeyConfig(
                source="batch",
                keys="target",  # 单个键也可以
                format="image",
                extension=".png",
                max_samples=-1,  # 无限制样本数
            ),
            "input_images": SaveKeyConfig(
                source="batch", keys="input", format="image", extension=".png"
            ),
            "custom_visualization": SaveKeyConfig(
                source="outputs",
                keys=["model_output"],
                format="custom",  # 需要 model 实现 save 方法
                extension=".png",
                custom_save_kwargs={
                    "visualization_type": "attention_map",
                    "colormap": "jet",
                },
            ),
        },
        default_save_frequency=100,
        default_max_samples=4,
        default_phases=["val"],
    )


def get_llm_example_config() -> SampleSaverCallbackConfig:
    """LLM 训练的示例配置"""
    return SampleSaverCallbackConfig(
        save_dir="./llm_samples",
        dir_structure="epoch",  # 按 epoch 组织目录
        save_keys={
            "generated_text": SaveKeyConfig(
                source="outputs",
                keys=["decoded_text"],
                format="text",
                extension=".txt",
                max_samples=-1,  # 保存所有生成的文本
            ),
            "target_text": SaveKeyConfig(
                source="batch", keys=["target_text"], format="text", extension=".txt"
            ),
            "model_logits": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "logits"],
                format="tensor",
                extension=".pt",
                phases=["val", "test"],  # 只在验证和测试时保存
            ),
            "attention_weights": SaveKeyConfig(
                source="outputs",
                keys=["model_output", "attention_weights"],
                format="custom",  # 使用 model.save 方法保存注意力可视化
                extension=".png",
                save_frequency=200,  # 较少保存
                custom_save_kwargs={"save_attention_heatmap": True},
            ),
        },
        default_save_frequency=100,
        default_max_samples=-1,  # 全局无限制
        default_phases=["val"],
    )


def get_complex_nested_example() -> SampleSaverCallbackConfig:
    """复杂嵌套数据的示例配置"""
    return SampleSaverCallbackConfig(
        save_dir="./complex_samples",
        dir_structure="step",  # 按 step 组织目录
        save_keys={
            "deep_nested_data": SaveKeyConfig(
                source="outputs",
                keys=[
                    "model_output",
                    "decoder",
                    "final_layer",
                    "activations",
                ],  # 深度嵌套
                format="tensor",
                extension=".pt",
            ),
            "batch_metadata": SaveKeyConfig(
                source="batch",
                keys=["metadata", "sample_info"],
                format="json",
                extension=".json",
            ),
        },
        default_save_frequency=50,
        default_max_samples=2,
        default_phases=["val"],
    )
