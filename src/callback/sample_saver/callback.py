# Training Sample Saver Callback - 训练样本保存回调
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pathlib import Path
from typing import Dict, List, Union, Any, Literal
import json
import torch
from PIL import Image
import numpy as np
from datetime import datetime
from .config import TrainingSampleSaverCallbackConfig, SaveKeyConfig

from typing import overload


class TrainingSampleSaverCallback(Callback):
    """
    训练样本保存回调 - 用于在训练过程中保存关键内容

    主要用途：
    - UNet 训练中的中间生成图片
    - LLM 训练中的生成文本样本
    - 其他需要可视化检查的训练内容

    新功能：
    - 支持嵌套键访问（keys 为 list）
    - 支持 custom 格式（调用 model.save 方法）
    - 支持每个键的独立配置
    - 支持无限制样本数（max_samples = -1）
    - 支持目录层级结构（epoch/step 目录）
    """

    @overload
    def __init__(self, config: TrainingSampleSaverCallbackConfig) -> None: ...
    @overload
    def __init__(
        self,
        *,
        save_dir: str = "./training_samples",
        save_keys: Dict[str, SaveKeyConfig] = ...,
        default_save_frequency: int = 100,
        default_max_samples: int = 4,
        default_phases: List[Literal["train", "val", "test"]] = ...,
        use_epoch_dirs: bool = True,
        use_step_dirs: bool = False,
        dir_structure: Literal["flat", "epoch", "step", "epoch_step"] = "epoch",
        auto_create_dirs: bool = True,
        add_timestamp: bool = True,
        add_step_info: bool = True,
    ) -> None: ...

    def __init__(
        self,
        config: TrainingSampleSaverCallbackConfig = None,
        *,
        save_dir: str = "./training_samples",
        save_keys: Dict[str, SaveKeyConfig] = None,
        default_save_frequency: int = 100,
        default_max_samples: int = 4,
        default_phases: List[Literal["train", "val", "test"]] = None,
        use_epoch_dirs: bool = True,
        use_step_dirs: bool = False,
        dir_structure: Literal["flat", "epoch", "step", "epoch_step"] = "epoch",
        auto_create_dirs: bool = True,
        add_timestamp: bool = True,
        add_step_info: bool = True,
    ):
        """
        初始化 TrainingSampleSaverCallback

        Args:
            config: TrainingSampleSaverCallbackConfig 配置对象。
                   如果提供此参数，将使用配置对象初始化，其他参数将被忽略。
                   示例: TrainingSampleSaverCallback(config=my_config)

            save_dir: 保存根目录。默认为 "./training_samples"
            save_keys: 保存配置字典。默认为空字典
            default_save_frequency: 默认保存频率（每N个batch保存一次）。默认为 100
            default_max_samples: 默认每次最多保存的样本数。默认为 4
            default_phases: 默认在哪些阶段保存。默认为 ["val"]
            use_epoch_dirs: 是否按 epoch 创建子目录。默认为 True
            use_step_dirs: 是否按 step 创建子目录。默认为 False
            dir_structure: 目录结构。默认为 "epoch"
            auto_create_dirs: 是否自动创建保存目录。默认为 True
            add_timestamp: 是否在文件名中添加时间戳。默认为 True
            add_step_info: 是否在文件名中添加步数信息。默认为 True

        使用示例:
            # 方式1: 使用配置对象（推荐用于复杂配置）
            config = TrainingSampleSaverCallbackConfig(...)
            callback = TrainingSampleSaverCallback(config=config)

            # 方式2: 直接传入参数（适合简单配置）
            callback = TrainingSampleSaverCallback(
                save_dir="./my_samples",
                save_keys={"output": SaveKeyConfig(...)},
                default_save_frequency=50
            )
        """
        if config is None:
            # 从参数构造配置对象
            if save_keys is None:
                save_keys = {}
            if default_phases is None:
                default_phases = ["val"]

            config = TrainingSampleSaverCallbackConfig(
                save_dir=save_dir,
                save_keys=save_keys,
                default_save_frequency=default_save_frequency,
                default_max_samples=default_max_samples,
                default_phases=default_phases,
                use_epoch_dirs=use_epoch_dirs,
                use_step_dirs=use_step_dirs,
                dir_structure=dir_structure,
                auto_create_dirs=auto_create_dirs,
                add_timestamp=add_timestamp,
                add_step_info=add_step_info,
            )

        self.config = config
        self.save_dir = Path(config.save_dir)

        if config.auto_create_dirs:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_nested_value(
        self, data: Dict[str, Any], keys: Union[str, List[str]]
    ) -> Any:
        """获取嵌套字典中的值

        Args:
            data: 数据字典
            keys: 键路径，可以是字符串或字符串列表

        Returns:
            嵌套访问的值
        """
        if isinstance(keys, str):
            keys = [keys]

        result = data
        for key in keys:
            if hasattr(result, key):
                result = getattr(result, key)
            elif isinstance(result, dict) and key in result:
                result = result[key]
            else:
                raise KeyError(f"Key path {keys} not found in data. Failed at '{key}'")

        return result

    def _should_save(
        self, save_key_config: SaveKeyConfig, batch_idx: int, phase: str
    ) -> bool:
        """判断是否应该保存

        Args:
            save_key_config: 保存键配置
            batch_idx: 批次索引
            phase: 训练阶段

        Returns:
            是否应该保存
        """
        # 检查阶段
        phases = save_key_config.phases or self.config.default_phases
        if phase not in phases:
            return False

        # 检查频率
        frequency = save_key_config.save_frequency or self.config.default_save_frequency
        return (batch_idx + 1) % frequency == 0

    def _get_max_samples(self, save_key_config: SaveKeyConfig) -> int:
        """获取最大样本数，-1 表示无限制"""
        max_samples = save_key_config.max_samples or self.config.default_max_samples
        return max_samples if max_samples != -1 else float("inf")

    def _get_save_directory(self, trainer: Trainer, phase: str) -> Path:
        """根据配置生成保存目录路径

        Args:
            trainer: PyTorch Lightning trainer
            phase: 训练阶段

        Returns:
            完整的保存目录路径
        """
        save_path = self.save_dir / phase

        if self.config.dir_structure == "flat":
            # 平铺结构，不创建子目录
            pass
        elif self.config.dir_structure == "epoch":
            # 按 epoch 创建目录
            save_path = save_path / f"epoch_{trainer.current_epoch}"
        elif self.config.dir_structure == "step":
            # 按 step 创建目录
            save_path = save_path / f"step_{trainer.global_step}"
        elif self.config.dir_structure == "epoch_step":
            # epoch/step 两级目录
            save_path = (
                save_path
                / f"epoch_{trainer.current_epoch}"
                / f"step_{trainer.global_step}"
            )

        # 自动创建目录
        if self.config.auto_create_dirs:
            save_path.mkdir(parents=True, exist_ok=True)

        return save_path

    def _save_data(
        self,
        data: Any,
        save_key_config: SaveKeyConfig,
        filename_base: str,
        model: LightningModule,
        save_dir: Path,
    ) -> None:
        """保存数据

        Args:
            data: 要保存的数据
            save_key_config: 保存配置
            filename_base: 文件名基础部分
            model: PyTorch Lightning 模型（用于 custom 格式）
            save_dir: 保存目录
        """
        if save_key_config.format == "custom":
            # 调用模型的 save 方法
            if hasattr(model, "save"):
                model.save(
                    data,
                    filename_base,
                    extension=save_key_config.extension,
                    save_dir=save_dir,
                    **save_key_config.custom_save_kwargs,
                )
            else:
                print(
                    "Warning: Model does not implement 'save' method for custom format"
                )
                return

        elif save_key_config.format == "image":
            self._save_as_image(
                data, filename_base, save_key_config.extension, save_dir
            )

        elif save_key_config.format == "text":
            self._save_as_text(data, filename_base, save_key_config.extension, save_dir)

        elif save_key_config.format == "tensor":
            self._save_as_tensor(
                data, filename_base, save_key_config.extension, save_dir
            )

        elif save_key_config.format == "json":
            self._save_as_json(data, filename_base, save_key_config.extension, save_dir)

    def _save_as_image(
        self, data: Any, filename_base: str, extension: str, save_dir: Path
    ) -> None:
        """保存为图像"""
        try:
            if isinstance(data, torch.Tensor):
                # 转换 tensor 为 numpy
                if data.dim() == 4:  # [B, C, H, W]
                    data = data.cpu().numpy()
                elif data.dim() == 3:  # [C, H, W]
                    data = data.unsqueeze(0).cpu().numpy()

                # 保存每个图像
                for i, img in enumerate(data):
                    if img.shape[0] == 3:  # RGB
                        img = np.transpose(img, (1, 2, 0))
                    elif img.shape[0] == 1:  # Grayscale
                        img = img.squeeze(0)

                    # 标准化到 0-255
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
                        np.uint8
                    )

                    filename = f"{filename_base}_{i}{extension}"
                    filepath = save_dir / filename
                    Image.fromarray(img).save(filepath)

            elif isinstance(data, np.ndarray):
                # 直接保存 numpy 数组
                for i, img in enumerate(data):
                    filename = f"{filename_base}_{i}{extension}"
                    filepath = save_dir / filename
                    Image.fromarray(img).save(filepath)

        except Exception as e:
            print(f"Error saving image {filename_base}: {e}")

    def _save_as_text(
        self, data: Any, filename_base: str, extension: str, save_dir: Path
    ) -> None:
        """保存为文本"""
        try:
            if isinstance(data, (list, tuple)):
                # 保存每个文本
                for i, text in enumerate(data):
                    filename = f"{filename_base}_{i}{extension}"
                    filepath = save_dir / filename
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(str(text))
            else:
                # 单个文本
                filename = f"{filename_base}{extension}"
                filepath = save_dir / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(str(data))

        except Exception as e:
            print(f"Error saving text {filename_base}: {e}")

    def _save_as_tensor(
        self, data: Any, filename_base: str, extension: str, save_dir: Path
    ) -> None:
        """保存为张量"""
        try:
            filename = f"{filename_base}{extension}"
            filepath = save_dir / filename
            torch.save(data, filepath)
        except Exception as e:
            print(f"Error saving tensor {filename_base}: {e}")

    def _save_as_json(
        self, data: Any, filename_base: str, extension: str, save_dir: Path
    ) -> None:
        """保存为JSON"""
        try:
            filename = f"{filename_base}{extension}"
            filepath = save_dir / filename

            # 转换为可序列化的格式
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy().tolist()
            elif isinstance(data, np.ndarray):
                data = data.tolist()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error saving JSON {filename_base}: {e}")

    def _generate_filename_base(
        self, trainer: Trainer, phase: str, save_name: str
    ) -> str:
        """生成文件名基础部分"""
        base_parts = [phase, save_name]

        if self.config.add_step_info:
            base_parts.append(f"step_{trainer.global_step}")
            base_parts.append(f"epoch_{trainer.current_epoch}")

        if self.config.add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_parts.append(timestamp)

        return "_".join(base_parts)

    def _process_save_keys(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        phase: str,
    ) -> None:
        """处理所有保存键配置"""
        # 获取当前阶段的保存目录
        current_save_dir = self._get_save_directory(trainer, phase)

        for save_name, save_key_config in self.config.save_keys.items():
            try:
                # 检查是否应该保存
                if not self._should_save(save_key_config, batch_idx, phase):
                    continue

                # 获取数据
                if save_key_config.source == "outputs":
                    data = self._get_nested_value(outputs, save_key_config.keys)
                elif save_key_config.source == "batch":
                    data = self._get_nested_value(batch, save_key_config.keys)
                else:
                    print(f"Unknown source: {save_key_config.source}")
                    continue

                # 限制样本数（如果不是无限制）
                max_samples = self._get_max_samples(save_key_config)
                if (
                    max_samples != float("inf")
                    and isinstance(data, (torch.Tensor, list, tuple, np.ndarray))
                    and len(data) > max_samples
                ):
                    data = data[: int(max_samples)]

                # 生成文件名
                filename_base = self._generate_filename_base(trainer, phase, save_name)

                # 保存数据（使用当前目录）
                self._save_data(
                    data, save_key_config, filename_base, pl_module, current_save_dir
                )

            except Exception as e:
                print(f"Error processing save key '{save_name}': {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """训练批次结束时的回调"""
        if not trainer.is_global_zero:
            return

        self._process_save_keys(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """验证批次结束时的回调"""
        if not trainer.is_global_zero:
            return

        self._process_save_keys(trainer, pl_module, outputs, batch, batch_idx, "val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """测试批次结束时的回调"""
        if not trainer.is_global_zero:
            return

        self._process_save_keys(trainer, pl_module, outputs, batch, batch_idx, "test")
