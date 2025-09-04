from ..base.dataset import BaseDataset
from .config import MnistDatasetConfig
from .model_input import MnistModelInput
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset
from typing import Optional, List
import torch
import os


class MnistDataset(BaseDataset):
    """MNIST 手写数字数据集

    使用 PyTorch 内置的 MNIST 数据集，提供标准化的数据加载和预处理。
    """

    def __init__(self, config: MnistDatasetConfig):
        self.config = config
        # 不在这里加载数据集，延迟到setup时加载
        self.dataset = None
        self.stage = None

    def _build_transform(self):
        """构建数据变换"""
        transform_list = []

        # 转换为Tensor
        transform_list.append(transforms.ToTensor())

        # 数据增强（仅训练集）
        if self.stage == "train":
            if self.config.random_rotation:
                transform_list.append(
                    transforms.RandomRotation(degrees=self.config.random_rotation)
                )
            if self.config.random_crop:
                height, width = self.config.random_crop
                transform_list.append(
                    transforms.RandomCrop(size=(height, width), padding=4)
                )

        # 归一化
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        self.transform = transforms.Compose(transform_list)

    def _load_dataset(self):
        """加载 MNIST 数据集"""
        # 确保数据目录存在
        os.makedirs(self.config.data_dir, exist_ok=True)

        # 创建原始数据集
        raw_dataset = MNIST(
            root=self.config.data_dir,
            train=(self.stage == "train" or self.stage == "val"),
            download=self.config.download,
            transform=self.transform,
        )

        if self.config.dataset_range and len(self.config.dataset_range) == 2:
            if isinstance(self.config.dataset_range[0], float) or isinstance(
                self.config.dataset_range[1], float
            ):
                min_val = int(self.config.dataset_range[0] * len(raw_dataset))
                max_val = int(self.config.dataset_range[1] * len(raw_dataset))
            else:
                min_val, max_val = self.config.dataset_range
            indices = [i for i in range(len(raw_dataset)) if min_val <= i <= max_val]
            self.dataset = Subset(raw_dataset, indices)

    def setup(self, stage: Optional[str] = None):
        """设置数据集（Lightning 回调）

        在这里加载数据集，确保stage信息正确。
        """
        self.stage = stage
        if self.dataset is None:  # 只加载一次
            self._build_transform()  # 先构建transform
            self._load_dataset()

    def __len__(self):
        """返回数据集大小"""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")
        return len(self.dataset)

    def __getitem__(self, idx) -> MnistModelInput:
        """获取单个数据样本"""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")
        image, label = self.dataset[idx]

        # 确保图像是正确的形状 [1, 28, 28]
        if image.dim() == 3 and image.shape[0] != 1:
            image = image.unsqueeze(0)  # 添加通道维度

        return MnistModelInput.from_tensors(image, torch.tensor(label))

    def collate_fn(self, batch: list[MnistModelInput]) -> MnistModelInput:
        """将batch中的样本堆叠成单个输入对象"""
        if not batch:
            raise ValueError("Empty batch")

        # 堆叠图像 [batch_size, 1, 28, 28]
        images = torch.stack([item.image for item in batch])

        # 堆叠标签 [batch_size]
        labels = torch.tensor([item.label for item in batch])

        return MnistModelInput(image=images, label=labels)

    @property
    def num_classes(self) -> int:
        """返回类别数量"""
        return 10

    @property
    def input_shape(self) -> List[int]:
        """返回输入数据的形状"""
        return [1, 28, 28]  # [channels, height, width]

    def get_sample_images(self, num_samples: int = 5) -> torch.Tensor:
        """获取样本图像用于可视化"""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")
        indices = torch.randperm(len(self))[:num_samples]
        images = []
        for idx in indices:
            sample = self[idx]
            images.append(sample.image)
        return torch.stack(images)

    def get_class_distribution(self) -> dict:
        """获取类别分布统计"""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")
        labels = []
        for i in range(len(self)):
            _, label = self.dataset[i]
            labels.append(label)

        from collections import Counter

        counter = Counter(labels)
        return dict(sorted(counter.items()))
