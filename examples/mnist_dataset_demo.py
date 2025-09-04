#!/usr/bin/env python3
"""
MNIST 数据集使用示例

展示如何使用框架中的 MNIST 数据集组件
"""

from src.dataset.tutorial_mnist.dataset import MnistDataset
from src.dataset.tutorial_mnist.config import MinistDatasetConfig


def main():
    """主函数：演示 MNIST 数据集的使用"""

    # 1. 创建配置
    config = MinistDatasetConfig(
        data_dir="./data/mnist", download=True, subset_size=1000  # 使用子集进行快速演示
    )

    # 2. 创建训练数据集
    train_dataset = MnistDataset(config, split="train")
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"输入形状: {train_dataset.input_shape}")
    print(f"类别数量: {train_dataset.num_classes}")

    # 3. 查看类别分布
    class_dist = train_dataset.get_class_distribution()
    print(f"类别分布: {class_dist}")

    # 4. 获取一个样本
    sample = train_dataset[0]
    print(f"样本图像形状: {sample.image.shape}")
    print(f"样本标签: {sample.label}")

    # 5. 获取多个样本用于可视化
    sample_images = train_dataset.get_sample_images(5)
    print(f"可视化样本形状: {sample_images.shape}")

    # 6. 创建测试数据集
    test_config = MinistDatasetConfig(
        data_dir="./data/mnist", download=False, subset_size=200  # 已经下载过了
    )
    test_dataset = MnistDataset(test_config, split="test")
    print(f"测试数据集大小: {len(test_dataset)}")

    print("\n✅ MNIST 数据集创建成功！")


if __name__ == "__main__":
    main()
