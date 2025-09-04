# MNIST 教学数据集

基于 PyTorch 内置 MNIST 数据集的教学实现，展示了框架的数据集组件使用方法。

## 🚀 特性

- ✅ **开箱即用**: 使用 PyTorch 内置 MNIST 数据集
- ✅ **自动下载**: 首次使用时自动下载数据
- ✅ **标准化预处理**: 自动归一化到标准分布
- ✅ **子集支持**: 支持快速实验的小数据集子集
- ✅ **类型安全**: 完整的类型提示和验证
- ✅ **框架集成**: 无缝集成到训练框架中

## 📁 文件结构

```
src/dataset/tutorial_mnist/
├── __init__.py
├── config.py          # 数据集配置
├── dataset.py         # 数据集实现
└── model_input.py     # 数据输入格式定义
```

## ⚙️ 配置选项

### 基础配置

```yaml
dataset:
  _target_: src.dataset.tutorial_mnist.dataset.MnistDataset
  data_dir: "./data/mnist"      # 数据存储目录
  download: true                # 是否下载数据集
  batch_size: 32                # 批次大小
  num_workers: 4                # 数据加载工作进程数
```

### 快速实验配置

```yaml
dataset:
  data_dir: "./data/mnist"
  download: true
  subset_size: 1000             # 只使用1000个样本进行快速实验
  batch_size: 32
  num_workers: 2                # 减少工作进程以加快启动
```

### 数据增强配置

```yaml
dataset:
  data_dir: "./data/mnist"
  download: true
  batch_size: 32
  random_rotation: 15.0         # 随机旋转 ±15度
  random_crop: [24, 24]         # 随机裁剪到24x24
```

## 💻 使用示例

### 基本使用

```python
from src.dataset.tutorial_mnist.dataset import MnistDataset
from src.dataset.tutorial_mnist.config import MinistDatasetConfig

# 创建配置
config = MinistDatasetConfig(
    data_dir="./data/mnist",
    download=True,
    subset_size=1000  # 可选：使用子集
)

# 创建数据集
train_dataset = MnistDataset(config, split="train")
test_dataset = MnistDataset(config, split="test")

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别数量: {train_dataset.num_classes}")
print(f"输入形状: {train_dataset.input_shape}")
```

### 与 DataLoader 结合

```python
from torch.utils.data import DataLoader

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 使用数据
for batch in train_loader:
    images = batch['image']  # [batch_size, 1, 28, 28]
    labels = batch['label']  # [batch_size]
    # 训练代码...
```

### 可视化样本

```python
import matplotlib.pyplot as plt

# 获取样本图像
sample_images = train_dataset.get_sample_images(5)

# 显示图像
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, img in enumerate(sample_images):
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"Sample {i+1}")
plt.show()
```

### 查看类别分布

```python
# 获取类别分布
class_dist = train_dataset.get_class_distribution()
print("MNIST 类别分布:")
for digit, count in class_dist.items():
    print(f"数字 {digit}: {count} 个样本")
```

## 🔧 技术细节

### 数据预处理

- **归一化**: 使用 MNIST 标准均值 (0.1307) 和标准差 (0.3081)
- **格式**: 转换为 PyTorch 张量格式
- **形状**: [batch_size, 1, 28, 28]

### 性能优化

- **内存固定**: `pin_memory=True` 加速 GPU 传输
- **持久工作进程**: `persistent_workers=True` 减少进程创建开销
- **多进程加载**: 使用多个工作进程并行加载数据

### 类型安全

```python
@dataclass
class MNISTModelInput(DictAccessMixin):
    image: Tensor    # [1, 28, 28] 或 [28, 28]
    label: int       # 0-9 数字标签
```

## 🎯 教学价值

这个实现展示了：

1. **数据集抽象**: 如何继承 `BaseDataset` 创建具体数据集
2. **配置管理**: 如何使用 Hydra 配置管理数据集参数
3. **数据流**: 从原始数据到模型输入的完整流程
4. **类型安全**: 如何使用类型提示确保数据正确性
5. **性能优化**: 数据加载的最佳实践

## 📊 数据集信息

- **训练样本**: 60,000 张 28x28 灰度图像
- **测试样本**: 10,000 张 28x28 灰度图像
- **类别数量**: 10 (数字 0-9)
- **数据大小**: ~50MB (压缩后)
- **来源**: Yann LeCun 的 MNIST 数据库

## 🔗 相关组件

- [Model 组件](../model/README.md) - 如何使用这个数据集训练模型
- [Network 组件](../network/README.md) - 适用于 MNIST 的网络架构
- [基础训练教程](../../docs/examples/basic-training.md) - 完整训练流程示例

## 🚀 运行示例

```bash
# 运行演示脚本
python examples/mnist_dataset_demo.py
```

这将创建一个 MNIST 数据集实例并展示其基本功能。
