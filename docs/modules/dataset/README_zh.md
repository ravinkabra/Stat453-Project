# Dataset Module

[English](README.md) | [中文](README_zh.md)

数据集模块专注于数据加载、预处理和模型输入的统一管理。

## 目录结构

```bash
dataset/
├── base/              # 基础数据集抽象类
│   ├── dataset.py     # BaseDataset 基类
│   ├── config.py      # 数据集配置类
│   └── model_input.py # 模型输入定义
└── tutorial_mnist/    # MNIST 教程实现
    ├── dataset.py     # MNIST 数据集实现
    ├── config.py      # MNIST 配置
    └── model_input.py # MNIST 模型输入
```

## 核心组件

### BaseDataset

所有数据集的基类，继承自 `torch.utils.data.Dataset`：

```python
class BaseDataset(Dataset, ABC):
    def __init__(self, config: BaseDatasetConfig):
        self.config = config

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> BaseModelInput:
        pass
```

### BaseDatasetConfig

数据集配置基类，其中一部分会用于构造dataloader，使用 Pydantic 进行类型安全：

```python
@dataclass
class BaseDatasetConfig:
    _target_: str = "src.dataset.base.dataset.BaseDataset"
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
```

### BaseModelInput

模型输入的基础定义，使用DictAccessMixin 保证dataclass可以同时作为dict访问数据：

```python
@dataclass
class BaseModelInput(DictAccessMixin):
    pass
```

## MNIST 数据集实现

详细的 MNIST 数据集实现请参考：[tutorial_mnist/README.md](tutorial_mnist/README.md)

该实现包含：

- 完整的数据加载和预处理流程
- 数据增强功能（随机旋转、裁剪）
- 子集采样支持（用于快速实验）
- 标准化的模型输入格式

## 主要功能

### 数据管道管理

- **统一接口**：标准化的数据集访问接口
- **预处理流程**：完整的数据预处理和增强
- **内存优化**：高效的数据加载和缓存机制
- **多进程支持**：支持多进程数据加载

### 配置管理

- **批量配置**：灵活的批处理大小设置
- **工作进程**：可配置的数据加载进程数
- **内存管理**：内存固定和数据打乱控制
- **类型安全**：基于 Pydantic 的配置验证

### 模型输入标准化

- **统一格式**：标准化的模型输入数据结构
- **字典访问**：支持字典风格的数据访问
- **类型转换**：自动处理数据类型转换
- **扩展性**：易于添加新的输入字段

## 扩展指南

要添加新的数据集，请遵循以下步骤：

1. **创建新文件夹**：在 `dataset/` 下创建新文件夹（如 `cifar10/`）
2. **实现数据集类**：继承 `BaseDataset` 并实现必要方法
3. **配置类继承**：继承 `BaseDatasetConfig` 并添加特定配置
4. **定义输入格式**：创建具体的 `ModelInput` 数据结构
5. **实现数据逻辑**：完成数据加载、预处理和增强功能
6. **导出更新**：在 `__init__.py` 中导出新数据集类
