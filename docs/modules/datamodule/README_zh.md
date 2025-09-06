# DataModule Module

[English](README.md) | [中文](README_zh.md)

数据模块专注于数据加载和准备，与训练逻辑解耦。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

## 目录结构

```text
_datamodule/
├── base/              # 基础数据模块抽象类
│   ├── datamodule.py  # BaseDataModule 基类
│   └── config.py      # 数据模块配置类
└── __init__.py        # 模块初始化
```

## 核心组件

### BaseDataModule

所有数据模块的基类，继承自 PyTorch Lightning 的 LightningDataModule，提供统一的数据加载接口：

```python
class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, config: BaseDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        """设置数据模块用于训练/验证/测试"""
        pass

    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器"""
        pass
```

### BaseDataModuleConfig

数据模块配置基类：

```python
@dataclass
class BaseDataModuleConfig:
    """数据模块配置基类"""

    _target_: Literal["src._datamodule.base.datamodule.BaseDataModule"]
    train: Optional[UnionDatasetConfig] = None
    val: Optional[UnionDatasetConfig] = None
    test: Optional[UnionDatasetConfig] = None
    predict: Optional[UnionDatasetConfig] = None
```

## 主要功能

### 数据管道管理

- **统一接口**：标准化的数据加载和准备流程
- **阶段分离**：支持训练、验证、测试、预测不同阶段
- **配置驱动**：通过配置类管理所有数据参数

### 数据加载器配置

- **批量大小**：可配置的批处理大小
- **工作进程**：支持多进程数据加载
- **内存优化**：支持内存固定和随机打乱

## 扩展指南

要添加新的数据模块实现：

1. 在 `_datamodule/` 下创建新文件夹
2. 继承 `BaseDataModule` 实现数据模块类（不强制）
3. 创建对应的配置类继承 `BaseDataModuleConfig`
4. 更新 `__init__.py` 导出新数据模块
