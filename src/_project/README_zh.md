# Project Module

[English](README.md) | [中文](README_zh.md)

项目模块专注于整个训练项目的组织和执行，统一管理所有训练组件。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../docs/architecture_zh.md)。

## 目录结构

```text
_project/
├── base/              # 基础项目管理抽象类
│   ├── project.py     # ProjectManager 核心类
│   └── config.py      # ProjectConfig 配置类
└── __init__.py        # 模块初始化
```

## 核心组件

### ProjectManager

项目管理器 - 统整所有训练组件的核心类：

```python
class ProjectManager:
    """
    项目管理器 - 统整所有组件的核心类

    设计理念：
    - 作为所有训练组件的统一入口
    - 直接使用 PyTorch Lightning 原生 Trainer
    - 通过配置管理所有参数，无需额外包装
    - 支持完整的训练生命周期管理
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        # 核心组件
        self.model: Optional[LightningModule] = None
        self.datamodule: Optional[LightningDataModule] = None
        self.trainer: Optional[Trainer] = None
```

### ProjectConfig

项目级配置类 - 统一管理整个训练项目的配置：

```python
@dataclass
class ProjectConfig:
    """项目级配置类 - 统一管理整个训练项目的配置"""

    # 项目元信息
    name: str = "training_project"
    description: str = ""
    version: str = "1.0.0"

    # 输出设置
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    # 核心组件配置
    model: UnionModelConfig
    datamodule: Optional[UnionDataModuleConfig] = None
    trainer: Optional[UnionTrainerConfig] = None
```

## 主要功能

- **统一组件管理**：协调 model、datamodule、trainer、callbacks、loggers 等
- **一站式训练接口**：提供完整的训练、验证、测试流程
- **实验管理**：支持实验管理和复现
- **配置驱动**：通过配置管理所有参数

## 使用示例

```python
from src._project.base.project import ProjectManager
from src._project.base.config import ProjectConfig

# 创建项目配置
config = ProjectConfig(
    name="my_experiment",
    model=model_config,
    datamodule=datamodule_config,
    trainer=trainer_config
)

# 创建项目管理器
project = ProjectManager(config)

# 执行训练
project.fit()
```

## 扩展指南

要扩展项目管理功能：

1. 在 `base/project.py` 中添加新的管理方法
2. 在 `base/config.py` 中添加新的配置字段
3. 更新相关文档和类型注解
4. 保持与 PyTorch Lightning 的兼容性
