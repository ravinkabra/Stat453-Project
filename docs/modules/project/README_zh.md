# Project Module

[English](README.md) | [中文](README_zh.md)

项目模块专注于整个训练项目的组织和执行，统一管理所有训练组件。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

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

### 统一组件管理

- **组件协调**：协调 model、datamodule、trainer、callbacks、loggers 等
- **生命周期管理**：完整的训练、验证、测试流程
- **配置驱动**：通过配置管理所有参数
- **模块化设计**：支持灵活的组件组合

### 一站式训练接口

- **简化流程**：统一的训练入口
- **自动配置**：自动处理组件间的依赖关系
- **状态管理**：维护训练过程中的状态信息
- **错误处理**：统一的异常处理机制

### 实验管理

- **实验跟踪**：记录实验配置和结果
- **版本控制**：支持实验版本管理和复现
- **结果管理**：自动保存和组织训练结果
- **对比分析**：支持多实验对比分析

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

1. 在 `ProjectManager` 中添加新的管理方法
2. 扩展 `ProjectConfig` 以支持新的配置选项
3. 添加新的生命周期管理钩子
4. 更新组件集成逻辑
