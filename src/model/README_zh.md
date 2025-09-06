# Model Module

[English](README.md) | [中文](README_zh.md)

模型模块是深度学习训练框架的核心抽象层，基于 PyTorch Lightning 构建，负责整合网络、优化器、学习率调度器和指标管理，提供统一的模型训练接口。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../docs/architecture_zh.md)。

## 目录结构

```text
model/
├── __init__.py                   # 模块初始化和类型定义
├── base/                         # 基础模型抽象
│   ├── __init__.py              # 子模块初始化
│   ├── config.py                # 基础模型配置类
│   └── model.py                 # 基础模型实现
├── tutorial_mnist/               # MNIST 模型示例
│   ├── __init__.py              # 子模块初始化
│   ├── config.py                # MNIST 模型配置类
│   └── model.py                 # MNIST 模型实现
├── README_zh.md                  # 中文文档
└── README.md                     # 英文文档
```

## 核心组件

### BaseModel

基础模型类 - 继承自 PyTorch Lightning 的 LightningModule：

```python
class BaseModel(LightningModule, ABC):
    """
    基础模型抽象类 - 提供统一的模型训练接口
    集成网络、优化器、调度器和指标管理
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.metric_manager = MetricManager(config.metric_manager)
```


### BaseModelConfig

基础模型配置类 - 定义模型的基本配置结构：

```python
@dataclass
class BaseModelConfig:
    _target_: str = Field(default="src.model.base.model.BaseModel")

    optimizer: Optional[UnionOptimizerParams] = Field(
        default_factory=BaseOptimizerParams,
        description="优化器配置"
    )

    lr_scheduler: Optional[UnionLRSchedulerParams] = Field(
        default_factory=BaseLRSchedulerParams,
        description="学习率调度器配置"
    )

    metric_manager: Optional[MetricManagerConfig] = Field(
        default_factory=MetricManagerConfig
    )
```

### MnistModel

MNIST 模型示例 - 具体模型实现的示例：

```python
class MnistModel(BaseModel):
    """
    MNIST 手写数字识别模型
    集成 LeNet 网络和交叉熵损失
    """
    def __init__(self, config: MnistModelConfig):
        super().__init__(config)
        self.network = LeNet(config.network)
        self.loss = nn.CrossEntropyLoss()
```

## 主要功能

### 模型抽象层

- **统一接口**：基于 PyTorch Lightning 提供标准化的训练接口
- **组件集成**：自动集成网络、优化器、调度器和指标管理
- **配置驱动**：支持通过配置类进行模型实例化
- **类型安全**：使用 Pydantic 提供完整的配置验证

### 优化器管理

- **灵活配置**：支持单个优化器或模块级优化器配置
- **自动实例化**：通过 Hydra 的 instantiate 自动创建优化器
- **参数传递**：自动将模型参数传递给优化器

### 学习率调度器

- **调度器集成**：与优化器自动关联
- **配置匹配**：支持与优化器结构匹配的调度器配置
- **动态调整**：支持训练过程中的学习率动态调整

### 指标管理集成

- **统一管理**：集成 MetricManager 进行指标计算和管理
- **自动记录**：在训练步骤中自动更新和记录指标
- **分组支持**：支持不同阶段和类型的指标分组

## 使用示例

### 基础使用

```python
from src.model import MnistModel, MnistModelConfig
from hydra.utils import instantiate

# 1. 配置模型
config = MnistModelConfig(
    network=LeNetConfig(),  # 网络配置
    optimizer={
        "_target_": "torch.optim.Adam",
        "lr": 0.001,
        "weight_decay": 1e-4
    },
    lr_scheduler={
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 30,
        "gamma": 0.1
    },
    metric_manager=MetricManagerConfig(
        metrics={
            "classification": {
                "metrics": {
                    "accuracy": "torchmetrics.Accuracy",
                    "precision": "torchmetrics.Precision",
                    "recall": "torchmetrics.Recall"
                }
            }
        }
    )
)

# 2. 实例化模型
model = instantiate(config)

# 3. 在训练中使用
trainer = Trainer(model=model)
trainer.fit(model, datamodule=datamodule)
```

### 高级配置示例

```python
# 复杂的模型配置
config = MnistModelConfig(
    network=LeNetConfig(
        num_classes=10,
        dropout_rate=0.5
    ),
    optimizer={
        "_target_": "torch.optim.AdamW",
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "weight_decay": 1e-4
    },
    lr_scheduler={
        "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 1e-6
    },
    metric_manager=MetricManagerConfig(
        metrics={
            "training_state": {
                "metrics": {
                    "loss": ValueRecorderParams(aggregation="mean"),
                    "learning_rate": ValueRecorderParams(aggregation="last")
                }
            },
            "classification": {
                "metrics": {
                    "accuracy": {
                        "_target_": "torchmetrics.Accuracy",
                        "task": "multiclass",
                        "num_classes": 10
                    },
                    "f1_score": {
                        "_target_": "torchmetrics.F1Score",
                        "task": "multiclass",
                        "num_classes": 10,
                        "average": "macro"
                    }
                }
            }
        }
    )
)
```

### 自定义模型实现

```python
from src.model.base import BaseModel, BaseModelConfig

class CustomModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        # 初始化自定义网络
        self.network = CustomNetwork()
        self.loss = CustomLoss()

    def forward(self, batch):
        """前向传播"""
        return self.network(batch)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        outputs = self.forward(batch)
        loss = self.loss(outputs, batch.targets)

        # 更新指标
        self.metric_manager.update(
            self, batch_idx, "training",
            loss=loss,
            custom_metric=custom_value
        )

        return {"loss": loss}
```

## 配置选项详解

### BaseModelConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optimizer` | `UnionOptimizerParams` | `BaseOptimizerParams()` | 优化器配置 |
| `lr_scheduler` | `UnionLRSchedulerParams` | `BaseLRSchedulerParams()` | 学习率调度器配置 |
| `metric_manager` | `MetricManagerConfig` | `MetricManagerConfig()` | 指标管理器配置 |

### MnistModelConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `network` | `LeNetConfig` | `LeNetConfig()` | LeNet 网络配置 |

## 设计理念

1. **抽象层设计**：通过 BaseModel 提供统一的模型抽象接口
2. **组件解耦**：网络、优化器、调度器和指标管理独立配置
3. **配置驱动**：通过配置类实现模型的声明式定义
4. **扩展性**：基于抽象基类支持自定义模型实现
5. **集成化**：自动集成所有训练相关组件

## 扩展指南

要添加新的模型类型：

1. 创建新的配置类，继承 `BaseModelConfig`
2. 实现具体的模型类，继承 `BaseModel`
3. 在配置中设置正确的 `_target_` 字段
4. 实现必要的抽象方法（`forward` 等）
5. 更新 `__init__.py` 中的类型定义
6. 添加相应的文档和使用示例

## 最佳实践

1. **配置组织**：将模型配置与训练配置分离
2. **组件复用**：合理利用现有的优化器和调度器配置
3. **指标集成**：充分利用 MetricManager 的分组功能
4. **类型安全**：使用正确的配置类型避免运行时错误
5. **抽象继承**：基于 BaseModel 实现具体模型逻辑
