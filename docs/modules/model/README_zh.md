# Model Module

[English](README.md) | [中文](README_zh.md)

模型模块是深度学习训练框架的核心抽象层，基于 PyTorch Lightning 构建，负责整合网络、优化器、学习率调度器和指标管理，提供统一的模型训练接口。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

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

- **调度集成**：无缝集成各种学习率调度策略
- **动态调整**：支持训练过程中的学习率动态调整
- **监控集成**：支持基于指标的学习率调度

### 指标管理

- **统一管理**：集中管理所有训练指标
- **自动计算**：自动计算和记录指标
- **日志集成**：与日志系统无缝集成

## 使用示例

### 基础模型实现

```python
from src.model.base import BaseModel, BaseModelConfig

class MyModel(BaseModel):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.network = MyNetwork(config.network)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
```

### 配置使用

```python
from src.model import UnionModelConfig, MnistModelConfig

# 创建模型配置
model_config = MnistModelConfig(
    optimizer=AdamParams(lr=1e-3),
    lr_scheduler=StepLRParams(step_size=30, gamma=0.1),
    metric_manager=MetricManagerConfig(...)
)

# 在项目配置中使用
project_config = ProjectConfig(
    model=model_config
)
```
