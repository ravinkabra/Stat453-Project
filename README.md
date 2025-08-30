# Model Training Framework

一个基于 PyTorch Lightning 和 Hydra 的模块化深度学习训练框架，旨在提供灵活、可扩展、易于配置的机器学习模型训练解决方案。

## 🏗️ 架构概览

本框架采用高度模块化的设计，将深度学习训练过程中的各个组件进行解耦，每个组件都遵循统一的接口设计和配置管理模式。

```
src/
├── model/           # 模型定义层 - 整合网络、优化器、调度器等
├── network/         # 网络架构层 - 纯粹的神经网络定义
├── optimizer/       # 优化器层 - 各种优化算法的封装
├── lr_scheduler/    # 学习率调度器层 - 学习率调整策略
├── metric/          # 指标管理层 - 训练过程中的指标计算和记录
├── callback/        # 回调函数层 - 训练过程中的自定义行为
└── datamodule/      # 数据模块层 - 数据加载和预处理
```

## 🎯 核心设计理念

### 1. 分层架构设计

框架采用清晰的分层架构，每层负责不同的职责：

- **Model Layer**: 最高层抽象，集成所有组件，定义训练流程
- **Network Layer**: 网络架构定义，专注于模型结构
- **Component Layer**: 各功能组件（优化器、调度器、指标等）
- **Data Layer**: 数据处理和加载

### 2. 配置驱动开发

- 使用 **Pydantic** 进行类型安全的配置管理
- 集成 **Hydra** 支持配置文件组合和命令行覆盖
- 每个组件都有对应的配置类，支持运行时实例化

### 3. 插件化扩展机制

- 基于抽象基类的设计，确保接口一致性
- 工厂模式的组件实例化（通过 `_target_` 字段）
- 支持热插拔式组件替换

## 🧩 核心组件详解

### Model 模块

**设计理念**: Model 是框架的核心协调者，继承自 PyTorch Lightning 的 `LightningModule`，负责整合所有组件并定义训练流程。

```python
# 抽象基类设计
class BaseModel(LightningModule, ABC):
    @abstractmethod
    def forward(self, batch):
        """定义前向传播逻辑"""
        
    @abstractmethod
    def compute_loss(self, model_output, batch):
        """定义损失计算逻辑"""
        
    @abstractmethod
    def decode(self, model_output):
        """定义输出解码逻辑（用于可解释性）"""
```

**核心特性**:
- 统一的训练/验证/测试步骤模板
- 自动配置优化器和学习率调度器
- 集成指标管理系统
- 支持分布式训练（DDP）

### Network 模块

**设计理念**: Network 专注于纯粹的神经网络架构定义，与训练逻辑解耦。

```python
# 网络组件独立于训练逻辑
class BaseNetwork(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """纯粹的网络前向传播"""
```

### Metric 管理系统

**设计理念**: 提供灵活的指标计算、记录和可视化管理。

**核心组件**:
```python
class MetricManager(torch.nn.Module):
    """DDP-aware 的指标管理器"""
    
    def __init__(self, config: MetricManagerConfig):
        # 使用 ModuleDict 确保 DDP 兼容性
        self.metrics = torch.nn.ModuleDict()
        self.log_configs = {}
```

**特性**:
- **DDP 兼容**: 使用 `ModuleDict` 确保分布式训练兼容性
- **灵活配置**: 支持不同阶段（train/val/test）的指标记录
- **可定制日志**: 支持步级别和epoch级别的日志记录
- **自动聚合**: 支持多种聚合方式（mean、sum、max、min）

### Callback 系统

**设计理念**: 基于 PyTorch Lightning 的回调机制，提供训练过程中的自定义行为。

**示例 - OutputLoggerCallback**:
```python
class OutputLoggerCallback(Callback):
    """输出记录回调，用于训练过程中的模型输出可视化"""
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 解码模型输出并记录
        decoded_preds = pl_module.decode(outputs["model_output"])
```

### 配置管理系统

**设计理念**: 使用 Pydantic + Hydra 构建类型安全、灵活的配置系统。

```python
@dataclass
class BaseModelConfig:
    _target_: str = "src.model.base.model.BaseModel"  # 运行时实例化目标
    optimizer: Optional[UnionOptimizerParams] = None
    lr_scheduler: Optional[UnionLRSchedulerParams] = None
    metrics: Optional[MetricManagerConfig] = None
```

**特性**:
- **类型安全**: Pydantic 提供运行时类型检查
- **组合式配置**: Hydra 支持配置文件组合和覆盖
- **工厂模式**: 通过 `_target_` 字段支持动态实例化

## 🔄 数据流和执行流程

### 1. 配置阶段
```
config.yaml → Hydra → Pydantic Config Classes → Component Instantiation
```

### 2. 训练阶段
```
DataModule → Model.forward() → Loss Computation → 
MetricManager.update() → Optimizer.step() → Callback.on_*()
```

### 3. 指标记录
```
Metric Computation → MetricManager.log() → Lightning Logger → 
External Tools (W&B, TensorBoard)
```

## 🚀 使用示例

### 基本使用流程

1. **定义你的网络架构**:
```python
class MyNetwork(BaseNetwork):
    def forward(self, x):
        return self.layers(x)
```

2. **实现你的模型**:
```python
class MyModel(BaseModel):
    def forward(self, batch):
        return self.network(batch["input"])
    
    def compute_loss(self, output, batch):
        return F.cross_entropy(output, batch["labels"])
    
    def decode(self, output):
        return torch.argmax(output, dim=-1)
```

3. **配置训练参数**:
```yaml
# config.yaml
model:
  _target_: src.model.example.model.MyModel
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  metrics:
    accuracy:
      metric:
        _target_: torchmetrics.Accuracy
        task: multiclass
        num_classes: 10
```

4. **开始训练**:
```python
# main.py
from hydra.utils import instantiate

model = instantiate(config.model)
trainer = pl.Trainer()
trainer.fit(model, datamodule)
```

## 🎨 扩展指南

### 添加新的优化器
1. 在 `src/optimizer/` 下创建新模块
2. 继承基类并实现必要方法
3. 更新配置类型定义

### 添加新的指标
1. 使用 torchmetrics 或自定义指标类
2. 在 MetricManager 配置中添加新指标
3. 配置记录策略

### 添加新的回调
1. 继承 `pytorch_lightning.Callback`
2. 实现相关的钩子函数
3. 在训练器中注册回调

## 🔧 技术特性

- **🔥 PyTorch Lightning**: 自动处理分布式训练、检查点、日志等
- **⚙️ Hydra**: 强大的配置管理和实验组织
- **📊 Type Safety**: Pydantic 提供运行时类型检查
- **🔌 Plugin Architecture**: 高度模块化，易于扩展
- **📈 DDP Ready**: 原生支持分布式数据并行训练
- **🎯 Metric Management**: 统一的指标计算和记录系统

## 📁 项目结构

```
model-training-framework/
├── src/                        # 源代码目录
│   ├── model/                  # 模型定义
│   │   ├── base/              # 基础抽象类
│   │   └── example/           # 示例实现
│   ├── network/               # 网络架构
│   ├── optimizer/             # 优化器组件
│   ├── lr_scheduler/          # 学习率调度器
│   ├── metric/                # 指标管理
│   │   ├── _manager/         # 指标管理器
│   │   └── base/             # 基础指标类
│   ├── callback/              # 回调函数
│   └── datamodule/           # 数据模块
├── config/                     # 配置文件目录
├── main.py                    # 训练入口点
├── pyproject.toml             # 项目配置
└── README.md                  # 项目文档
```

## 🎯 设计优势

1. **高内聚，低耦合**: 每个模块职责单一，依赖关系清晰
2. **配置驱动**: 无需修改代码即可调整训练策略
3. **易于测试**: 模块化设计便于单元测试
4. **生产就绪**: 支持分布式训练和实验管理
5. **可扩展性**: 插件化架构，易于添加新功能

## 🔮 后续规划

- [ ] 完善数据模块实现
- [ ] 添加更多优化器和调度器示例
- [ ] 集成更多实验跟踪工具
- [ ] 添加模型导出和部署功能
- [ ] 完善单元测试和文档

---

这个框架体现了现代深度学习工程的最佳实践，通过合理的抽象和模块化设计，为快速原型开发和生产环境部署提供了坚实的基础。