# Optimizer Module

[English](README.md) | [中文](README_zh.md)

优化器模块专注于模型参数优化的配置和管理，与训练逻辑解耦。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

## 目录结构

```text
_optimizer/
├── base/              # 基础优化器参数抽象类
│   └── config.py      # BaseOptimizerParams 基类
├── adam/              # Adam 优化器
│   └── config.py      # AdamParams 参数类
├── sgd/               # SGD 优化器
│   └── config.py      # SGDParams 参数类
├── adamw/             # AdamW 优化器
│   └── config.py      # AdamWParams 参数类
├── adadelta/          # AdaDelta 优化器
│   └── config.py      # AdaDeltaParams 参数类
├── adagrad/           # AdaGrad 优化器
│   └── config.py      # AdaGradParams 参数类
├── rmsprop/           # RMSProp 优化器
│   └── config.py      # RMSPropParams 参数类
├── rprop/             # RProp 优化器
│   └── config.py      # RPropParams 参数类
├── lbfgs/             # LBFGS 优化器
│   └── config.py      # LBFGSParams 参数类
└── __init__.py        # 模块初始化
```

## 核心组件

### BaseOptimizerParams

所有优化器参数的基类：

```python
@dataclass
class BaseOptimizerParams:
    """优化器参数基类"""

    _target_: str = Field("torch.optim.Adam", description="Optimizer class")
    lr: float = Field(1e-3, description="Learning rate")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
```

### UnionOptimizerParams

统一的优化器参数类型：

```python
UnionOptimizerParams = Union[
    BaseOptimizerParams,
    AdamParams,
    SGDParams,
    AdamWParams,
    AdaDeltaParams,
    AdaGradParams,
    RMSPropParams,
    RPropParams,
    LBFGSParams,
    dict,  # Allow dict for backward compatibility
]
```

### 支持的优化器

- **Adam**: 自适应矩估计优化器
- **SGD**: 随机梯度下降
- **AdamW**: Adam with weight decay regularization
- **AdaDelta**: 自适应学习率优化器
- **AdaGrad**: 自适应梯度优化器
- **RMSProp**: 均方根传播优化器
- **RProp**: 弹性反向传播优化器
- **LBFGS**: 有限内存 BFGS 优化器

## 主要功能

### 优化算法多样性

- **一阶优化**：基于梯度的优化算法
- **自适应优化**：根据参数历史调整学习率
- **正则化**：支持权重衰减等正则化技术
- **动量优化**：支持动量和 Nesterov 动量

### 配置管理

- **学习率控制**：灵活的学习率设置
- **权重衰减**：L2 正则化参数
- **算法参数**：各优化器特有的参数配置
- **类型安全**：基于 Pydantic 的配置验证

## 使用示例

```python
from src._optimizer import UnionOptimizerParams, AdamParams

# 创建 Adam 优化器参数
adam_params = AdamParams(
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# 在配置中使用
config = SomeConfig(
    optimizer=adam_params
)
```
