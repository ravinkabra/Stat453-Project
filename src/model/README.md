# Model 组件 - 快速参考

> 📚 **详细文档**: [Model 组件完整文档](../../docs/components/model.md)

## 🚀 快速开始

Model 组件是基于 PyTorch Lightning 的模型定义，支持灵活的网络组合和训练配置。

### 基础用法

```python
# 1. 定义配置
@dataclass
class MyModelConfig(BaseModelConfig):
    network: NetworkConfig
    optimizer: OptimizerConfig
    loss_fn: LossFunctionConfig

# 2. 实现模型
class MyModel(BaseModel):
    config_class = MyModelConfig
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

# 3. 配置使用
# config/model/my_model.yaml
_target_: path.to.MyModel
network:
  _target_: src.network.example.MyNetwork
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
```

## 🔧 核心API

### BaseModel 类

```python
class BaseModel(LightningModule):
    """基础模型类，提供标准化的训练接口"""
    
    # 必须实现
    config_class: Type[BaseModelConfig]
    
    # 自动构建的属性
    self.network: nn.Module          # 从配置构建的网络
    self.loss_fn: nn.Module         # 损失函数
    self.metrics: Dict[str, Metric] # 指标字典
    
    # 核心方法
    def training_step(self, batch, batch_idx) -> Tensor
    def validation_step(self, batch, batch_idx) -> Tensor
    def configure_optimizers(self) -> Dict
```

### 配置结构

```python
@dataclass
class BaseModelConfig:
    network: Union[NetworkConfig, Dict[str, NetworkConfig]]
    optimizer: Union[OptimizerConfig, Dict[str, OptimizerConfig]]
    lr_scheduler: Optional[Union[LRSchedulerConfig, Dict[str, LRSchedulerConfig]]] = None
    loss_fn: LossFunctionConfig
    metrics: Optional[Dict[str, MetricConfig]] = None
```

## 🎯 常用功能

### 多网络支持

```yaml
network:
  encoder:
    _target_: src.network.example.Encoder
  decoder:
    _target_: src.network.example.Decoder
```

### 多优化器支持

```yaml
optimizer:
  main:
    _target_: torch.optim.Adam
    lr: 0.001
  auxiliary:
    _target_: torch.optim.SGD
    lr: 0.01
```

### 自动指标记录

```python
def training_step(self, batch, batch_idx):
    # ... 计算 y_hat, loss
    self._log_metrics(y_hat, y, prefix='train')  # 自动记录所有配置的指标
    return loss
```

## 📁 文件结构

```
src/model/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── config.py      # BaseModelConfig
│   └── model.py       # BaseModel
└── example/
    ├── __init__.py
    ├── config.py      # 示例配置
    └── model.py       # 示例实现
```

## 🔗 相关组件

- **Network**: [快速参考](../network/README.md) | [详细文档](../../docs/components/network.md)
- **Optimizer**: [详细文档](../../docs/components/optimizer.md)
- **Metric**: [详细文档](../../docs/components/metric.md)

## 💡 示例

查看 `src/model/example/` 中的完整示例，或参考：
- [基础训练教程](../../docs/examples/basic-training.md)
- [多GPU训练指南](../../docs/examples/multi-gpu-training.md)
