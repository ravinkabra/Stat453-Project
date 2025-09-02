# Model 组件详细文档

Model 组件是训练框架的核心，负责定义模型的结构、训练逻辑和推理过程。

## 📋 目录

- [概述](#概述)
- [基础配置](#基础配置)
- [训练逻辑定义](#训练逻辑定义)
- [与其他组件集成](#与其他组件集成)
- [高级功能](#高级功能)
- [示例代码](#示例代码)
- [常见问题](#常见问题)

## 📖 概述

Model 组件基于 PyTorch Lightning 的 `LightningModule`，提供了：

- **统一的模型接口**: 标准化的训练、验证、测试流程
- **灵活的配置系统**: 支持 Hydra 配置和类型验证
- **多网络支持**: 可以组合多个网络组件
- **多优化器支持**: 支持不同部分使用不同优化器
- **丰富的回调机制**: 与框架的其他组件无缝集成

## ⚙️ 基础配置

### 配置结构

```python
from src.model.base.config import BaseModelConfig

@dataclass
class MyModelConfig(BaseModelConfig):
    # 网络配置 - 可以是单个或多个网络
    network: Union[NetworkConfig, Dict[str, NetworkConfig]]
    
    # 优化器配置 - 支持单个或多个优化器
    optimizer: Union[OptimizerConfig, Dict[str, OptimizerConfig]]
    
    # 学习率调度器配置（可选）
    lr_scheduler: Optional[Union[LRSchedulerConfig, Dict[str, LRSchedulerConfig]]] = None
    
    # 损失函数配置
    loss_fn: LossFunctionConfig
    
    # 指标配置
    metrics: Optional[Dict[str, MetricConfig]] = None
```

### Hydra 配置示例

```yaml
# config/model/my_model.yaml
_target_: src.model.example.MyModel
_partial_: false

# 网络配置
network:
  _target_: src.network.example.MyNetwork
  hidden_dims: [256, 128, 64]
  activation: relu
  dropout: 0.1

# 优化器配置
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 1e-4

# 学习率调度器
lr_scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 30
  gamma: 0.1

# 损失函数
loss_fn:
  _target_: torch.nn.CrossEntropyLoss

# 指标配置
metrics:
  accuracy:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
  f1:
    _target_: torchmetrics.F1Score
    task: multiclass
    num_classes: 10
```

## 🏗️ 训练逻辑定义

### 基础模型实现

```python
from src.model.base import BaseModel
from src.model.base.config import BaseModelConfig

class MyModel(BaseModel):
    config_class = MyModelConfig
    
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        
        # 网络会自动从配置中构建
        # self.network 已经可用
        
        # 如果需要多个网络
        if isinstance(config.network, dict):
            self.encoder = self.network['encoder']
            self.decoder = self.network['decoder']
        else:
            self.main_network = self.network
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # 自动记录指标
        self.log('train_loss', loss, prog_bar=True)
        self._log_metrics(y_hat, y, prefix='train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self._log_metrics(y_hat, y, prefix='val')
        
        return loss
```

### 自定义训练逻辑

```python
class AdvancedModel(BaseModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # 获取当前优化器（多优化器场景）
        if self.config.multi_optimizer:
            opt_main, opt_aux = self.optimizers()
        
        # 主网络前向传播
        features = self.network.encoder(x)
        y_hat = self.network.decoder(features)
        
        # 主损失
        main_loss = self.loss_fn(y_hat, y)
        
        # 辅助损失（如果有）
        if hasattr(self.network, 'aux_head'):
            aux_out = self.network.aux_head(features)
            aux_loss = self.aux_loss_fn(aux_out, y)
            total_loss = main_loss + 0.4 * aux_loss
            self.log('aux_loss', aux_loss)
        else:
            total_loss = main_loss
        
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
```

## 🔗 与其他组件集成

### 与 Network 组件集成

Model 组件通过配置自动构建 Network：

```python
# 单网络模式
network:
  _target_: src.network.example.ResNet
  layers: [2, 2, 2, 2]
  num_classes: 10

# 多网络模式  
network:
  encoder:
    _target_: src.network.example.Encoder
    input_dim: 784
    hidden_dim: 256
  decoder:
    _target_: src.network.example.Decoder  
    hidden_dim: 256
    output_dim: 10
```

**详细信息**: [Network 组件文档](./network.md#model-integration)

### 与 Optimizer 组件集成

支持多优化器配置：

```python
# 单优化器
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

# 多优化器 - 不同部分使用不同优化器
optimizer:
  main:
    _target_: torch.optim.Adam
    lr: 0.001
  auxiliary:
    _target_: torch.optim.SGD
    lr: 0.01
    momentum: 0.9
```

**详细信息**: [Optimizer 组件文档](./optimizer.md#multi-optimizer)

### 与 Metric 组件集成

自动指标计算和记录：

```python
metrics:
  accuracy:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
  f1_macro:
    _target_: torchmetrics.F1Score
    task: multiclass
    num_classes: 10
    average: macro
```

**详细信息**: [Metric 组件文档](./metric.md#auto-logging)

## 🚀 高级功能

### 模型检查点管理

```python
class MyModel(BaseModel):
    def configure_callbacks(self):
        """配置模型专用回调"""
        callbacks = super().configure_callbacks()
        
        # 添加模型检查点
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            filename='{epoch}-{val_loss:.2f}'
        )
        callbacks.append(checkpoint)
        
        return callbacks
```

### 自定义学习率调度

```python
def configure_optimizers(self):
    """自定义优化器和调度器配置"""
    optimizer = self._build_optimizer()
    
    # 自定义调度器
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5
        ),
        'monitor': 'val_loss',
        'frequency': 1
    }
    
    return {'optimizer': optimizer, 'lr_scheduler': scheduler}
```

### 模型融合

```python
class EnsembleModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 构建多个子模型
        self.models = nn.ModuleList([
            self._build_single_model(model_config) 
            for model_config in config.sub_models
        ])
    
    def forward(self, x):
        # 模型融合推理
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
```

## 💡 示例代码

### 完整的分类模型示例

```python
# models/image_classifier.py
from dataclasses import dataclass
from src.model.base import BaseModel, BaseModelConfig

@dataclass  
class ImageClassifierConfig(BaseModelConfig):
    num_classes: int = 10
    dropout: float = 0.5

class ImageClassifier(BaseModel):
    config_class = ImageClassifierConfig
    
    def __init__(self, config: ImageClassifierConfig):
        super().__init__(config)
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.network.output_dim, config.num_classes)
        )
    
    def forward(self, x):
        features = self.network(x)
        return self.classifier(features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self._log_metrics(y_hat, y, prefix='train')
        
        return loss
```

### 配置文件示例

```yaml
# config/model/image_classifier.yaml
_target_: models.image_classifier.ImageClassifier

num_classes: 10
dropout: 0.5

network:
  _target_: src.network.example.ResNet18
  pretrained: true
  num_classes: ${model.num_classes}

optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 1e-4

lr_scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100

loss_fn:
  _target_: torch.nn.CrossEntropyLoss
  label_smoothing: 0.1

metrics:
  accuracy:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: ${model.num_classes}
  top5_acc:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: ${model.num_classes}
    top_k: 5
```

## ❓ 常见问题

### Q: 如何实现多任务学习？

A: 配置多个输出头和对应的损失函数：

```python
class MultiTaskModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.task1_head = nn.Linear(self.network.output_dim, config.task1_classes)
        self.task2_head = nn.Linear(self.network.output_dim, config.task2_classes)
    
    def training_step(self, batch, batch_idx):
        x, y1, y2 = batch
        features = self.network(x)
        
        out1 = self.task1_head(features)
        out2 = self.task2_head(features)
        
        loss1 = F.cross_entropy(out1, y1)
        loss2 = F.cross_entropy(out2, y2)
        
        total_loss = loss1 + loss2
        self.log_dict({
            'task1_loss': loss1,
            'task2_loss': loss2,
            'total_loss': total_loss
        })
        
        return total_loss
```

### Q: 如何使用预训练模型？

A: 在网络配置中指定预训练参数：

```yaml
network:
  _target_: torchvision.models.resnet50
  pretrained: true
  num_classes: ${model.num_classes}
```

### Q: 如何实现模型的增量训练？

A: 使用 PyTorch Lightning 的检查点机制：

```python
# 从检查点恢复训练
model = MyModel.load_from_checkpoint('path/to/checkpoint.ckpt')

# 或在配置中指定
trainer = Trainer(resume_from_checkpoint='path/to/checkpoint.ckpt')
```

---

## 🔗 相关文档

- [Network 组件文档](./network.md) - 了解如何设计网络架构
- [基础训练教程](../examples/basic-training.md) - 完整的训练流程示例
- [多GPU训练指南](../examples/multi-gpu-training.md) - 分布式训练配置
- [自定义组件开发](../examples/custom-components.md) - 开发自定义模型组件

## 📝 API 参考

- [BaseModel API](../api/base-classes.md#basemodel)
- [BaseModelConfig API](../api/config.md#basemodelconfig)
