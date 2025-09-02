# Network 组件 - 快速参考

> 📚 **详细文档**: [Network 组件完整文档](../../docs/components/network.md)

## 🚀 快速开始

Network 组件定义神经网络的具体架构，支持各种网络类型和预训练模型。

### 基础用法

```python
# 1. 定义配置
@dataclass
class MyNetworkConfig(BaseNetworkConfig):
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

# 2. 实现网络
class MyNetwork(BaseNetwork):
    config_class = MyNetworkConfig
    
    def __init__(self, config: MyNetworkConfig):
        super().__init__(config)
        self.layers = self._build_layers()
    
    def forward(self, x):
        return self.layers(x)

# 3. 配置使用
# config/network/my_network.yaml
_target_: path.to.MyNetwork
input_dim: 784
output_dim: 10
hidden_dims: [512, 256, 128]
```

## 🔧 核心API

### BaseNetwork 类

```python
class BaseNetwork(nn.Module):
    """基础网络类，提供标准化的网络接口"""
    
    # 必须实现
    config_class: Type[BaseNetworkConfig]
    
    # 核心方法
    def forward(self, x) -> Tensor      # 前向传播
    
    # 推荐实现的属性
    @property
    def output_dim(self) -> int         # 输出维度
```

### 配置结构

```python
@dataclass  
class BaseNetworkConfig:
    """基础网络配置"""
    # 子类需要根据具体网络添加配置项
    pass
```

## 🎯 常用网络类型

### MLP 网络

```yaml
_target_: src.network.example.MLP
input_dim: 784
output_dim: 10
hidden_dims: [512, 256, 128]
activation: relu
dropout: 0.1
batch_norm: true
```

### CNN 网络

```yaml
_target_: src.network.example.SimpleCNN
input_channels: 3
num_classes: 10
base_filters: 64
num_blocks: 3
```

### 预训练模型

```yaml
# 使用 torchvision 预训练模型
_target_: torchvision.models.resnet50
pretrained: true
num_classes: 10
```

## 🔗 与Model集成

### 单网络模式

```yaml
# 在 model 配置中
network:
  _target_: src.network.example.ResNet18
  num_classes: 10
```

### 多网络模式

```yaml
# 编码器-解码器架构
network:
  encoder:
    _target_: src.network.example.Encoder
    input_dim: 784
    latent_dim: 128
  decoder:
    _target_: src.network.example.Decoder
    latent_dim: 128
    output_dim: 784
```

在Model中使用：

```python
class AutoEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = self.network['encoder']
        self.decoder = self.network['decoder']
```

## 📁 文件结构

```
src/network/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── config.py      # BaseNetworkConfig
│   └── network.py     # BaseNetwork
└── example/
    ├── __init__.py
    ├── config.py      # 示例配置
    └── network.py     # 示例实现 (MLP, CNN等)
```

## 🔗 相关组件

- **Model**: [快速参考](../model/README.md) | [详细文档](../../docs/components/model.md)
- **Dataset**: [快速参考](../dataset/README.md) | [详细文档](../../docs/components/dataset.md)

## 💡 示例

查看 `src/network/example/` 中的完整示例，或参考：

- [基础训练教程](../../docs/examples/basic-training.md)
- [自定义网络开发](../../docs/examples/custom-components.md)
