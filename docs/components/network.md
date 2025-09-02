# Network 组件详细文档

Network 组件负责定义神经网络的具体架构，是模型的核心计算部分。

## 📋 目录

- [概述](#概述)
- [基础配置](#基础配置)
- [网络架构设计](#网络架构设计)
- [与Model组件集成](#与model组件集成)
- [预训练模型支持](#预训练模型支持)
- [示例代码](#示例代码)
- [最佳实践](#最佳实践)

## 📖 概述

Network 组件提供了标准化的网络架构定义方式：

- **模块化设计**: 可组合的网络组件
- **灵活配置**: 支持动态网络结构配置
- **预训练支持**: 无缝集成各种预训练模型
- **类型安全**: 完整的类型提示和验证
- **可扩展性**: 易于添加新的网络架构

## ⚙️ 基础配置

### 配置结构

```python
from src.network.base.config import BaseNetworkConfig

@dataclass
class MyNetworkConfig(BaseNetworkConfig):
    # 输入维度
    input_dim: int
    
    # 输出维度  
    output_dim: int
    
    # 隐藏层配置
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # 激活函数
    activation: str = "relu"
    
    # Dropout率
    dropout: float = 0.0
    
    # 批归一化
    batch_norm: bool = True
```

### Hydra 配置示例

```yaml
# config/network/mlp.yaml
_target_: src.network.example.MLP
_partial_: false

input_dim: 784
output_dim: 256
hidden_dims: [512, 256, 128]
activation: relu
dropout: 0.1
batch_norm: true
```

## 🏗️ 网络架构设计

### 基础网络实现

```python
from src.network.base import BaseNetwork
from src.network.base.config import BaseNetworkConfig

class MLP(BaseNetwork):
    config_class = MyNetworkConfig
    
    def __init__(self, config: MyNetworkConfig):
        super().__init__(config)
        
        layers = []
        input_dim = config.input_dim
        
        # 构建隐藏层
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(config.activation),
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    @property
    def output_dim(self):
        """输出维度属性"""
        return self.config.output_dim
```

### 卷积网络示例

```python
@dataclass
class CNNConfig(BaseNetworkConfig):
    input_channels: int = 3
    num_classes: int = 10
    base_filters: int = 64
    num_blocks: int = 3

class SimpleCNN(BaseNetwork):
    config_class = CNNConfig
    
    def __init__(self, config: CNNConfig):
        super().__init__(config)
        
        self.features = self._make_feature_layers(config)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 计算特征维度
        feature_dim = config.base_filters * (2 ** (config.num_blocks - 1))
        self.classifier = nn.Linear(feature_dim, config.num_classes)
    
    def _make_feature_layers(self, config):
        layers = []
        in_channels = config.input_channels
        
        for i in range(config.num_blocks):
            out_channels = config.base_filters * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.adaptive_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)
```

## 🔗 与Model组件集成

### 单网络集成

在Model配置中直接指定网络：

```yaml
# config/model/image_classifier.yaml
_target_: src.model.example.ImageClassifier

network:
  _target_: src.network.example.ResNet18
  num_classes: 10
  pretrained: true

# Model会自动构建网络
```

Model中的使用：

```python
class ImageClassifier(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.network 已经从配置中自动构建
        
    def forward(self, x):
        return self.network(x)
```

### 多网络架构

复杂模型可能需要多个网络组件：

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

Model中的使用：

```python
class AutoEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # 多网络会构建为字典
        self.encoder = self.network['encoder']
        self.decoder = self.network['decoder']
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
```

**详细信息**: [Model 组件文档](./model.md#network-integration)

## 🤖 预训练模型支持

### 使用 torchvision 预训练模型

```yaml
# ResNet 预训练模型
network:
  _target_: torchvision.models.resnet50
  pretrained: true
  num_classes: 1000  # 或自定义类别数

# EfficientNet 预训练模型
network:
  _target_: torchvision.models.efficientnet_b0
  pretrained: true
  num_classes: 10
```

### 自定义预训练加载

```python
class PretrainedResNet(BaseNetwork):
    def __init__(self, config):
        super().__init__(config)
        
        # 加载预训练模型
        self.backbone = torchvision.models.resnet50(pretrained=config.pretrained)
        
        # 替换分类头
        if config.num_classes != 1000:
            self.backbone.fc = nn.Linear(
                self.backbone.fc.in_features, 
                config.num_classes
            )
        
        # 冻结部分层（可选）
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 只训练分类头
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
```

### 加载自定义检查点

```python
class CustomPretrainedNetwork(BaseNetwork):
    def __init__(self, config):
        super().__init__(config)
        
        # 构建网络架构
        self.network = self._build_network()
        
        # 加载预训练权重
        if config.checkpoint_path:
            self.load_pretrained_weights(config.checkpoint_path)
    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理权重名称不匹配的情况
        state_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint['state_dict'].items() 
            if k in state_dict and state_dict[k].shape == v.shape
        }
        
        state_dict.update(pretrained_dict)
        self.load_state_dict(state_dict)
        
        print(f"Loaded {len(pretrained_dict)} pretrained weights")
```

## 💡 示例代码

### ResNet风格的网络

```python
@dataclass
class ResNetConfig(BaseNetworkConfig):
    block_type: str = "basic"  # "basic" or "bottleneck"
    layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_classes: int = 1000
    zero_init_residual: bool = False

class ResNet(BaseNetwork):
    config_class = ResNetConfig
    
    def __init__(self, config: ResNetConfig):
        super().__init__(config)
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet 块
        self.layer1 = self._make_layer(64, config.layers[0])
        self.layer2 = self._make_layer(128, config.layers[1], stride=2)
        self.layer3 = self._make_layer(256, config.layers[2], stride=2)
        self.layer4 = self._make_layer(512, config.layers[3], stride=2)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, config.num_classes)
        
        self._initialize_weights(config)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        # 实现ResNet层的构建逻辑
        # ... (省略具体实现)
        pass
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

### Transformer风格的网络

```python
@dataclass  
class TransformerConfig(BaseNetworkConfig):
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 5000

class TransformerNetwork(BaseNetwork):
    config_class = TransformerConfig
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            config.d_model, config.dropout, config.max_seq_len
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Linear(config.d_model, config.output_dim)
    
    def forward(self, src, src_mask=None):
        # src: [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask)
        
        # 取序列的最后一个时间步或进行池化
        if len(output.shape) == 3:  # [seq_len, batch, d_model]
            output = output[-1]  # 取最后时间步
        
        return self.output_proj(output)
```

## 🎯 最佳实践

### 1. 网络设计原则

```python
class WellDesignedNetwork(BaseNetwork):
    def __init__(self, config):
        super().__init__(config)
        
        # ✅ 好的做法：模块化设计
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = self._build_classifier()
        
        # ✅ 好的做法：记录重要信息
        self.logger.info(f"Network initialized with {self.count_parameters()} parameters")
    
    def _build_feature_extractor(self):
        """分离特征提取部分"""
        return nn.Sequential(
            # ... 特征提取层
        )
    
    def _build_classifier(self):
        """分离分类部分"""
        return nn.Sequential(
            # ... 分类层
        )
    
    def count_parameters(self):
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 2. 配置验证

```python
@dataclass
class ValidatedNetworkConfig(BaseNetworkConfig):
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.0
    
    def __post_init__(self):
        # ✅ 配置验证
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
```

### 3. 动态网络构建

```python
class DynamicNetwork(BaseNetwork):
    def __init__(self, config):
        super().__init__(config)
        
        # ✅ 根据配置动态构建
        if config.architecture_type == "cnn":
            self.backbone = self._build_cnn_backbone()
        elif config.architecture_type == "transformer":
            self.backbone = self._build_transformer_backbone()
        else:
            raise ValueError(f"Unknown architecture: {config.architecture_type}")
    
    def _build_cnn_backbone(self):
        # CNN架构构建逻辑
        pass
    
    def _build_transformer_backbone(self):
        # Transformer架构构建逻辑  
        pass
```

### 4. 网络状态监控

```python
class MonitoredNetwork(BaseNetwork):
    def forward(self, x):
        # ✅ 添加形状检查（开发时）
        if self.training and hasattr(self.config, 'debug') and self.config.debug:
            print(f"Input shape: {x.shape}")
        
        x = self.features(x)
        
        if self.training and hasattr(self.config, 'debug') and self.config.debug:
            print(f"Features shape: {x.shape}")
        
        output = self.classifier(x)
        
        # ✅ 检查输出合理性
        if torch.isnan(output).any():
            self.logger.warning("NaN detected in network output")
        
        return output
```

---

## 🔗 相关文档

- [Model 组件文档](./model.md) - 了解如何在模型中使用网络
- [基础训练教程](../examples/basic-training.md) - 网络在训练中的应用
- [自定义组件开发](../examples/custom-components.md) - 开发自定义网络架构

## 📝 API 参考

- [BaseNetwork API](../api/base-classes.md#basenetwork)
- [BaseNetworkConfig API](../api/config.md#basenetworkconfig)
