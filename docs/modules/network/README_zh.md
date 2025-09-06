# Network Module

[English](README.md) | [中文](README_zh.md)

网络模块专注于神经网络架构的定义，与训练逻辑解耦。

## 目录结构

```text
network/
├── base/              # 基础网络抽象类
│   ├── network.py     # BaseNetwork 基类
│   └── config.py      # 网络配置类
└── tutorial_lenet/    # LeNet 教程实现
    ├── network.py     # LeNet 网络实现
    └── config.py      # LeNet 配置
```

## 核心组件

### BaseNetwork

所有网络架构的基类，提供统一的接口：

```python
class BaseNetwork(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """网络前向传播"""
        pass
```

### 使用示例

```python
from src.network.tutorial_lenet.network import LeNet
from src.network.tutorial_lenet.config import LeNetConfig

# 配置网络
config = LeNetConfig(num_classes=10)
network = LeNet(config)

# 使用网络
output = network(input_tensor)
```

## 主要功能

### 网络架构抽象

- **统一接口**：标准化的网络前向传播接口
- **模块化设计**：支持复杂的网络架构组合
- **配置管理**：通过配置类管理网络参数
- **类型安全**：基于 Pydantic 的配置验证

### LeNet 实现

LeNet 网络的具体实现：

```python
class LeNet(BaseNetwork):
    def __init__(self, config: LeNetConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config.num_classes)

    def forward(self, x):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # 全连接层
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 扩展指南

要添加新的网络架构：

1. 在 `network/` 下创建新文件夹
2. 继承 `BaseNetwork` 实现网络类（不强制）
3. 创建对应的配置类
4. 更新 `__init__.py` 导出新网络
