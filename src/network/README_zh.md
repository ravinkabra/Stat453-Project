# Network Module

[English](README.md) | [中文](README_zh.md)

网络模块专注于神经网络架构的定义，与训练逻辑解耦。

## 目录结构

```
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

## 扩展指南

要添加新的网络架构：

1. 在 `network/` 下创建新文件夹
2. 继承 `BaseNetwork` 实现网络类（不强制）
3. 创建对应的配置类
4. 更新 `__init__.py` 导出新网络
