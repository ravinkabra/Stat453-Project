# 基础训练教程

本教程将指导你使用框架完成一个完整的图像分类任务。

## 📋 目录

- [环境准备](#环境准备)
- [项目配置](#项目配置)
- [组件配置](#组件配置)
- [运行训练](#运行训练)
- [结果分析](#结果分析)

## 🚀 环境准备

确保已安装必要的依赖：

```bash
pip install torch torchvision pytorch-lightning hydra-core
```

## ⚙️ 项目配置

创建主配置文件 `config/config.yaml`：

```yaml
defaults:
  - model: image_classifier
  - network: resnet18
  - datamodule: cifar10
  - optimizer: adam
  - trainer: default

# 项目设置
project:
  name: "image-classification"
  experiment_name: "cifar10-resnet18"
  tags: ["classification", "cifar10", "resnet"]

# 随机种子
seed: 42
```

## 🧩 组件配置

### 网络配置 (`config/network/resnet18.yaml`)

```yaml
_target_: torchvision.models.resnet18
pretrained: false
num_classes: 10
```

> 💡 **更多网络选项**: 查看 [Network 组件文档](../components/network.md) 了解自定义网络

### 模型配置 (`config/model/image_classifier.yaml`)

```yaml
_target_: src.model.example.ImageClassifier

network: ${network}

optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 1e-4

loss_fn:
  _target_: torch.nn.CrossEntropyLoss

metrics:
  accuracy:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
  top5_acc:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
    top_k: 5
```

> 💡 **更多模型选项**: 查看 [Model 组件文档](../components/model.md) 了解高级功能

## 🏃 运行训练

### 基础训练

```bash
python main.py
```

### 自定义参数

```bash
# 修改学习率
python main.py model.optimizer.lr=0.01

# 使用不同网络
python main.py network=resnet50

# 修改训练轮数
python main.py trainer.max_epochs=50
```

### 多GPU训练

```bash
python main.py trainer.devices=2 trainer.strategy=ddp
```

## 📊 结果分析

训练完成后，你可以：

1. **查看训练日志**：
   ```
   outputs/2024-01-01/12-00-00/
   ├── .hydra/          # Hydra配置
   ├── checkpoints/     # 模型检查点
   └── logs/           # 训练日志
   ```

2. **加载最佳模型**：
   ```python
   from src.model.example import ImageClassifier
   
   model = ImageClassifier.load_from_checkpoint('path/to/best.ckpt')
   model.eval()
   ```

3. **进行推理**：
   ```python
   with torch.no_grad():
       predictions = model(test_images)
   ```

## 🔗 下一步

- [多GPU分布式训练](./multi-gpu-training.md)
- [自定义组件开发](./custom-components.md)
- [配置文件最佳实践](./config-best-practices.md)
