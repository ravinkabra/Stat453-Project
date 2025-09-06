# 模块文档索引

[English](README.md) | [中文](README_zh.md)

本目录包含了模型训练框架所有模块的详细文档说明。

## 核心模块

### [Model Module](model/README_zh.md)

模型模块是深度学习训练框架的核心抽象层，基于 PyTorch Lightning 构建。

### [Network Module](network/README_zh.md)

网络模块专注于神经网络架构的定义。

### [Dataset Module](dataset/README_zh.md)

数据集模块专注于数据加载、预处理和模型输入的统一管理。

## 工具模块

### [Callback Module](callback/README_zh.md)

回调模块专注于 PyTorch Lightning 训练过程中的回调功能配置。

### [DataModule Module](datamodule/README_zh.md)

数据模块专注于数据加载和准备，与训练逻辑解耦。

### [Logger Module](logger/README_zh.md)

日志器模块专注于实验日志记录和管理。

### [LR Scheduler Module](lr_scheduler/README_zh.md)

学习率调度器模块专注于训练过程中的学习率调整策略。

### [Optimizer Module](optimizer/README_zh.md)

优化器模块专注于模型参数优化的配置和管理。

### [Trainer Module](trainer/README_zh.md)

训练器模块专注于 PyTorch Lightning Trainer 的配置和管理。

### [Utils Module](utils/README_zh.md)

工具模块提供通用的配置基础类和工具函数。

### [Project Module](project/README_zh.md)

项目模块专注于整个训练项目的组织和执行。

### [Metric Module](metric/README_zh.md)

指标模块专注于指标计算、记录和管理。

## 文档导航

- [项目架构规范](../../architecture_zh.md) - 整体项目规范和架构说明
- [快速开始](../../README.md) - 项目使用指南
- [配置说明](../../config/) - 配置示例和说明

## 贡献指南

如需为特定模块添加或更新文档，请：

1. 在对应的模块文件夹中编辑 `README_zh.md`
2. 保持文档格式的一致性
3. 更新此索引文件中的链接
4. 提交 Pull Request 进行审查
