# 📚 模型训练框架文档

欢迎使用基于 PyTorch Lightning 的模型训练框架！这是一个完整的文档导航。

## 🚀 快速开始

- [安装指南](./installation.md)
- [5分钟快速上手](./quickstart.md)
- [项目架构概览](./architecture.md)

## 📖 核心组件文档

### 🤖 模型相关
- [**Model 组件**](./components/model.md) - 模型定义与配置
- [**Network 组件**](./components/network.md) - 网络架构设计

### 🗃️ 数据相关
- [**DataModule 组件**](./components/datamodule.md) - 数据加载与处理
- [**Dataset 组件**](./components/dataset.md) - 数据集定义

### ⚙️ 训练相关
- [**Optimizer 组件**](./components/optimizer.md) - 优化器配置
- [**LR Scheduler 组件**](./components/lr_scheduler.md) - 学习率调度
- [**Metric 组件**](./components/metric.md) - 指标计算

### 🔧 系统相关
- [**Logger 组件**](./components/logger.md) - 日志记录
- [**Callback 组件**](./components/callback.md) - 训练回调
- [**Project Manager**](./components/project.md) - 项目管理

## 🎯 实践指南

- [基础训练流程](./examples/basic-training.md)
- [多GPU分布式训练](./examples/multi-gpu-training.md)
- [自定义组件开发](./examples/custom-components.md)
- [配置文件最佳实践](./examples/config-best-practices.md)

## 🔗 API 参考

- [配置类 API](./api/config.md)
- [基础类 API](./api/base-classes.md)
- [工具函数 API](./api/utilities.md)

## 🤝 贡献指南

- [开发环境搭建](./contributing/development.md)
- [代码规范](./contributing/code-style.md)
- [测试指南](./contributing/testing.md)

---

## 💡 提示

- 🔍 **组件快速参考**: 每个组件在源码目录下都有简短的 README，可以快速查看核心API
- 📚 **详细文档**: 本目录下的组件文档包含完整的使用说明和示例
- 🔗 **交叉引用**: 文档间通过链接相互关联，便于深入学习
