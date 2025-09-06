# Module Documentation Index

[English](README.md) | [中文](README_zh.md)

This directory contains detailed documentation for all modules in the model training framework.

## Core Modules

### [Model Module](model/README.md)

The model module is the core abstraction layer of the deep learning training framework, built on PyTorch Lightning.

### [Network Module](network/README.md)

The network module focuses on neural network architecture definitions.

### [Dataset Module](dataset/README.md)

The dataset module focuses on unified management of data loading, preprocessing, and model input.

## Tool Modules

### [Callback Module](callback/README.md)

The callback module focuses on callback function configuration during PyTorch Lightning training.

### [DataModule Module](datamodule/README.md)

The datamodule focuses on data loading and preparation, decoupled from training logic.

### [Logger Module](logger/README.md)

The logger module focuses on experiment logging and management.

### [LR Scheduler Module](lr_scheduler/README.md)

The learning rate scheduler module focuses on learning rate adjustment strategies during training.

### [Optimizer Module](optimizer/README.md)

The optimizer module focuses on configuration and management of model parameter optimization.

### [Trainer Module](trainer/README.md)

The trainer module focuses on PyTorch Lightning Trainer configuration and management.

### [Utils Module](utils/README.md)

The utils module provides general configuration base classes and utility functions.

### [Project Module](project/README.md)

The project module focuses on the organization and execution of the entire training project.

### [Metric Module](metric/README.md)

The metric module focuses on metric calculation, recording, and management.

## Documentation Navigation

- [Project Architecture Specification](../../architecture.md) - Overall project specifications and architecture description
- [Quick Start](../../README.md) - Project usage guide
- [Configuration Examples](../../config/) - Configuration examples and instructions

## Contribution Guidelines

To add or update documentation for a specific module, please:

1. Edit the `README.md` in the corresponding module folder
2. Maintain consistency in documentation format
3. Update the links in this index file
4. Submit a Pull Request for review
