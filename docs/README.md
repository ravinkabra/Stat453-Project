# Documentation

[English](README.md) | [中文](README_zh.md)

Welcome to the Model Training Framework documentation! This directory contains comprehensive documentation for the framework, including architecture specifications, module guides, and usage examples.

## Documentation Structure

### 📋 Project Architecture

- **[Project Specifications (中文)](architecture_zh.md)** - Project folder and file naming conventions, design principles
- **[Project Specifications (English)](architecture.md)** - English version of project specifications

### 📚 Module Documentation

- **[Modules Overview (中文)](modules/README_zh.md)** - Index of all framework modules in Chinese
- **[Modules Overview (English)](modules/README.md)** - Index of all framework modules in English

#### Core Modules

- **[Model Module](modules/model/)** - Core model abstraction layer
- **[Network Module](modules/network/)** - Neural network architecture definitions
- **[Dataset Module](modules/dataset/)** - Data loading and preprocessing

#### Tool Modules

- **[Callback Module](modules/callback/)** - Training callbacks and hooks
- **[DataModule Module](modules/datamodule/)** - Data module management
- **[Logger Module](modules/logger/)** - Experiment logging and tracking
- **[LR Scheduler Module](modules/lr_scheduler/)** - Learning rate scheduling
- **[Optimizer Module](modules/optimizer/)** - Model optimization algorithms
- **[Trainer Module](modules/trainer/)** - Training process configuration
- **[Utils Module](modules/utils/)** - Utility functions and base classes
- **[Project Module](modules/project/)** - Project management and orchestration
- **[Metric Module](modules/metric/)** - Metrics calculation and monitoring

## Quick Start

1. **New to the framework?** Start with the [Project Specifications](architecture_zh.md) to understand the design principles
2. **Looking for specific modules?** Check the [Modules Overview](modules/README_zh.md) for detailed guides
3. **Need implementation examples?** Each module contains usage examples and configuration samples

## Language Support

All documentation is available in both languages:

- 🇨🇳 **Chinese (中文)**: `README_zh.md` files
- 🇺🇸 **English**: `README.md` files

## Contributing to Documentation

We welcome contributions to improve the documentation! Here's how you can help:

### For Module Documentation

1. Navigate to the specific module directory under `modules/`
2. Edit the corresponding `README.md` (English) or `README_zh.md` (Chinese)
3. Follow the established format and include:
   - Module overview and purpose
   - Directory structure
   - Core components and APIs
   - Usage examples
   - Extension guidelines

### For General Documentation

1. Edit files in the root `docs/` directory
2. Maintain consistency with existing documentation style
3. Update links and references as needed

### Guidelines

- Keep technical content accurate and up-to-date
- Include code examples where helpful
- Use clear, concise language
- Test links and cross-references
- Follow Markdown best practices

## Support

If you have questions about the documentation or need help with the framework:

- Check the [Modules Overview](modules/README_zh.md) for detailed guides
- Review the [Project Specifications](architecture_zh.md) for design principles
- Explore the source code in the `src/` directory for implementation details

---

*This documentation is maintained by the Model Training Framework development team.*
