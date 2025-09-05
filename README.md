# Model Training Framework

[English](README.md) | [中文](README_zh.md)

A modular deep learning training framework based on PyTorch Lightning and Hydra, designed to provide flexible, scalable, and easy-to-configure machine learning model training solutions.

## 🏗️ Architecture Overview

This framework adopts a highly modular design, decoupling various components in the deep learning training process, with each component following unified interface design and configuration management patterns.

For detailed folder and file specifications, please refer to: [docs/architecture.md](docs/architecture.md).

```
src/
├── model/           # Model definition layer - integrates network, optimizer, scheduler, etc.
├── network/         # Network architecture layer - pure neural network definitions
├── optimizer/       # Optimizer layer - encapsulation of various optimization algorithms
├── lr_scheduler/    # Learning rate scheduler layer - learning rate adjustment strategies
├── metric/          # Metric management layer - metric calculation and recording during training
├── callback/        # Callback layer - custom behaviors during training
└── datamodule/      # Data module layer - data loading and preprocessing
```

## 🎯 Core Design Philosophy

### 1. Layered Architecture Design

The framework adopts a clear layered architecture, with each layer responsible for different duties:

- **Model Layer**: Highest level abstraction, integrates all components, defines training flow
- **Network Layer**: Network architecture definition, focuses on model structure
- **Component Layer**: Various functional components (optimizer, scheduler, metrics, etc.)
- **Data Layer**: Data processing and loading

### 2. Configuration-Driven Development

- Uses **Pydantic** for type-safe configuration management
- Integrates **Hydra** to support configuration file composition and command-line overrides
- Each component has a corresponding configuration class, supporting runtime instantiation

### 3. Plugin Extension Mechanism

- Design based on abstract base classes to ensure interface consistency
- Factory pattern component instantiation (via `_target_` field)
- Supports hot-swappable component replacement

## 🧩 Core Components Details

### Model Module

**Design Philosophy**: Model is the core coordinator of the framework, inheriting from PyTorch Lightning's `LightningModule`, responsible for integrating all components and defining the training flow.

```python
# Abstract base class design
class BaseModel(LightningModule, ABC):
    @abstractmethod
    def forward(self, batch):
        """Define forward propagation logic"""
        
    @abstractmethod
    def compute_loss(self, model_output, batch):
        """Define loss computation logic"""
        
    @abstractmethod
    def decode(self, model_output):
        """Define output decoding logic (for interpretability)"""
```

**Core Features**:
- Unified training/validation/testing step templates
- Automatic optimizer and learning rate scheduler configuration
- Integrated metric management system
- Supports distributed training (DDP)

### Network Module

**Design Philosophy**: Network focuses on pure neural network architecture definition, decoupled from training logic.

```python
# Network components independent of training logic
class BaseNetwork(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """Pure network forward propagation"""
```

### Metric Management System

**Design Philosophy**: Provides flexible metric calculation, recording, and visualization management.

**Core Components**:
```python
class MetricManager(torch.nn.Module):
    """DDP-aware metric manager"""
    
    def __init__(self, config: MetricManagerConfig):
        # Use ModuleDict to ensure DDP compatibility
        self.metrics = torch.nn.ModuleDict()
        self.log_configs = {}
```

**Features**:
- **DDP Compatible**: Uses `ModuleDict` to ensure distributed training compatibility
- **Flexible Configuration**: Supports metrics recording for different stages (train/val/test)
- **Customizable Logging**: Supports step-level and epoch-level logging
- **Automatic Aggregation**: Supports multiple aggregation methods (mean, sum, max, min)

### Callback System

**Design Philosophy**: Based on PyTorch Lightning's callback mechanism, provides custom behaviors during training.

**Example - OutputLoggerCallback**:
```python
class OutputLoggerCallback(Callback):
    """Output logging callback for model output visualization during training"""
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Decode model output and record
        decoded_preds = pl_module.decode(outputs["model_output"])
```

### Configuration Management System

**Design Philosophy**: Uses Pydantic + Hydra to build a type-safe, flexible configuration system.

```python
@dataclass
class BaseModelConfig:
    _target_: str = "src.model.base.model.BaseModel"  # Runtime instantiation target
    optimizer: Optional[UnionOptimizerParams] = None
    lr_scheduler: Optional[UnionLRSchedulerParams] = None
    metrics: Optional[MetricManagerConfig] = None
```

**Features**:
- **Type Safety**: Pydantic provides runtime type checking
- **Compositional Configuration**: Hydra supports configuration file composition and overrides
- **Factory Pattern**: Supports dynamic instantiation via `_target_` field

## 🔄 Data Flow and Execution Process

### 1. Configuration Phase
```
config.yaml → Hydra → Pydantic Config Classes → Component Instantiation
```

### 2. Training Phase
```
DataModule → Model.forward() → Loss Computation → 
MetricManager.update() → Optimizer.step() → Callback.on_*()
```

### 3. Metric Recording
```
Metric Computation → MetricManager.log() → Lightning Logger → 
External Tools (W&B, TensorBoard)
```

## 🚀 Usage Examples

### Basic Usage Flow

1. **Define your network architecture**:
```python
class MyNetwork(BaseNetwork):
    def forward(self, x):
        return self.layers(x)
```

2. **Implement your model**:
```python
class MyModel(BaseModel):
    def forward(self, batch):
        return self.network(batch["input"])
    
    def compute_loss(self, output, batch):
        return F.cross_entropy(output, batch["labels"])
    
    def decode(self, output):
        return torch.argmax(output, dim=-1)
```

3. **Configure training parameters**:
```yaml
# config.yaml
model:
  _target_: src.model.example.model.MyModel
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  metrics:
    accuracy:
      metric:
        _target_: torchmetrics.Accuracy
        task: multiclass
        num_classes: 10
```

4. **Start training**:
```python
# main.py
from hydra.utils import instantiate

model = instantiate(config.model)
trainer = pl.Trainer()
trainer.fit(model, datamodule)
```

## 🎨 Extension Guide

### Adding a New Optimizer
1. Create a new module under `src/optimizer/`
2. Inherit from base class and implement necessary methods
3. Update configuration type definitions

### Adding a New Metric
1. Use torchmetrics or custom metric class
2. Add new metric in MetricManager configuration
3. Configure recording strategy

### Adding a New Callback
1. Inherit from `pytorch_lightning.Callback`
2. Implement relevant hook functions
3. Register callback in trainer

## 🔧 Technical Features

- **🔥 PyTorch Lightning**: Automatically handles distributed training, checkpoints, logging, etc.
- **⚙️ Hydra**: Powerful configuration management and experiment organization
- **📊 Type Safety**: Pydantic provides runtime type checking
- **🔌 Plugin Architecture**: Highly modular, easy to extend
- **📈 DDP Ready**: Native support for distributed data parallel training
- **🎯 Metric Management**: Unified metric calculation and recording system

## 📁 Project Structure

```
model-training-framework/
├── src/                        # Source code directory
│   ├── model/                  # Model definitions
│   │   ├── base/              # Base abstract classes
│   │   └── example/           # Example implementations
│   ├── network/               # Network architectures
│   ├── optimizer/             # Optimizer components
│   ├── lr_scheduler/          # Learning rate schedulers
│   ├── metric/                # Metric management
│   │   ├── _manager/         # Metric manager
│   │   └── base/             # Base metric classes
│   ├── callback/              # Callback functions
│   └── datamodule/           # Data modules
├── config/                     # Configuration files directory
├── main.py                    # Training entry point
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```

## 🎯 Design Advantages

1. **High Cohesion, Low Coupling**: Each module has a single responsibility, clear dependencies
2. **Configuration-Driven**: Adjust training strategies without modifying code
3. **Easy to Test**: Modular design facilitates unit testing
4. **Production Ready**: Supports distributed training and experiment management
5. **Extensibility**: Plugin architecture, easy to add new features

## 🔮 Future Plans

- [ ] Complete data module implementation
- [ ] Add more optimizer and scheduler examples
- [ ] Integrate more experiment tracking tools
- [ ] Add model export and deployment features
- [ ] Improve unit tests and documentation

---

This framework embodies modern deep learning engineering best practices, providing a solid foundation for rapid prototyping and production deployment through reasonable abstraction and modular design.