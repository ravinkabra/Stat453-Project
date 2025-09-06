# Utils Module

[English](README.md) | [中文](README_zh.md)

The utils module focuses on providing universal configuration base classes and utility functions, supporting dictionary access functionality for multiple configuration types.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes that are commonly used but rarely modified. For more information, please refer to [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_utils/
├── config_base.py      # Configuration base classes and dictionary access mixins
└── __init__.py         # Module initialization
```

## Core Components

### DictAccessMixin

Dictionary access mixin class - provides dictionary-style access interface for configuration classes:

```python
class DictAccessMixin:
    """Mixin class that provides dictionary access functionality for configuration classes"""

    def __getitem__(self, key: str) -> Any:
        """Support config['key'] access"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Support config['key'] = value assignment"""
        if self._has_field(key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __contains__(self, key: str) -> bool:
        """Support 'key' in config check"""
        return self._has_field(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Support config.get('key', default) access"""
        return getattr(self, key, default)

    def keys(self) -> Iterator[str]:
        """Return all field names"""
        return iter(self._get_field_names())

    def values(self) -> Iterator[Any]:
        """Return all field values"""
        return (getattr(self, key) for key in self.keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return all field (key, value) pairs"""
        return ((key, getattr(self, key)) for key in self.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary"""
        return dict(self.items())

    def update(self, other: Dict[str, Any]) -> None:
        """Batch update field values"""
        for key, value in other.items():
            self[key] = value
```

### TypedDictAccessMixin

Enhanced dictionary access mixin class - supports TypedDict type safety:

```python
class TypedDictAccessMixin(DictAccessMixin):
    """Enhanced mixin class with TypedDict support for better type safety"""

    def to_typed_dict(self) -> Dict[str, Any]:
        """Convert to TypedDict-compatible dictionary"""
        return self.to_dict()

    def update_from_typed_dict(self, typed_dict: Dict[str, Any]) -> None:
        """Update from TypedDict-compatible dictionary"""
        self.update(typed_dict)
```

### ConfigBase

Pydantic BaseModel-based configuration base class:

```python
class ConfigBase(BaseModel, DictAccessMixin):
    """Pydantic BaseModel-based configuration base class with dictionary access support"""

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
```

### DataclassConfigBase

Standard dataclass-based configuration base class:

```python
@dataclass
class DataclassConfigBase(DictAccessMixin):
    """Standard dataclass-based configuration base class with dictionary access support"""
    pass
```

## Supported Configuration Types

The module supports multiple configuration types with unified dictionary access interface:

1. **Pydantic DataClass** - Using `@pydantic_dataclass` decorator
2. **Standard DataClass** - Using `@dataclass` decorator
3. **Pydantic BaseModel** - Inheriting from `BaseModel`
4. **Custom Classes** - Through mixing in `DictAccessMixin`

## Main Features

- **Dictionary Access Interface**: Support for `config['key']`, `config['key'] = value` operations
- **Type Safety**: Support for TypedDict type hints and validation
- **Bidirectional Synchronization**: Automatic synchronization between attribute and dictionary access
- **Batch Operations**: Support for `update()`, `to_dict()` batch operations
- **Field Validation**: Integration with Pydantic field validation
- **Iteration Support**: Support for `keys()`, `values()`, `items()` iteration

## Usage Examples

### Basic Dictionary Access

```python
from src._utils.config_base import DictAccessMixin
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field

@pydantic_dataclass
class ModelConfig(DictAccessMixin):
    name: str = Field(default="model", description="Model name")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")

# Create configuration instance
config = ModelConfig()

# Dictionary-style access
print(config['name'])  # 'model'
config['learning_rate'] = 0.01
print(config.learning_rate)  # 0.01

# Batch update
config.update({
    'name': 'new_model',
    'batch_size': 64
})

# Convert to dictionary
config_dict = config.to_dict()
```

### TypedDict Type Safety

```python
from typing import TypedDict
from src._utils.config_base import TypedDictAccessMixin

class ModelConfigDict(TypedDict):
    name: str
    learning_rate: float
    batch_size: int

@pydantic_dataclass
class ModelConfig(TypedDictAccessMixin):
    name: str = Field(default="model")
    learning_rate: float = Field(default=0.001)
    batch_size: int = Field(default=32)

config = ModelConfig()

# Type-safe dictionary conversion
config_dict: ModelConfigDict = config.to_typed_dict()

# Update from TypedDict
config.update_from_typed_dict({
    'name': 'typed_model',
    'learning_rate': 0.005,
    'batch_size': 128
})
```

### Support for Different Configuration Types

```python
# Pydantic BaseModel
class MyConfig(ConfigBase):
    name: str
    value: int

config1 = MyConfig(name="test", value=42)
print(config1['name'])  # 'test'

# Standard DataClass
@dataclass
class MyDataclassConfig(DataclassConfigBase):
    name: str
    value: int

config2 = MyDataclassConfig(name="test", value=42)
print(config2['name'])  # 'test'
```

## Extension Guide

To extend the utils module functionality:

1. Add new mixin classes or base classes in `config_base.py`
2. Update type annotations and docstrings
3. Add corresponding test cases
4. Update usage examples and documentation
5. Maintain compatibility with existing configuration types
