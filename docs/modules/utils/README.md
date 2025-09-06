# Utils Module

[English](README.md) | [中文](README_zh.md)

The utils module focuses on providing general configuration base classes and utility functions that support multiple configuration types with dictionary access functionality.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_utils/
├── config_base.py      # Configuration base classes and dictionary access mixin classes
└── __init__.py         # Module initialization
```

## Core Components

### DictAccessMixin

Dictionary access mixin class - provides dictionary-style access interfaces for configuration classes:

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
        """Return all (key, value) pairs of fields"""
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
    """Enhanced mixin class that supports TypedDict type safety"""

    def to_typed_dict(self) -> Dict[str, Any]:
        """Convert to TypedDict compatible dictionary"""
        return self.to_dict()

    def update_from_typed_dict(self, typed_dict: Dict[str, Any]) -> None:
        """Update from TypedDict compatible dictionary"""
        self.update(typed_dict)
```

### ConfigBase

Configuration base class based on Pydantic BaseModel:

```python
class ConfigBase(BaseModel, DictAccessMixin):
    """Configuration base class based on Pydantic BaseModel"""

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
```

### DataclassConfigBase

Configuration base class based on dataclasses:

```python
@dataclass
class DataclassConfigBase(DictAccessMixin):
    """Configuration base class based on dataclasses"""
    pass
```

## Main Features

### Dictionary Access Interface

- **Compatibility**: Supports both object attribute and dictionary access simultaneously
- **Flexibility**: Supports dynamic field access and modification
- **Security**: Field existence checks and type validation
- **Convenience**: Provides commonly used dictionary operation methods

### Configuration Management

- **Type safety**: Type validation and checking using Pydantic
- **Data conversion**: Automatic handling of data type conversion
- **Configuration inheritance**: Supports inheritance and extension of configuration classes
- **Serialization**: Supports serialization and deserialization of configurations

### Utility Functions

- **Field operations**: Batch field updates and queries
- **Data conversion**: Mutual conversion between configuration objects and dictionaries
- **Validation mechanisms**: Automatic validation of configuration parameters
- **Error handling**: Friendly error prompts and exception handling

## Usage Example

```python
from src._utils import DictAccessMixin, ConfigBase

# Use dictionary access mixin class
@dataclass
class MyConfig(DictAccessMixin):
    learning_rate: float = 0.01
    batch_size: int = 32

config = MyConfig()
config['learning_rate'] = 0.001  # Dictionary-style access
lr = config.get('learning_rate', 0.01)  # Get value

# Use configuration base class
class ModelConfig(ConfigBase):
    lr: float = 0.01
    name: str = "model"

model_config = ModelConfig(lr=0.001, name="my_model")
```
