# Utils Module

[English](README.md) | [中文](README_zh.md)

工具模块专注于提供通用的配置基础类和工具函数，支持多种配置类型的字典访问功能。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../docs/architecture_zh.md)。

## 目录结构

```text
_utils/
├── config_base.py      # 配置基础类和字典访问混入类
└── __init__.py         # 模块初始化
```

## 核心组件

### DictAccessMixin

字典访问混入类 - 为配置类提供字典风格的访问接口：

```python
class DictAccessMixin:
    """为配置类提供字典访问功能的混入类"""

    def __getitem__(self, key: str) -> Any:
        """支持 config['key'] 访问"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """支持 config['key'] = value 赋值"""
        if self._has_field(key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __contains__(self, key: str) -> bool:
        """支持 'key' in config 检查"""
        return self._has_field(key)

    def get(self, key: str, default: Any = None) -> Any:
        """支持 config.get('key', default) 访问"""
        return getattr(self, key, default)

    def keys(self) -> Iterator[str]:
        """返回所有字段名"""
        return iter(self._get_field_names())

    def values(self) -> Iterator[Any]:
        """返回所有字段值"""
        return (getattr(self, key) for key in self.keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        """返回所有字段 (key, value) 对"""
        return ((key, getattr(self, key)) for key in self.keys())

    def to_dict(self) -> Dict[str, Any]:
        """转换为标准字典"""
        return dict(self.items())

    def update(self, other: Dict[str, Any]) -> None:
        """批量更新字段值"""
        for key, value in other.items():
            self[key] = value
```

### TypedDictAccessMixin

增强的字典访问混入类 - 支持 TypedDict 类型安全：

```python
class TypedDictAccessMixin(DictAccessMixin):
    """增强的混入类，支持 TypedDict 类型安全"""

    def to_typed_dict(self) -> Dict[str, Any]:
        """转换为 TypedDict 兼容的字典"""
        return self.to_dict()

    def update_from_typed_dict(self, typed_dict: Dict[str, Any]) -> None:
        """从 TypedDict 兼容的字典更新"""
        self.update(typed_dict)
```

### ConfigBase

基于 Pydantic BaseModel 的配置基类：

```python
class ConfigBase(BaseModel, DictAccessMixin):
    """基于 Pydantic BaseModel 的配置基类"""

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
```

### DataclassConfigBase

基于标准 dataclass 的配置基类：

```python
@dataclass
class DataclassConfigBase(DictAccessMixin):
    """基于标准 dataclass 的配置基类"""
    pass
```

## 支持的配置类型

模块支持多种配置类型，提供统一的字典访问接口：

1. **Pydantic DataClass** - 使用 `@pydantic_dataclass` 装饰器
2. **标准 DataClass** - 使用 `@dataclass` 装饰器
3. **Pydantic BaseModel** - 继承自 `BaseModel`
4. **自定义类** - 通过混入 `DictAccessMixin`

## 主要功能

- **字典访问接口**：支持 `config['key']`、`config['key'] = value` 等操作
- **类型安全**：支持 TypedDict 类型提示和验证
- **双向同步**：属性访问和字典访问自动同步
- **批量操作**：支持 `update()`、`to_dict()` 等批量操作
- **字段验证**：集成 Pydantic 的字段验证功能
- **迭代支持**：支持 `keys()`、`values()`、`items()` 迭代

## 使用示例

### 基本字典访问

```python
from src._utils.config_base import DictAccessMixin
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field

@pydantic_dataclass
class ModelConfig(DictAccessMixin):
    name: str = Field(default="model", description="模型名称")
    learning_rate: float = Field(default=0.001, description="学习率")
    batch_size: int = Field(default=32, description="批次大小")

# 创建配置实例
config = ModelConfig()

# 字典风格访问
print(config['name'])  # 'model'
config['learning_rate'] = 0.01
print(config.learning_rate)  # 0.01

# 批量更新
config.update({
    'name': 'new_model',
    'batch_size': 64
})

# 转换为字典
config_dict = config.to_dict()
```

### TypedDict 类型安全

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

# 类型安全的字典转换
config_dict: ModelConfigDict = config.to_typed_dict()

# 从 TypedDict 更新
config.update_from_typed_dict({
    'name': 'typed_model',
    'learning_rate': 0.005,
    'batch_size': 128
})
```

### 不同配置类型的支持

```python
# Pydantic BaseModel
class MyConfig(ConfigBase):
    name: str
    value: int

config1 = MyConfig(name="test", value=42)
print(config1['name'])  # 'test'

# 标准 DataClass
@dataclass
class MyDataclassConfig(DataclassConfigBase):
    name: str
    value: int

config2 = MyDataclassConfig(name="test", value=42)
print(config2['name'])  # 'test'
```

## 扩展指南

要扩展工具模块功能：

1. 在 `config_base.py` 中添加新的混入类或基类
2. 更新类型注解和文档字符串
3. 添加相应的测试用例
4. 更新使用示例和文档
5. 保持与现有配置类型的兼容性
