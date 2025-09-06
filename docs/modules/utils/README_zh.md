# Utils Module

[English](README.md) | [中文](README_zh.md)

工具模块专注于提供通用的配置基础类和工具函数，支持多种配置类型的字典访问功能。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

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

基于数据类的配置基类：

```python
@dataclass
class DataclassConfigBase(DictAccessMixin):
    """基于数据类的配置基类"""
    pass
```

## 主要功能

### 字典访问接口

- **兼容性**：同时支持对象属性和字典访问
- **灵活性**：支持动态字段访问和修改
- **安全性**：字段存在性检查和类型验证
- **便利性**：提供常用的字典操作方法

### 配置管理

- **类型安全**：基于 Pydantic 的类型验证
- **数据转换**：自动处理数据类型转换
- **配置继承**：支持配置类的继承和扩展
- **序列化**：支持配置的序列化和反序列化

### 工具函数

- **字段操作**：批量字段更新和查询
- **数据转换**：配置对象与字典的相互转换
- **验证机制**：配置参数的自动验证
- **错误处理**：友好的错误提示和异常处理

## 使用示例

```python
from src._utils import DictAccessMixin, ConfigBase

# 使用字典访问混入类
@dataclass
class MyConfig(DictAccessMixin):
    learning_rate: float = 0.01
    batch_size: int = 32

config = MyConfig()
config['learning_rate'] = 0.001  # 字典风格访问
lr = config.get('learning_rate', 0.01)  # 获取值

# 使用配置基类
class ModelConfig(ConfigBase):
    lr: float = 0.01
    name: str = "model"

model_config = ModelConfig(lr=0.001, name="my_model")
```
