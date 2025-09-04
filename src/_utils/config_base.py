"""
Universal configuration base classes with dictionary access support.
Supports pydantic dataclass, standard dataclass, and pydantic BaseModel.
"""

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import Any, Iterator, Dict, TypedDict, TypeVar, overload
from dataclasses import dataclass

# TypeVar for generic type support
T = TypeVar("T")


class TypedDictAccessMixin:
    """Enhanced mixin class with TypedDict support for better type safety.

    Provides both runtime dictionary access and compile-time type checking.
    Can work with TypedDict definitions for better IDE support and type checking.
    """

    def __getitem__(self, key: str) -> Any:
        """Support config['key'] access with type safety"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Support config['key'] = value assignment with validation"""
        if self._has_field(key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __contains__(self, key: str) -> bool:
        """Support 'key' in config check"""
        return self._has_field(key)

    @overload
    def get(self, key: str) -> Any: ...

    @overload
    def get(self, key: str, default: T) -> T: ...

    def get(self, key: str, default: Any = None) -> Any:
        """Support config.get('key', default) access with type preservation"""
        return getattr(self, key, default)

    def _has_field(self, key: str) -> bool:
        """Check if field exists (supports various dataclass types)"""
        # 1. Check pydantic dataclass
        if hasattr(self, "__pydantic_fields__"):
            return key in self.__pydantic_fields__
        # 2. Check standard dataclass
        elif hasattr(self, "__dataclass_fields__"):
            return key in self.__dataclass_fields__
        # 3. Check pydantic BaseModel
        elif hasattr(self, "model_fields"):
            return key in self.model_fields
        # 4. Other cases, check attributes
        else:
            return hasattr(self, key)

    def keys(self) -> Iterator[str]:
        """Return all field names"""
        # 1. pydantic dataclass
        if hasattr(self, "__pydantic_fields__"):
            return iter(self.__pydantic_fields__.keys())
        # 2. standard dataclass
        elif hasattr(self, "__dataclass_fields__"):
            return iter(self.__dataclass_fields__.keys())
        # 3. pydantic BaseModel
        elif hasattr(self, "model_fields"):
            return iter(self.model_fields.keys())
        # 4. other cases, return non-private attributes
        else:
            return iter(k for k in self.__dict__.keys() if not k.startswith("_"))

    def values(self) -> Iterator[Any]:
        """Return all field values"""
        return (getattr(self, key) for key in self.keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return all field (key, value) pairs"""
        return ((key, getattr(self, key)) for key in self.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary"""
        return dict(self.items())

    def to_typed_dict(self) -> Dict[str, Any]:
        """Convert to TypedDict-compatible dictionary for type-safe representation

        Returns a regular dict that can be typed as TypedDict in user code
        """
        return self.to_dict()

    def update(self, other: Dict[str, Any]) -> None:
        """Batch update field values"""
        for key, value in other.items():
            self[key] = value

    def update_from_typed_dict(self, typed_dict: Dict[str, Any]) -> None:
        """Update from TypedDict-compatible dictionary with type safety"""
        self.update(typed_dict)


class DictAccessMixin:
    """Mixin class to add dictionary access functionality to configuration classes.

    Supports:
    1. pydantic.dataclasses.dataclass
    2. Standard dataclass
    3. pydantic BaseModel
    4. Field-defined fields
    5. Bidirectional synchronization (attribute <-> dictionary access)
    """

    def __getitem__(self, key: str) -> Any:
        """Support config['key'] access"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Support config['key'] = value assignment

        Automatically syncs to object attributes, supports pydantic validation
        """
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

    def _has_field(self, key: str) -> bool:
        """Check if field exists (supports various dataclass types)"""
        # 1. Check pydantic dataclass
        if hasattr(self, "__pydantic_fields__"):
            return key in self.__pydantic_fields__
        # 2. Check standard dataclass
        elif hasattr(self, "__dataclass_fields__"):
            return key in self.__dataclass_fields__
        # 3. Check pydantic BaseModel
        elif hasattr(self, "model_fields"):
            return key in self.model_fields
        # 4. Other cases, check attributes
        else:
            return hasattr(self, key)

    def keys(self) -> Iterator[str]:
        """Return all field names"""
        # 1. pydantic dataclass
        if hasattr(self, "__pydantic_fields__"):
            return iter(self.__pydantic_fields__.keys())
        # 2. standard dataclass
        elif hasattr(self, "__dataclass_fields__"):
            return iter(self.__dataclass_fields__.keys())
        # 3. pydantic BaseModel
        elif hasattr(self, "model_fields"):
            return iter(self.model_fields.keys())
        # 4. other cases, return non-private attributes
        else:
            return iter(k for k in self.__dict__.keys() if not k.startswith("_"))

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


class ConfigBase(BaseModel, DictAccessMixin):
    """Pydantic BaseModel-based configuration base class with dictionary access support"""

    class Config:
        # Allow arbitrary types
        arbitrary_types_allowed = True
        # Validate assignments
        validate_assignment = True


# Support for dataclass as well
@dataclass
class DataclassConfigBase(DictAccessMixin):
    """Standard dataclass-based configuration base class with dictionary access support"""

    pass


# Pydantic dataclass base class
def create_pydantic_dataclass_base():
    """Create pydantic dataclass base class with dictionary access support"""

    @pydantic_dataclass
    class PydanticDataclassBase(DictAccessMixin):
        """Pydantic dataclass-based configuration base class with dictionary access support"""

        pass

    return PydanticDataclassBase


# Usage examples and tests
if __name__ == "__main__":
    print("=== Testing TypedDict Integration ===")

    # Define a TypedDict for type safety
    class ModelConfigDict(TypedDict):
        name: str
        learning_rate: float
        batch_size: int
        epochs: int
        use_gpu: bool
        tags: list[str]

    # Use with pydantic dataclass
    @pydantic_dataclass
    class ModelConfig(TypedDictAccessMixin):
        name: str = Field(default="test_model", description="Model name")
        learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
        batch_size: int = Field(default=32, gt=0, description="Batch size")
        epochs: int = Field(default=10, gt=0, description="Training epochs")
        use_gpu: bool = Field(default=True, description="Use GPU for training")
        tags: list[str] = Field(default_factory=list, description="Model tags")

    # Create instance
    config = ModelConfig()
    print(f"Initial config: {config.to_dict()}")

    # Test TypedDict-style access with type safety
    config_dict: ModelConfigDict = config.to_typed_dict()  # Type hint for IDE support
    print(f"As TypedDict: {config_dict}")

    # Modify using dictionary access
    config["learning_rate"] = 0.01
    config["tags"] = ["transformer", "nlp"]
    print(f"After modification: {config.to_dict()}")

    # Update from TypedDict-compatible data
    typed_update: ModelConfigDict = {
        "name": "bert_model",
        "learning_rate": 0.005,
        "batch_size": 64,
        "epochs": 20,
        "use_gpu": True,
        "tags": ["bert", "transformer"],
    }

    config.update_from_typed_dict(typed_update)
    print(f"After TypedDict update: {config.to_dict()}")

    # Type-safe get with default
    learning_rate: float = config.get("learning_rate", 0.001)
    model_name: str = config.get("name", "default")
    print(f"Type-safe access - LR: {learning_rate}, Name: {model_name}")

    print("\n=== Testing Original DictAccessMixin ===")
    from typing import Optional

    print("=== Testing Pydantic DataClass Support ===")

    # Create pydantic dataclass base class
    PydanticDataclassBase = create_pydantic_dataclass_base()

    @pydantic_dataclass
    class TestPydanticDataclass(DictAccessMixin):
        name: str = Field(default="test", description="Name")
        age: int = Field(default=25, ge=0, le=120, description="Age")
        email: Optional[str] = Field(default=None, description="Email")
        tags: list[str] = Field(default_factory=list, description="Tags")

    # Test instantiation
    config = TestPydanticDataclass()
    print(f"Initial config: {config.to_dict()}")

    # Test attribute access
    print(f"Attribute access name: {config.name}")

    # Test dictionary access
    print(f"Dictionary access age: {config['age']}")

    # Test bidirectional sync: attribute modification -> dictionary access
    config.name = "Alice"
    print(f"After attribute modification, dictionary access: {config['name']}")

    # Test bidirectional sync: dictionary modification -> attribute access
    config["age"] = 30
    print(f"After dictionary modification, attribute access: {config.age}")

    # Test Field validation (age must be between 0-120)
    try:
        config["age"] = 150  # Should fail
    except Exception as e:
        print(f"Validation failed (expected): {e}")

    # Test batch update
    config.update({"email": "alice@example.com", "tags": ["python", "ml"]})
    print(f"After batch update: {config.to_dict()}")

    # Test iteration
    print("Iterating all fields:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"Contains check 'name': {'name' in config}")
    print(f"Contains check 'nonexistent': {'nonexistent' in config}")

    # Test your actual config class
    print("\n=== Testing Actual BaseModelConfig ===")

    # First let BaseModelConfig support dictionary access
    # from src.model.base.config import BaseModelConfig

    # # Add dictionary access capability to BaseModelConfig
    # class EnhancedBaseModelConfig(BaseModelConfig, DictAccessMixin):
    #     pass

    # model_config = EnhancedBaseModelConfig()
    # print(f"Model config: {model_config.to_dict()}")
    # print(f"Dictionary access _target_: {model_config['_target_']}")

    # # Test modification
    # original_target = model_config._target_
    # model_config["_target_"] = "test.target"
    # print(f"Modified _target_: {model_config._target_}")

    # # Restore
    # model_config._target_ = original_target
    # print(f"Restored via dictionary access: {model_config['_target_']}")


# Usage examples
if __name__ == "__main__":
    from typing import Optional

    # Pydantic version
    class MyConfig(ConfigBase):
        name: str
        age: int
        email: Optional[str] = None

    config = MyConfig(name="Alice", age=25)

    # Attribute access
    print(f"Name: {config.name}")

    # Dictionary access
    print(f"Age: {config['age']}")

    # Set value
    config["email"] = "alice@example.com"
    print(f"Email: {config.email}")

    # Check existence
    print(f"Has name: {'name' in config}")

    # Iteration
    for key, value in config.items():
        print(f"{key}: {value}")

    # Convert to dictionary
    print(f"Dict: {config.to_dict()}")

    # dataclass version
    @dataclass
    class MyDataclassConfig(DataclassConfigBase):
        name: str
        age: int
        email: str = ""

    dc_config = MyDataclassConfig(name="Bob", age=30)

    print(f"Dataclass name: {dc_config['name']}")
