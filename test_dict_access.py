"""
全面测试 DictAccessMixin 对 pydantic dataclass 的支持
测试内容：
1. pydantic dataclass 支持
2. Field 支持和验证
3. 双向同步（属性 <-> 字典）
4. 各种边界情况
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config_base import DictAccessMixin
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field, ValidationError
from typing import Optional, List


def test_pydantic_dataclass_support():
    """测试 pydantic dataclass 支持"""
    print("=== 测试 Pydantic DataClass 支持 ===")

    @pydantic_dataclass
    class PydanticConfig(DictAccessMixin):
        name: str = Field(default="default", description="名称")
        age: int = Field(default=25, ge=0, le=120, description="年龄必须在0-120之间")
        email: Optional[str] = Field(default=None, description="可选邮箱")
        tags: List[str] = Field(default_factory=list, description="标签列表")
        score: float = Field(default=0.0, ge=0.0, le=100.0, description="分数0-100")

    # 测试1: 基本实例化
    config = PydanticConfig()
    print(f"✓ 基本实例化成功: {config.to_dict()}")

    # 测试2: 字段检测
    expected_fields = {"name", "age", "email", "tags", "score"}
    actual_fields = set(config.keys())
    assert (
        expected_fields == actual_fields
    ), f"字段检测失败: 期望{expected_fields}, 实际{actual_fields}"
    print(f"✓ 字段检测正确: {actual_fields}")

    # 测试3: 属性访问
    assert config.name == "default"
    print(f"✓ 属性访问正常: config.name = {config.name}")

    # 测试4: 字典访问
    assert config["age"] == 25
    print(f"✓ 字典访问正常: config['age'] = {config['age']}")

    return config


def test_field_validation():
    """测试 Field 验证功能"""
    print("\n=== 测试 Field 验证功能 ===")

    @pydantic_dataclass(config={"validate_assignment": True})
    class ValidatedConfig(DictAccessMixin):
        age: int = Field(ge=0, le=120, description="年龄限制")
        email: str = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$", description="邮箱格式验证")
        score: float = Field(ge=0.0, le=100.0, description="分数范围")

    config = ValidatedConfig(age=25, email="test@example.com", score=85.5)

    # 测试合法值
    config["age"] = 30
    assert config.age == 30
    print("✓ 合法值设置成功")

    # 测试非法年龄
    try:
        config["age"] = 150
        assert False, "应该抛出验证错误"
    except ValidationError as e:
        print(f"✓ 验证错误正确捕获: {str(e).split(chr(10))[0]}")

    # 测试非法邮箱
    try:
        config["email"] = "invalid-email"
        assert False, "应该抛出验证错误"
    except ValidationError as e:
        print(f"✓ 邮箱验证错误正确捕获: {str(e).split(chr(10))[0]}")

    # 验证当前值依然正确
    assert config.age == 30  # 应该还是30，因为150被拒绝了
    assert config.email == "test@example.com"  # 应该还是原值
    print("✓ 验证失败后值保持不变")


def test_bidirectional_sync():
    """测试双向同步功能"""
    print("\n=== 测试双向同步功能 ===")

    @pydantic_dataclass
    class SyncConfig(DictAccessMixin):
        name: str = Field(default="initial")
        value: int = Field(default=10)

    config = SyncConfig()

    # 测试1: 属性修改 -> 字典访问
    config.name = "modified_via_attr"
    assert config["name"] == "modified_via_attr"
    print("✓ 属性修改 -> 字典访问 同步成功")

    # 测试2: 字典修改 -> 属性访问
    config["value"] = 20
    assert config.value == 20
    print("✓ 字典修改 -> 属性访问 同步成功")

    # 测试3: 批量更新
    config.update({"name": "batch_update", "value": 30})
    assert config.name == "batch_update" and config.value == 30
    print("✓ 批量更新同步成功")


def test_dict_methods():
    """测试字典方法"""
    print("\n=== 测试字典方法 ===")

    @pydantic_dataclass
    class DictMethodsConfig(DictAccessMixin):
        a: str = Field(default="value_a")
        b: int = Field(default=100)
        c: Optional[str] = Field(default=None)

    config = DictMethodsConfig()

    # 测试 keys()
    keys = list(config.keys())
    assert set(keys) == {"a", "b", "c"}
    print(f"✓ keys() 方法正确: {keys}")

    # 测试 values()
    values = list(config.values())
    expected_values = ["value_a", 100, None]
    assert values == expected_values
    print(f"✓ values() 方法正确: {values}")

    # 测试 items()
    items = list(config.items())
    expected_items = [("a", "value_a"), ("b", 100), ("c", None)]
    assert items == expected_items
    print(f"✓ items() 方法正确: {items}")

    # 测试 in 操作符
    assert "a" in config
    assert "nonexistent" not in config
    print("✓ in 操作符正确")

    # 测试 get() 方法
    assert config.get("a") == "value_a"
    assert config.get("nonexistent", "default") == "default"
    print("✓ get() 方法正确")


def test_with_actual_config():
    """测试与实际项目配置的集成"""
    print("\n=== 测试与实际项目配置集成 ===")

    try:
        from src.model.base.config import BaseModelConfig

        # 创建增强版本
        class EnhancedConfig(BaseModelConfig, DictAccessMixin):
            pass

        config = EnhancedConfig()

        # 测试基本功能
        print(f"✓ 实际配置创建成功: {type(config)}")
        print(f"✓ 字段列表: {list(config.keys())}")

        # 测试字典访问
        target = config["_target_"]
        print(f"✓ 字典访问 _target_: {target}")

        # 测试修改和同步
        original = config._target_
        config["_target_"] = "test.target"
        assert config._target_ == "test.target"
        config._target_ = original  # 恢复
        assert config["_target_"] == original
        print("✓ 双向同步在实际配置中工作正常")

    except ImportError as e:
        print(f"⚠ 跳过实际配置测试（导入失败）: {e}")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    @pydantic_dataclass
    class EdgeCaseConfig(DictAccessMixin):
        normal_field: str = Field(default="test")

    config = EdgeCaseConfig()

    # 测试访问不存在的字段
    try:
        _ = config["nonexistent"]
        assert False, "应该抛出 KeyError"
    except KeyError:
        print("✓ 访问不存在字段正确抛出 KeyError")

    # 测试设置不存在的字段
    try:
        config["nonexistent"] = "value"
        assert False, "应该抛出 KeyError"
    except KeyError:
        print("✓ 设置不存在字段正确抛出 KeyError")


def run_comprehensive_test():
    """运行全面测试"""
    print("开始全面测试 DictAccessMixin 对 pydantic dataclass 的支持\n")

    try:
        test_pydantic_dataclass_support()
        test_field_validation()
        test_bidirectional_sync()
        test_dict_methods()
        test_with_actual_config()
        test_edge_cases()

        print("\n🎉 所有测试通过！")
        print("\n总结:")
        print("✓ 完全支持 pydantic.dataclasses.dataclass")
        print("✓ 完全支持 Field 定义和验证")
        print("✓ 属性和字典访问完全双向同步")
        print("✓ 所有字典方法正常工作")
        print("✓ 与现有项目配置兼容")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
