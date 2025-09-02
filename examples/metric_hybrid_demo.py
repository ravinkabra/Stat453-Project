"""
演示 MetricManager 的混合配置方案

展示如何使用四种不同方式配置 metrics：
1. BaseMetricParams (标准配置)
2. 直接实例化的 Metric
3. 字符串方式
4. 字典方式
"""

import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall

# 导入我们的配置类
from src.metric._manager.config import (
    MetricManagerConfig,
    ManagedMetricConfig,
    MetricLogConfig,
)
from src.metric.base.config import BaseMetricParams
from src.metric._manager.manager import MetricManager


def demo_hybrid_metric_configuration():
    """演示混合配置方案"""

    print("🎯 MetricManager 混合配置方案演示")
    print("=" * 50)

    # 创建不同类型的 metric 配置
    metrics_config = {
        # 方式1: 标准配置方式 (推荐用于配置文件)
        "accuracy_config": ManagedMetricConfig(
            metric=BaseMetricParams(
                _target_="torchmetrics.Accuracy",
                task="multiclass",
                num_classes=10,
                top_k=1,
            ),
            log_config=MetricLogConfig(
                phase=["train", "val"], prog_bar=True, on_epoch=True
            ),
        ),
        # 方式2: 直接实例化 (推荐用于代码开发)
        "f1_instance": ManagedMetricConfig(
            metric=F1Score(task="multiclass", num_classes=10, average="macro"),
            log_config=MetricLogConfig(
                phase=["val", "test"], frequency=2, prog_bar=False
            ),
        ),
        # 方式3: 字符串方式 (快速原型)
        "precision_string": ManagedMetricConfig(
            metric="torchmetrics.Precision",  # 使用默认参数
            log_config=MetricLogConfig(phase=["train"], on_step=False, on_epoch=True),
        ),
        # 方式4: 字典方式 (向后兼容)
        "recall_dict": ManagedMetricConfig(
            metric={
                "_target_": "torchmetrics.Recall",
                "task": "multiclass",
                "num_classes": 10,
                "average": "weighted",
            },
            log_config=MetricLogConfig(phase=["val"], reduce_fx="mean"),
        ),
    }

    # 创建 MetricManagerConfig
    manager_config = MetricManagerConfig(metrics=metrics_config)

    # 实例化 MetricManager
    print("📊 创建 MetricManager...")
    try:
        metric_manager = MetricManager(manager_config)
        print("✅ MetricManager 创建成功!")

        # 显示已注册的 metrics
        print(f"\n📈 已注册的 metrics: {list(metric_manager.metrics.keys())}")

        # 显示每个 metric 的类型
        print("\n🔍 Metrics 详细信息:")
        for name, metric in metric_manager.metrics.items():
            print(f"  - {name}: {type(metric).__name__}")

    except Exception as e:
        print(f"❌ 创建 MetricManager 失败: {e}")
        return

    # 测试 metric 更新
    print("\n🔄 测试 metrics 更新...")
    try:
        # 模拟预测和目标数据
        batch_size = 16
        num_classes = 10

        preds = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # 更新所有 metrics
        metric_manager.update(preds, targets)
        print("✅ Metrics 更新成功!")

        # 计算结果
        print("\n📊 当前 metrics 结果:")
        for name, metric in metric_manager.metrics.items():
            try:
                value = metric.compute()
                print(f"  - {name}: {value:.4f}")
            except Exception as e:
                print(f"  - {name}: 计算失败 ({e})")

    except Exception as e:
        print(f"❌ Metrics 更新失败: {e}")
        return

    print("\n🎉 混合配置方案演示完成!")
    print("\n💡 总结:")
    print("1. ✅ 配置方式: 支持 4 种不同的配置方式")
    print("2. ✅ 类型安全: 所有方式都能正确实例化")
    print("3. ✅ 灵活性: 开发者可以根据场景选择最适合的方式")
    print("4. ✅ 兼容性: 保持了向后兼容")


def demo_usage_scenarios():
    """演示不同使用场景"""

    print("\n" + "=" * 50)
    print("🎭 不同使用场景演示")
    print("=" * 50)

    # 场景1: 配置文件驱动 (生产环境)
    print("\n📄 场景1: 配置文件驱动")
    production_config = {
        "accuracy": ManagedMetricConfig(
            metric=BaseMetricParams(
                _target_="torchmetrics.Accuracy", task="multiclass", num_classes=1000
            )
        )
    }
    print("  适用于: 生产环境、Hydra 配置、可序列化需求")

    # 场景2: 快速开发 (原型阶段)
    print("\n⚡ 场景2: 快速开发")
    prototype_config = {
        "quick_accuracy": ManagedMetricConfig(
            metric=Accuracy(task="binary")  # 直接实例化
        ),
        "simple_f1": ManagedMetricConfig(metric="torchmetrics.F1Score"),  # 字符串方式
    }
    print("  适用于: 快速原型、Jupyter notebook、交互式开发")

    # 场景3: 复杂自定义 (研究场景)
    print("\n🔬 场景3: 复杂自定义")
    research_config = {
        "custom_accuracy": ManagedMetricConfig(
            metric=Accuracy(
                task="multiclass",
                num_classes=100,
                top_k=5,
                average="macro",
                multidim_average="samplewise",
            )
        )
    }
    print("  适用于: 研究实验、复杂参数配置、精细控制")


if __name__ == "__main__":
    demo_hybrid_metric_configuration()
    demo_usage_scenarios()
