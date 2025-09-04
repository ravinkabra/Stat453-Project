"""
MetricManager 使用演示
展示新的分组设计和使用方式
"""

import torch
import torch.nn.functional as F
from src.metric._manager.config import (
    MetricManagerConfig,
    ManagedMetricConfig,
    MetricLogConfig,
)
from src.metric._manager.manager import MetricManager


def demo_metric_manager():
    """演示 MetricManager 的新设计"""

    # 创建配置
    config = MetricManagerConfig(
        metrics={
            # 预测指标组
            "prediction_metrics": ManagedMetricConfig(
                metrics={
                    "train_accuracy": "torchmetrics.Accuracy",
                    "val_accuracy": "torchmetrics.Accuracy",
                    "precision": "torchmetrics.Precision",
                    "recall": "torchmetrics.Recall",
                },
                log_config=MetricLogConfig(
                    phase=["train", "val"], frequency=1, on_step=True, on_epoch=True
                ),
            ),
            # 值记录器组
            "value_recorders": ManagedMetricConfig(
                metrics={
                    "loss": "torchmetrics.MeanMetric",
                    "learning_rate": "torchmetrics.MeanMetric",
                },
                log_config=MetricLogConfig(
                    phase=["train", "val", "test"],
                    frequency=1,
                    on_step=True,
                    on_epoch=False,
                ),
            ),
        }
    )

    # 创建 MetricManager
    manager = MetricManager(config)

    print("=== MetricManager 初始化成功 ===")
    print(f"可用分组: {manager.list_groups()}")
    for group_name in manager.list_groups():
        print(f"  {group_name}: {manager.list_metrics_in_group(group_name)}")

    # 展示双层结构的直观访问
    print("\n=== 双层结构访问示例 ===")
    print(f"预测指标组中的指标: {list(manager.metrics['prediction_metrics'].keys())}")
    print(f"值记录器组中的指标: {list(manager.metrics['value_recorders'].keys())}")

    # 直接获取特定指标
    train_acc_metric = manager.get_metric("prediction_metrics", "train_accuracy")
    print(f"直接获取的训练准确率指标: {type(train_acc_metric).__name__}")

    # 模拟数据
    batch_size = 32
    num_classes = 10
    preds = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    probs = F.softmax(preds, dim=1)

    loss_value = F.cross_entropy(preds, target)
    lr_value = 0.001

    print("\n=== 演示统一的更新接口 ===")

    # 1. 更新所有指标（group_name=None，默认行为）
    print("1. 更新所有指标（默认行为）...")
    manager.update(preds=probs, target=target, loss=loss_value, learning_rate=lr_value)

    # 2. 更新特定分组
    print("2. 更新特定分组...")
    manager.update(group_name="prediction_metrics", preds=probs, target=target)
    manager.update(
        group_name="value_recorders",
        loss=loss_value * 0.9,
        learning_rate=lr_value * 1.1,
    )

    # 3. 混合更新 - 既有通用参数，又有按名称的参数
    print("3. 混合更新...")
    manager.update(
        group_name="prediction_metrics",
        preds=probs,
        target=target,
        # 即使在特定分组更新中，也可以按名称传递值
    )

    # 4. 向后兼容的方法仍然可用
    print("4. 使用向后兼容的 update_group...")
    manager.update_group(
        "value_recorders", loss=loss_value * 0.8, learning_rate=lr_value * 1.2
    )

    print("\n=== 指标结果 ===")

    # 展示分组计算的便利性
    print("预测指标组结果:")
    pred_results = manager.compute_group("prediction_metrics")
    for name, value in pred_results.items():
        if value is not None:
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: 无法计算")

    print("\n值记录器组结果:")
    value_results = manager.compute_group("value_recorders")
    for name, value in value_results.items():
        if value is not None:
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: 无法计算")

    print("\n=== 演示统一的重置接口 ===")

    # 1. 重置特定分组
    print("1. 重置预测指标组...")
    manager.reset(group_name="prediction_metrics")

    # 2. 重置所有指标
    print("2. 重置所有指标...")
    manager.reset()  # group_name=None，默认行为

    # 3. 向后兼容的方法
    print("3. 使用向后兼容的 reset_group...")
    manager.reset_group("prediction_metrics")

    # 验证重置效果 - 展示双层结构的清晰性
    print("\n重置后的预测指标组:")
    pred_results_after_reset = manager.compute_group("prediction_metrics")
    for name, value in pred_results_after_reset.items():
        print(f"  {name}: {'已重置' if value is None else f'{value:.4f}'}")


if __name__ == "__main__":
    demo_metric_manager()
