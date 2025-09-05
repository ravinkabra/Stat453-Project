import inspect
import torch
from torchmetrics import Metric
from hydra.utils import instantiate, get_class
from pytorch_lightning import LightningModule
from .config import MetricLogConfig, MetricManagerConfig

# from torchmetrics.utilities import dim_zero_cat


class MetricManager(torch.nn.Module):
    """
    Manages a collection of metrics, including their state, computation,
    and logging strategy.
    This module is DDP-aware thanks to its use of ModuleDict.
    """

    def __init__(self, config: MetricManagerConfig):
        super().__init__()
        self.config = config

        # 双层 ModuleDict 结构：group_name -> metric_name -> Metric
        # 保持 DDP 兼容性，每个分组都是一个 ModuleDict
        self.metrics: dict[str, torch.nn.ModuleDict] = {}

        # 分组级别的日志配置
        self.log_configs: dict[str, MetricLogConfig] = {}

        for group_name, managed_config in config.metrics.items():
            # 为每个分组创建一个 ModuleDict
            group_metrics = torch.nn.ModuleDict()

            # 处理 ManagedMetricConfig 中的指标字典
            for metric_name, metric_config in managed_config.metrics.items():
                metric = self._instantiate_metric(metric_config)
                group_metrics[metric_name] = metric

            # 注册分组 ModuleDict 为子模块（重要！保证 DDP 兼容性）
            self.add_module(group_name, group_metrics)
            self.metrics[group_name] = group_metrics
            self.log_configs[group_name] = managed_config.log_config

    def _instantiate_metric(self, metric_input) -> Metric:
        """
        根据不同的输入类型实例化 metric

        Args:
            metric_input: 可以是 BaseMetricParams, Metric, str, 或 dict

        Returns:
            实例化的 Metric 对象
        """
        if isinstance(metric_input, Metric):
            # 方式2: 直接传入已实例化的 Metric
            return metric_input

        elif isinstance(metric_input, str):
            # 方式3: 字符串方式，使用默认参数
            try:
                metric_class = get_class(metric_input)
                return metric_class()
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate metric from string '{metric_input}': {e}"
                )

        elif isinstance(metric_input, dict):
            # 方式4: 字典方式 (向后兼容)
            return instantiate(metric_input)

        else:
            # 方式1: BaseMetricParams 或其他配置对象
            return instantiate(metric_input)

    def update(
        self,
        model: LightningModule,
        batch_idx: int,
        group_name: str = None,
        preds=None,
        target=None,
        **kwargs,
    ) -> None:
        """
        支持额外参数的更新和选择性更新

        Args:
            group_name: 分组名称，如果为 None 则更新所有分组
            preds: 预测值 (可选，某些指标如 value recorder 不需要)
            target: 目标值 (可选，某些指标如 value recorder 不需要)
            **kwargs: 其他参数，支持：
                - 指标名=值: 直接为特定指标提供值 (如 loss=loss_value)
                - 其他参数会传递给需要额外参数的指标
        """
        # 确定要更新的分组
        if group_name is None:
            # 更新所有分组
            groups_to_update = self.metrics.items()
        else:
            # 更新特定分组
            if group_name not in self.metrics:
                raise ValueError(
                    f"Group '{group_name}' not found. Available groups: {list(self.metrics.keys())}"
                )
            groups_to_update = [(group_name, self.metrics[group_name])]

        for current_group_name, group_metrics in groups_to_update:
            log_config = self.log_configs[current_group_name]
            # current_global_step = model.global_step
            # if current_phase not in log_config.phase:
            #     continue

            if (batch_idx + 1) % log_config.update_frequency != 0:
                continue

            for metric_name, metric in group_metrics.items():
                metric: Metric

                # 优先使用指标名称匹配的值
                if metric_name in kwargs:
                    try:
                        metric.update(kwargs[metric_name])
                        # print(metric.values)
                        continue
                    except Exception as e:
                        print(
                            f"Warning: Failed to update {current_group_name}/{metric_name} with direct value: {e}"
                        )

                # 使用标准的 preds/target
                try:
                    if preds is not None and target is not None:
                        metric.update(preds, target)
                    elif preds is not None:
                        metric.update(preds)
                    else:
                        # 跳过需要输入但没有提供输入的指标
                        continue
                except TypeError:
                    # 如果失败，尝试带额外参数
                    metric_signature = inspect.signature(metric.update)
                    if len(metric_signature.parameters) > 2:
                        # 只传入metric需要的参数
                        valid_kwargs = {
                            k: v
                            for k, v in kwargs.items()
                            if k in metric_signature.parameters
                        }
                        try:
                            if preds is not None and target is not None:
                                metric.update(preds, target, **valid_kwargs)
                            elif preds is not None:
                                metric.update(preds, **valid_kwargs)
                            else:
                                metric.update(**valid_kwargs)
                        except Exception as e:
                            print(
                                f"Warning: Failed to update {current_group_name}/{metric_name}: {e}"
                            )

    def reset(self, group_name: str = None) -> None:
        """
        重置指标状态

        Args:
            group_name: 分组名称，如果为 None 则重置所有指标
        """
        if group_name is None:
            # 重置所有指标
            for group_metrics in self.metrics.values():
                for metric in group_metrics.values():
                    metric: Metric
                    metric.reset()
        else:
            # 重置特定分组
            if group_name not in self.metrics:
                raise ValueError(
                    f"Group '{group_name}' not found. Available groups: {list(self.metrics.keys())}"
                )
            for metric in self.metrics[group_name].values():
                metric: Metric
                metric.reset()

    def log(
        self, model: "LightningModule", phase: str = "train", batch_idx: int = 0
    ) -> None:
        """Log all metrics to the Lightning model."""
        for group_name, group_metrics in self.metrics.items():
            log_config = self.log_configs[group_name]

            if phase not in log_config.phase:
                continue

            if (batch_idx + 1) % log_config.compute_frequency != 0:
                continue

            if not log_config.on_step and not log_config.on_epoch:
                continue

            for metric_name, metric in group_metrics.items():
                metric: Metric
                metric_computed = metric.compute()
                if log_config.on_step and metric_computed is not None:
                    model.log(
                        f"{phase}/step/{metric_name}",
                        metric_computed,
                        on_step=log_config.on_step,
                        on_epoch=False,
                        prog_bar=log_config.prog_bar,
                        reduce_fx=log_config.reduce_fx,
                        sync_dist=log_config.sync_dist,
                        logger=True,
                    )
                if log_config.on_epoch and metric_computed is not None:
                    model.log(
                        f"{phase}/epoch/{metric_name}",
                        metric_computed,
                        on_step=False,
                        on_epoch=log_config.on_epoch,
                        prog_bar=log_config.prog_bar,
                        reduce_fx=log_config.reduce_fx,
                        sync_dist=log_config.sync_dist,
                        logger=True,
                    )

    def get_metric(self, group_name: str, metric_name: str) -> Metric:
        """
        获取特定的指标对象

        Args:
            group_name: 分组名称
            metric_name: 指标名称

        Returns:
            指标对象
        """
        if group_name not in self.metrics:
            raise ValueError(f"Group '{group_name}' not found")
        if metric_name not in self.metrics[group_name]:
            raise ValueError(
                f"Metric '{metric_name}' not found in group '{group_name}'"
            )
        return self.metrics[group_name][metric_name]

    def get_group_metrics(self, group_name: str) -> torch.nn.ModuleDict:
        """
        获取整个分组的指标

        Args:
            group_name: 分组名称

        Returns:
            分组的 ModuleDict
        """
        if group_name not in self.metrics:
            raise ValueError(f"Group '{group_name}' not found")
        return self.metrics[group_name]

    def list_groups(self) -> list[str]:
        """列出所有分组名称"""
        return list(self.metrics.keys())

    def list_metrics_in_group(self, group_name: str) -> list[str]:
        """
        列出指定分组中的所有指标名称

        Args:
            group_name: 分组名称

        Returns:
            指标名称列表
        """
        if group_name not in self.metrics:
            raise ValueError(f"Group '{group_name}' not found")
        return list(self.metrics[group_name].keys())
