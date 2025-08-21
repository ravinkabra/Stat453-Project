from .config import MetricLogConfig, ManagedMetricConfig, MetricManagerConfig

import torch
from torchmetrics import Metric
from hydra.utils import instantiate
from typing import TYPE_CHECKING, Mapping
from pytorch_lightning import LightningModule

# if TYPE_CHECKING:
#     from ...model.base.model import BaseModel


class MetricManager(torch.nn.Module):
    """
    Manages a collection of metrics, including their state, computation,
    and logging strategy.
    This module is DDP-aware thanks to its use of ModuleDict.
    """

    def __init__(self, config: MetricManagerConfig):
        super().__init__()
        self.config = config

        # ModuleDict is crucial! It registers the metrics as submodules,
        # making them visible to PyTorch Lightning for device placement and DDP.
        self.metrics: Mapping[str, Metric] = torch.nn.ModuleDict()

        # Store logging configs in a regular dict, as they don't have state.
        self.log_configs: dict[str, MetricLogConfig] = {}

        for name, managed_config in config.metrics.items():
            # Instantiate the actual torchmetric object, passing extra args like num_classes
            self.metrics[name] = instantiate(managed_config.metric)
            self.log_configs[name] = managed_config.log_config

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the state of all managed metrics."""
        for metric in self.metrics.values():
            metric.update(preds, target)

    def log(
        self, model: "LightningModule", phase: str = "train", current_step: int = 0
    ) -> None:
        # current_epoch = model.current_epoch
        for name, metric in self.metrics.items():
            log_config = self.log_configs[name]

            if phase not in log_config.phase:
                continue

            if (current_step + 1) % log_config.frequency != 0:
                continue

            if not log_config.on_step and not log_config.on_epoch:
                continue

            model.log(
                f"{phase}/{name}/step",
                metric,
                on_step=log_config.on_step,
                on_epoch=False,
                prog_bar=log_config.prog_bar,
                reduce_fx=log_config.reduce_fx,
            )
            
            model.log(
                f"{phase}/{name}/epoch",  
                metric,
                on_step=False,
                on_epoch=log_config.on_epoch,
                prog_bar=log_config.prog_bar,
                reduce_fx=log_config.reduce_fx,
            )
            # model.log(
            #     f"{phase}/{name}",
            #     metric,
            #     on_step=log_config.on_step,
            #     on_epoch=log_config.on_epoch,
            #     prog_bar=log_config.prog_bar,
            #     reduce_fx=log_config.reduce_fx,
            # )
    def reset(self) -> None:
        """Reset the state of all managed metrics."""
        for metric in self.metrics.values():
            metric.reset()
