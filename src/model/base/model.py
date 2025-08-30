import torch
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall
from abc import ABC, abstractmethod
from typing import Any, Optional, Mapping, Union
from pathlib import Path
from .config import BaseModelConfig


class BaseModel(LightningModule, ABC):
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch):
        """
        Abstract method for the forward pass. Must be implemented by subclasses.
        Should return the raw model output (logits).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, model_output, batch) -> torch.Tensor:
        """
        Abstract method to compute the loss. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _step(self, batch, batch_idx):
        """
        Generic step for training, validation, and test.
        Returns the loss and model output.
        """
        model_output = self.forward(batch)
        loss = self.compute_loss(model_output, batch)
        return loss, model_output

    def training_step(self, batch, batch_idx):
        loss, output = self._step(batch, batch_idx)
        self.train_metrics.update(
            output, batch["labels"]
        )  # Assuming batch has 'labels'
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output = self._step(batch, batch_idx)
        self.val_metrics.update(output, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, output = self._step(batch, batch_idx)
        self.test_metrics.update(output, batch["labels"])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Instantiate and configure the optimizer and learning rate scheduler."""
        # Instantiate optimizer, passing model parameters as an override
        # This avoids modifying the config object
        optimizer = instantiate(self.config.optimizer, params=self.parameters())

        # If no scheduler is defined in the config, just return the optimizer
        if not self.config.lr_scheduler or not getattr(
            self.config.lr_scheduler, "_target_", None
        ):
            return optimizer

        # Instantiate scheduler, passing the created optimizer instance
        scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer)

        # Build the scheduler dictionary for PyTorch Lightning
        # We extract only the relevant lightning-specific keys from the config
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": getattr(self.config.lr_scheduler, "interval", "epoch"),
            "frequency": getattr(self.config.lr_scheduler, "frequency", 1),
            "monitor": getattr(self.config.lr_scheduler, "monitor", "val_loss"),
            "strict": getattr(self.config.lr_scheduler, "strict", True),
            "name": getattr(self.config.lr_scheduler, "name", None),
        }

        # For schedulers like ReduceLROnPlateau, the monitor key is essential.
        # For others, it might not be present, so we only add it if it's not None.
        if lr_scheduler_config["monitor"] is None:
            del lr_scheduler_config["monitor"]

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    @abstractmethod
    def decode(self, model_output) -> Any:
        """
        Abstract method to decode the model output into a more interpretable format.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


    @abstractmethod
    def save(
        self,
        data: Any,
        filename_base: str,
        extension: str,
        save_dir: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Abstract method to save the model output.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
