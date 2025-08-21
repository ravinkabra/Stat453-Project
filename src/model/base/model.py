import torch
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall
from abc import ABC, abstractmethod
from typing import Any, Optional, Mapping
from .config import BaseModelConfig


class BaseModel(LightningModule, ABC):
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        # self._setup_metrics()

    # def _setup_metrics(self):
    #     """
    #     Initialize metrics for training, validation, and testing.
    #     This is an example for a classification task. You should override this
    #     method in your specific model to define task-appropriate metrics.
    #     """
    #     # Example: get num_classes from config, assuming it's defined there.
    #     # You might need to add `num_classes` to your model's specific config.
    #     num_classes = getattr(self.config, "num_classes", 10)

    #     metrics = MetricCollection(
    #         {
    #             "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
    #             "precision_macro": Precision(task="multiclass", num_classes=num_classes, average="macro"),
    #             "recall_macro": Recall(task="multiclass", num_classes=num_classes, average="macro"),
    #         }
    #     )

    #     # Create separate metric instances for each phase to avoid conflicts
    #     self.train_metrics = metrics.clone(prefix="train_")
    #     self.val_metrics = metrics.clone(prefix="val_")
    #     self.test_metrics = metrics.clone(prefix="test_")

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
        self.train_metrics.update(output, batch["labels"])  # Assuming batch has 'labels'
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
        if not self.config.lr_scheduler or not getattr(self.config.lr_scheduler, "_target_", None):
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
    def decode(self,model_output)->Any:
        """
        Abstract method to decode the model output into a more interpretable format.
        Must be implemented by subclasses.
        """
        raise NotImplementedError