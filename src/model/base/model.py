from pytorch_lightning import LightningModule
from hydra.utils import instantiate, get_class
from abc import ABC, abstractmethod
from typing import Any, Optional, Mapping, Union
from pathlib import Path
from .config import BaseModelConfig
from ...metric._manager.manager import MetricManager
from dataclasses import asdict
import inspect


class BaseModel(LightningModule, ABC):
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.metric_manager = MetricManager(config.metric_manager)

    @abstractmethod
    def forward(self, batch):
        """
        Abstract method for the forward pass. Must be implemented by subclasses.
        Should return the raw model output (logits).
        """
        raise NotImplementedError

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
        # scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer)
        # valid_kwargs = {k: v for k, v in asdict(self.config.lr_scheduler).items() if k != "_target_"}
        # scheduler = get_class(self.config.lr_scheduler._target_)(optimizer=optimizer, **valid_kwargs)
        scheduler_cls = get_class(self.config.lr_scheduler._target_)
        valid_kwargs = {
            k: v
            for k, v in asdict(self.config.lr_scheduler).items()
            if k != "_target_"
            and k != "optimizer"
            and k in inspect.signature(scheduler_cls.__init__).parameters
        }
        scheduler = scheduler_cls(optimizer=optimizer, **valid_kwargs)

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

    # @abstractmethod
    # def decode(self, model_output) -> Any:
    #     """
    #     Abstract method to decode the model output into a more interpretable format.
    #     Must be implemented by subclasses.
    #     """
    #     raise NotImplementedError

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
        Will be called by SampleSaverCallback.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _training_forward(self, inputs):
        """训练时的前向传播，子类可重写处理细节差异"""
        return self.forward(inputs)

    def _inference_forward(self, inputs):
        """推理时的前向传播，子类可重写处理细节差异"""
        return self.forward(inputs)
