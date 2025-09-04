from .config import MnistModelConfig
from ..base.model import BaseModel
from ...network.tutorial_lenet.network import LeNet
from ...dataset.tutorial_mnist.model_input import MnistModelInput

import torch.nn as nn
import json

# from typing import overload
from typing_extensions import override
import torch


class MnistModel(BaseModel):
    def __init__(self, config: MnistModelConfig):
        super().__init__(config)
        self.network = LeNet(config.network)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch: MnistModelInput) -> MnistModelInput:
        return self.network(batch)

    def _training_forward(self, inputs):
        return self.network(inputs)

    def _inference_forward(self, inputs):
        return self.network(inputs)

    def training_step(self, batch: MnistModelInput, batch_idx: int):
        image = batch.image
        label = batch.label
        logits = self._training_forward(image)
        loss = self.loss(logits, label)
        self.metric_manager.forward(
            "training_state",
            loss=loss.mean(),
            learning_rate=self.lr_schedulers().get_last_lr()[0],
        )
        self.metric_manager.log(self, phase="train", current_step=self.global_step)
        return {"loss": loss, "preds": logits}

    def on_train_epoch_end(self):
        # self.metric_manager.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch: MnistModelInput, batch_idx: int):
        image = batch.image
        label = batch.label
        logits = self._inference_forward(image)
        loss = self.loss(logits, label)
        self.metric_manager.forward("training_state", loss=loss.mean())
        self.metric_manager.forward("classification", target=label, preds=logits)
        # self.metric_manager.update("training_state", loss=loss)
        # self.metric_manager.update("classification", target=label, preds=logits)
        self.metric_manager.log(self, phase="val", current_step=self.global_step)
        return {"loss": loss, "preds": logits}

    def on_validation_epoch_end(self):
        # self.metric_manager.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch: MnistModelInput, batch_idx: int):
        image = batch.image
        label = batch.label
        logits = self._inference_forward(image)
        loss = self.loss(logits, label)
        self.metric_manager.update("training_state", loss=loss)
        self.metric_manager.update("classification", target=label, preds=logits)
        self.metric_manager.log(self, phase="test", current_step=self.global_step)
        return {"loss": loss, "preds": logits}

    def predict_step(
        self, batch: MnistModelInput, batch_idx: int, dataloader_idx: int = 0
    ):
        return super().predict_step(batch, batch_idx, dataloader_idx)

    @override
    def decode(self, model_output: torch.Tensor) -> torch.Tensor:
        model_output = model_output.detach()
        return model_output.argmax(dim=1)

    @override
    def save(self, data: torch.Tensor, filename_base, extension, save_dir, **kwargs):
        decoded_data = self.decode(data)
        json_data = json.dumps(decoded_data.tolist())
        with open(f"{save_dir}/{filename_base}{extension}", "w") as f:
            f.write(json_data)
        return super().save(decoded_data, filename_base, extension, save_dir, **kwargs)
