from .config import BaseDataModuleConfig
from hydra.utils import get_class
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from ...dataset import UnionDataset
from abc import ABC
from typing import Optional


class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, config: BaseDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        """Setup the data module for training/validation/testing."""
        if stage == "fit" or stage is None:
            if self.config.train:
                cls = get_class(self.config.train._target_)
                self.train_dataset: UnionDataset = cls(self.config.train)
                self.train_dataset.setup(stage="train")
            if self.config.val:
                cls = get_class(self.config.val._target_)
                self.val_dataset: UnionDataset = cls(self.config.val)
                self.val_dataset.setup(stage="validate")
        if stage == "validate" or stage is None:
            if self.config.val:
                cls = get_class(self.config.val._target_)
                self.val_dataset: UnionDataset = cls(self.config.val)
                self.val_dataset.setup(stage="validate")
        if stage == "test" or stage is None:
            if self.config.test:
                cls = get_class(self.config.test._target_)
                self.test_dataset: UnionDataset = cls(self.config.test)
                self.test_dataset.setup(stage="test")
        if stage == "predict" or stage is None:
            if self.config.predict:
                cls = get_class(self.config.predict._target_)
                self.predict_dataset: UnionDataset = cls(self.config.predict)
                self.predict_dataset.setup(stage="predict")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError(
                "Train dataset not initialized. Call setup('fit') first."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            sampler=(
                None if not self.config.train.sampler else self.config.train.sampler
            ),
            num_workers=self.config.train.num_workers,
            pin_memory=self.config.train.pin_memory,
            drop_last=self.config.train.drop_last,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup('fit') first."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val.batch_size,
            shuffle=self.config.val.shuffle,
            sampler=None if not self.config.val.sampler else self.config.val.sampler,
            num_workers=self.config.val.num_workers,
            pin_memory=self.config.val.pin_memory,
            drop_last=self.config.val.drop_last,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError(
                "Test dataset not initialized. Call setup('test') first."
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test.batch_size,
            shuffle=self.config.test.shuffle,
            sampler=None if not self.config.test.sampler else self.config.test.sampler,
            num_workers=self.config.test.num_workers,
            pin_memory=self.config.test.pin_memory,
            drop_last=self.config.test.drop_last,
            collate_fn=self.test_dataset.collate_fn,
        )

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        if self.predict_dataset is None:
            raise RuntimeError(
                "Predict dataset not initialized. Call setup('predict') first."
            )

        return DataLoader(
            self.predict_dataset,
            batch_size=self.config.predict.batch_size,
            shuffle=self.config.predict.shuffle,
            sampler=(
                None if not self.config.predict.sampler else self.config.predict.sampler
            ),
            num_workers=self.config.predict.num_workers,
            pin_memory=self.config.predict.pin_memory,
            drop_last=self.config.predict.drop_last,
            collate_fn=self.predict_dataset.collate_fn,
        )
