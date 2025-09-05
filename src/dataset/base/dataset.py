from hydra.utils import get_class
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from typing import Optional
from .model_input import BaseModelInput
from .config import BaseDatasetConfig


class BaseDataset(Dataset, ABC):
    def __init__(self, config: BaseDatasetConfig):
        self.config = config

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """Setup the dataset for train/validate/test/predict."""
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> BaseModelInput:
        pass

    def collate_fn(self, batch: list[BaseModelInput]) -> list[BaseModelInput]:
        """Custom collate function if needed."""
        return batch
