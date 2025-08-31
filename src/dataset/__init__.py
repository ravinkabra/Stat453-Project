from typing import Union

from .base.config import BaseDatasetConfig
from .base.dataset import BaseDataset
from torch.utils.data import Dataset

# ChainDataset ...
UnionDatasetConfig = Union[BaseDatasetConfig, dict]
UnionDataset = Union[BaseDataset, Dataset]