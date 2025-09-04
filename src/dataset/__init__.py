from typing import Union

from .base.config import BaseDatasetConfig
from .base.dataset import BaseDataset
from torch.utils.data import Dataset
from .tutorial_mnist.config import MnistDatasetConfig
from .tutorial_mnist.dataset import MnistDataset

# ChainDataset ...
UnionDatasetConfig = Union[BaseDatasetConfig, MnistDatasetConfig, dict]
UnionDataset = Union[BaseDataset, MnistDataset, Dataset]
