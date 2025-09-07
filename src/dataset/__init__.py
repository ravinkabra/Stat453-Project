from typing import Union

from .base.config import BaseDatasetConfig
from .base.dataset import BaseDataset
from torch.utils.data import Dataset
from .tutorial_mnist.config import MnistDatasetConfig
from .tutorial_mnist.dataset import MnistDataset
from .new_pt.config import NewPtDatasetConfig
from .new_pt.dataset import NewPtDataset
from .mel_acc_remi_json.config import MelAccRemiJsonDatasetConfig
from .mel_acc_remi_json.dataset import MelAccRemiJsonDataset
from .old_pt.config import OldPtDatasetConfig
from .old_pt.dataset import OldPtDataset

# ChainDataset ...
UnionDatasetConfig = Union[
    BaseDatasetConfig,
    MnistDatasetConfig,
    NewPtDatasetConfig,
    OldPtDatasetConfig,
    MelAccRemiJsonDatasetConfig,
    dict,
]
UnionDataset = Union[
    BaseDataset,
    MnistDataset,
    NewPtDataset,
    OldPtDataset,
    MelAccRemiJsonDataset,
    Dataset,
]
