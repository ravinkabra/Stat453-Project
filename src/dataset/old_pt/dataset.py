from ..base.dataset import BaseDataset
from .config import OldPtDatasetConfig
from .model_input import OldPtModelInput
from typing import Optional, List
import torch


class OldPtDataset(BaseDataset):
    """Migrated OldPt dataset in project dataset style.

    Delayed setup to load backing .pt files and expose `setup()`/`collate_fn`.
    """

    def __init__(self, config: OldPtDatasetConfig):
        self.config = config
        self.file_path = config.file_path
        self.target_length = config.target_length
        self.split_ratio = config.split_ratio
        self.stage = config.stage

        # lazy-loaded
        self.length = None
        self.start = None
        self.data_acc = None
        self.data_mel = None
        self.pitch_shift_range_acc = None
        self.pitch_shift_range_mel = None
        self.valid_indices = None

    def setup(self, stage: Optional[str] = None):
        if stage is not None:
            self.stage = stage

        try:
            self.length = torch.load(self.file_path[:-3] + ".length.pt", mmap=True)
            self.start = torch.cumsum(self.length, dim=0) - self.length
            self.data_acc = torch.load(self.file_path, mmap=True)
            self.data_mel = torch.load(self.file_path.replace("acc", "mel"), mmap=True)
            self.pitch_shift_range_acc = torch.load(
                self.file_path[:-3] + ".pitch_shift_range.pt", mmap=True
            ).reshape(-1, 2)
            self.pitch_shift_range_mel = torch.load(
                self.file_path.replace("acc", "mel")[:-3]
                + ".pitch_shift_range.pt",
                mmap=True,
            ).reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(
                f"Error loading .pt files for {self.file_path} in OldPtDataset: {e}"
            )

        # clamp
        self.pitch_shift_range_acc[self.pitch_shift_range_acc[:, 0] < -5, 0] = -5
        self.pitch_shift_range_acc[self.pitch_shift_range_acc[:, 1] > 6, 1] = 6
        self.pitch_shift_range_mel[self.pitch_shift_range_mel[:, 0] < -5, 0] = -5
        self.pitch_shift_range_mel[self.pitch_shift_range_mel[:, 1] > 6, 1] = 6

        if self.stage in ("val", "test"):
            self.pitch_shift_range_acc = torch.zeros_like(self.pitch_shift_range_acc)
            self.pitch_shift_range_mel = torch.zeros_like(self.pitch_shift_range_mel)

        is_valid = self.length >= self.target_length
        self.valid_indices = torch.arange(len(self.start))[is_valid]

        if self.stage == "all":
            pass
        elif self.stage == "train":
            self.valid_indices = self.valid_indices[
                self.valid_indices % self.split_ratio != 0
            ]
        elif self.stage in ("val", "test"):
            self.valid_indices = self.valid_indices[
                self.valid_indices % self.split_ratio == 0
            ]

        self.valid_song_count = len(self.valid_indices)

    def __len__(self):
        if self.valid_indices is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        return int(self.valid_song_count)

    def __getitem__(self, idx) -> OldPtModelInput:
        if self.valid_indices is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        raw_id = int(self.valid_indices[idx])
        segment_relative_start = int(
            torch.floor(
                torch.rand(1) * (self.length[raw_id] - self.target_length)
            ).long()
        )
        actual_segment_start_in_full_data = segment_relative_start + int(
            self.start[raw_id]
        )
        index_vector = (
            torch.arange(self.target_length) + actual_segment_start_in_full_data
        )

        mel_segment = self.data_mel[index_vector]
        acc_segment = self.data_acc[index_vector]

        minmax = torch.minimum(
            self.pitch_shift_range_mel[raw_id, 1], self.pitch_shift_range_acc[raw_id, 1]
        )
        maxmin = torch.maximum(
            self.pitch_shift_range_mel[raw_id, 0], self.pitch_shift_range_acc[raw_id, 0]
        )
        single_pitch_shift = int(
            torch.floor(torch.rand(1) * (minmax - maxmin + 1)).long()
        ) + int(maxmin)

        return OldPtModelInput.from_tensors(
            mel=mel_segment,
            acc=acc_segment,
            pitch_shift=torch.tensor(single_pitch_shift),
        )

    def collate_fn(self, batch: List[OldPtModelInput]) -> OldPtModelInput:
        if not batch:
            raise ValueError("Empty batch")

        mel_data = torch.stack([item.mel_data for item in batch])
        acc_data = torch.stack([item.acc_data for item in batch])
        pitch_shift = torch.stack([item.pitch_shift for item in batch])

        return OldPtModelInput(
            mel_data=mel_data, acc_data=acc_data, pitch_shift=pitch_shift
        )

