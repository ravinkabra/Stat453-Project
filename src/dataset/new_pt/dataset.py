from ..base.dataset import BaseDataset
from .config import NewPtDatasetConfig
from .model_input import NewPtModelInput
from typing import Optional, List
import torch


class NewPtDataset(BaseDataset):
    """Adapted NewPt dataset following project conventions.

    This implementation is ported from StreamMUSE's `new_pt_datamodule.py` and
    adapted to fit `src.dataset` patterns: delayed setup, `stage` handling,
    collate_fn as instance method, and returning `NewPtModelInput` objects.
    """

    def __init__(self, config: NewPtDatasetConfig):
        self.config = config
        self.file_path = config.file_path
        self.target_length = config.target_length
        self.split_ratio = config.split_ratio
        self.stage = config.stage
        self.sequence_shift = config.sequence_shift

        # lazy-loaded members
        self.length = None
        self.start = None
        self.input_acc_data_full = None
        self.input_mel_data_full = None
        self.pitch_shift_range_acc = None
        self.pitch_shift_range_mel = None
        self.valid_indices = None

    def setup(self, stage: Optional[str] = None):
        """Load backing .pt files and compute valid indices."""
        # allow external caller to override stage
        if stage is not None:
            self.stage = stage

        try:
            self.length = torch.load(self.file_path[:-3] + ".length.pt", mmap=True)
            self.start = torch.cumsum(self.length, dim=0) - self.length

            self.input_acc_data_full = torch.load(self.file_path, mmap=True)
            self.input_mel_data_full = torch.load(
                self.file_path.replace("acc", "mel"), mmap=True
            )
            self.pitch_shift_range_acc = torch.load(
                self.file_path[:-3] + ".pitch_shift_range.pt", mmap=True
            ).reshape(-1, 2)
            self.pitch_shift_range_mel = torch.load(
                self.file_path.replace("acc.pt", "mel.pt")[:-3]
                + ".pitch_shift_range.pt",
                mmap=True,
            ).reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(
                f"Error loading .pt files for {self.file_path} in NewPtDataset: {e}"
            )

        # clamp pitch ranges
        self.pitch_shift_range_acc[self.pitch_shift_range_acc[:, 0] < -5, 0] = -5
        self.pitch_shift_range_acc[self.pitch_shift_range_acc[:, 1] > 6, 1] = 6
        self.pitch_shift_range_mel[self.pitch_shift_range_mel[:, 0] < -5, 0] = -5
        self.pitch_shift_range_mel[self.pitch_shift_range_mel[:, 1] > 6, 1] = 6

        if self.stage in ("val", "test"):
            self.pitch_shift_range_acc = torch.zeros_like(self.pitch_shift_range_acc)
            self.pitch_shift_range_mel = torch.zeros_like(self.pitch_shift_range_mel)

        # validate lengths considering sequence_shift
        is_valid = (self.length - self.sequence_shift) >= self.target_length
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

    def __getitem__(self, idx) -> NewPtModelInput:
        if self.valid_indices is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        raw_id = int(self.valid_indices[idx])

        max_input_start_offset = int(
            self.length[raw_id] - self.target_length - self.sequence_shift
        )
        segment_relative_input_start = int(
            torch.floor(torch.rand(1) * (max_input_start_offset + 1)).long()
        )
        actual_input_start_in_full_data = segment_relative_input_start + int(
            self.start[raw_id]
        )
        actual_target_end_in_full_data = (
            actual_input_start_in_full_data + self.target_length + self.sequence_shift
        )

        combined_segment_start_index = actual_input_start_in_full_data
        combined_segment_end_index = actual_target_end_in_full_data

        combined_mel_segment = self.input_mel_data_full[
            combined_segment_start_index:combined_segment_end_index
        ]
        combined_acc_segment = self.input_acc_data_full[
            combined_segment_start_index:combined_segment_end_index
        ]

        minmax = torch.minimum(
            self.pitch_shift_range_mel[raw_id, 1], self.pitch_shift_range_acc[raw_id, 1]
        )
        maxmin = torch.maximum(
            self.pitch_shift_range_mel[raw_id, 0], self.pitch_shift_range_acc[raw_id, 0]
        )
        single_pitch_shift = int(
            torch.floor(torch.rand(1) * (minmax - maxmin + 1)).long()
        ) + int(maxmin)

        return NewPtModelInput.from_tensors(
            mel=combined_mel_segment,
            acc=combined_acc_segment,
            pitch_shift=torch.tensor(single_pitch_shift),
        )

    def collate_fn(self, batch: List[NewPtModelInput]) -> NewPtModelInput:
        if not batch:
            raise ValueError("Empty batch")

        mel_data = torch.stack([item.mel_data for item in batch])
        acc_data = torch.stack([item.acc_data for item in batch])
        pitch_shift = torch.stack([item.pitch_shift for item in batch])

        return NewPtModelInput(
            mel_data=mel_data, acc_data=acc_data, pitch_shift=pitch_shift
        )