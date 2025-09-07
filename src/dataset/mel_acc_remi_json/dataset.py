"""Remi-json dataset adapted to project conventions.

Ported from StreamMUSE's remi_json_datamodule.py but shaped to the
`src/dataset/*` package style: BaseDataset subclass, delayed
`setup()` and an instance `collate_fn`.
"""

import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..base.dataset import BaseDataset
from .config import MelAccRemiJsonDatasetConfig
from .model_input import MelAccRemiJsonModelInput


BAR_NONE_TOKEN_ID = 4
EXTRACT_BAR_NUM = 4
ACC_BAR_BIAS = 1


def random_clip_and_pad(
    mel_sequence: np.ndarray,
    acc_sequence: np.ndarray,
    target_length: int,
    padding_token_id: int = 0,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    padding_probability: float = 0.0,
    exp_bias_lambda: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bar_indices = np.where(mel_sequence == BAR_NONE_TOKEN_ID)[0]

    mel_start_bar = (
        np.random.choice(bar_indices[: -EXTRACT_BAR_NUM + 1])
        if len(bar_indices) >= EXTRACT_BAR_NUM
        else 0
    )
    mel_end_bar = mel_start_bar + EXTRACT_BAR_NUM
    mel_seq = mel_sequence[mel_start_bar:mel_end_bar]
    acc_start_bar = (
        mel_start_bar + ACC_BAR_BIAS
        if mel_start_bar + ACC_BAR_BIAS < len(acc_sequence)
        else mel_start_bar
    )
    acc_end_bar = (
        mel_end_bar + ACC_BAR_BIAS
        if mel_end_bar + ACC_BAR_BIAS < len(acc_sequence)
        else mel_end_bar
    )
    acc_seq = acc_sequence[acc_start_bar:acc_end_bar]

    max_content_len = target_length - 2

    use_random_shortening = np.random.rand() < padding_probability
    len_shortening_rand_val = np.random.rand()

    def process_sequence(seq: np.ndarray) -> torch.Tensor:
        if use_random_shortening:
            content_len = (
                int(
                    max_content_len
                    * (1 - np.exp(-len_shortening_rand_val * exp_bias_lambda))
                )
                + 1
            )
            content_len = np.clip(content_len, 1, max_content_len)
        else:
            content_len = max_content_len

        if len(seq) > content_len:
            start_index = np.random.randint(0, len(seq) - content_len + 1)
            clipped_seq = seq[start_index : start_index + content_len]
        else:
            clipped_seq = seq

        final_seq = np.full(target_length, padding_token_id, dtype=np.int64)
        final_seq[0] = sos_token_id
        final_seq[1 : 1 + len(clipped_seq)] = clipped_seq
        final_seq[1 + len(clipped_seq)] = eos_token_id

        return torch.tensor(final_seq, dtype=torch.long)

    return process_sequence(mel_seq), process_sequence(acc_seq)


def deterministic_clip_and_pad(
    mel_sequence: np.ndarray,
    acc_sequence: np.ndarray,
    target_length: int,
    padding_token_id: int = 0,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    start_index: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bar_indices = np.where(mel_sequence == BAR_NONE_TOKEN_ID)[0]

    start_bar_idx = start_index
    end_bar_idx = start_bar_idx + EXTRACT_BAR_NUM

    if len(bar_indices) > end_bar_idx:
        mel_start_token = bar_indices[start_bar_idx]
        mel_end_token = bar_indices[end_bar_idx]
    elif len(bar_indices) > start_bar_idx:
        mel_start_token = bar_indices[start_bar_idx]
        mel_end_token = len(mel_sequence)
    else:
        mel_start_token = 0
        mel_end_token = 0

    mel_seq = mel_sequence[mel_start_token:mel_end_token]

    acc_start_token = (
        mel_start_token + ACC_BAR_BIAS
        if mel_start_token + ACC_BAR_BIAS < len(acc_sequence)
        else mel_start_token
    )
    acc_end_token = (
        mel_end_token + ACC_BAR_BIAS
        if mel_end_token + ACC_BAR_BIAS < len(acc_sequence)
        else mel_end_token
    )
    acc_seq = acc_sequence[acc_start_token:acc_end_token]

    max_content_len = target_length - 2

    def process_sequence(seq: np.ndarray) -> torch.Tensor:
        clipped_seq = seq[:max_content_len]
        final_seq = np.full(target_length, padding_token_id, dtype=np.int64)
        final_seq[0] = sos_token_id
        final_seq[1 : 1 + len(clipped_seq)] = clipped_seq
        if 1 + len(clipped_seq) < target_length:
            final_seq[1 + len(clipped_seq)] = eos_token_id

        return torch.tensor(final_seq, dtype=torch.long)

    return process_sequence(mel_seq), process_sequence(acc_seq)


class MelAccRemiJsonDataset(BaseDataset):
    """Dataset for melody/accompaniment JSON pairs.

    Expects JSON files with a top-level key `ids` containing a 1D list/array.
    """

    def __init__(self, config: MelAccRemiJsonDatasetConfig) -> None:
        self.config = config
        self.mel_dir = config.mel_dir
        self.acc_dir = config.acc_dir
        self.file_pattern = config.file_pattern

        self.transform = config.transform
        self.max_seq_len = config.max_seq_len
        self.tokenization_type = config.tokenization_type
        self.stage = config.stage

        self.midi_file_pairs: List[Tuple[str, str]] = self._collect_file_pairs()
        self.midi_file_pairs.sort()

        data_range = config.data_range
        if isinstance(data_range[0], int) and isinstance(data_range[1], int):
            start_index, end_index = data_range
        elif isinstance(data_range[0], float) and isinstance(data_range[1], float):
            start_index = int(data_range[0] * len(self.midi_file_pairs))
            end_index = int(data_range[1] * len(self.midi_file_pairs))
        else:
            raise ValueError(
                "data_range must be a tuple of two integers or two floats."
            )
        self.midi_file_pairs = self.midi_file_pairs[start_index:end_index]

    def _collect_file_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        mel_files = glob.glob(os.path.join(self.mel_dir, self.file_pattern))

        for mel_path in mel_files:
            file_name = os.path.basename(mel_path)
            acc_path = os.path.join(self.acc_dir, file_name)

            if os.path.exists(acc_path):
                pairs.append((mel_path, acc_path))
            else:
                print(
                    f"Warning: Corresponding accompaniment file not found for {mel_path} at {acc_path}"
                )
        return pairs

    def setup(self, stage: Optional[str] = None):
        if stage is not None:
            self.stage = stage

    def __len__(self) -> int:
        return len(self.midi_file_pairs)

    def __getitem__(self, idx: int) -> MelAccRemiJsonModelInput:
        mel_path, acc_path = self.midi_file_pairs[idx]

        try:
            with open(mel_path, "r") as f:
                mel_data = json.load(f)["ids"]
            with open(acc_path, "r") as f:
                acc_data = json.load(f)["ids"]

            mel_seq_np = np.array(mel_data).squeeze()
            acc_seq_np = np.array(acc_data).squeeze()

            if self.stage == "train":
                mel_tensor, acc_tensor = random_clip_and_pad(
                    mel_seq_np,
                    acc_seq_np,
                    target_length=self.max_seq_len,
                    padding_token_id=0,
                    padding_probability=0.2,
                    exp_bias_lambda=5.0,
                )
            else:
                mel_tensor, acc_tensor = deterministic_clip_and_pad(
                    mel_seq_np,
                    acc_seq_np,
                    target_length=self.max_seq_len,
                    padding_token_id=0,
                )

            if self.transform:
                mel_tensor = self.transform(mel_tensor)
                acc_tensor = self.transform(acc_tensor)
            return MelAccRemiJsonModelInput(mel_data=mel_tensor, acc_data=acc_tensor)
        except Exception as e:
            print(f"Error loading JSON files, {mel_path}, {acc_path}: {e}")
            raise RuntimeError(
                f"Error processing data for {mel_path} and {acc_path}: {e}"
            )

    # Provide both names so callers (datamodules or factories) find one they expect
    def collate_fn(
        self, batch: List[MelAccRemiJsonModelInput]
    ) -> MelAccRemiJsonModelInput:
        mel_data = [item.mel_data for item in batch if item is not None]
        acc_data = [item.acc_data for item in batch if item is not None]
        if not mel_data or not acc_data:
            raise ValueError("Batch contains no valid data items.")
        mel_data = torch.stack(mel_data)
        acc_data = torch.stack(acc_data)
        return MelAccRemiJsonModelInput(mel_data=mel_data, acc_data=acc_data)
