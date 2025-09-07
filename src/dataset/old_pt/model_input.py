from ..._utils.config_base import DictAccessMixin
from pydantic.dataclasses import dataclass, ConfigDict
from torch import Tensor


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OldPtModelInput(DictAccessMixin):
    """Model input for OldPt dataset.

    Fields:
        mel_data: Tensor
        acc_data: Tensor
        pitch_shift: Tensor
    """

    mel_data: Tensor
    acc_data: Tensor
    pitch_shift: Tensor

    def to_dict(self):
        return {
            "mel_data": self.mel_data,
            "acc_data": self.acc_data,
            "pitch_shift": self.pitch_shift,
        }

    @classmethod
    def from_tensors(
        cls, mel: Tensor, acc: Tensor, pitch_shift: Tensor
    ) -> "OldPtModelInput":
        return cls(mel_data=mel, acc_data=acc, pitch_shift=pitch_shift)
