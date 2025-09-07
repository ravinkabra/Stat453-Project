from ..._utils.config_base import DictAccessMixin
from pydantic.dataclasses import dataclass, ConfigDict
from torch import Tensor


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MelAccRemiJsonModelInput(DictAccessMixin):
    mel_data: Tensor
    acc_data: Tensor

    def to_dict(self):
        return {"mel_data": self.mel_data, "acc_data": self.acc_data}

    @classmethod
    def from_tensors(cls, mel: Tensor, acc: Tensor) -> "MelAccRemiJsonModelInput":
        return cls(mel_data=mel, acc_data=acc)
