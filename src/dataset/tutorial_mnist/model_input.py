from ..._utils.config_base import DictAccessMixin, TypedDictAccessMixin
from pydantic.dataclasses import dataclass, ConfigDict
from torch import Tensor


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MnistModelInput(DictAccessMixin):
    """MNIST 数据输入格式"""

    image: Tensor  # [1, 28, 28] 或 [28, 28]
    label: Tensor  # 0-9 数字标签

    def to_dict(self):
        """转换为字典格式"""
        return {"image": self.image, "label": self.label}

    @classmethod
    def from_tensors(cls, image: Tensor, label: Tensor) -> "MnistModelInput":
        """从张量创建输入对象"""
        return cls(image=image, label=label)
