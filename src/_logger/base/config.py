from typing import Optional, Union
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..._utils.config_base import DictAccessMixin


@dataclass
class BaseLoggerParams(DictAccessMixin):
    """
    🔧 Logger 参数基类

    所有 logger 参数的基础类，除了 _target_ 外的所有字段
    都会直接传递给对应的 Logger 类构造函数
    """

    _target_: str = Field(..., description="Logger 类的完整路径")
    version: Optional[Union[int, str]] = Field(None, description="实验版本, None 则由Project自动编号")
    prefix: str = Field("", description="日志前缀")
