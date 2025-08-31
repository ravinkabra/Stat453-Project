from ..._utils.config_base import DictAccessMixin
from pydantic import Field
from pydantic.dataclasses import dataclass, ConfigDict


@dataclass
class BaseModelInput(DictAccessMixin): ...
