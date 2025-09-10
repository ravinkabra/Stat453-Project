"""
🔧 项目配置 - 统整所有组件的配置类
"""

from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field, model_validator
from typing import Dict, Any, Optional, List, Literal

import dataclasses
import copy
import os

try:
    import yaml
except Exception:  # pragma: no cover - graceful fallback if PyYAML not installed
    yaml = None

from ..._logger import UnionLoggerParams
from ..._datamodule import UnionDataModuleConfig
from ..._trainer import UnionTrainerConfig
from ..._callback import UnionCallbackConfig
from ...model import UnionModelConfig
from ._utils import compute_and_set_versions


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProjectConfig:
    """
    项目级配置类 - 统一管理整个训练项目的配置

    设计理念：
    - 作为所有组件配置的顶层容器
    - 直接配置 PyTorch Lightning 原生 Trainer
    - 提供项目级的元信息和设置
    - 支持组件间的配置协调
    """

    # 项目元信息
    name: str = Field(default="training_project", description="项目名称")
    description: str = Field(default="", description="项目描述")

    # 输出设置
    save_dir: str = Field(default="./outputs", description="输出根目录，如果logger的save_dir为None，则用此覆盖")
    experiment_name: Optional[str] = Field(
        default=None, description="实验名称，None则使用项目名称，如果logger的name为None，则用此覆盖"
    )
    version: Optional[str] = Field(
        default=None,
        description="项目版本，如果version存在目标的文件夹中，不做额外修改，如果为其他格式则为{version}x,如果为none则是version_x，如果logger的version为None，则用此覆盖",
    )
    save_experiment_version_dir: Optional[str] = Field(
        default=None,
        description="最终影响保存位置的参数，同trainer的default_root_dir，如果为None则自动推断为{save_dir}/{experiment_name}/{version}",
    )

    # 核心组件配置
    model: UnionModelConfig = Field(description="模型配置")
    datamodule: Optional[UnionDataModuleConfig] = Field(default=None, description="数据模块配置")

    # PyTorch Lightning Trainer 配置
    trainer: Optional[UnionTrainerConfig] = Field(default=None, description="PyTorch Lightning Trainer 原生配置")

    # 回调配置
    callbacks: Optional[dict[str, UnionCallbackConfig]] = Field(default=None, description="回调配置字典，key为回调名称")

    # 日志器配置 - 使用类型化的 logger 配置
    loggers: Optional[Dict[str, UnionLoggerParams]] = Field(
        default=None, description="日志器配置字典，key为日志器名称，value为类型化配置"
    )

    # 运行模式
    mode: Literal["train", "validate", "test", "predict"] = Field(default="train", description="运行模式")

    # 实验管理
    tags: List[str] = Field(default_factory=list, description="实验标签")
    notes: str = Field(default="", description="实验备注")

    # 重现性设置
    seed: Optional[int] = Field(default=None, description="随机种子")
    deterministic: bool = Field(default=False, description="是否启用确定性模式")

    def get_experiment_name(self) -> str:
        """获取实验名称"""
        return self.experiment_name or self.name

    def _save_original_conifg(self) -> None:
        # Use a deep copy to preserve the original config object.
        # Re-instantiating from a dict can attempt to set read-only properties
        # on nested objects (for example HuggingFace `RoFormerConfig.use_return_dict`),
        # which raises AttributeError. deepcopy avoids re-running constructors.
        self._original_config = copy.deepcopy(self)

    def to_dict(
        self,
        exclude_none: bool = True,
        exclude_original_config: bool = True,
        use_original_config: bool = False,
    ) -> Dict[str, Any]:
        """把 ProjectConfig 转成原生 Python 字典。

        - 使用 dataclasses.asdict 递归地把 dataclass 转为 dict。
        - 当 exclude_none=True 时，会递归移除值为 None 的条目，便于生成更简洁的 yaml。
        - 新增参数 `use_original_config`：当为 True 且实例保存了 `_original_config` 时，
          会把保存的原始配置作为序列化对象（而不是当前可能已被修改的实例）。
        """

        # 支持序列化原始保存的配置或当前实例（向后兼容：默认序列化当前实例）
        source = (
            getattr(self, "_original_config") if (use_original_config and hasattr(self, "_original_config")) else self
        )
        result = dataclasses.asdict(source)

        # 可选：从序列化结果中排除内部保存的原始配置对象（以避免把不可序列化或只读属性一并写出）
        if exclude_original_config and "_original_config" in result:
            # 如果原始配置存在，移除它（do not serialize internal state by default）
            result.pop("_original_config", None)

        if not exclude_none:
            return result

        def _clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                new = {}
                for k, v in obj.items():
                    v2 = _clean(v)
                    if v2 is not None:
                        new[k] = v2
                return new
            if isinstance(obj, (list, tuple)):
                cleaned = [_clean(v) for v in obj]
                return [v for v in cleaned if v is not None]
            return obj

        return _clean(result)

    def to_yaml(
        self,
        path: Optional[str] = None,
        sort_keys: bool = False,
        exclude_original_config: bool = True,
        use_original_config: bool = False,
    ) -> str:
        """把 ProjectConfig 导出为 YAML 字符串，并可选择写入文件。

        Args:
            path: 如果提供，将把 YAML 写入到该路径（会创建父目录）。
            sort_keys: 是否对字典的键进行排序写入。
            use_original_config: 是否序列化保存的 `_original_config` 而不是当前实例。

        Returns:
            生成的 YAML 字符串。

        注意:
            该方法依赖 PyYAML（yaml）。若未安装，会退回到使用 str() 表示对象并生成简单字符串。
        """
        # 将 exclude_original_config 和 use_original_config 传递给 to_dict，以控制序列化内容
        data = self.to_dict(
            exclude_none=True,
            exclude_original_config=exclude_original_config,
            use_original_config=use_original_config,
        )

        # 如果 PyYAML 可用，使用它并为不可序列化对象提供默认的字符串表示。
        if yaml is not None:

            class _Dumper(yaml.SafeDumper):
                pass

            def _repr_unknown(dumper, value):
                # 把任意对象都当作字符串标量处理，避免序列化失败
                return dumper.represent_scalar("tag:yaml.org,2002:str", str(value))

            # 为所有未特别处理的对象注册 representer
            _Dumper.add_multi_representer(object, _repr_unknown)

            yaml_str = yaml.dump(
                data,
                Dumper=_Dumper,
                sort_keys=sort_keys,
                allow_unicode=True,
            )
        else:
            # 最小回退：把 dict 转为 str，尽量可读
            yaml_str = str(data)

        if path:
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)

        return yaml_str

    def _setup_logger_save_dirs(self):
        for name, logger_config in self.loggers.items():
            if logger_config.save_dir is None:
                logger_config.save_dir = self.save_dir

    def _setup_logger_names(self):
        for name, logger_config in self.loggers.items():
            if logger_config.name is None:
                logger_config.name = self.experiment_name

    def _setup_logger_versions(self):
        if self.version is not None:
            compute_and_set_versions(self.loggers, self.save_dir, self.experiment_name, version=self.version)

    @model_validator(mode="after")
    def _post_init(self) -> "ProjectConfig":
        self._save_original_conifg()

        if self.experiment_name is None:
            self.experiment_name = self.name
        self.save_experiment_version_dir = os.path.join(self.save_dir, self.name, self.version)
        if self.loggers:
            self._setup_logger_save_dirs()
            self._setup_logger_names()
            self._setup_logger_versions()
        return self
