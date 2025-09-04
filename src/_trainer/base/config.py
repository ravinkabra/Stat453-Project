from pydantic.dataclasses import dataclass, Field, ConfigDict
from typing import Union, Optional, List, Dict, Any, Literal
from datetime import timedelta


@dataclass(config=ConfigDict(extra="allow"))
class BaseTrainerConfig:
    # 硬件和加速器配置
    accelerator: Union[str, Any] = Field(
        default="auto",
        description="加速器类型 ('cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto')",
    )
    strategy: Union[str, Any] = Field(default="auto", description="训练策略")
    devices: Union[List[int], str, int] = Field(
        default="auto", description="使用的设备"
    )
    num_nodes: int = Field(default=1, description="分布式训练的GPU节点数量")

    # 精度配置
    precision: Union[
        Literal[64, 32, 16],
        Literal[
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
        ],
        Literal["64", "32", "16", "bf16"],
        None,
    ] = Field(default="32-true", description="训练精度 (64, 32, 16, 'bf16' 等)")

    # 日志和记录器
    logger: Union[Any, List[Any], bool, None] = Field(
        default=True, description="日志记录器"
    )
    log_every_n_steps: Optional[int] = Field(
        default=50, description="每N步记录一次日志"
    )

    # 回调和插件
    callbacks: Union[List[Any], Any, None] = Field(default=None, description="训练回调")
    plugins: Union[Any, List[Any], None] = Field(
        default=None, description="Lightning插件"
    )

    # 快速开发和调试
    fast_dev_run: Union[int, bool] = Field(
        default=False, description="快速开发运行 (运行n批次或1批次)"
    )

    # 训练轮数控制
    max_epochs: Optional[int] = Field(default=None, description="最大训练轮数")
    min_epochs: Optional[int] = Field(default=None, description="最小训练轮数")
    max_steps: int = Field(default=-1, description="最大训练步数 (-1表示无限制)")
    min_steps: Optional[int] = Field(default=None, description="最小训练步数")

    # 时间限制
    max_time: Union[str, timedelta, Dict[str, int], None] = Field(
        default=None,
        description="最大训练时间 (格式: DD:HH:MM:SS 或 timedelta 或 dict)",
    )

    # 数据批次限制
    limit_train_batches: Union[int, float, None] = Field(
        default=1.0, description="训练数据批次限制 (int=批次数, float=比例)"
    )
    limit_val_batches: Union[int, float, None] = Field(
        default=1.0, description="验证数据批次限制 (int=批次数, float=比例)"
    )
    limit_test_batches: Union[int, float, None] = Field(
        default=1.0, description="测试数据批次限制 (int=批次数, float=比例)"
    )
    limit_predict_batches: Union[int, float, None] = Field(
        default=1.0, description="预测数据批次限制 (int=批次数, float=比例)"
    )

    # 过拟合测试
    overfit_batches: Union[int, float] = Field(
        default=0.0, description="过拟合测试批次 (int=批次数, float=比例)"
    )

    # 验证控制
    val_check_interval: Union[int, float, None] = Field(
        default=1.0, description="验证检查间隔 (int=步数, float=比例)"
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=1, description="每N个epoch进行一次验证"
    )
    num_sanity_val_steps: Optional[int] = Field(default=2, description="完整性验证步数")

    # 检查点和进度条
    enable_checkpointing: Optional[bool] = Field(
        default=True, description="是否启用检查点"
    )
    enable_progress_bar: Optional[bool] = Field(
        default=True, description="是否启用进度条"
    )
    enable_model_summary: Optional[bool] = Field(
        default=True, description="是否启用模型摘要"
    )

    # 梯度累积和裁剪
    accumulate_grad_batches: int = Field(default=1, description="梯度累积批次数")
    gradient_clip_val: Union[int, float, None] = Field(
        default=None, description="梯度裁剪值"
    )
    gradient_clip_algorithm: Optional[str] = Field(
        default="norm", description="梯度裁剪算法 ('value' 或 'norm')"
    )

    # 确定性和性能
    deterministic: Union[bool, Literal["warn"], None] = Field(
        default=None, description="确定性模式"
    )
    benchmark: Optional[bool] = Field(default=None, description="性能基准模式")
    inference_mode: bool = Field(default=True, description="推理模式")

    # 分布式采样器
    use_distributed_sampler: bool = Field(
        default=True, description="是否使用分布式采样器"
    )

    # 性能分析和调试
    profiler: Union[Any, str, None] = Field(default=None, description="性能分析器")
    detect_anomaly: bool = Field(default=False, description="启用异常检测")

    # 精简模式
    barebones: bool = Field(default=False, description="精简模式 (禁用影响速度的功能)")

    # 批归一化同步
    sync_batchnorm: bool = Field(default=False, description="同步批归一化")

    # 数据加载器重载
    reload_dataloaders_every_n_epochs: int = Field(
        default=0, description="每N个epoch重新加载数据加载器"
    )

    # 目录和注册表
    default_root_dir: Union[str, Any, None] = Field(
        default=None, description="默认根目录"
    )
    model_registry: Optional[str] = Field(default=None, description="模型注册表名称")
