from src._project.base.config import ProjectConfig
from src._datamodule.base.config import BaseDataModuleConfig
from src.dataset.old_pt.config import OldPtDatasetConfig

# 1. 导入你的新模型配置
from src.model.straight_forword_stanley.config import OldPtSFSTransformerConfig

# 2. 直接从 transformers 库导入 RoFormerConfig
from transformers import RoFormerConfig

from src._optimizer.adam.config import AdamParams
from src._lr_scheduler.onecycle.config import OneCycleLRParams
from src.metric._manager import (
    MetricManagerConfig,
    ManagedMetricConfig,
    MetricLogConfig,
)
from src.metric.value_recoder.config import ValueRecorderParams
from src._logger import TensorBoardLoggerParams, CSVLoggerParams
from src._trainer.base.config import BaseTrainerConfig

# --- 数据集配置 (与之前保持一致) ---
train_dataset = OldPtDatasetConfig(
    file_path="/home/ubuntu/ugrip/data/pop909/pop909_acc_cp4.pt",
    split_ratio=9,
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    target_length=384,
    shuffle=True,
)

val_dataset = OldPtDatasetConfig(
    file_path="/home/ubuntu/ugrip/data/pop909/pop909_acc_cp4.pt",
    split_ratio=9,
    batch_size=8,
    num_workers=0,
    shuffle=False,
    target_length=384,
    persistent_workers=False,
)
datamodule = BaseDataModuleConfig(train=train_dataset, val=val_dataset)

# --- 优化器和学习率调度器 ---
optimizer = AdamParams(_target_="torch.optim.Adam", lr=1e-4, weight_decay=1e-5)
lr_scheduler = OneCycleLRParams(
    _target_="torch.optim.lr_scheduler.OneCycleLR",
    max_lr=1e-4,
    total_steps=1000000,
    pct_start=0.005,
    frequency=1,
    interval="step",
)

# --- 指标管理器 ---
metric_manager = MetricManagerConfig(
    metrics={
        "training_state": ManagedMetricConfig(
            log_config=MetricLogConfig(phase=["train", "val"], prog_bar=True, update_frequency=5, on_step=False),
            metrics={
                "loss": ValueRecorderParams(),
                "learning_rate": ValueRecorderParams(),
            },
        ),
    }
)

# --- 网络配置 ---
# 直接使用 RoFormerConfig，并更新 vocab_size
# N_NORMAL_TOKENS (3200) + 7 special tokens = 3207
VOCAB_SIZE = 3207

local_encoder_network = RoFormerConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,  # 局部编码器处理较短序列
)

global_network = RoFormerConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=2048,  # 全局编码器处理长序列
)

local_decoder_network = RoFormerConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
)

# --- 模型配置 ---
# 实例化你的新模型配置，并传入新参数
model = OldPtSFSTransformerConfig(
    local_encoder_network=local_encoder_network,
    global_network=global_network,
    local_decoder_network=local_decoder_network,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    metric_manager=metric_manager,
    # 添加新的切分点参数
    cut_point_start=8,
    cut_point_end=None,  # 示例值，可以设为 None
)

# --- 日志和训练器配置 ---
loggers = {
    "csv": CSVLoggerParams(flush_logs_every_n_steps=1000),
    "tensorboard": TensorBoardLoggerParams(),
}
trainer = BaseTrainerConfig(
    log_every_n_steps=10,
)

# --- 项目总配置 ---
project = ProjectConfig(
    project_name="Straightforward-M2A",
    save_dir="./output/straightforward_m2a",
    log_level="INFO",
    experiment_name="sfs_m2a_experiment",
    datamodule=datamodule,
    model=model,
    mode="train",
    loggers=loggers,
    trainer=trainer,
    version="0.0.1",
)

# --- 生成 YAML 文件 ---
# 6. 指定新的 YAML 文件名
project.to_yaml("./config/sfs_transformer_v0.yaml", use_original_config=True)
