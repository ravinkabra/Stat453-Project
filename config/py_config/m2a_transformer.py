from src._project.base.config import ProjectConfig
from src._datamodule.base.config import BaseDataModuleConfig

# from src.dataset.tutorial_mnist.config import MnistDatasetConfig
# from src.model.tutorial_mnist.config import MnistModelConfig
# from src.network.tutorial_lenet.config import LeNetConfig
from src.dataset.old_pt.config import OldPtDatasetConfig
from src.model.old_pt_m2a_transformer.config import OldPtM2ATransformerConfig
from src.network.customized_roformer.config import CustomizedRoFormerEncoderParams

from src._optimizer.adam.config import AdamParams
from src._lr_scheduler.onecycle.config import OneCycleLRParams
from src.metric._manager import (
    MetricManagerConfig,
    ManagedMetricConfig,
    MetricLogConfig,
)
from src.metric.value_recoder.config import ValueRecorderParams
from torchmetrics import MeanMetric

# from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from src.metric.classification.config import (
    AccuracyParams,
    F1ScoreParams,
    PrecisionParams,
    RecallParams,
)
from src._logger import TensorBoardLoggerParams, CSVLoggerParams
from src._trainer.base.config import BaseTrainerConfig
from src._callback.sample_saver.config import (
    SampleSaverCallbackConfig,
    SaveKeyConfig,
)

target_lenth = 384
train_dataset = OldPtDatasetConfig(
    file_path="/home/ubuntu/ugrip/data/pop909/pop909_acc_cp4.pt",
    split_ratio=9,
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    target_length=target_lenth,
    shuffle=True,
)

val_dataset = OldPtDatasetConfig(
    file_path="/home/ubuntu/ugrip/data/pop909/pop909_acc_cp4.pt",
    split_ratio=9,
    batch_size=12,
    num_workers=0,
    shuffle=False,
    target_length=target_lenth,
    persistent_workers=False,
)
datamodule = BaseDataModuleConfig(train=train_dataset, val=val_dataset)

optimizer = AdamParams(_target_="torch.optim.Adam", lr=1e-4, weight_decay=1e-5)
lr_scheduler = OneCycleLRParams(
    _target_="torch.optim.lr_scheduler.OneCycleLR",
    max_lr=1e-4,  # 对应你代码中的 max_lr
    total_steps=1000000,  # 对应你代码中的 MAX_STEPS
    pct_start=0.005,  # 对应你代码中的 pct_start
    frequency=1,  # OneCycleLR 通常每步更新
    interval="step",  # OneCycleLR 按步更新而不是按 epoch
)
metric_manager = MetricManagerConfig(
    metrics={
        "training_state": ManagedMetricConfig(
            log_config=MetricLogConfig(
                phase=["train", "val"],
                prog_bar=True,
                update_frequency=5,
                on_step=False,
                # on_step=True,
            ),
            metrics={
                "loss": ValueRecorderParams(),
                "learning_rate": ValueRecorderParams(),
            },
        ),
    }
)
from transformers.models.roformer import RoFormerConfig

local_encoder_network = CustomizedRoFormerEncoderParams(
    config=RoFormerConfig(
        vocab_size=3000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
)

main_encoder_network = CustomizedRoFormerEncoderParams(
    config=RoFormerConfig(
        vocab_size=3000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
)

local_decoder_network = CustomizedRoFormerEncoderParams(
    config=RoFormerConfig(
        vocab_size=3000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
)

model = OldPtM2ATransformerConfig(
    local_encoder_network=local_encoder_network,
    global_network=main_encoder_network,
    local_decoder_network=local_decoder_network,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    metric_manager=metric_manager,
)
loggers = {
    "csv": CSVLoggerParams(flush_logs_every_n_steps=1000),
    "tensorboard": TensorBoardLoggerParams(),
}
trainer = BaseTrainerConfig(
    max_epochs=-1,
    log_every_n_steps=10,
)
from src._callback.checkpoint.config import ModelCheckpointParams

callbacks = {
    "model_checkpoint": ModelCheckpointParams(
        monitor="val/epoch/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}-{val/epoch/loss:.4f}",
    )
}
project = ProjectConfig(
    project_name="M2A-Example",
    save_dir="./output/m2a_example",
    log_level="INFO",
    experiment_name=f"m2a_experiment-len:{target_lenth}",
    datamodule=datamodule,
    model=model,
    mode="train",
    loggers=loggers,
    callbacks=callbacks,
    trainer=trainer,
    version="0.0.",
)

project.to_yaml("./config/m2a_retrain_basline_v2.yaml", use_original_config=True)
