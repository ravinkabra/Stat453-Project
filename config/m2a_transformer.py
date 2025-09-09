from src._project.base.config import ProjectConfig
from src._datamodule.base.config import BaseDataModuleConfig

# from src.dataset.tutorial_mnist.config import MnistDatasetConfig
# from src.model.tutorial_mnist.config import MnistModelConfig
# from src.network.tutorial_lenet.config import LeNetConfig
from src.dataset.old_pt.config import OldPtDatasetConfig
from src.model.old_pt_m2a_transformer.config import OldPtM2ATransformerConfig
from src.network.customized_roformer.config import CustomizedRoFormerEncoderParams

from src._optimizer.adam.config import AdamParams
from src._lr_scheduler.cosine.config import CosineAnnealingLRParams
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

train_dataset = OldPtDatasetConfig(
    file_path="data/pop909/pop909_acc_cp4.pt",
    split_ratio=0.7,
    batch_size=128,
    num_workers=0,
    persistent_workers=False,
    shuffle=True,
)

val_dataset = OldPtDatasetConfig(
    file_path="data/pop909/pop909_acc_cp4.pt",
    split_ratio=0.15,
    batch_size=2,
    num_workers=0,
    shuffle=False,
    persistent_workers=False,
)
datamodule = BaseDataModuleConfig(train=train_dataset, val=val_dataset)

optimizer = AdamParams(_target_="torch.optim.Adam", lr=1e-3, weight_decay=1e-5)
lr_scheduler = CosineAnnealingLRParams(
    _target_="torch.optim.lr_scheduler.CosineAnnealingLR",
    frequency=10,
)
metric_manager = MetricManagerConfig(
    metrics={
        "training_state": ManagedMetricConfig(
            log_config=MetricLogConfig(
                phase=["train", "val"],
                prog_bar=True,
                update_frequency=5,
                on_step=True,
            ),
            metrics={
                "loss": ValueRecorderParams(),
                "learning_rate": ValueRecorderParams(),
            },
        ),
        # "classification": ManagedMetricConfig(
        #     log_config=MetricLogConfig(
        #         phase=["train"],
        #         prog_bar=True,
        #         update_frequency=1,
        #         compute_frequency=1,
        #         on_step=True,
        #     ),
        #     metrics={
        #         "accuracy": AccuracyParams(
        #             task="multiclass", num_classes=10, average="macro"
        #         ),
        #         "f1_score": F1ScoreParams(
        #             task="multiclass", num_classes=10, average="macro"
        #         ),
        #         "precision": PrecisionParams(
        #             task="multiclass", num_classes=10, average="macro"
        #         ),
        #         "recall": RecallParams(
        #             task="multiclass", num_classes=10, average="macro"
        #         ),
        #     },
        # ),
    }
)
from transformers.models.roformer import RoFormerConfig

local_encoder_network = CustomizedRoFormerEncoderParams(
    config=RoFormerConfig(
        vocab_size=3000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
    )
)

main_encoder_network = CustomizedRoFormerEncoderParams(
    config=RoFormerConfig(
        vocab_size=3000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
)

local_decoder_network = CustomizedRoFormerEncoderParams(
    config=RoFormerConfig(
        vocab_size=3000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
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
    "csv": CSVLoggerParams(save_dir="./output/mnist_example/logs", flush_logs_every_n_steps=50, version=0),
    "tensorboard": TensorBoardLoggerParams(save_dir="./output/mnist_example/logs", version=0),
}
trainer = BaseTrainerConfig(
    max_epochs=3,
    log_every_n_steps=5,
)
callbacks = {
    # "sample_saver": SampleSaverCallbackConfig(
    #     save_dir="./output/mnist_example/samples",
    #     save_keys={
    #         "image": SaveKeyConfig(
    #             source="batch",
    #             keys=["image"],
    #             format="image",
    #             extension=".png",
    #             phases=["val", "test"],
    #             save_frequency=10,
    #         ),
    #         "label": SaveKeyConfig(
    #             source="batch",
    #             keys=["label"],
    #             format="json",
    #             extension=".json",
    #             phases=["val", "test"],
    #             save_frequency=10,
    #         ),
    #         "preds": SaveKeyConfig(
    #             source="outputs",
    #             keys=["preds"],
    #             format="json",
    #             extension=".json",
    #             phases=["val", "test"],
    #             save_frequency=10,
    #         ),
    #     },
    # )
}
project = ProjectConfig(
    project_name="M2A-Example",
    output_dir="./output/m2a_example",
    log_level="INFO",
    experiment_name="m2a_experiment",
    datamodule=datamodule,
    model=model,
    mode="train",
    loggers=loggers,
    # callbacks=callbacks,
    trainer=trainer,
)

project.to_yaml("./config/m2a_example.yaml")
