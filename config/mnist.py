from src._project.config import ProjectConfig
from src._datamodule.base.config import BaseDataModuleConfig
from src.dataset.tutorial_mnist.config import MnistDatasetConfig
from src.model.tutorial_mnist.config import MnistModelConfig
from src.network.tutorial_lenet.config import LeNetConfig
from src._optimizer.base.config import BaseOptimizerParams
from src._lr_scheduler.base.config import BaseLRSchedulerParams
from src.metric._manager import (
    MetricManagerConfig,
    ManagedMetricConfig,
    MetricLogConfig,
)
from src.metric.value_recoder.config import ValueRecorderParams 
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from src._logger.config import TensorBoardLoggerParams, CSVLoggerParams
from src._trainer.base.config import BaseTrainerConfig
from src.callback.sample_saver.config import (
    TrainingSampleSaverCallbackConfig,
    SaveKeyConfig,
)

train_dataset = MnistDatasetConfig(
    batch_size=128,
    num_workers=0,
    persistent_workers=False,
    shuffle=True,
    data_dir="./data/mnist",
    dataset_range=[0.0, 0.95],
)
val_dataset = MnistDatasetConfig(
    batch_size=32,
    num_workers=0,
    shuffle=False,
    persistent_workers=False,
    data_dir="./data/mnist",
    dataset_range=[0.95, 1.0],
)
test_dataset = MnistDatasetConfig(
    batch_size=32,
    num_workers=0,
    shuffle=False,
    persistent_workers=False,
    data_dir="./data/mnist",
    dataset_range=[0.0, 1.0],
)
datamodule = BaseDataModuleConfig(
    train=train_dataset, val=val_dataset, test=test_dataset
)

network = LeNetConfig(in_channels=1, num_classes=10)
optimizer = BaseOptimizerParams(_target_="torch.optim.Adam", lr=1e-3, weight_decay=1e-5)
lr_scheduler = BaseLRSchedulerParams(
    _target_="torch.optim.lr_scheduler.CosineAnnealingLR", step_size=10, gamma=0.1
)
metric_manager = MetricManagerConfig(
    metrics={
        "training_state": ManagedMetricConfig(
            log_config=MetricLogConfig(
                phase=["train", "val", "test"],
                prog_bar=True,
                frequency=50,
            ),
            metrics={
                "loss": ValueRecorderParams(),
                "learning_rate": ValueRecorderParams(),
            },
        ),
        "classification": ManagedMetricConfig(
            log_config=MetricLogConfig(
                phase=["val", "test"],
                prog_bar=True,
                # frequency=3,
            ),
            metrics={
                "accuracy": Accuracy.__new__(
                    Accuracy, task="multiclass", num_classes=10
                ),
                "f1_score": F1Score.__new__(F1Score, task="multiclass", num_classes=10),
                "precision": Precision.__new__(
                    Precision, task="multiclass", num_classes=10
                ),
                "recall": Recall.__new__(Recall, task="multiclass", num_classes=10),
            },
        ),
    }
)
model = MnistModelConfig(
    network=network,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    metric_manager=metric_manager,
)
loggers = {
    "csv": CSVLoggerParams(
        save_dir="./output/mnist_example/logs", flush_logs_every_n_steps=50, version=0
    ),
    "tensorboard": TensorBoardLoggerParams(
        save_dir="./output/mnist_example/logs", version=0
    ),
}
trainer = BaseTrainerConfig(
    max_epochs=10,
    log_every_n_steps=1,
)
callbacks = {
    "sample_saver": TrainingSampleSaverCallbackConfig(
        save_dir="./output/mnist_example/samples",
        save_keys={
            "image": SaveKeyConfig(
                source="batch",
                keys=["image"],
                format="image",
                extension=".png",
                phases=["val", "test"],
                save_frequency=10,
            ),
            "label": SaveKeyConfig(
                source="batch",
                keys=["label"],
                format="json",
                extension=".json",
                phases=["val", "test"],
                save_frequency=10,
            ),
            "preds": SaveKeyConfig(
                source="outputs",
                keys=["preds"],
                format="json",
                extension=".json",
                phases=["val", "test"],
                save_frequency=10,
            ),
        },
    )
}
project = ProjectConfig(
    project_name="MNIST-Example",
    output_dir="./output/mnist_example",
    log_level="INFO",
    experiment_name="mnist_experiment",
    datamodule=datamodule,
    model=model,
    mode="train",
    loggers=loggers,
    callbacks=callbacks,
    trainer=trainer,
)
