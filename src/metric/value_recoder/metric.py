from torchmetrics import Metric
import torch
from torchmetrics.segmentation.dice import DiceScore
from torchmetrics import MeanMetric
from torchmetrics.utilities import dim_zero_cat


class ValueRecorderMetric(Metric):
    """
    数值记录指标 - 专门用于记录和聚合训练过程中的标量值

    适用场景：
    - 损失值记录
    - 学习率监控
    - 梯度范数跟踪
    - 自定义标量指标
    """

    def __init__(self, aggregation: str = "mean"):
        super().__init__()
        self.aggregation = aggregation
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value, _=None):  # 第二个参数被忽略
        """更新记录值"""
        if isinstance(value, torch.Tensor):
            self.values.append(value.detach().clone())
        else:
            self.values.append(torch.tensor(value, dtype=torch.float))

    def compute(self):
        """计算聚合结果"""
        if not self.values:
            return None

        stacked_values = torch.stack(self.values)
        if self.aggregation == "mean":
            return stacked_values.mean()
        elif self.aggregation == "sum":
            return stacked_values.sum()
        elif self.aggregation == "last":
            return stacked_values[-1]
        else:
            return stacked_values.mean()  # 默认
