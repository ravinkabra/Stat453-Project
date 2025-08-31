"""
🔧 统一优化器配置演示
展示 optimizer 和 lr_scheduler 的统一设计
"""

from src.model.base.config import BaseModelConfig
from src._optimizer.base.config import BaseOptimizerParams
from src._lr_scheduler.base.config import BaseLRSchedulerParams


def demo_unified_optimizer_config():
    """演示统一的优化器配置设计"""

    print("🎯 统一优化器配置演示")
    print("=" * 50)

    # 场景 1: 简单模型 - 单个优化器（优化整个模型）
    print("\n📝 场景 1: 单个优化器 - 优化整个模型")
    single_optimizer_config = BaseModelConfig(
        optimizer=BaseOptimizerParams(
            _target_="torch.optim.Adam", lr=0.001, weight_decay=1e-4
        ),
        lr_scheduler=BaseLRSchedulerParams(
            _target_="torch.optim.lr_scheduler.StepLR", step_size=30, gamma=0.1
        ),
    )

    print("配置：")
    print(f"  优化器: {single_optimizer_config.optimizer._target_}")
    print(f"  学习率: {single_optimizer_config.optimizer.lr}")
    print(f"  调度器: {single_optimizer_config.lr_scheduler._target_}")
    print("  作用范围: 整个模型的所有参数")

    # 场景 2: GAN 模型 - 多个优化器（不同模块独立优化）
    print("\n📝 场景 2: 多个优化器 - GAN 训练")
    gan_optimizer_config = BaseModelConfig(
        optimizer={
            "generator": BaseOptimizerParams(
                _target_="torch.optim.Adam", lr=0.0002, betas=(0.5, 0.999)
            ),
            "discriminator": BaseOptimizerParams(
                _target_="torch.optim.Adam", lr=0.0001, betas=(0.5, 0.999)
            ),
        },
        lr_scheduler={
            "generator": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.StepLR", step_size=50, gamma=0.5
            ),
            "discriminator": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.CosineAnnealingLR", T_max=100
            ),
        },
    )

    print("配置：")
    for name, opt_config in gan_optimizer_config.optimizer.items():
        sched_config = gan_optimizer_config.lr_scheduler[name]
        print(f"  {name}:")
        print(f"    优化器: {opt_config._target_} (lr={opt_config.lr})")
        print(f"    调度器: {sched_config._target_}")

    # 场景 3: 多任务学习 - 不同组件不同策略
    print("\n📝 场景 3: 多任务学习 - 不同学习率策略")
    multitask_config = BaseModelConfig(
        optimizer={
            "backbone": BaseOptimizerParams(
                _target_="torch.optim.Adam",
                lr=1e-4,  # 预训练部分用小学习率
                weight_decay=1e-5,
            ),
            "task1_head": BaseOptimizerParams(
                _target_="torch.optim.Adam",
                lr=1e-3,  # 新任务头用大学习率
                weight_decay=1e-4,
            ),
            "task2_head": BaseOptimizerParams(
                _target_="torch.optim.SGD",
                lr=1e-2,  # 第二个任务用不同的优化器
                momentum=0.9,
            ),
        },
        lr_scheduler={
            "backbone": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.CosineAnnealingLR", T_max=200
            ),
            "task1_head": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.StepLR", step_size=30, gamma=0.1
            ),
            "task2_head": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.ReduceLROnPlateau",
                factor=0.5,
                patience=10,
            ),
        },
    )

    print("配置：")
    for name, opt_config in multitask_config.optimizer.items():
        sched_config = multitask_config.lr_scheduler[name]
        print(f"  {name}:")
        print(f"    优化器: {opt_config._target_} (lr={opt_config.lr})")
        print(f"    调度器: {sched_config._target_}")


def demo_config_validation():
    """演示配置验证逻辑"""
    print("\n🔍 配置验证演示")
    print("=" * 50)

    def validate_optimizer_config(config: BaseModelConfig) -> bool:
        """验证优化器配置的一致性"""

        # 检查类型一致性
        optimizer_is_dict = isinstance(config.optimizer, dict)
        scheduler_is_dict = isinstance(config.lr_scheduler, dict)

        if optimizer_is_dict != scheduler_is_dict:
            print("❌ 错误：optimizer 和 lr_scheduler 类型不匹配")
            print(f"   optimizer 是字典: {optimizer_is_dict}")
            print(f"   lr_scheduler 是字典: {scheduler_is_dict}")
            return False

        # 如果都是字典，检查键的一致性
        if optimizer_is_dict:
            opt_keys = set(config.optimizer.keys())
            sched_keys = set(config.lr_scheduler.keys())

            if opt_keys != sched_keys:
                print("❌ 错误：optimizer 和 lr_scheduler 的键不匹配")
                print(f"   optimizer 键: {opt_keys}")
                print(f"   lr_scheduler 键: {sched_keys}")
                return False

            print(f"✅ 多优化器配置验证通过，模块: {opt_keys}")
        else:
            print("✅ 单优化器配置验证通过")

        return True

    # 测试正确配置
    correct_config = BaseModelConfig(
        optimizer={
            "generator": BaseOptimizerParams(_target_="torch.optim.Adam", lr=0.002),
            "discriminator": BaseOptimizerParams(_target_="torch.optim.SGD", lr=0.001),
        },
        lr_scheduler={
            "generator": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.StepLR"
            ),
            "discriminator": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.CosineAnnealingLR"
            ),
        },
    )

    validate_optimizer_config(correct_config)

    # 测试错误配置
    print("\n测试错误配置：")
    incorrect_config = BaseModelConfig(
        optimizer=BaseOptimizerParams(_target_="torch.optim.Adam"),  # 单个
        lr_scheduler={  # 字典
            "generator": BaseLRSchedulerParams(
                _target_="torch.optim.lr_scheduler.StepLR"
            )
        },
    )

    validate_optimizer_config(incorrect_config)


def demo_implementation_logic():
    """演示在模型中的实现逻辑"""
    print("\n🔧 实现逻辑演示")
    print("=" * 50)

    print(
        """
在 BaseModel 的 configure_optimizers 方法中：

```python
def configure_optimizers(self):
    if isinstance(self.config.optimizer, dict):
        # 多优化器模式
        return self._configure_multiple_optimizers()
    else:
        # 单优化器模式
        return self._configure_single_optimizer()

def _configure_single_optimizer(self):
    # 优化整个模型的参数
    optimizer = instantiate(self.config.optimizer, params=self.parameters())
    scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer)
    return [optimizer], [scheduler]

def _configure_multiple_optimizers(self):
    optimizers = []
    schedulers = []
    
    for module_name, opt_config in self.config.optimizer.items():
        # 获取指定模块的参数
        if hasattr(self, module_name):
            module_params = getattr(self, module_name).parameters()
        else:
            raise ValueError(f"模块 '{module_name}' 不存在")
        
        # 创建优化器和调度器
        optimizer = instantiate(opt_config, params=module_params)
        scheduler = instantiate(self.config.lr_scheduler[module_name], optimizer=optimizer)
        
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    return optimizers, schedulers
```
    """
    )


if __name__ == "__main__":
    demo_unified_optimizer_config()
    demo_config_validation()
    demo_implementation_logic()
