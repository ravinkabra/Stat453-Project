"""
🚀 项目统整器和增强训练器演示

展示新的架构组织：
- _project: 🔧 项目统整器 (工具)
- _trainer: 🔧 增强训练器 (工具)
- 统一的项目管理和执行流程
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src._project import (
    create_training_project,
    ProjectManager,
    get_unet_project_config,
    get_llm_project_config,
    get_minimal_project_config,
)
from src._trainer import (
    create_enhanced_trainer,
    get_development_trainer_config,
    get_production_trainer_config,
    get_research_trainer_config,
)


def demo_project_manager():
    """演示项目管理器功能"""
    print("🔧 === 项目管理器演示 ===")

    # 方式1: 快速创建项目
    project = create_training_project(
        model_config={
            "_target_": "src.model.example.model.MyModel",
            "hidden_size": 256,
        },
        output_dir="./demo_outputs/quick_project",
        project_name="quick_demo",
    )

    print("快速创建的项目:")
    project.print_summary()

    # 方式2: 使用预定义配置
    unet_config = get_unet_project_config()
    unet_project = ProjectManager(unet_config)

    print("UNet 项目配置:")
    unet_project.print_summary()

    # 方式3: 完整配置
    llm_config = get_llm_project_config()
    llm_project = ProjectManager(llm_config)

    print("LLM 项目配置:")
    llm_project.print_summary()


def demo_enhanced_trainer():
    """演示增强训练器功能"""
    print("\n🔧 === 增强训练器演示 ===")

    # 方式1: 快速创建
    trainer = create_enhanced_trainer(
        experiment_name="demo_experiment", max_epochs=20, devices=1, accelerator="cpu"
    )

    print("快速创建的增强训练器:")
    trainer.print_summary()

    # 方式2: 使用预定义配置
    dev_config = get_development_trainer_config()
    dev_trainer = create_enhanced_trainer(**dev_config.__dict__)

    print("开发环境训练器:")
    dev_trainer.print_summary()

    prod_config = get_production_trainer_config()
    print("生产环境配置摘要:")
    print(f"  - 实验名称: {prod_config.experiment_name}")
    print(f"  - 最大轮数: {prod_config.max_epochs}")
    print(f"  - 精度: {prod_config.precision}")
    print(f"  - 早停: {prod_config.early_stopping}")
    print(f"  - 确定性: {prod_config.deterministic}")


def demo_naming_conventions():
    """演示命名约定的实际应用"""
    print("\n📝 === 命名约定应用演示 ===")

    print("🔧 _xxx (工具/内部/稳定):")
    print("  ✓ src/_project/     - 项目统整器工具")
    print("  ✓ src/_trainer/     - 增强训练器工具")
    print("  ✓ src/_utils/       - 通用工具模块 (建议)")
    print("  ✓ src/metric/_manager/ - 指标管理器 (已存在)")

    print("\n🧪 xxx_ (实验/临时/开发中):")
    print("  ⚠️  optimizer_experimental_.py - 实验性优化器")
    print("  ⚠️  callback_beta_.py          - 测试版回调")
    print("  ⚠️  features_/                 - 新功能开发目录")
    print("  ⚠️  model_v2_.py               - 模型v2草稿")

    print("\n📁 建议的目录结构:")
    structure = """
    src/
    ├── _project/           # 🔧 项目统整器
    ├── _trainer/           # 🔧 增强训练器
    ├── _utils/             # 🔧 通用工具
    ├── _core/              # 🔧 核心基础设施
    ├── model/              # 📦 模型组件
    │   ├── base/
    │   ├── example/
    │   └── experimental_/  # 🧪 实验性模型
    ├── optimizer/          # 📦 优化器组件
    │   └── research_/      # 🧪 研究中的优化器
    ├── callback/           # 📦 回调组件
    │   ├── sample_saver/
    │   └── beta_/          # 🧪 测试版回调
    └── features_/          # 🧪 新功能开发
    """
    print(structure)


def demo_integration_workflow():
    """演示集成工作流程"""
    print("\n🔄 === 集成工作流程演示 ===")

    print("步骤1: 创建项目配置")
    project_config = {
        "name": "integration_demo",
        "model": {"_target_": "src.model.example.model.DemoModel"},
        "trainer": {"max_epochs": 5, "devices": 1, "accelerator": "cpu"},
    }

    print("步骤2: 创建项目管理器")
    # 注意：这里展示概念，实际使用时需要有真实的模型类
    try:
        from src._project.base.config import ProjectConfig

        config = ProjectConfig(**project_config)
        project = ProjectManager(config)

        print("步骤3: 项目摘要")
        project.print_summary()

        print("步骤4: 如果有真实模型，可以这样训练:")
        print("  project.train()  # 一键启动训练")
        print("  project.validate()  # 运行验证")
        print("  project.test()  # 运行测试")

    except ImportError as e:
        print(f"⚠️ 导入错误 (预期): {e}")
        print("这是正常的，因为我们还没有实际的模型实现")


def demo_configuration_flexibility():
    """演示配置的灵活性"""
    print("\n⚙️ === 配置灵活性演示 ===")

    print("1. 最小化配置:")
    minimal = get_minimal_project_config()
    print(f"   - 项目名称: {minimal.name}")
    print(f"   - 模型目标: {minimal.model.get('_target_', 'N/A')}")
    print(f"   - 训练轮数: {minimal.trainer.get('max_epochs', 'N/A')}")

    print("\n2. 完整的UNet配置:")
    unet = get_unet_project_config()
    print(f"   - 项目名称: {unet.name}")
    print(f"   - 描述: {unet.description}")
    print(f"   - 回调数量: {len(unet.callbacks or {})}")
    print(f"   - 标签: {unet.tags}")
    print(f"   - 随机种子: {unet.seed}")

    print("\n3. LLM项目配置:")
    llm = get_llm_project_config()
    print(f"   - 项目名称: {llm.name}")
    print(f"   - 确定性模式: {llm.deterministic}")
    print(f"   - 训练器精度: {llm.trainer.get('precision', 'N/A')}")
    print(f"   - 样本保存策略: 按 epoch 组织目录")


def run_comprehensive_demo():
    """运行综合演示"""
    print("🚀 项目统整器和增强训练器 - 综合演示")
    print("=" * 60)

    try:
        demo_project_manager()
        demo_enhanced_trainer()
        demo_naming_conventions()
        demo_integration_workflow()
        demo_configuration_flexibility()

        print("\n" + "=" * 60)
        print("✅ 演示完成！")

        print("\n📋 关键要点总结:")
        print("1. 🔧 _project: 统整所有组件的项目管理器")
        print("2. 🔧 _trainer: 增强的训练器包装")
        print("3. 📝 命名约定: _xxx=工具, xxx_=实验")
        print("4. ⚙️ 配置驱动: 通过配置控制所有行为")
        print("5. 🔌 插件化: 易于扩展和定制")

        print("\n🎯 使用建议:")
        print("- 新手: 使用预定义配置和便捷函数")
        print("- 进阶: 自定义配置类和组件")
        print("- 专家: 扩展基础类和工具模块")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_demo()
