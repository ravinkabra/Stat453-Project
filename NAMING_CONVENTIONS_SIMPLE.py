"""
Model Training Framework - 统一命名约定指南

🎯 简化的二元命名约定：
- _xxx : 工具/内部/核心组件 (稳定)
- xxx_ : 实验/临时/开发中功能 (不稳定)
"""

# ============================================================================
# 🔧 _xxx (下划线开头) = 工具/内部/核心组件
# ============================================================================

"""
用途：稳定的、内部的、工具性的组件
特点：
✓ 不直接对外暴露的API
✓ 框架内部使用的工具
✓ 核心基础设施
✓ 稳定且经过测试的代码
✓ 生产环境可靠

目录结构示例：
src/
├── _utils/                 # 🔧 工具模块
│   ├── config_base.py     # 配置基类工具
│   ├── registry.py        # 组件注册工具  
│   ├── validation.py      # 验证工具
│   └── logging.py         # 日志工具
│
├── _core/                  # 🔧 核心内部实现
│   ├── base_classes.py    # 抽象基类
│   ├── interfaces.py      # 接口定义
│   ├── protocols.py       # 协议定义
│   └── exceptions.py      # 异常定义
│
├── model/
│   └── _internal/         # 🔧 模型内部工具
│
├── metric/
│   ├── _manager/          # 🔧 指标管理器（已存在）
│   └── _aggregators/      # 🔧 聚合器工具
│
└── callback/
    └── _base/             # 🔧 基础回调工具

文件级别：
_constants.py              # 🔧 全局常量
_types.py                  # 🔧 类型定义  
_compat.py                 # 🔧 兼容性工具
"""

# ============================================================================
# 🧪 xxx_ (下划线结尾) = 实验/临时/开发中功能
# ============================================================================

"""
用途：实验性、临时性、开发中的功能
特点：
⚠️  实验性功能，API可能变化
⚠️  临时文件或备份文件
⚠️  开发中的新特性
⚠️  可能会被移除或重构的代码
⚠️  不建议在生产环境使用

目录结构示例：
src/
├── optimizer/
│   ├── experimental_/     # 🧪 实验性优化器
│   ├── research_/         # 🧪 研究阶段的优化器
│   └── beta_/             # 🧪 测试阶段的优化器
│
├── model/
│   ├── prototype_/        # 🧪 原型模型
│   ├── draft_/            # 🧪 草稿实现
│   └── wip_/              # 🧪 工作进行中 (work in progress)
│
├── network/
│   ├── experimental_/     # 🧪 实验性网络架构
│   └── research_/         # 🧪 研究网络
│
└── features_/             # 🧪 新功能开发

文件级别：
config_new_.py             # 🧪 新的配置实现
model_v2_.py               # 🧪 模型v2草稿
optimizer_research_.py     # 🧪 研究中的优化器
callback_experimental_.py # 🧪 实验性回调

临时/备份文件：
backup_.py                 # 🧪 备份文件
temp_.py                   # 🧪 临时文件
old_.py                    # 🧪 旧版本文件

避免关键字冲突：
class Config:
    type_: str = "model"       # 避免与 type 冲突
    class_: str = "BaseModel"  # 避免与 class 冲突
    import_: str = "module"    # 避免与 import 冲突
"""

# ============================================================================
# 📋 实际应用示例
# ============================================================================

"""
现有框架重构建议：

🔄 立即可实施的移动：
   当前：src/config_base.py
   建议：src/_utils/config_base.py
   
   当前：src/metric/_manager/
   保持：src/metric/_manager/    # 已经符合约定

🧪 实验性功能组织：
   # 如果你在开发新的优化器
   src/optimizer/adam_experimental_.py
   src/optimizer/research_/new_scheduler_.py
   
   # 如果你在测试新的回调
   src/callback/wandb_experimental_.py
   src/callback/tensorboard_beta_.py

📚 版本管理：
   # 保留旧版本用于兼容
   src/model/base/config_v1_.py
   
   # 新版本开发
   src/model/base/config_v2_.py

🔧 工具函数重组：
   src/_utils/
   ├── __init__.py
   ├── config_base.py         # 从根目录移动
   ├── dict_access.py         # 字典访问工具
   ├── hydra_helpers.py       # Hydra帮助函数
   └── validation.py          # 验证工具
"""

# ============================================================================
# 📝 导入约定
# ============================================================================

"""
导入规则：

1. 🔧 工具模块 (_xxx)：
   from src._utils.config_base import DictAccessMixin
   from src._core.interfaces import ModelInterface
   
2. 🧪 实验模块 (xxx_)：
   # 明确标记为实验性
   from src.optimizer.experimental_ import NewOptimizer  # EXPERIMENTAL
   from src.features_.new_callback import BetaCallback   # BETA

3. 📖 文档约定：
   # 工具模块
   '''🔧 Internal utility for configuration management'''
   
   # 实验模块  
   '''🧪 EXPERIMENTAL: This feature is under development and may change'''
"""

# ============================================================================
# 📁 .gitignore 建议
# ============================================================================

"""
# 临时和无意义的实验文件
*_temp_.py
*_backup_.py
*_old_.py
*_trash_.py
temp_/
backup_/
old_/

# 但保留有意义的实验文件
!*experimental_.py
!*research_.py
!*beta_.py
!*prototype_.py
"""

# ============================================================================
# 🚀 快速参考
# ============================================================================


def quick_reference():
    """快速参考指南"""
    print("🔧 _xxx = 工具/内部/稳定     (生产可用)")
    print("🧪 xxx_ = 实验/临时/开发中   (谨慎使用)")
    print()
    print("示例：")
    print("✓ _utils/config_base.py      # 🔧 稳定的配置工具")
    print("✓ optimizer_experimental_.py # 🧪 实验性优化器")
    print("✓ features_/                 # 🧪 新功能开发目录")
    print("✓ _core/exceptions.py        # 🔧 核心异常定义")


if __name__ == "__main__":
    print("Model Training Framework - 统一命名约定")
    quick_reference()
