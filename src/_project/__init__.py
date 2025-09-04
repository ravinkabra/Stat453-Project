"""
🔧 项目统整器模块

提供统一的项目管理和执行接口：
- ProjectManager: 核心项目管理器
- ProjectConfig: 项目配置类
- 便捷的创建函数

使用示例：
```python
from src._project import create_training_project

# 快速创建项目
project = create_training_project(
    model_config={"_target_": "src.model.example.model.MyModel"},
    output_dir="./my_experiment"
)

# 开始训练
project.train()
```
"""

from .manager import ProjectManager, ProjectBuilder, create_training_project
from .config import (
    ProjectConfig,
    get_unet_project_config,
    get_llm_project_config,
    # get_minimal_project_config,
)

__all__ = [
    # 核心类
    "ProjectManager",
    "ProjectBuilder",
    "ProjectConfig",
    # 便捷函数
    "create_training_project",
    # 预定义配置
    "get_unet_project_config",
    "get_llm_project_config",
    "get_minimal_project_config",
]
