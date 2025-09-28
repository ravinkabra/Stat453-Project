# from config.mnist import project
# from src._project.base.project import ProjectManager

# project_manager = ProjectManager(project)
# project_manager.run()
# import multidict

from src._project.base.project import ProjectManager
from src._project.base.config import ProjectConfig

# 使用新的 from_yaml 方法加载配置
project = ProjectConfig.from_yaml("config/m2a_new_Yuan.yaml")
print("Loaded project config:")
print(f"  - Name: {project.name}")
print(f"  - Experiment: {project.experiment_name}")
print(f"  - Model: {project.model._target_}")
print(f"  - Mode: {project.mode}")

ProjectManager(project).run()
