from config.mnist import project
from src._project.base.project import ProjectManager

project_manager = ProjectManager(project)
project_manager.run()
# import multidict