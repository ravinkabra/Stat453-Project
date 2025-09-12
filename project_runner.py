# from config.mnist import project
# from src._project.base.project import ProjectManager

# project_manager = ProjectManager(project)
# project_manager.run()
# # import multidict

from src._project.base.project import ProjectConfig, ProjectManager
import yaml 
project = yaml.load(open("./config/m2a_example.yaml","r"),Loader=yaml.FullLoader)
print(project)
project = ProjectConfig(**project)
project_manager = ProjectManager(project)
project_manager.run()