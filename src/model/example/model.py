

from .config import ExampleModelConfig
from ..base.model import BaseModel
from ...network.example import ExampleNetwork

class ExampleModel(BaseModel):
    def __init__(self, config: ExampleModelConfig):
        super().__init__(config)
        self.example_param = config.example_param
        self.example_network = ExampleNetwork(config.example_network)

    def forward(self, x):
        # Example forward method using the example network
        self.log
        return self.example_network(x)