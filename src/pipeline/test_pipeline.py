from src.config.configuration import Configuration
from src.components.testing.testing import TestModule


class TestPipeline():
    def __init__(self):

        configuration = Configuration()
        self.test_config = configuration.test_config()

    def run_testing(self):

        testing = TestModule(self.test_config)
        testing.initiate_testing()


if __name__=="__main__":
    
    test_pipeline=TestPipeline()
    test_pipeline.run_testing()
    


