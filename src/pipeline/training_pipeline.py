from src.config.configuration import Configuration
from src.components.training.training import Training


class TrainingPipeline():
    def __init__(self):

        configuration = Configuration()
        self.training_config = configuration.training_config()

    def run_training(self):

        training = Training(self.training_config)
        training.start_training_with_mlflow()
        
        


if __name__=="__main__":
    
    training_pipeline=TrainingPipeline()
    training_pipeline.run_training()
