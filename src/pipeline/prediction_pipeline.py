from src.config.configuration import Configuration
from src.components.predict.predict import PredictionModule

class PredictionPipeline():
    def __init__(self):

        configuration=Configuration()
        self.prediction_config=configuration.prediction_config()
    
    def run_prediction_pipeline(self):
    
        prediction=PredictionModule(self.prediction_config)
        return prediction.initiate_predict()
                