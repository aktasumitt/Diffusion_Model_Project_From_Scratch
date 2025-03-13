from torchvision.datasets import CIFAR10
from src.logger import logger
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import DataIngestionConfig


class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            _ = CIFAR10(root=self.config.train_data_path,
                        train=True,
                        download=True)
            logger.info(f"Train dataset oluşturuldu, dir: [{self.config.train_data_path}] ")
            
            
        except Exception as e:
            raise ExceptionNetwork(e, sys)



if __name__=="__main__":
    
    
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()