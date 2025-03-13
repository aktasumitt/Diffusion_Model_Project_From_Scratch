from torchvision.transforms import transforms
from src.components.data_transformation.dataset_module import DatasetModule
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from src.utils import save_obj
import torch
from pathlib import Path
from src.exception.exception import ExceptionNetwork,sys
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def transformer(self):
        try:
            transformer = transforms.Compose([transforms.Normalize(self.config.normalize_mean, self.config.normalize_std)])
            
            return transformer
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def load_ingested_data(self, train: bool, local_dir: Path):
        try:   
            dataset = CIFAR10(root=local_dir, train=train, download=False)
            images = dataset.data
            labels = dataset.targets
            images = torch.from_numpy(images).float()
            labels = torch.tensor(labels)
            return images, labels
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def create_and_save_dataset_transformed(self):
        try:
            transformer = self.transformer()
            train_images, train_labels = self.load_ingested_data(train=True, local_dir=self.config.train_data_path)
            
            train_dataset = DatasetModule(train_images, train_labels, transformer) 
            save_obj(train_dataset, save_path=self.config.train_dataset_save_path)
            
        except Exception as e:
            raise ExceptionNetwork(e, sys)

if __name__ == "__main__":
    config = DataTransformationConfig()
    data_transformation = DataTransformation(config)
    data_transformation.create_and_save_dataset_transformed()
