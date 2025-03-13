from src.components.training.training_module import training_model
from src.components.training.validation_module import validation_model
from src.utils import (save_as_json, load_obj, load_checkpoints,save_checkpoints, save_obj)

from torch.utils.data import DataLoader
import torch
from src.logger import logger
from src.exception.exception import ExceptionNetwork, sys
import mlflow
from src.entity.config_entity import TrainingConfig

import dagshub
dagshub.init(repo_owner='umitaktas', repo_name='Diffusion_Model_Project_From_Scratch', mlflow=True)



class Training():
    def __init__(self, config: TrainingConfig, TEST_MODE: bool = False):
        self.config = config
        self.TEST_MODE = TEST_MODE
        self.unet_model = load_obj(self.config.unet_load_path).to(self.config.device)
        self.diffusion = load_obj(self.config.diffusion_load_path)
        self.optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=self.config.learning_rate, betas=self.config.betas)
        self.loss_fn=torch.nn.MSELoss()

    def load_object(self):
        try:
            train_dataset = load_obj(self.config.train_dataset_path)
            return DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        except Exception as e:
            ExceptionNetwork(e, sys)

    def load_checkpoints(self, load: bool):
        starting_epoch = 1
        if load:
            starting_epoch = load_checkpoints(path=self.config.checkpoint_path,
                                              model=self.unet_model,
                                              optimizer=self.optimizer)
            
            logger.info(f"Checkpoints loaded. Training starts from epoch {starting_epoch}.")
        return starting_epoch

    def initiate_training(self):
        try:
            result_list = []
            train_dataloader = self.load_object()
            
            starting_epoch = self.load_checkpoints(load=False)   
            
            for epoch in range(starting_epoch, self.config.epochs+1):
                loss_train = training_model(train_Dataloader=train_dataloader,
                                                     unet_model=self.unet_model,
                                                     diffussion_Model=self.diffusion,
                                                     optimizer=self.optimizer,
                                                     loss_fn=self.loss_fn,
                                                     devices=self.config.device)
                
                save_checkpoints(
                    save_path=self.config.checkpoint_path,
                    epoch=epoch,
                    model=self.unet_model,
                    optimizer=self.optimizer)
                
                images_generated = validation_model(model_unet=self.unet_model,
                                                    difussion_model=self.diffusion,
                                                    img_size=self.config.img_size,
                                                    device=self.config.device)
                
                
                logger.info(f"Checkpoint saved at epoch {epoch}.")
                
                metrics = {"loss_train": loss_train}
                result_list.append(metrics)
                
                mlflow.log_metrics(metrics=metrics, step=epoch)
                mlflow.log_image(images_generated, key=f"epoch_{epoch}_batch_grid.png", step=epoch)
            
            save_as_json(data=result_list, save_path=self.config.results_save_path)
            logger.info("Training results saved as JSON.")
            
            save_obj(self.unet_model, save_path=self.config.final_unet_model_path)
            save_obj(self.diffusion, save_path=self.config.final_diffusion_path)
            logger.info("Final models saved.")
            
        except Exception as e:
            ExceptionNetwork(e, sys)

    def start_training_with_mlflow(self):
        try:
            uri = "https://dagshub.com/umitaktas/Diffusion_Model_Project_From_Scratch.mlflow"
            mlflow.set_tracking_uri(uri=uri)
            logger.info(f"MLflow tracking started on {uri}.")
            
            mlflow.set_experiment("MLFLOW MyFirstExperiment")
            params = {
                "Batch_size": self.config.batch_size, "Learning_rate": self.config.learning_rate,
                "Betas": self.config.betas, "Epoch": self.config.epochs, "Labels":self.config.labels,
            }
            
            
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.set_tag("Pytorch Training Info", "Image classification training")
                self.initiate_training()
                logger.info("Training completed. Metrics, parameters, and model saved in MLflow.")
        except Exception as e:
            ExceptionNetwork(e, sys)

if __name__ == "__main__":
    config = TrainingConfig()
    training = Training(config)
    training.start_training_with_mlflow()
