from src.entity.config_entity import (
    TestConfig, 
    TrainingConfig,
    DataIngestionConfig,
    ModelIngestionConfig,
    DataTransformationConfig
)

from src.constants.config import Config
from src.constants.params import Params
from src.constants.schema import Schema


class Configuration:

    def __init__(self):
        self.config = Config
        self.params = Params
        self.schema = Schema

    def data_ingestion_config(self):
        return DataIngestionConfig(
            train_data_path=self.config.train_data_path
        )

    def data_transformation_config(self):
        return DataTransformationConfig(
            train_data_path=self.config.train_data_path,
            train_dataset_save_path=self.config.train_dataset_save_path,
            normalize_mean=self.params.normalize_mean,
            normalize_std=self.params.normalize_std,
            img_size=self.params.img_size
        )

    def model_config(self):
        return ModelIngestionConfig(
            u_net_save_path=self.config.u_net_save_path,
            diffusion_save_path=self.config.diffusion_save_path,
            beta_start=self.params.beta_start,
            beta_end=self.params.beta_end,
            n_timesteps=self.params.n_timesteps,
            img_size=self.params.img_size,
            device=self.params.device,
            channel_size=self.params.channel_size,
            label_size=self.params.label_size,
            embed_size=self.params.embed_size
        )

    def training_config(self):
        return TrainingConfig(
            unet_load_path=self.config.u_net_save_path,
            diffusion_load_path=self.config.diffusion_save_path,
            train_dataset_path=self.config.train_dataset_save_path,
            checkpoint_path=self.config.checkpoint_save_path,
            final_unet_model_path=self.config.final_unet_model_path,
            final_diffusion_path=self.config.final_diffusion_path,
            results_save_path=self.config.results_save_path,
            batch_size=self.params.batch_size,
            device=self.params.device,
            learning_rate=self.params.learning_rate,
            betas=self.params.betas,
            epochs=self.params.epochs,
            img_size=self.params.img_size,
            labels=self.schema.labels
        )

    def test_config(self):
        return TestConfig(
            u_net_model_path=self.config.final_unet_model_path,
            diffusion_path=self.config.final_diffusion_path,
            img_size=self.params.img_size,
            img_save_path=self.config.test_img_save_path
        )

    # def prediction_config(self):
        
    #     configuration = PredictionConfig(noise_size=self.params.noise_dim,
    #                                      predicted_img_save_path=self.config.predicted_img_save_path,
    #                                      model_path=self.config.final_generator_model_path,
    #                                      device=self.params.device,
    #                                      labels=self.schema.labels
    #                                     )

    #     return configuration
    