from src.components.model.diffusion import Diffusion
from src.components.model.u_net import Unet
from src.utils import save_obj
from src.logger import logger
from src.entity.config_entity import ModelIngestionConfig
from src.exception.exception import ExceptionNetwork,sys


class ModelIngestion():
    
    def __init__(self, config: ModelIngestionConfig):
        self.config = config
        self.diffusion = Diffusion(beta_start=self.config.beta_start,
                                    beta_end=self.config.beta_end,
                                    n_timesteps=self.config.n_timesteps,
                                    img_size=self.config.img_size)
        self.Unet = Unet(
            channel_size=self.config.channel_size,
            devices=self.config.device,
            num_classes=self.config.label_size,
            embed_dim=self.config.embed_size  
        )
        
    def initiate_and_save_model(self):
      try:
        save_obj(self.Unet, self.config.u_net_save_path)
        save_obj(self.diffusion, self.config.diffusion_save_path)
        logger.info("Generator ve Discriminator modelleri artifacts içerisine kaydedildi")
      except Exception as e:
        raise ExceptionNetwork(e,sys)
        
if __name__ == "__main__":
    config = ModelIngestionConfig()
    model_ingestion = ModelIngestion(config)
    model_ingestion.initiate_and_save_model()

