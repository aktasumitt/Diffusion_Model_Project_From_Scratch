from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str 
    

@dataclass
class DataTransformationConfig:
    train_data_path: Path 
    train_dataset_save_path: Path 
    normalize_mean: tuple 
    normalize_std: tuple 
    img_size: int
    
    
@dataclass
class ModelIngestionConfig:
    u_net_save_path:Path
    diffusion_save_path:Path
    beta_start:int
    beta_end:int
    n_timesteps:int
    img_size:int
    device:str
    channel_size:int
    label_size:int
    embed_size:int
    
    
@dataclass
class TrainingConfig:
    unet_load_path:Path
    diffusion_load_path:Path
    learning_rate:float
    betas:tuple
    train_dataset_path:Path
    batch_size:int
    checkpoint_path:Path
    device:str
    epochs:int
    img_size:int
    results_save_path:Path
    final_unet_model_path:Path
    final_diffusion_path:Path
    labels:dict
 
    
@dataclass
class TestConfig:
    u_net_model_path: Path 
    diffusion_path: Path 
    img_size: int 
    img_save_path:Path



# @dataclass
# class PredictionConfig:
#     noise_size: int 
#     predicted_img_save_path: Path 
#     model_path: Path 
#     device: str 
#     labels: dict











