import torch
from torchvision.utils import make_grid
from src.exception.exception import ExceptionNetwork,sys
import numpy as np
import PIL.Image
from src.utils import load_obj
from src.entity.config_entity import TestConfig
import torch
import numpy as np
import PIL.Image

class TestModule():
    def __init__(self, config: TestConfig):
        self.config = config

    def test_model(self,model_unet,difussion_model,img_size,test_image_save_path,device:str="cuda"):  # Generator
        # We will see generated fake images for each labels 5 times
       
        try:    
            with torch.no_grad():
                random_noise=torch.randn(50,3,img_size,img_size).to(device)
                labels=torch.tensor([0,1,2,3,4,5,6,7,8,9],dtype=torch.int).repeat(5).to(device)
                img_pred=difussion_model.denoising(model_unet,random_noise,labels,device)
                
                # Tensorları grid haline getir (nrow belirterek sütun sayısını ayarlıyoruz)
                grid_tensor = make_grid(img_pred, nrow=10)
                grid_tensor = (grid_tensor*255).type(torch.uint8)  # Normalizasyonu kaldır, uint8 formatına çevir

                # Grid görüntüsünü NumPy formatına çevir
                grid_np = grid_tensor.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

                # NumPy array’i PIL Image'e çevir
                grid_pil = PIL.Image.fromarray(grid_np)
                
                grid_pil.save(test_image_save_path)
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        

        
    def initiate_testing(self):
        
        model = load_obj(path=self.config.u_net_model_path).to("cuda")
        diffusion_model = load_obj(path=self.config.diffusion_path)
        self.test_model(model,difussion_model=diffusion_model,
                        img_size=self.config.img_size,test_image_save_path=self.config.img_save_path)
        
    
        
        