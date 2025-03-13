import torch
import matplotlib.pyplot as plt
from src.utils import load_obj
from src.entity.config_entity import PredictionConfig
from src.exception.exception import ExceptionNetwork, sys
import os

class PredictionModule():
    def __init__(self, config: PredictionConfig):
        self.config = config

    def predict_model(self, model_generator):
        try:
            os.makedirs(self.config.predicted_img_save_path,exist_ok=True)
            
            model_generator.eval()
            random_noise = torch.randn((5, self.config.noise_size, 1, 1))
            
            with torch.no_grad():
                fake_imgs = model_generator(random_noise)
            
            image_paths = []
            for i, img_tensor in enumerate(fake_imgs):
                
                img_tensor=(img_tensor-img_tensor.min())/(img_tensor.max()-img_tensor.min()) # normalization
                img_tensor=img_tensor.squeeze(0)
                im_denormalized = (img_tensor*255).type(torch.uint8)
                
                img_path = os.path.join(self.config.predicted_img_save_path, f"prediction_{i}.jpg")
                plt.imsave(img_path,im_denormalized,cmap="gray")
                
                image_paths.append(img_path)
                
            return image_paths
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def initiate_predict(self):
        model = load_obj(path=self.config.model_path)
        return self.predict_model(model_generator=model)


        
    
        