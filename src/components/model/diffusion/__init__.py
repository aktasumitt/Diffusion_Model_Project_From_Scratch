import torch.nn as nn
import torch
from src.exception.exception import ExceptionNetwork,sys
import tqdm


class Diffusion():
    def __init__(self,beta_start,beta_end,n_timesteps,img_size):        
        try:  
            self.n_timesteps=n_timesteps
            self.img_size=img_size
            
            # For the formula to aplly noising and denoising process
            self.beta=torch.linspace(beta_start,beta_end,n_timesteps)
            self.alpha=1. - self.beta
            self.alpha_hat=torch.cumprod(self.alpha,dim=0)
            
        except Exception as e:
            raise ExceptionNetwork(e,sys) 
                
    def noising_to_Image(self,x,t,device):
        try:    
            sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None].to(device) # Boyutlandırma
            sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None].to(device)
            
            noise=torch.randn_like(x).to(device)
            
            noisy_img=(sqrt_alpha_hat*x)+(sqrt_one_minus_alpha_hat*noise)
            
            return noisy_img,noise
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    
    def random_Timesteps(self,batch_size,device):
        try:
            return torch.randint(1,self.n_timesteps,(batch_size,))
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    
    def denoising(self,model:None,random_noise,labels:None,device:None): # Test the model with random noisy img
        try:
            x=random_noise
            prog_bar=tqdm.tqdm(range(self.n_timesteps),"validation step")
            model.eval()

            
            for i in reversed(range(1,self.n_timesteps)):
                
                T=(torch.ones(x.shape[0])*i).long()
        
                predicted_noise=model(x,T,labels)
                
                # CFG predicted_noise. This process about, if we train conditional, after we need to predict uncoditional.
                # We use torch lerp to aproach conditional prediction from unconditional smoothly with 3 scale factor
                predicted_noise_unc=model(x,T,None)
                predicted_noise=torch.lerp(predicted_noise_unc,predicted_noise,3) 
                
                beta=self.beta[T][:,None,None,None].to(device)
                alpha=self.alpha[T][:,None,None,None].to(device)
                alpha_hat=self.alpha_hat[T][:,None,None,None].to(device)
                
                noise=(torch.randn_like(x) if i>1 else torch.zeros_like(x)).to(device)
                
                x = (1/alpha) * (x-((1-alpha)/torch.sqrt(1-alpha_hat))*predicted_noise) +(torch.sqrt(beta)*noise)
                prog_bar.update(1)
            
            prog_bar.close()
            
            model.train()
            
            # x=(x.clamp(-1,1) + 1) / 2
            # x=(x*255).type(torch.uint8)
            
            return x
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)

        