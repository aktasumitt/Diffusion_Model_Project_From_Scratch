from src.exception.exception import ExceptionNetwork,sys

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
        
    def update_model_average(self, ema_model, current_model):
        try:   
            
            for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
                old_weight, up_weight = ema_params.data, current_params.data
                ema_params.data = old_weight * self.beta + (1 - self.beta) * up_weight
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    
    def step_ema(self, ema_model, unet_model, step_start_ema=2000):
        try:
            
            unet_model.eval()        
            if self.step < step_start_ema:
                ema_model.load_state_dict(unet_model.state_dict())
                self.step += 1
            
            self.update_model_average(ema_model, unet_model)
            self.step += 1       
             
        except Exception as e:
            raise ExceptionNetwork(e,sys)