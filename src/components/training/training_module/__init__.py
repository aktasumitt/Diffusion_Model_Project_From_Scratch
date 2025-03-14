import torch
import tqdm
from src.exception.exception import ExceptionNetwork, sys

# Training Func
def training_model(train_Dataloader, unet_model, diffussion_Model, ema_model,ema, optimizer, loss_fn, devices):

    try:
        unet_model.train()
        train_loss_value = 0

        progress_bar = tqdm.tqdm(range(len(train_Dataloader)), "Training Progress")

        for batch_train, (img, label) in enumerate(train_Dataloader):

            img_train = img.to(devices)
            label_train = label.to(devices)

            optimizer.zero_grad()
            CFG_SCALE = torch.randint(1, 101, (1,)).item()
            if CFG_SCALE < 10:
                label_train = None
                
            t = diffussion_Model.random_Timesteps(img_train.shape[0],devices)  # Create Random Tİmesteps
            noisy_img, noise = diffussion_Model.noising_to_Image(img_train, t,devices)  # image noising Step
            
            # Pred noise with model VAE
            pred_noise = unet_model(noisy_img, t, label_train)
            
            # Loss MSE between pred_noise and real noise
            loss_train = loss_fn(pred_noise, noise)

            train_loss_value += loss_train.item()

            loss_train.backward()
            optimizer.step()
            
            # Upgrade EMA
            with torch.no_grad():
                ema.step_ema(ema_model, unet_model)
                
            progress_bar.update(1)
            
        loss=train_loss_value/(batch_train+1)
        
        progress_bar.set_postfix({"Loss_Train": loss})

        progress_bar.close()
        return loss

    except Exception as e:
        raise ExceptionNetwork(e, sys)
