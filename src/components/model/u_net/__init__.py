import torch.nn as nn
import torch
from src.exception.exception import ExceptionNetwork, sys
from src.components.model.u_net.layers  import DoubleConv,DownBlock,MultiheadAttention,UpBlock


class Unet(nn.Module):
    def __init__(self,channel_size,devices,num_classes:None,embed_dim=256):
        super().__init__()
        self.devices=devices
        self.channel_size=channel_size
        self.embed_dim=embed_dim
        
        self.initial_block=DoubleConv(in_channels=channel_size,out_channels=64)
        
        self.down1=DownBlock(64,128)
        self.mha1=MultiheadAttention(128)
        
        self.down2=DownBlock(128,256)
        self.mha2=MultiheadAttention(256)
        
        self.down3=DownBlock(256,256)
        self.mha3=MultiheadAttention(256)
        
        self.bottleneck1=DoubleConv(256,512)
        self.bottleneck2=DoubleConv(512,512)
        self.bottleneck3=DoubleConv(512,256)
        
        self.up1=UpBlock(256*2,128)
        self.mha4=MultiheadAttention(128)
        
        self.up2=UpBlock(128*2,64)
        self.mha5=MultiheadAttention(64)
        
        self.up3=UpBlock(64*2,64)
        self.mha6=MultiheadAttention(64)
        
        self.out_block=nn.Conv2d(64,channel_size,kernel_size=1)
        
        if num_classes !=None:
            self.embed_label=nn.Embedding(num_classes,embed_dim)
        
        
    def Positional_encoding(self, t):
        try:
            inverse = 1 / 10000 ** (torch.arange(1, self.embed_dim, 2, dtype=torch.float) / self.embed_dim).to(self.devices)        
            
            # Tekrarlama işlemi
            repeated_t = t.unsqueeze(-1).repeat(1, (self.embed_dim // 2)).to(self.devices)  
            
            
            # pos_A ve pos_B'nin oluşturulması
            pos_A = torch.sin(repeated_t * inverse)
            pos_B = torch.cos(repeated_t * inverse)
            
            return torch.cat([pos_A, pos_B], dim=-1)
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
        
    def forward(self,x,t,label):
        try:
            t=self.Positional_encoding(t) # Positional Encoding for time
            
            if label!=None:
              label_embed=self.embed_label(label)
              t+=label_embed

            xi=self.initial_block(x) # 64,64,64
            
            d1=self.down1(xi,t) # 128,32,32
            s1=self.mha1(d1)

            d2=self.down2(s1,t)  # 256,16,16
            s2=self.mha2(d2)
            
            d3=self.down3(s2,t)  # 128,8,8
            s3=self.mha3(d3)
            
            bn1=self.bottleneck1(s3)
            bn2=self.bottleneck2(bn1)
            bn3=self.bottleneck3(bn2)
            
            u1=self.up1(bn3,s2,t)
            s4=self.mha4(u1)
            
            u2=self.up2(s4,s1,t)
            s5=self.mha5(u2)
            
            u3=self.up3(s5,xi,t)
            s6=self.mha6(u3)
            
            out=self.out_block(s6)
            
            return out
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
    

        
        
        
    
    

