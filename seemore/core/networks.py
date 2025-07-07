import torch
import torch.nn as nn
from .layers import LayerNorm, ResGroup



######################
# Meta Architecture
######################
class SeemoRe(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_experts: int = 6,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = False,
                 global_kernel_size: int = 11,
                 recursive: int = 2,
                 lr_space: str = 'linear',
                 topk: int = 2,):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        
        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)
        
        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [ResGroup(in_ch=embedding_dim, 
                       num_experts=num_experts, 
                       use_shuffle=use_shuffle,
                       topk=topk,
                       lr_space=lr_space,
                       recursive=recursive,
                       global_kernel_size=global_kernel_size) for i in range(num_layers)]
        )
        
        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale**2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x
        
        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)
                
        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x