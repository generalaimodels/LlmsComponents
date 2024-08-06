import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ImageGenerationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        use_batch_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        assert len(hidden_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All parameter lists must have the same length"
        
        self.layers = nn.ModuleList()
        in_ch = in_channels
        
        for i, out_ch in enumerate(hidden_channels):
            conv = nn.Conv2d(in_ch, out_ch, kernel_sizes[i], strides[i], paddings[i])
            self.layers.append(conv)
            
            if use_batch_norm:
                self.layers.append(nn.BatchNorm2d(out_ch))
            
            self.layers.append(activation)
            in_ch = out_ch
        
        self.final_conv = nn.Conv2d(in_ch, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_batch_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class ImageGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: List[int],
        image_size: Tuple[int, int],
        use_batch_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        
        # Calculate initial spatial dimensions
        self.init_height = image_size[0] // (2 ** (len(channels) - 1))
        self.init_width = image_size[1] // (2 ** (len(channels) - 1))
        
        self.linear = nn.Linear(latent_dim, channels[0] * self.init_height * self.init_width)
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(UpsampleBlock(
                channels[i], channels[i+1],
                use_batch_norm=use_batch_norm,
                activation=activation
            ))
        
        self.final_conv = nn.Conv2d(channels[-1], 3, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.linear(z)
        x = x.view(x.size(0), self.channels[0], self.init_height, self.init_width)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)
        return torch.tanh(x)


# Example usage:
latent_dim = 100
channels = [512, 256, 128, 64]
image_size = (64, 64)

generator = ImageGenerator(latent_dim, channels, image_size)
print(generator)
# Generate a batch of 16 images
batch_size = 16
z = torch.randn(batch_size, latent_dim)
generated_images = generator(z)

print(generated_images.shape)  # Should output: torch.Size([16, 3, 64, 64])