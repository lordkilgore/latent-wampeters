import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class AutoEncoder(nn.Module):
    """Symmetric convolutional autoencoder. Expects (3, 224, 224) input.
    - factor of two spatial downsampling and channel upsampling
    - pooling omitted (loses precise coordinate location of features)
    - gelu for smooth latent space, better interpolation
    - tanh for normalized geometry in latent space, makes interpolation more predictable
      and avoids cases where interpolations lie in regions the encoder never maps to.
    """
    def __init__(self, 
                hidden_dim: int,
                *,
                device: "cuda" if torch.cuda.is_available() else None
        ) -> None:

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, device=device), # -> (32, 112, 112) 
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, device=device), # -> (64, 56, 56)
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, device=device), # -> (128, 28, 28)
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, device=device), # -> (256, 14, 14)
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, device=device), # -> (512, 7, 7)
            nn.GELU(),
            nn.Flatten(), # -> (512*7*7)
            nn.Linear(in_features=25088, out_features=hidden_dim, device=device), # -> (hidden_dim)
            nn.Tanh()
        )

        self.decoder_input = nn.Linear(in_features=hidden_dim, out_features=25088, device=device) # -> (512*7*7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, device=device), # -> (256, 14, 14)
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, device=device), # -> (128, 28, 28)
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, device=device), # -> (64, 56, 56)
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, device=device), # -> (32, 112, 112)
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, device=device), # -> (3, 224, 224)
            nn.Sigmoid()
        )

    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        decoder_input = self.decoder_input(latent).view(-1, 512, 7, 7)
        x_hat = self.decoder(decoder_input)
        return x_hat, latent
