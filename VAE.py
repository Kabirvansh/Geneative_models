import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[64, 128, 256, 512], 
                 latent_dim: int=1, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]

        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # TODO: handle the label embedding here
        self.class_emb = nn.Embedding(num_classes, self.mid_size[0] * self.mid_size[1] * self.mid_size[2])
        
        # TODO: define the encoder part of your network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[2], mid_channels[3], kernel_size=1, stride=1, padding=0), 
            nn.ReLU()
        )
        
        # TODO: define the network/layer for estimating the mean
        self.mean_net = nn.Linear(mid_channels[-1] * self.mid_size[1] * self.mid_size[2], latent_dim)
        
        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = nn.Linear(mid_channels[-1] * self.mid_size[1] * self.mid_size[2], latent_dim)

        # TODO: define the decoder part of your network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + mid_channels[-1], mid_channels[-2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-2], mid_channels[-3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-3], mid_channels[-4], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels[-4], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = self.encoder(x).view(x.size(0), -1)

        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        # TODO: decoding the sample
        label_emb = self.class_emb(label).view(x.size(0), *self.mid_size)
        sample = sample.view(x.size(0), -1, 1, 1).expand(-1, -1, self.mid_size[1], self.mid_size[2])
        sample = torch.cat([sample, label_emb], dim=1)
        out = self.decoder(sample)

        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)  #####
        noise = torch.randn_like(std)  #####
        sample = noise * std + mean

        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # TODO: sample from standard Normal distrubution
        noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        label_emb = self.class_emb(labels).view(num_samples, self.mid_size[0], self.mid_size[1], self.mid_size[2])     #################
        noise = noise.expand(-1, -1, self.mid_size[1], self.mid_size[2])###############
        noise = torch.cat([noise, label_emb], dim=1) #####################

        # TODO: decode the noise based on the given labels
        out = self.decode(noise, labels)

        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        out = self.decoder(sample)

        return out
