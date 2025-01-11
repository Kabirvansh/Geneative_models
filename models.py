import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple


class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            self.betas = torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), num_steps) ** 2
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device

        # TODO: sample a random noise
        noise = torch.randn_like(x, device=device)
        time_step = time_step.view(-1)
        alpha_bar_t = self.alpha_bars.to(device)[time_step].view(-1, 1, 1, 1)


        # TODO: construct the noisy sample
        noisy_input = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb_scale = torch.exp(-torch.arange(half_dim, device=device).float() * math.log(10_000) / half_dim)
        scaled_time = time[:, None].float() * emb_scale
        embeddings = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)

        return embeddings

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, 
                 down_channels: List[int] = [64, 128, 128, 128, 128], 
                 up_channels: List[int] = [128, 128, 128, 128, 64], 
                 time_emb_dim: int = 128,
                 num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.time_emb_dim = time_emb_dim

        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.label_embedding = nn.Embedding(num_classes, time_emb_dim)

        adjusted_in_channels = in_channels + time_emb_dim

        # Downsampling layers
        self.encoder_blocks = nn.ModuleList()
        previous_channels = adjusted_in_channels
        for channels in down_channels:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous_channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=channels),
                    nn.GELU(),
                )
            )
            previous_channels = channels

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(previous_channels, previous_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=32, num_channels=previous_channels), 
            nn.Conv2d(previous_channels, previous_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )


        # Upsampling layers
        self.decoder_blocks = nn.ModuleList()
        for channels in up_channels:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous_channels + channels, channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=channels),
                    nn.GELU(),
                )
            )
            previous_channels = channels

        # Output layer
        self.final_conv = nn.Conv2d(previous_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Process time and label embeddings
        time_emb = SinusoidalPositionEmbeddings(self.time_emb_dim).to(x.device)(timestep)
        time_emb = self.time_embedding(time_emb)
        label_emb = self.label_embedding(label)
        context = time_emb + label_emb

        # Expand and concatenate context embeddings with input
        context_expanded = context.view(context.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, context_expanded], dim=1)

        # Downsample through encoder
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsample through decoder
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # Generate output
        return self.final_conv(x)


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
            nn.GELU(),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=4, stride=2, padding=1), 
            nn.GELU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=4, stride=2, padding=1), 
            nn.GELU(),
            nn.Conv2d(mid_channels[2], mid_channels[3], kernel_size=1, stride=1, padding=0), 
            nn.GELU()
        )
        
        # TODO: define the network/layer for estimating the mean
        self.mean_net = nn.Linear(mid_channels[-1] * self.mid_size[1] * self.mid_size[2], latent_dim)
        
        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = nn.Linear(mid_channels[-1] * self.mid_size[1] * self.mid_size[2], latent_dim)

        # TODO: define the decoder part of your network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + mid_channels[-1], mid_channels[-2], kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(mid_channels[-2], mid_channels[-3], kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(mid_channels[-3], mid_channels[-4], kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels[-4], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = torch.flatten(self.encoder(x), start_dim=1)

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
        std = torch.sqrt(torch.exp(logvar)) 
        noise = torch.randn_like(std)  
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
        l_embedding = self.class_emb(labels).view(num_samples, self.mid_size[0], self.mid_size[1], self.mid_size[2])    
        noise = noise.expand(-1, -1, self.mid_size[1], self.mid_size[2])
        noise = torch.cat([noise, l_embedding], dim=1) 

        # TODO: decode the noise based on the given labels
        out = self.decode(noise, labels)

        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        out = self.decoder(sample)

        return out



class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        sample = ...

        sample = self.vae.decode(sample, labels)
        
        return sample

class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        batch_size = x.size(0)
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=x.device)

        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        estimated_noise = self.network(noisy_input, t, label)

        loss = F.l1_loss(estimated_noise, noise)
        return loss


    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
         # Ensure timestep is on the same device as the scheduler tensors
        timestep = timestep.to(self.var_scheduler.betas.device)

        beta_t = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_t = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)

        # Compute mean (mu_t)
        mu_t = (1 / torch.sqrt(alpha_t)) * (
            noisy_sample - (beta_t / torch.sqrt(1 - alpha_bar_t)) * estimated_noise)

        # Add noise
        if timestep.min() > 0:
            sigma_t = torch.sqrt(beta_t)
            sample = mu_t + sigma_t * torch.randn_like(noisy_sample)
        else:
            sample = mu_t
        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        x = torch.randn((num_samples, 1, 32, 32), device=device)

        if labels is None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)

        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(x, t_tensor, labels)
            x = self.recover_sample(x, estimated_noise, t_tensor)

        return x


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()
        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        batch_size = x.size(0)
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss
        loss = F.mse_loss(estimated_noise, noise)
        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        timestep = timestep.to(self.var_scheduler.betas.device)
        beta_t = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_bar_t_prev = self.var_scheduler.alpha_bars[torch.clamp(timestep - 1, min=0)].view(-1, 1, 1, 1).to(noisy_sample.device)
     
        x = (noisy_sample - torch.sqrt(1 - alpha_bar_t) * estimated_noise) / torch.sqrt(alpha_bar_t) 
        sample = torch.sqrt(alpha_bar_t_prev) * x + torch.sqrt(1 - alpha_bar_t_prev) * estimated_noise
        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None):
        # Initialize random noise
        x = torch.randn((num_samples, 1, 32, 32), device=device)

        # Generate random labels if none are provided
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        else:
            labels = None

        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        for i in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), i, device=device, dtype=torch.long)
            estimated_noise = self.network(x, t_tensor, labels)
            x = self.recover_sample(x, estimated_noise, t_tensor)

        return x

    
