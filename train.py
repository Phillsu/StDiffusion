import numpy as np
from torchvision import transforms
from torch.optim import Adam
from model import Unet
from hrlr_dataset import Image_Data
from torch.utils.data import DataLoader
import torch
from torch.functional import F
import visdom
from torchvision.utils import save_image
import os
import math




def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

betas = cosine_beta_schedule(timesteps=1000)

# alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

device = "cuda" if torch.cuda.is_available() else "cpu"

viz = visdom.Visdom()

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(hr, lr, t, noise=None):
    if noise is None:
        noise = torch.randn_like(hr)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, hr.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, hr.shape
    )
    noise_img = sqrt_alphas_cumprod_t * hr + sqrt_one_minus_alphas_cumprod_t * noise
    cat_img = torch.cat((noise_img, lr), dim=1)
    return cat_img

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  return x_noisy


def p_losses(denoise_model, hr, lr, t, noise=None, loss_type="l1"):
    
    if noise is None:
        noise = torch.randn_like(hr)

    
    x_noisy = q_sample(hr=hr, lr=lr, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)


    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)

    else:
        raise NotImplementedError()

    return loss


def train():
    root = './data/train/'
    DS = Image_Data(root, 256)
    batch_size = 8

    loader = DataLoader(DS, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Unet(
        dim=128,
        dim_mults=(1, 2, 4,))

    model = model.cuda()

    T_loss = np.Inf
    gloab_step = 0
    learning_rate = 1e-4
    optimizer = Adam(model.parameters(), lr=learning_rate)
    for epoch in range(0, 201):

        if (epoch % 50) == 0 and epoch != 0:
            learning_rate = learning_rate / 2
            optimizer = Adam(model.parameters(), lr=learning_rate)
        for step, (lr, hr) in enumerate(loader):

            optimizer.zero_grad()

            lr, hr = lr.cuda(), hr.cuda()

            t = torch.randint(0, 1000, (batch_size,), device=device).long()
            loss = p_losses(model, hr=hr, lr=lr, t=t, loss_type="huber")

            if step % 2000 == 0:
                print("epoch:", epoch, "Loss:", loss.item())

                if loss.item() < 10:
                    viz.line([loss.item()], [gloab_step], win='loss', update='append', name='loss',
                             opts=dict(showlegend=True))
                    gloab_step = gloab_step + 1
            if loss.item() < T_loss:
                T_loss = loss.item()
                torch.save(model.state_dict(), os.path.join('./results/', '256ddpm.pt'),
                           _use_new_zipfile_serialization=False)

            loss.backward()
            optimizer.step()





if __name__ == '__main__':
    train()
