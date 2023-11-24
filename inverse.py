import math
from model import Unet
import cv2
from hrlr_dataset import Image_Data
from torch.utils.data import DataLoader
import torch
from torch.functional import F
import torchvision
from torchvision.utils import save_image
import os


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(timesteps=1000)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(hr, lr, t, noise=None):
    # torch.manual_seed(111)
    if noise is None:
        noise = torch.randn_like(hr)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, hr.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, hr.shape
    )
    noise_img = sqrt_alphas_cumprod_t * hr + sqrt_one_minus_alphas_cumprod_t * noise
    #viz.images(noise_img, nrow=2, win='noisy', opts={'title': 'noisy'})

    cat_img = torch.cat((noise_img, lr), dim=1)
    return cat_img

def get_noisy_image(hr, lr, t):
  # add noise
  x_noisy = q_sample(hr, lr, t=t)

  return x_noisy

@torch.no_grad()
def p_sample(model, x, lr, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    pred = x[:, :3, :, :]
    model_mean = sqrt_recip_alphas_t * (
            pred - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x[:, :3, :, :])

        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def Decryption(model, shape, hr, lr, t):
    device = next(model.parameters()).device

    b = shape[0]

    img = get_noisy_image(hr, lr, t)

    step = t.item()
    print('Total step:', step)
    for i in range(step, -1, -1):
        print('step:',i)
        if i == step:
            img2 = p_sample(model, img, lr, torch.full((b,), i, device=device, dtype=torch.long), i)
            img2 = torch.clamp(img2, 0, 1)
            if i % 100 == 0:
                img2_pil = torchvision.transforms.ToPILImage()(img2[0].cpu())  # Assuming batch_size=1, adjust accordingly
                img2_pil.save('./results/img/'+str(i)+'_output_image.png')  # Adjust the filename as needed
                
        else:
            img2 = torch.cat((img2, lr), dim = 1)
            img2 = p_sample(model, img2, lr, torch.full((b,), i, device=device, dtype=torch.long), i)
            img2 = torch.clamp(img2, 0, 1)
            if i % 100 == 0:
                img2_pil = torchvision.transforms.ToPILImage()(img2[0].cpu())  # Assuming batch_size=1, adjust accordingly
                img2_pil.save('./results/img/'+str(i)+'_output_image.png')  # Adjust the filename as needed
    return img

@torch.no_grad()
def E_sample(model, hr, lr, t, image_size=256, batch_size=1, channels=6):
    return Decryption(model, shape=(batch_size, channels, image_size, image_size), hr=hr, lr=lr, t=t)

def inverse():
    root = './data/test/'
    DS = Image_Data(root, 256)
    print(DS)
    batch_size = 1

    loader = DataLoader(DS, batch_size=batch_size, shuffle=True)
    model = Unet(
        dim=128,
        dim_mults=(1, 2, 4,))
    model.load_state_dict(
        torch.load((os.path.join('./results/', '256ddpm.pt'))))
    model = model.cuda()

    for step, (lr, hr) in enumerate(loader):
        t = torch.randint(999, 1000, (batch_size,), device=device).long()
        lr, hr = lr.cuda(), hr.cuda()
        E_sample(model, hr=hr, lr=lr, t=t)
        break


if __name__ == '__main__':
    inverse()
    