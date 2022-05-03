import copy
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

class Diffusion():

    def __init__(self,
                 model,
                 timesteps: int=4000,
                 device: str='cuda') -> None:
        self.model = model
        self.timesteps = timesteps

        self.betas = self.get_betas(timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_sqrt = torch.sqrt(self.alphas)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.one_minus_alpha_bar_sqrt = torch.sqrt(self.one_minus_alpha_bar)
        
        self.one_over_alpha_sqrt = 1 / self.alpha_sqrt
        self.one_over_alpha_bar = 1 / self.alpha_bar
        self.one_over_one_minus_alpha_bar = 1 / self.one_minus_alpha_bar
        
    def gather(self, res, t, shape):
        res = res.gather(-1, t)
        while len(res.shape) < len(shape):
            res = res[..., None]

        return res

    def get_q_xt(self, x_start, timestep: torch.Tensor):
        mean = self.gather(self.alpha_bar_sqrt, timestep, x_start.shape) * x_start
        variance = self.gather(self.one_minus_alpha_bar, timestep, x_start.shape)

        return mean, variance
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        mean, variance = self.get_q_xt(x_start, t)
        return mean + (variance ** 0.5) * noise

    def get_p_xt_prev(self, x_t, timestep, model=None):
        if model is None:
            model = self.model
        model_output = model(x_t, timestep)

        one_minus_alpha_bar_sqrt = self.gather(self.one_minus_alpha_bar_sqrt, timestep, x_t.shape)
        betas = self.gather(self.betas, timestep, x_t.shape)

        eps_coef = betas / one_minus_alpha_bar_sqrt
        one_over_alpha_sqrt = self.gather(self.one_over_alpha_sqrt, timestep, x_t.shape)

        mean = one_over_alpha_sqrt * (x_t - eps_coef * model_output)
        variance = self.gather(self.betas, timestep, x_t.shape)

        return mean, variance
        
    def p_sample(self, x_t, timestep: torch.Tensor, model=None):

        mean, variance = self.get_p_xt_prev(x_t, timestep, model)

        noise = torch.randn(x_t.shape, device=x_t.device)

        return mean + (variance ** 0.5) * noise

    
    def loss(self, x_start, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device, dtype=torch.long)
        x_t = self.q_sample(x_start, timesteps, noise)

        model_eps = self.model(x_t, timesteps)

        return F.mse_loss(noise, model_eps)

    def get_betas(self, timesteps: int) -> torch.Tensor:

        scale = 1000 / timesteps
        beta_start = 0.0001 * scale
        beta_end = 0.02 * scale
        betas = torch.linspace(beta_start, beta_end, timesteps)

        return betas

def main():
    diffusion = Diffusion(None, timesteps=4001, device='cpu')

    img = cv2.imread('./images/10000_2004.jpg').astype(np.float32) / 127.5 - 1
    print(img.shape)
    img2 = cv2.resize(img, (128, 128))

    cv2.imwrite('./0.png', (img2 + 1) * 127.5)

    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    print(img2.shape)
    x_1 = diffusion.q_sample(img2, torch.tensor([1]), noise=None)
    x_10 = diffusion.q_sample(img2, torch.tensor([10]), noise=None)
    x_100 = diffusion.q_sample(img2, torch.tensor([100]), noise=None)
    x_400 = diffusion.q_sample(img2, torch.tensor([400]), noise=None)
    x_1000 = diffusion.q_sample(img2, torch.tensor([1000]), noise=None)
    x_4000 = diffusion.q_sample(img2, torch.tensor([4000]), noise=None)

    cv2.imwrite('./1.png', (x_1[0].permute(1, 2, 0).numpy() + 1 )  * 127.5)
    cv2.imwrite('./10.png', (x_10[0].permute(1, 2, 0).numpy() + 1) * 127.5)
    cv2.imwrite('./100.png', (x_100[0].permute(1, 2, 0).numpy() + 1) * 127.5)
    cv2.imwrite('./400.png', (x_400[0].permute(1, 2, 0).numpy() + 1) * 127.5)
    cv2.imwrite('./1000.png', (x_1000[0].permute(1, 2, 0).numpy() + 1) * 127.5)
    cv2.imwrite('./4000.png', (x_4000[0].permute(1, 2, 0).numpy() + 1) * 127.5)



if __name__ == "__main__":
    main()