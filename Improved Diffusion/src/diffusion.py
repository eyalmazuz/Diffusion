import math

import cv2
import numpy as np
from scipy import cov
import torch
from torch.nn import functional as F

from utils import normal_kl, mean_flat, discretized_gaussian_log_likelihood, betas_for_alpha_bar

class Diffusion():

    def __init__(self,
                 model,
                 timesteps: int=4000,
                 schedule: str='linear',
                 max_beta: float=0.999,
                 device: str='cuda') -> None:

        self.model = model
        self.timesteps = timesteps

        self.betas = self.get_betas(timesteps, schedule, max_beta).to(device)
        self.alphas = 1 - self.betas
        self.alpha_sqrt = torch.sqrt(self.alphas)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.one_minus_alpha_bar_sqrt = torch.sqrt(self.one_minus_alpha_bar)
        
        self.one_over_alpha_sqrt = 1 / self.alpha_sqrt
        self.one_over_alpha_bar = 1 / self.alpha_bar
        self.one_over_one_minus_alpha_bar = 1 / self.one_minus_alpha_bar

        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_bar[:-1]])
        self.alpha_bar_next = torch.cat([self.alpha_bar[1:], torch.tensor([0.0]).to(device)])
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alpha_bar)
        )
         
    def gather(self, res, t, shape):
        res = res.gather(-1, t)
        while len(res.shape) < len(shape):
            res = res[..., None]

        return res

    def get_q_xt(self, x_start, timestep: torch.Tensor):
        mean = self.gather(self.alpha_bar_sqrt, timestep, x_start.shape) * x_start
        variance = self.gather(self.one_minus_alpha_bar, timestep, x_start.shape)

        return mean, variance
    
    def get_q_xt_prev(self, x_start, x_t, timestep):
        x_start_coef = self.gather(self.posterior_mean_coef1, timestep, x_start.shape)
        x_t_coef = self.gather(self.posterior_mean_coef2, timestep, x_t.shape)
        
        mean = (x_start_coef * x_start) + (x_t_coef * x_t)
        
        variance = self.gather(self.posterior_log_variance_clipped, timestep, x_t.shape)
        
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

        model_output, variance = torch.split(model_output, x_t.shape[1], dim=1)

        # calculate the variance
        var_coef = torch.log(self.gather(self.betas, timestep, variance.shape))
        var_minus_coef = torch.log(self.gather(self.posterior_log_variance_clipped, timestep, variance.shape))

        frac = (variance + 1) / 2
        covariance = var_coef * frac + var_minus_coef * (1.0 - frac)

        # calculate the mean
        one_minus_alpha_bar_sqrt = self.gather(self.one_minus_alpha_bar_sqrt, timestep, x_t.shape)
        betas = self.gather(self.betas, timestep, x_t.shape)

        eps_coef = betas / one_minus_alpha_bar_sqrt
        one_over_alpha_sqrt = self.gather(self.one_over_alpha_sqrt, timestep, x_t.shape)

        mean = one_over_alpha_sqrt * (x_t - eps_coef * model_output)

        return mean, covariance
        
    def p_sample(self, x_t, timestep: torch.Tensor, model=None):

        mean, variance = self.get_p_xt_prev(x_t, timestep, model)

        noise = torch.randn(x_t.shape, device=x_t.device)

        return mean + torch.exp(variance * 0.5) * noise

    
    def loss(self, x_start, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device, dtype=torch.long)
        x_t = self.q_sample(x_start, timesteps, noise)

        model_eps = self.model(x_t, timesteps)

        model_eps, model_var = torch.split(model_eps, x_start.shape[1], dim=1)

        frozen_out = torch.cat([model_eps.detach(), model_var], dim=1)

        mse = F.mse_loss(noise, model_eps)
        
        true_mean, true_var = self.get_q_xt_prev(x_start, x_t, timesteps)
        
        model_mean, model_var = self.get_p_xt_prev(x_t, timesteps,
                                                    lambda *args, r=frozen_out: r)
        
        kl = normal_kl(true_mean, true_var, model_mean, model_var)
        
        kl = mean_flat(kl) / np.log(2.0)
        
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_var
        )
        
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        
        vb = torch.where((timesteps == 0), decoder_nll, kl)
        
        loss = mse + (vb / 1000.0)
        
        return loss

    def get_betas(self, timesteps: int, schedule: str, max_beta: float=0.999) -> np.ndarray:

        if schedule == 'linear':
            scale = 1000 / timesteps
            beta_start = 0.0001 * scale
            beta_end = 0.02 * scale
            betas = torch.linspace(beta_start, beta_end, timesteps)

        elif schedule == 'cosine':
            betas = betas_for_alpha_bar(
            timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
            betas = torch.tensor(betas).float()
        
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