import copy
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from utils import sum_flat

class Diffusion():

    def __init__(self,
                 model,
                 timesteps: int=4000,
                 num_classes: int=32,
                 device: str='cuda') -> None:
        self.model = model
        self.timesteps = timesteps

        self.betas = self.get_betas(timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.one_minus_alpha_bar = 1 - self.alpha_bar

        self.num_classes = num_classes
        
    def gather(self, res, t, shape):
        res = res.gather(-1, t)
        while len(res.shape) < len(shape):
            res = res[..., None]

        return res

    def get_q_xt_from_prev(self, x_prev, timestep: torch.Tensor):
        x_t = self.gather(self.alphas, timestep, x_prev.shape) * x_prev \
            + self.gather(self.betas, timestep, x_prev.shape) / self.num_classes

        return x_t
        
    def get_q_xt_from_start(self, x_start, timestep: torch.Tensor):
        x_t = self.gather(self.alpha_bar, timestep, x_start.shape) * x_start \
            + self.gather(self.one_minus_alpha_bar, timestep, x_start.shape) / self.num_classes

        return x_t
    
    def get_q_xt_prev_from_t_and_start(self, x_t, x_start, timestep: torch.Tensor):
        timestep_minus_1 = timestep - 1
        timestep_minus_1 = torch.where(timestep_minus_1 < 0, torch.zeros_like(timestep_minus_1), timestep_minus_1)

        prior_x_t = self.get_q_xt_from_prev(x_t, timestep)
        prior_x_start = self.get_q_xt_from_start(x_start, timestep_minus_1) 
        prior_x_start = torch.where(timestep_minus_1 == 0, x_start, prior_x_start)

        logprobs = prior_x_t + prior_x_start
        normed_logprobs = logprobs - torch.logsumexp(logprobs, dim=1, keepdim=True)

        return normed_logprobs

    def p_pred(self, x_t, timestep: torch.Tensor, model=None):
        x_t_one_hot = self.onehot_to_idx(x_t)
        if model is None:
            model = self.model
        model_x_0 = model(x_t_one_hot, timestep)
        x_t = self.get_q_xt_prev_from_t_and_start(x_t, model_x_0, timestep)

        return x_t

    def categorical_kl(self, prob_a, prob_b):
        return torch.sum(prob_a * torch.log(prob_a / prob_b), dim=1)

    def loss(self, x_start):
        timestep = torch.randint(0, self.timesteps, (x_start.shape[0],), deivce=x_start.device)

        x_start_one_hot = self.idx_to_onehot(x_start)

        x_t = self.get_q_xt_from_start(x_start_one_hot, timestep)
        x_t_prev = self.get_q_xt_prev_from_t_and_start(x_t, x_start_one_hot, timestep)
        model_x_start = self.p_pred(x_t, timestep)

        kl = self.categorical_kl(x_t_prev, model_x_start)
        loss = sum_flat(kl)

        nll = -(x_start_one_hot * model_x_start).sum(dim=1)
        nll = sum_flat(nll)

        loss = torch.where(timestep == 0, nll, loss)

        loss /= np.log(2)

        return loss


    def idx_to_onehot(self, x: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(x.shape[0], self.num_classes)
        onehot.scatter_(1, x.unsqueeze(1), 1)
        return onehot

    def onehot_to_idx(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(1)


    def sample_p(self, x, timestep: torch.Tensor):
        model_probs = self.p_pred(x, timestep)
        out = F.gumbel_softmax(model_probs, tau=1, hard=True)

        out = self.idx_to_onehot(out)

        return out

    def get_betas(self, timesteps: int) -> torch.Tensor:

        scale = 1000 / timesteps
        beta_start = 0.0001 * scale
        beta_end = 0.02 * scale
        betas = torch.linspace(beta_start, beta_end, timesteps)

        return betas