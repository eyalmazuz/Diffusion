import copy
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from dataset import Text8Dataset
from utils import sum_flat, cosine_beta_schedule, log_add, onehot_to_idx, index_to_onehot
class Diffusion():

    def __init__(self,
                 model,
                 timesteps: int=4000,
                 num_classes: int=32,
                 schedule: str='linear',
                 use_log: bool=False,
                 device: str='cuda') -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.use_log = use_log

        if schedule == 'linear':
            self.alphas = 1 - torch.linspace(0.0001, 0.02, timesteps)
        
        elif schedule == 'cosine':
            self.alphas = torch.tensor(cosine_beta_schedule(timesteps)).to(device)

        self.log_alphas = torch.log(self.alphas)

        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.log_alpha_bar = torch.cumprod(self.log_alphas, dim=0)

        self.one_minus_alpha = 1 - self.alphas
        self.log_one_minus_alpha = torch.log(self.one_minus_alpha)

        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.log_one_minus_alpha_bar = torch.log(self.one_minus_alpha_bar)

        self.num_classes = num_classes

        print(self.alphas[0], self.alpha_bar[0], self.one_minus_alpha[0], self.one_minus_alpha_bar[0])

    def gather(self, res, t, shape):
        res = res.gather(-1, t)
        while len(res.shape) < len(shape):
            res = res[..., None]

        return res

    def log_categorical(self, x_start, log_prob):
        if self.use_log:
            x_start = x_start.exp()

        return torch.sum(x_start * log_prob, dim=-1)

    def categorical_kl(self, prob_a, prob_b):
        if self.use_log:
            return torch.sum(prob_a.exp() * (prob_a - prob_b), dim=-1)

        else:
            return torch.sum(prob_a * torch.log(prob_a / prob_b), dim=-1)

    def get_q_xt_from_prev(self, x_prev, timestep: torch.Tensor):
        if self.use_log:
            alphas = self.gather(self.log_alphas, timestep, x_prev.shape)
            one_minus_alpha = self.gather(self.log_one_minus_alpha, timestep, x_prev.shape)

            x_t = log_add(x_prev + alphas, one_minus_alpha - np.log(self.num_classes))

        else:
            alphas = self.gather(self.alphas, timestep, x_prev.shape)
            one_minus_alpha = self.gather(self.one_minus_alpha, timestep, x_prev.shape)

            x_t = alphas * x_prev + one_minus_alpha / self.num_classes

        return x_t
        
    def get_q_xt_from_start(self, x_start, timestep: torch.Tensor):
        if self.use_log:
            alpha_bar_log = self.gather(self.log_alpha_bar, timestep, x_start.shape)
            one_minus_alpha_bar_log = self.gather(self.log_one_minus_alpha_bar, timestep, x_start.shape)

            x_t = log_add(x_start + alpha_bar_log, one_minus_alpha_bar_log - np.log(self.num_classes))

        else:
            alpha_bar = self.gather(self.alpha_bar, timestep, x_start.shape)
            one_minus_alpha_bar = self.gather(self.one_minus_alpha_bar, timestep, x_start.shape)

            x_t = alpha_bar * x_start + one_minus_alpha_bar / self.num_classes

        return x_t

    def predict_start(self, x_t, timestep: torch.Tensor, model=None):
        x_t_idx = onehot_to_idx(x_t)

        if model is None:
            model = self.model
        out = model(x_t_idx, timestep)

        model_x_0 = F.log_softmax(out, dim=-1)
        # Not Sure About this, since we don't use log space
        # I think it's needed to always have X_start be one-hot vector
        if not self.use_log:
            model_x_0 = model_x_0.argmax(-1)
        # x_t_prev = self.get_q_xt_prev_from_xt_and_start(x_t, model_x_0, timestep)

        return model_x_0
    
    def get_q_xt_prev_from_xt_and_start(self, x_t, x_start, timestep: torch.Tensor):

        timestep_minus_1 = timestep - 1
        timestep_minus_1 = torch.where(timestep_minus_1 < 0, torch.zeros_like(timestep_minus_1), timestep_minus_1)

        prior_x_start = self.get_q_xt_from_start(x_start, timestep_minus_1) 
        prior_x_start = torch.where(timestep == 0, x_start, prior_x_start)

        prior_x_t = self.get_q_xt_from_prev(x_t, timestep)

        if self.use_log:
            logprobs = prior_x_t + prior_x_start
            normed_logprobs = logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)
        
        else:
            logprobs = prior_x_t * prior_x_start
            normed_logprobs = logprobs / torch.sum(logprobs, dim=-1, keepdim=True)

        return normed_logprobs

    def pred_p(self, x_t, timestep: torch.Tensor, model=None):
        x_start = self.predict_start(x_t, timestep, model)
        x_t_prev = self.get_q_xt_prev_from_xt_and_start(x_t, x_start, timestep)

        return x_t_prev

    @torch.no_grad()
    def sample_p(self, x, timestep: torch.Tensor, model=None):
        model_probs = self.pred_p(x, timestep, model)
        if not self.use_log:
            model_probs = torch.log(model_probs)
        out = F.gumbel_softmax(model_probs, tau=1, hard=False)

        return out

    def sample_q(self, x_start, timestep: torch.Tensor):
        x_t = self.get_q_xt_from_start(x_start, timestep)

        if not self.use_log:
            x_t = torch.log(x_t)

        out = F.gumbel_softmax(x_t, tau=1, hard=False)

        return out
    
    def kl_prior(self, x_start):
        ones = torch.ones(x_start.size(0), device=x_start.device).long()

        q_xt_prob = self.get_q_xt_from_start(x_start, (self.timesteps - 1) * ones)
        q_xt_half = -torch.log(self.num_classes * torch.ones_like(q_xt_prob))

        kl_prior = self.categorical_kl(q_xt_prob, q_xt_half)

        return sum_flat(kl_prior)

    def compute_Lt(self, x_start, x_t, timestep):
        true_prob = self.get_q_xt_prev_from_xt_and_start(x_t, x_start, timestep)

        model_prob = self.pred_p(x_t, timestep)

        kl = self.categorical_kl(true_prob, model_prob)    
        kl = sum_flat(kl)

        nll = self.log_categorical(x_start, model_prob) 
        nll = sum_flat(nll)

        loss = torch.where(timestep == 0, nll, kl)
        
        return loss

    def loss(self, x_start):
        timestep = torch.randint(0, self.timesteps, (x_start.shape[0],), deivce=x_start.device)

        x_start_one_hot = index_to_onehot(x_start, self.num_classes, self.use_log)
        
        x_t = self.sample_q(x_start_one_hot, timestep)

        kl = self.compute_Lt(x_start_one_hot, x_t, timestep)
        kl_prior = self.kl_prior(x_start,)

        loss = kl + kl_prior
        loss /= np.log(2)

        return loss
    
if __name__ == '__main__':
    print('loading dataset')
    use_log = False
    dataset = Text8Dataset('../text8', seq_len=256)
    diffusion = Diffusion(None, timesteps=1000, num_classes=dataset.vocab_size, schedule='cosine', use_log=use_log, device='cpu')

    print('getting first sample')
    x_start = dataset[0]
    x_start = x_start.unsqueeze(0)
    print(f'{x_start.size()=}')
    print('step: 0', ''.join([dataset.itos[i.item()] for i in x_start[0]]))

    x_start_one_hot = index_to_onehot(x_start, dataset.vocab_size, use_log)
    

    print()
    print(x_start_one_hot)
    for i in [1, 10, 50, 100, 200, 500, 999]:
        x_i = diffusion.sample_q(x_start_one_hot, torch.tensor([i]))
        argmax = onehot_to_idx(x_i)
        print(f'step: {i}', ''.join([dataset.itos[i.item()] for i in argmax[0]]))
        print()
