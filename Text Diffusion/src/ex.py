import math
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

stoi = {" ": 0, "a": 1, "b": 2, "c": 3, "d": 4,
        "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
        "j": 10, "k": 11, "l": 12, "m": 13, "n": 14,
        "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
        "t": 20, "u": 21, "v": 22, "w": 23, "x": 24,
        "y": 25, "z": 26, }
                    
itos = {v: k for k, v in stoi.items()}

def index_to_onehot(x, num_classes):
    x_onehot = F.one_hot(x, num_classes)

    return x_onehot

def onehot_to_idx(x: torch.Tensor) -> torch.Tensor:
    return x.argmax(-1)

def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas

class Diffusion():

    def __init__(self, model, timesteps: int=4000, num_classes: int=27, device: str='cuda') -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        self.alphas = torch.tensor(cosine_beta_schedule(timesteps)).to(device)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.one_minus_alpha = 1 - self.alphas
        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.num_classes = num_classes

    def gather(self, res, t, shape):
        res = res.gather(-1, t)
        while len(res.shape) < len(shape):
            res = res[..., None]

        return res

    def get_q_xt_from_prev(self, x_prev, timestep: torch.Tensor):
        alphas = self.gather(self.alphas, timestep, x_prev.shape)
        one_minus_alpha = self.gather(self.one_minus_alpha, timestep, x_prev.shape)

        x_t = alphas * x_prev + one_minus_alpha / self.num_classes

        return x_t
        
    def get_q_xt_from_start(self, x_start, timestep: torch.Tensor):
        alpha_bar = self.gather(self.alpha_bar, timestep, x_start.shape)
        one_minus_alpha_bar = self.gather(self.one_minus_alpha_bar, timestep, x_start.shape)

        x_t = alpha_bar * x_start + one_minus_alpha_bar / self.num_classes

        return x_t
    
    def sample_q(self, x_start, timestep: torch.Tensor):
        x_t = self.get_q_xt_from_start(x_start, timestep)
        out = self.log_sample_categorical(x_t)

        return out
        
    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=-1)
        log_sample = index_to_onehot(sample, self.num_classes)
        return log_sample

if __name__ == '__main__':
    text = 'anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act'
    
    x_start = torch.tensor([stoi[s] for s in text]).long().unsqueeze(0)
    diffusion = Diffusion(None, timesteps=1000, num_classes=len(stoi), device='cpu')

    print('step: 0', ''.join([itos[i.item()] for i in x_start[0]]))

    x_start_one_hot = index_to_onehot(x_start, len(stoi))

    print()
    for i in [1, 10, 50, 100, 200, 500, 999]:
        x_i = diffusion.sample_q(x_start_one_hot, torch.tensor([i]))
        argmax = onehot_to_idx(x_i)
        print(f'step: {i}', ''.join([itos[i.item()] for i in argmax[0]]))
