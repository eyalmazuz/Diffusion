import numpy as np
import torch
from torch.nn import functional as F

def log_add(log_x, log_y):
    """
    Add two log-values.
    :param log_x: a log-value.
    :param log_y: a log-value.
    :return: the log-value of log_x + log_y.
    """
    maximum = torch.max(log_x, log_y)
    return maximum + torch.log(torch.exp(log_x - maximum) + torch.exp(log_y - maximum))

def index_to_onehot(x, num_classes, use_log=False):
    x_onehot = F.one_hot(x, num_classes)

    if use_log:
        log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_onehot

def onehot_to_idx(x: torch.Tensor) -> torch.Tensor:
    return x.argmax(-1)

def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas