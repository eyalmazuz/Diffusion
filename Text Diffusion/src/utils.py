import torch

def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))
