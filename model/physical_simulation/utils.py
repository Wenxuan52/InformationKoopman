from torch.nn import Module
import torch
import torch.nn as nn


class View(Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class weighted_MSELoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return ((inputs - targets)**2 ) * weights

def count_parameters(model:nn.Module)->int:
    num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Number of parameters: {num_params}")
    return num_params


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def is_symmetric(matrix, tol=1e-8):
    return torch.allclose(matrix, matrix.T, atol=tol)
