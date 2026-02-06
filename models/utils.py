import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.mlp import MLP
from models.nir import NIR


def get_learnable_transforms(num_samples, coordinate_dim=2, zeros=True, freeze_first=True):
    if zeros:
        if freeze_first:
            params = [nn.Parameter(torch.zeros(coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
        else:
            params = [nn.Parameter(torch.zeros(coordinate_dim), requires_grad=True) for i in range(num_samples)]
    else:
        if freeze_first:
            params = [nn.Parameter(torch.ones(coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
        else:
            params = [nn.Parameter(torch.ones(coordinate_dim), requires_grad=True) for i in range(num_samples)]
    return nn.ParameterList(params)


def get_direct_variances(num_samples, dimensions, freeze_first=True):
    if freeze_first:
        params = [nn.Parameter(torch.zeros(dimensions) if i != 0 else nn.Parameter(torch.zeros(dimensions)), requires_grad=(i != 0)) for i in range(num_samples)]
    else:
        params = [nn.Parameter(torch.zeros(dimensions) if i != 0 else nn.Parameter(torch.zeros(dimensions)), requires_grad=True) for i in range(num_samples)]
    return nn.ParameterList(params)


def get_decoder(network_name, network_depth, input_dim, network_hidden_dim, output_dim=3):
    if network_name == "mlp":
        return MLP(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "nir":
        return NIR(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    else:
        raise ValueError(f"Network name {network_name} not recognized. Use 'mlp' or 'nir'.")
