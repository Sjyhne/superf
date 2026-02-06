import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FourierProjection(nn.Module):
    """Random Fourier features (sin/cos) for coords. Good for high-freq detail."""
    def __init__(self, input_dim=2, output_dim=256, scale=10.0, device=None):
        super(FourierProjection, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim // 2  # sin + cos each
        self.scale = scale
        self.device = device
        B = torch.randn(self.output_dim, self.input_dim)
        if device is not None:
            B = B.to(device)
        self.register_buffer("B", B)
    
    def input_mapping(self, x, B):
        B = B.to(x.device)
        x_proj = (2. * np.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x, fs=None):
        if fs is not None:
            current_scale = fs
            if isinstance(fs, torch.Tensor):
                fs_val = fs.item()
                if fs_val != self.scale:
                    self.scale = fs_val
            else:
                if fs != self.scale:
                    self.scale = fs
        else:
            current_scale = self.scale
        B_scaled = self.B * current_scale
        x = self.input_mapping(x, B_scaled)

        return x

