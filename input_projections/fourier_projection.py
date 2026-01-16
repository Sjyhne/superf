import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FourierProjection(nn.Module):
    """
    Projects input coordinates to a higher dimensional space using random Fourier features.
    
    Fourier features map low-dimensional inputs to a higher dimensional space by
    projecting them with random frequencies and computing sine and cosine activations.
    This approach helps models learn high-frequency functions more effectively.
    """
    def __init__(self, input_dim=2, output_dim=256, scale=10.0, device=None):
        """
        Initialize the Fourier feature projection.
        
        Args:
            input_dim: Dimension of input coordinates (typically 2 for 2D coordinates)
            output_dim: Number of Fourier features (output will be 2*output_dim)
            scale: Scaling factor for the frequency matrix (higher values = higher frequencies)
        """
        super(FourierProjection, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim // 2 # Beacuse we are using sin and cos
        self.scale = scale
        self.device = device
        
        # Initialize B once without scaling (fixed base frequencies)
        # Scaling will be applied dynamically in forward() if fs is provided
        B = torch.randn(self.output_dim, self.input_dim)
        if device is not None:
            B = B.to(device)
        self.register_buffer("B", B)
    
    def input_mapping(self, x, B):
        """
        Apply Fourier feature mapping to input coordinates.
        
        Args:
            x: Input coordinates tensor of shape [..., input_dim]
            B: Random Fourier feature matrix of shape [output_dim, input_dim]
            
        Returns:
            Tensor of shape [..., output_dim * 2] with Fourier features
        """
        # Move B to the same device as x
        B = B.to(x.device)
        
        # Project input coordinates
        x_proj = (2. * np.pi * x) @ B.T
        
        # Apply sine and cosine to get Fourier features
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x, fs=None):
        """
        Map input coordinates to Fourier features.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            fs: Optional frequency scale (if provided and different from self.scale, scales B dynamically)
            
        Returns:
            Tensor of shape [..., 2*output_dim] with Fourier features
        """
        # Determine which scale to use (must use fs tensor directly for gradient flow)
        if fs is not None:
            current_scale = fs  # Use fs tensor directly so gradients flow through
            # Update Python variable for reference (detached from graph, only for reference)
            if isinstance(fs, torch.Tensor):
                fs_val = fs.item()
                if fs_val != self.scale:
                    self.scale = fs_val
            else:
                if fs != self.scale:
                    self.scale = fs
        else:
            current_scale = self.scale
        
        # Apply scaling to B (B is fixed, we scale it using the tensor so gradients flow)
        # CRITICAL: Use current_scale (which is fs tensor) directly, not self.scale (Python float)
        B_scaled = self.B * current_scale

        # Apply Fourier feature mapping using the scaled B matrix
        x = self.input_mapping(x, B_scaled)

        return x

