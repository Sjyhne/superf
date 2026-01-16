import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Optional, Tuple

from utils import bilinear_resize_torch

from models.utils import get_learnable_transforms, get_direct_variances
from models.nir import apply_affine, Affine
import einops

import math

class INR(nn.Module):
    def __init__(self,
                 input_projection,
                 decoder,
                 num_samples,
                 coordinate_dim=2,
                 use_gnll=False,
                 use_separate_ud=False,
                 use_base_frame=True,
                 use_direct_param_T=True,
                 use_color_shift=True,
                 optimize_with_hr=False):
        super(INR, self).__init__()

        self.input_projection = input_projection
        self.decoder = decoder

        self.coordinate_dim = coordinate_dim

        self.optimize_with_hr = optimize_with_hr

        self.log_fs = nn.Parameter(torch.as_tensor(math.log(5.0)))
        self.log_fs.requires_grad = True

        self.num_samples = num_samples
        self.time_vectors = torch.FloatTensor(np.linspace(0, 1, self.num_samples))

        self.use_base_frame = use_base_frame
        self.use_color_shift = use_color_shift
        
        self.use_gnll = use_gnll

        if use_direct_param_T:
            self.shift_vectors = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim, zeros=True, freeze_first=use_base_frame)
            self.rotation_angle = get_learnable_transforms(num_samples=num_samples, coordinate_dim=1, zeros=True, freeze_first=use_base_frame)
        else:
            self.affine_mlp = Affine(hidden_features=256, hidden_layers=2)

        self.use_direct_param_T = use_direct_param_T
        self.use_base_frame = use_base_frame
        self.use_separate_ud = use_separate_ud
        
        ct = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        self.color_transforms = nn.ModuleList([ct for _ in range(num_samples)])

        self.color_transforms[0].requires_grad = False

        # Initialize all biases to 0
        for color_transform in self.color_transforms:
            for ct in color_transform:
                ct.bias.data.zero_()

        # Initialize all weights to 1
        for color_transform in self.color_transforms:
            for ct in color_transform:
                ct.weight.data.fill_(1)

        if self.use_gnll:
            # Only use MLP variance predictors if not using direct GNLL
            self.variance_predictor = nn.Sequential(
                nn.Linear(3 + 3, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
            # Initialize to start neutral: output near zero so exp(0) = 1.0 (neutral variance)
    
    
    def get_affine_transform(self, sample_id):

        if self.time_vectors.device != sample_id.device:
            self.time_vectors = self.time_vectors.to(sample_id.device)

        if self.use_direct_param_T:
            return self.get_direct_affine(sample_id)
        else:
            return self.get_neural_affine(sample_id)


    def get_direct_affine(self, sample_id):

        B = sample_id.shape[0]

        # Handle batch indexing of ParameterList
        shifts = []
        angles = []
        
        for i, idx in enumerate(sample_id):
            shifts.append(self.shift_vectors[idx.item()])
            angles.append(self.rotation_angle[idx.item()])
        
        shift = torch.stack(shifts)  # [B, 2]
        angle = torch.stack(angles)  # [B, 1]

        a1 = torch.stack([torch.cos(angle), -torch.sin(angle), shift[:, 0].unsqueeze(-1)], dim=1).squeeze(-1)
        a2 = torch.stack([torch.sin(angle), torch.cos(angle), shift[:, 1].unsqueeze(-1)], dim=1).squeeze(-1)

        A = torch.stack([a1, a2], dim=1)

        assert A.shape == (B, 2, 3), f"A.shape: {A.shape}"

        return A


    def get_neural_affine(self, sample_id):

        time_vector = self.time_vectors[sample_id].unsqueeze(-1)

        affine_params = self.affine_mlp(time_vector) # [B, 6]

        B, C = affine_params.shape

        A = affine_params.reshape(B, 2, 3) # [B, 2, 3]
        A = torch.stack([
            torch.stack([affine_params[:, 0], affine_params[:, 1], affine_params[:, 4]], dim=1),  # [a11, a12, tx]
            torch.stack([affine_params[:, 2], affine_params[:, 3], affine_params[:, 5]], dim=1)   # [a21, a22, ty]
        ], dim=1)

        assert A.shape == (B, 2, 3)

        # Override affine parameters for sample_id 0 (base frame) to apply no transformation
        if hasattr(self, 'use_base_frame') and self.use_base_frame:
            # Find indices where sample_id is 0 (base frame)
            base_frame_mask = (sample_id == 0)
            if base_frame_mask.any():
                # Set identity transformation for base frame: [1, 0, 0], [0, 1, 0]
                A[base_frame_mask, 0, 0] = 1.0  # cos(0) = 1
                A[base_frame_mask, 0, 1] = 0.0  # -sin(0) = 0  
                A[base_frame_mask, 0, 2] = 0.0  # no translation
                A[base_frame_mask, 1, 0] = 0.0  # sin(0) = 0
                A[base_frame_mask, 1, 1] = 1.0  # cos(0) = 1
                A[base_frame_mask, 1, 2] = 0.0  # no translation
        
        return A

    def apply_affine(self, coords, A):

        B, H, W, C = coords.shape
        
        coords = coords.reshape(B, -1, C) # [B, H*W, C]

        homogenous_coords = torch.cat([coords, torch.ones(B, coords.shape[1], 1, device=coords.device)], dim=2) # B, HW, 3 - Homoegenous coordinates
        transformed_coords = torch.matmul(homogenous_coords, A.mT) # B, HW, 2

        return transformed_coords.reshape(B, H, W, C)

    def apply_color_transform(self, x, sample_idx):
        """Apply per-channel color scaling."""
        result = x.clone()

        for i, idx in enumerate(sample_idx):
            if idx != 0:  # Skip reference sample
                for channel in range(3):
                    transformed = self.color_transforms[idx][channel](x[i, :, :, channel].unsqueeze(-1))
                    result[i, :, :, channel] = transformed.squeeze(-1)

        return result

    def forward(self, coords, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        B, H, W, C = coords.shape

        # Initialize shift lists
        dx_list = None
        dy_list = None

        raw_coords = coords.clone()

        # raw_coords = coords.clone().permute(0, 3, 1, 2)
        if training:

            A = self.get_affine_transform(sample_idx) # [B, 2, 3]

            dx_list = A[:, 0, 2]
            dy_list = A[:, 1, 2]

            not_affined_coords = coords.clone()

            coords = self.apply_affine(coords, A)
        
        if not training:
            A = self.get_affine_transform(sample_idx) # [B, 2, 3]
            dx_list = A[:, 0, 2]
            dy_list = A[:, 1, 2]
            if not self.use_direct_param_T:
                coords = self.apply_affine(coords, A)


        if self.input_projection is not None and self.optimize_with_hr:
            coords = self.input_projection(coords, torch.exp(self.log_fs))
            raw_coords = self.input_projection(raw_coords, torch.exp(self.log_fs))
            if training:
                not_affined_coords = self.input_projection(not_affined_coords, torch.exp(self.log_fs))
        
        else:
            if self.input_projection is not None:
                coords = self.input_projection(coords)
                raw_coords = self.input_projection(raw_coords)
                if training:
                    not_affined_coords = self.input_projection(not_affined_coords)

        # Reshape coordinates from [B, H, W, C] to [B*H*W, C] for decoder
        B, H, W, C = coords.shape
        coords_flat = coords.reshape(B * H * W, C)
        output_flat = self.decoder(coords_flat)
        # Reshape output back to [B, H, W, output_dim]
        output = output_flat.reshape(B, H, W, -1)
        
        if training:
            B_na, H_na, W_na, C_na = not_affined_coords.shape
            not_affined_coords_flat = not_affined_coords.reshape(B_na * H_na * W_na, C_na)
            hr_output_flat = self.decoder(not_affined_coords_flat)
            hr_output = hr_output_flat.reshape(B_na, H_na, W_na, -1)
        else:
            hr_output = output

        
        logvars = None
        if self.use_gnll and not self.use_separate_ud:
            # Check if output has enough channels for logvars
            if output.shape[-1] >= 3 + self.num_samples * 3:
                rgb = output[:, :, :, :3]
                
                # We need to partition the logvars into num_samples of 3 channels each
                logvars_list = []

                for i in range(self.num_samples):
                    logvars_list.append(output[:, :, :, 3 + i * 3: 6 + i * 3])

                logvars = torch.stack(logvars_list, dim=0)  # [num_samples, B, H, W, 3]

                output = rgb
            else:
                # If decoder doesn't output logvars, create dummy logvars
                B, H, W, _ = output.shape
                logvars = torch.zeros(self.num_samples, B, H, W, 3, device=output.device)

        if self.use_color_shift:
            if self.optimize_with_hr:
                hr_output = self.apply_color_transform(hr_output, torch.tensor(0, device=output.device))
            output = self.apply_color_transform(output, sample_idx)

        shifts = [dx_list, dy_list] if dx_list is not None else None

        if training: # pool the supersampled output to the LR resolution

            if scale_factor.unique().shape[0] == 1:
                scale_factor = scale_factor.unique().item()
            else:
                raise ValueError("Not implemented functionality that supports multiple scale factors in the same batch")

            output = F.interpolate(output.permute(0, 3, 1, 2), 
                             scale_factor=scale_factor, 
                             mode='area').permute(0, 2, 3, 1)

        if self.use_gnll and lr_frames is not None:
            variances = []
            if not self.use_separate_ud:
                # logvars has shape [num_samples, B, H, W, 3] where first dim is LR frame index
                # We need to select logvars for each batch element
                if logvars is not None and logvars.shape[1] > 0:
                    batch_size = logvars.shape[1]
                    selected_logvars = []
                    for b in range(batch_size):
                        # Map sample_idx to LR frame index using modulo
                        frame_idx = (sample_idx[b] % self.num_samples).item()
                        # Clamp to valid range [0, num_samples-1]
                        frame_idx = min(max(0, frame_idx), self.num_samples - 1)
                        selected_logvars.append(logvars[frame_idx, b])  # [H, W, 3]
                    if selected_logvars:
                        variances = torch.stack(selected_logvars, dim=0)  # [B, H, W, 3]
                        variances = F.interpolate(variances.permute(0, 3, 1, 2), scale_factor=scale_factor).permute(0, 2, 3, 1)
                        variances = torch.exp(variances)
                    else:
                        # Fallback: create dummy variances
                        B, H, W = output.shape[:3]
                        variances = torch.ones(B, H, W, 3, device=output.device) * 0.1
                else:
                    # Fallback: create dummy variances
                    B, H, W = output.shape[:3]
                    variances = torch.ones(B, H, W, 3, device=output.device) * 0.1
            elif self.use_separate_ud:
                for idx in sample_idx:
                    logvar = self.variance_predictor(torch.cat([output[0], lr_frames[0]], dim=-1))
                    variances.append(logvar)
                variances = torch.stack(variances, dim=0)
                variances = torch.exp(variances)
            
            if self.optimize_with_hr:
                return output, hr_output, shifts, variances
            else:
                return output, shifts, variances

        else:
            if self.optimize_with_hr:
                return output, hr_output, shifts
            else:
                return output, shifts