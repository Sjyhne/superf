from input_projections.fourier_projection import FourierProjection

import torch.nn.functional as F


def get_input_projection(input_projection_name, input_dim, output_dim, device, fourier_scale=10.0, legendre_max_degree=10, activation=F.relu):
    if input_projection_name == "fourier":
        return FourierProjection(input_dim=input_dim, output_dim=output_dim, scale=fourier_scale, device=device)
    raise ValueError(f"Unknown projection: {input_projection_name}. Use 'fourier'.")
