import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math
from typing import Tuple


def apply_shift_cv2(image, dx, dy):
    """Apply translation using cv2 for data generation"""
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return cv2.warpAffine(image, M, (cols, rows))

def apply_shift_torch(img, dx, dy):
    """Shift img by (dx, dy) pixels. dx positive = right, dy positive = down. img: [B,C,H,W]."""
    dx_norm = 2 * dx / img.shape[3]
    dy_norm = 2 * dy / img.shape[2]

    theta = torch.zeros(img.shape[0], 2, 3, device=img.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = dx_norm
    theta[:, 1, 2] = dy_norm
    
    grid = F.affine_grid(theta, img.size(), align_corners=True)
    output = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    
    return output

def downsample_cv2(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def bilinear_resize_torch(image, size, antialiasing=True):
    return F.interpolate(image, size=size, mode='bilinear', align_corners=False, antialias=antialiasing)

def sentinel2_psf_downsample_torch(image: torch.Tensor, factor: int, truncate: float = 4.0) -> torch.Tensor:
    """Simulate Sentinel-2 lower-resolution observations with PSF blur and box downsampling.

    Follows the DSen2 reduced-resolution setup: blur with a Gaussian of sigma=1/factor
    pixels, then average over non-overlapping factor x factor windows.
    """
    factor_int = int(factor)
    if factor_int != factor:
        raise ValueError(f"Downsampling factor must be an integer, got {factor}.")
    factor = factor_int
    if factor < 1:
        raise ValueError(f"Downsampling factor must be >= 1, got {factor}.")
    if factor == 1:
        return image.clone()
    if image.dim() not in (3, 4):
        raise ValueError(f"Expected CHW or NCHW tensor, got shape {tuple(image.shape)}.")
    if not image.is_floating_point():
        raise ValueError(f"Expected a floating point image tensor, got dtype {image.dtype}.")
    if image.shape[-2] < factor or image.shape[-1] < factor:
        raise ValueError(
            f"Image spatial size {tuple(image.shape[-2:])} is smaller than factor {factor}."
        )

    squeeze_batch = image.dim() == 3
    if squeeze_batch:
        image = image.unsqueeze(0)

    sigma = 1.0 / factor
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = image.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    pad_mode_x = "reflect" if image.shape[-1] > radius else "replicate"
    blurred = F.pad(image, (radius, radius, 0, 0), mode=pad_mode_x)
    blurred = F.conv2d(blurred, kernel_x, groups=channels)

    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    pad_mode_y = "reflect" if blurred.shape[-2] > radius else "replicate"
    blurred = F.pad(blurred, (0, 0, radius, radius), mode=pad_mode_y)
    blurred = F.conv2d(blurred, kernel_y, groups=channels)

    downsampled = F.avg_pool2d(blurred, kernel_size=factor, stride=factor)
    return downsampled.squeeze(0) if squeeze_batch else downsampled

# Color transfer (histogram match). Ref: https://gist.github.com/ProGamerGov/d032aa6780d8ef234f3ce67b177f3c14
def color_transfer(
    input: torch.Tensor,
    source: torch.Tensor,
    mode: str = "pca",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Match input image colors to source. NCHW or CHW. mode: 'pca' | 'cholesky' | 'sym'. Clamp output yourself."""

    assert input.dim() == 3 or input.dim() == 4
    assert source.dim() == 3 or source.dim() == 4
    input = input.unsqueeze(0) if input.dim() == 3 else input
    source = source.unsqueeze(0) if source.dim() == 3 else source
    assert input.shape[:2] == source.shape[:2]

    # Handle older versions of PyTorch
    torch_cholesky = (
        torch.linalg.cholesky if torch.__version__ >= "1.9.0" else torch.cholesky
    )

    def torch_symeig_eigh(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.__version__ >= "1.9.0":
            L, V = torch.linalg.eigh(x, UPLO="U")
        else:
            L, V = torch.symeig(x, eigenvectors=True, upper=True)
        return L, V

    def get_mean_vec_and_cov(
        x_input: torch.Tensor, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mean = x_input.mean(3).mean(2)[:, :, None, None]
        B, C = x_input.shape[:2]
        x_vec = (x_input - x_mean).reshape(B, C, -1)

        x_cov = torch.bmm(x_vec, x_vec.permute(0, 2, 1)) / x_vec.shape[2]
        x_cov = x_cov + (eps * torch.eye(C, device=x_input.device)[None, :])  # stabilizes if you see artifacts
        return x_mean, x_vec, x_cov

    def pca(x: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch_symeig_eigh(x)
        e = torch.sqrt(torch.diag_embed(eigenvalues.reshape(eigenvalues.size(0), -1)))
        if torch.isnan(e).any():
            e = torch.where(torch.isnan(e), torch.zeros_like(e), e)
        return torch.bmm(torch.bmm(eigenvectors, e), eigenvectors.permute(0, 2, 1))

    _, input_vec, input_cov = get_mean_vec_and_cov(input, eps)
    source_mean, _, source_cov = get_mean_vec_and_cov(source, eps)

    if mode == "pca":
        new_cov = torch.bmm(pca(source_cov), torch.inverse(pca(input_cov)))
    elif mode == "cholesky":
        new_cov = torch.bmm(
            torch_cholesky(source_cov), torch.inverse(torch_cholesky(input_cov))
        )
    elif mode == "sym":
        p = pca(input_cov)
        pca_out = pca(torch.bmm(torch.bmm(p, source_cov), p))
        new_cov = torch.bmm(torch.bmm(torch.inverse(p), pca_out), torch.inverse(p))
    else:
        raise ValueError(
            "mode has to be one of 'pca', 'cholesky', or 'sym'."
            + " Received '{}'.".format(mode)
        )

    new_vec = torch.bmm(new_cov, input_vec)
    return new_vec.reshape(input.shape) + source_mean




def align_spatial(input: torch.Tensor, reference: torch.Tensor, mode: str = "ECC") -> torch.Tensor:
    """Align input to reference via ECC (affine). Both (3, H, W) RGB."""
    if mode != "ECC":
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")
    
    input_np = (input.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    reference_np = (reference.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    input_gray = cv2.cvtColor(input_np, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference_np, cv2.COLOR_RGB2GRAY)
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    _, warp_matrix = cv2.findTransformECC(reference_gray, input_gray, warp_matrix, warp_mode, criteria)
    aligned_img_np = cv2.warpAffine(input_np, warp_matrix, (reference_np.shape[1], reference_np.shape[0]),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    aligned_img_tensor = torch.tensor(aligned_img_np, dtype=torch.float32) / 255.0
    aligned_img_tensor = aligned_img_tensor.permute(2, 0, 1)
    
    return aligned_img_tensor


def align_spectral(input: torch.Tensor, reference: torch.Tensor, mode: str = "shift_scale") -> torch.Tensor:
    """Match input per-channel mean/std to reference. Both (3, H, W)."""
    if mode != "shift_scale":
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")
    
    input_mean, input_std = input.mean(dim=(1, 2), keepdim=True), input.std(dim=(1, 2), keepdim=True)
    reference_mean, reference_std = reference.mean(dim=(1, 2), keepdim=True), reference.std(dim=(1, 2), keepdim=True)
    adjusted_input = (input - input_mean) / (input_std + 1e-6) * reference_std + reference_mean

    adjusted_input = torch.clamp(adjusted_input, 0, 1)
    
    return adjusted_input

def align_output_to_target(input: torch.Tensor, reference: torch.Tensor,
                           spectral: bool = True, spatial: bool = True) -> torch.Tensor:
    """Spectral (color) then spatial (ECC) alignment to reference. (3, H, W)."""
    input = input.squeeze(0)
    reference = reference.squeeze(0)

    if spectral:
        aligned = align_spectral(input, reference, mode="shift_scale")
    else:
        aligned = input
    
    if spatial:
        aligned = align_spatial(aligned, reference, mode="ECC")

    return aligned.unsqueeze(0).cuda()


def get_valid_mask(input: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mask where both input and reference have all channels > 0. (1, 3, H, W) -> (1, 1, H, W)."""
    input = input.squeeze(0)
    reference = reference.squeeze(0)
    input_valid = (input > 0).all(dim=0)
    reference_valid = (reference > 0).all(dim=0)
    
    valid_mask = (input_valid & reference_valid)[None, None, ...]
    return valid_mask
