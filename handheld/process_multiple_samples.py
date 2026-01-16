#!/usr/bin/env python3
"""
Process multiple satellite image samples through the handheld super-resolution pipeline.
Supports RGB images with arbitrary scale factors, shift amounts, and augmentation types.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import json
import time
import re
import traceback
import matplotlib.pyplot as plt  # For creating comparison images
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
import warnings
from numba import cuda


from evals import match_colors, get_gaussian_kernel
from evals_2 import align_kornia_brute_force


def color_match_mean_std(pred, gt, eps=1e-8, clip=(0.0, 1.0)):
    """
    Per-channel affine: y ≈ s*x + b, with s=std_gt/std_pred, b=mean_gt - s*mean_pred.
    """
    p = pred.reshape(-1, 3).astype(np.float32)
    g = gt.reshape(-1, 3).astype(np.float32)

    mp, sp = p.mean(axis=0), p.std(axis=0) + eps
    mg, sg = g.mean(axis=0), g.std(axis=0)

    s = sg / sp
    b = mg - s * mp

    out = pred * s + b  # broadcast
    if clip is not None:
        out = np.clip(out, *clip)
    return out, np.diag(s), b


def linear_color_match(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    add_bias: bool = True,
    ridge: float = 1e-3,
    robust_trim_percentile: float | None = 95.0,
    clip_range: tuple[float, float] | None = (0.0, 1.0),
):
    """
    Fit a global linear color transform that maps pred_rgb -> gt_rgb.

    Args:
        pred_rgb:  HxWx3 float array (prediction in linear RGB).
        gt_rgb:    HxWx3 float array (ground truth in linear RGB), spatially aligned to pred_rgb.
        mask:      Optional HxW boolean array of valid pixels (e.g., from flow/confidence).
        add_bias:  If True, fit Y ≈ A*X + b (3x3 + bias); else fit Y ≈ A*X (3x3).
        ridge:     L2 regularization strength (λ) for stability.
        robust_trim_percentile:
                   If set (e.g., 95), do a two-pass fit:
                     1) fit on all valid pixels,
                     2) compute residuals, keep pixels below given percentile, refit.
                   Set to None to disable trimming.
        clip_range: If not None, clip the corrected image to this range.

    Returns:
        corrected_rgb: HxWx3 array = color-corrected pred_rgb.
        A:             3x3 matrix.
        b:             3-vector (zeros if add_bias=False).
        inlier_mask:   HxW boolean mask actually used in the final fit.
    """
    assert pred_rgb.shape == gt_rgb.shape and pred_rgb.shape[-1] == 3
    H, W, _ = pred_rgb.shape

    # Build initial valid mask
    valid = np.isfinite(pred_rgb).all(axis=-1) & np.isfinite(gt_rgb).all(axis=-1)
    if mask is not None:
        valid &= mask.astype(bool)

    # Flatten valid pixels
    X = pred_rgb[valid].reshape(-1, 3)
    Y = gt_rgb[valid].reshape(-1, 3)
    
    # Ensure X and Y have the same number of pixels
    assert X.shape[0] == Y.shape[0], f"X and Y must have same number of pixels: X={X.shape[0]}, Y={Y.shape[0]}"

    def _fit(X_, Y_):
        # Ensure X_ and Y_ have the same number of samples
        assert X_.shape[0] == Y_.shape[0], f"X_ and Y_ must have same number of samples: X_={X_.shape[0]}, Y_={Y_.shape[0]}"
        
        # Design matrix: [X | 1] if bias, else [X]
        if add_bias:
            ones = np.ones((X_.shape[0], 1), dtype=X_.dtype)
            DM = np.hstack([X_, ones])       # N x 4
            I = np.eye(4, dtype=X_.dtype)
        else:
            DM = X_                           # N x 3
            I = np.eye(3, dtype=X_.dtype)

        # Ridge LS solve for each channel jointly: W = (DM^T DM + λI)^-1 DM^T Y
        # W has shape (4x3) if bias else (3x3); last row is bias if add_bias.
        XtX = DM.T @ DM
        W = np.linalg.solve(XtX + ridge * I, DM.T @ Y_)

        if add_bias:
            A = W[:3, :].T   # 3x3
            b = W[3, :]      # 3
        else:
            A = W.T          # 3x3
            b = np.zeros(3, dtype=X_.dtype)
        return A, b

    # First fit
    A, b = _fit(X, Y)

    # Optional robust trimming (discard high-residual pixels and refit)
    inlier_mask = valid.copy()
    if robust_trim_percentile is not None:
        Y_hat = (X @ A.T) + b  # N x 3
        resid = np.mean((Y_hat - Y) ** 2, axis=1)  # per-pixel MSE
        thresh = np.percentile(resid, robust_trim_percentile)
        keep = resid <= thresh
        
        # Refit on inliers using the same valid pixels
        X2 = X[keep]  # Use the same valid pixels that were kept
        Y2 = Y[keep]  # Use the same valid pixels that were kept
        if X2.shape[0] >= 16:  # minimal safety check
            A, b = _fit(X2, Y2)
            # Update inlier mask in image space
            inlier_idx = np.where(valid.ravel())[0]
            inlier_mask = valid.ravel().copy()
            inlier_mask[inlier_idx[~keep]] = False
            inlier_mask = inlier_mask.reshape(H, W)
        else:
            # Fallback: keep first fit if too few points
            inlier_mask = valid

    # Apply transform
    corrected = (pred_rgb @ A.T) + b

    if clip_range is not None:
        lo, hi = clip_range
        corrected = np.clip(corrected, lo, hi)

    return corrected, A, b, inlier_mask

# import ssim from torchmetrics
from torchmetrics.functional import structural_similarity_index_measure

# For metric calculations
try:
    import lpips
except ImportError:
    print("LPIPS module not available. Install with: pip install lpips")
    
try:
    from skimage.metrics import structural_similarity
except ImportError:
    print("skimage.metrics module not available. Install with: pip install scikit-image")

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import handheld super-resolution modules
from handheld_super_resolution.super_resolution import process  # Import the main processing function

# Import data loaders
from data import SRData, SyntheticBurstVal, WorldStratDatasetFrame, WorldStratTestDataset

# Add the parent directory to the path so we can import from data.py
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the dataset loaders
try:
    from data import get_dataset, SyntheticBurstVal, SRData
except ImportError:
    print("Warning: Unable to import dataset loaders from data.py")

# Import evaluation metrics from evals.py
try:
    from evals import PSNR, SSIM, LPIPS as LPIPS_Eval
except ImportError:
    print("Warning: Unable to import evaluation metrics from evals.py")
    print("LPIPS and SSIM metrics will not be available")
    HAS_EVALS = False
else:
    HAS_EVALS = True


def crop_borders(img, margin=16):
    """
    Crop border pixels from an image to avoid edge artifacts in evaluation.
    
    Args:
        img: Input image tensor [C, H, W] or [B, C, H, W]
        margin: Number of pixels to crop from each edge
        
    Returns:
        Cropped image tensor with same batch and channel dimensions
    """
    if len(img.shape) == 3:  # [C, H, W]
        return img[:, margin:-margin, margin:-margin]
    elif len(img.shape) == 4:  # [B, C, H, W]
        return img[:, :, margin:-margin, margin:-margin]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def calculate_metrics(pred, target, loss_fn_alex=None, crop_margin=16):
    """
    Calculate all evaluation metrics for the given prediction and target.
    This implementation matches the one in main.py for consistency.
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
        loss_fn_alex: Optional pre-initialized LPIPS model
        crop_margin: Number of pixels to crop from border before calculating metrics (0 for no cropping)
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are in correct format
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
        
    # Ensure tensors are on the right device
    device = pred.device
    
    # Crop borders to avoid edge artifacts in evaluation
    if crop_margin > 0:
        pred = crop_borders(pred, crop_margin)
        target = crop_borders(target, crop_margin)
    
    # Initialize metrics from evals.py
    psnr_metric = PSNR(max_value=1.0)
    
    # Calculate MSE, PSNR and SSIM on the original inputs
    # These metrics work for any number of channels
    mse = F.mse_loss(pred, target)
    psnr_value = psnr_metric(pred, target)
    ssim_value = structural_similarity_index_measure(pred, target)
    
    # For LPIPS, handle RGGB format (4 channels) by converting to RGB (3 channels)
    # LPIPS requires RGB inputs
    pred_lpips = pred
    target_lpips = target
    
    # Convert RGGB to RGB if necessary
    if pred.shape[1] == 4:
        # Extract R, G1, G2, B channels
        R_pred = pred[:, 0:1]
        G1_pred = pred[:, 1:2]
        G2_pred = pred[:, 2:3]
        B_pred = pred[:, 3:4]
        
        # Average G1 and G2 to create RGB
        G_pred = (G1_pred + G2_pred) / 2
        pred_lpips = torch.cat([R_pred, G_pred, B_pred], dim=1)
    
    if target.shape[1] == 4:
        # Extract R, G1, G2, B channels
        R_target = target[:, 0:1]
        G1_target = target[:, 1:2]
        G2_target = target[:, 2:3]
        B_target = target[:, 3:4]
        
        # Average G1 and G2 to create RGB
        G_target = (G1_target + G2_target) / 2
        target_lpips = torch.cat([R_target, G_target, B_target], dim=1)
    
    # For LPIPS, either use the provided model or create a new one from evals
    if loss_fn_alex is not None and isinstance(loss_fn_alex, lpips.LPIPS):
        # Use the existing model but wrap with our interface
        lpips_value = loss_fn_alex(pred_lpips, target_lpips).mean()
    else:
        # Create a new LPIPS from evals
        lpips_metric = LPIPS_Eval(type='alex').to(device)
        lpips_value = lpips_metric(pred_lpips, target_lpips)
    
    return {
        'mse': mse.item(),
        'psnr': psnr_value.item(),
        'lpips': lpips_value.item(),
        'ssim': ssim_value.item()
    }

# Import utilities from main
from utils import apply_shift_torch, bilinear_resize_torch, align_output_to_target, get_valid_mask

def process_rgb(burst, exposures, options, params):
    """
    Process an RGB burst by separately processing each color channel.
    
    Args:
        burst: RGB burst with shape [B, C, H, W]
        exposures: Exposure values with shape [B] or None
        options: Processing options dictionary
        params: Processing parameters dictionary
        
    Returns:
        Dictionary containing the processed image and metadata
    """
    print("Processing RGB image by channels...")
    # Ensure burst is in [B, C, H, W] format
    if burst.shape[1] != 3:
        print(f"Transposing burst from shape {burst.shape} to [B, C, H, W] format")
        burst = burst.transpose(0, 3, 1, 2)
    
    # Extract channels
    r_channel = burst[:, 0]  # Shape: [B, H, W]
    g_channel = burst[:, 1]  # Shape: [B, H, W]
    b_channel = burst[:, 2]  # Shape: [B, H, W]

    print(f"R channel shape: {r_channel.shape}, range: {r_channel.min()} to {r_channel.max()}")
    print(f"G channel shape: {g_channel.shape}, range: {g_channel.min()} to {g_channel.max()}")
    print(f"B channel shape: {b_channel.shape}, range: {b_channel.min()} to {b_channel.max()}")
    
    # Process each channel separately
    print("Processing R channel...")
    r_output = process(r_channel, exposures, options, params.copy())
    print("Processing G channel...")
    g_output = process(g_channel, exposures, options, params.copy())
    print("Processing B channel...")
    b_output = process(b_channel, exposures, options, params.copy())
    
    # Check for NaN values in outputs
    if np.isnan(r_output).any():
        print("WARNING: R channel output contains NaN values! Replacing with zeros.")
        r_output = np.nan_to_num(r_output, nan=0.0)
    if np.isnan(g_output).any():
        print("WARNING: G channel output contains NaN values! Replacing with zeros.")
        g_output = np.nan_to_num(g_output, nan=0.0)
    if np.isnan(b_output).any():
        print("WARNING: B channel output contains NaN values! Replacing with zeros.")
        b_output = np.nan_to_num(b_output, nan=0.0)
    
    # Combine channels back into RGB
    output_rgb = np.stack([r_output, g_output, b_output], axis=2)
    
    # Return a dictionary with the image and metadata
    return {
        'image': output_rgb,
        'scale_factor': params.get('scale', 1.0),
        'shift_amount': params.get('shift', 0.0)
    }

def find_matching_datasets(data_dir, dataset_type='burstsr', scale_factor=None, shift_amount=None, aug_type=None, sample_id=None, synth_dir=None, worldstrat_test_dir=None):
    """
    Directly construct paths for datasets using the specified criteria instead of searching.
    
    Args:
        data_dir: Base directory for datasets
        dataset_type: Type of dataset ('burstsr', 'synthetic', or 'worldstrat_test')
        scale_factor: Scale factor (e.g., 2, 4)
        shift_amount: Shift amount (e.g., 0.0, 1.0)
        aug_type: Augmentation type (e.g., 'none', 'light', 'medium', 'heavy')
        sample_id: Filter by specific sample ID
        synth_dir: Directory for synthetic datasets
        worldstrat_test_dir: Directory for worldstrat_test datasets
        
    Returns:
        List of Path objects for matching datasets
    """
    data_dir = Path(data_dir)
    matching_datasets = []
    
    if dataset_type == 'burstsr':
        # First, get the list of available samples the same way as in main.py
        available_samples = list(data_dir.glob("*"))
        
        # If a specific sample_id is requested, filter the list
        if sample_id is not None:
            available_samples = [s for s in available_samples if s.name == sample_id]
            
        if not available_samples:
            print(f"Error: No matching samples found in {data_dir}")
            print(f"Available samples: {[s.name for s in data_dir.glob('*')]}")
            return []
        
        # If scale_factor and shift_amount are provided, construct complete paths
        if scale_factor is not None and shift_amount is not None:
            # Set default aug_type if not provided
            if aug_type is None:
                aug_type = 'none'
                
            for sample_path in available_samples:
                # Construct path: data/<sample_id>/scale_<scale>_shift_<shift>_aug_<aug>
                dataset_path = sample_path / f"scale_{scale_factor}_shift_{shift_amount}px_aug_{aug_type}"
                
                if dataset_path.exists():
                    matching_datasets.append(dataset_path)
                else:
                    print(f"Warning: Dataset path does not exist: {dataset_path}")
        else:
            # If scale_factor and shift_amount aren't specified, just return the sample directories
            matching_datasets = available_samples
            
    elif dataset_type == 'synthetic':
        # For synthetic datasets, the structure is simple:
        # SyntheticBurstVal/bursts/<sample_id> - contains burst frames
        # SyntheticBurstVal/gt/<sample_id> - contains ground truth

        # Determine the base directory
        base_dir = synth_dir if synth_dir is not None else data_dir
        
        # If sample_id is provided, construct path for that specific sample
        if sample_id is not None:
            # Format the sample_id with leading zeros if it's a number
            try:
                sample_id_formatted = f"{int(sample_id):04d}"
            except ValueError:
                # If not a number, use as is
                sample_id_formatted = sample_id
                
            # Check if this sample exists
            burst_path = base_dir / 'bursts' / sample_id_formatted
            gt_path = base_dir / 'gt' / sample_id_formatted
            
            if burst_path.exists():
                matching_datasets.append(burst_path)
            else:
                print(f"Warning: Sample path does not exist: {burst_path}")
        else:
            # If no specific sample_id, get all available samples
            bursts_dir = base_dir / 'bursts'
            for sample_path in sorted(bursts_dir.glob('*')):
                if sample_path.is_dir():
                    matching_datasets.append(sample_path)
    
    elif dataset_type == 'worldstrat_test':
        # For worldstrat_test datasets, the structure is:
        # worldstrat_test_data/<sample_id>/hr/ - contains HR image
        # worldstrat_test_data/<sample_id>/lr/ - contains LR images

        # Determine the base directory
        base_dir = worldstrat_test_dir if worldstrat_test_dir is not None else data_dir
        
        # If sample_id is provided, construct path for that specific sample
        if sample_id is not None:
            # Check if this sample exists
            sample_path = base_dir / sample_id
            hr_path = sample_path / 'hr'
            lr_path = sample_path / 'lr'
            
            if sample_path.exists() and hr_path.exists() and lr_path.exists():
                matching_datasets.append(sample_path)
            else:
                print(f"Warning: Sample path does not exist: {sample_path}")
        else:
            # If no specific sample_id, get all available samples
            for sample_path in sorted(base_dir.glob('*')):
                if sample_path.is_dir():
                    hr_path = sample_path / 'hr'
                    lr_path = sample_path / 'lr'
                    if hr_path.exists() and lr_path.exists():
                        matching_datasets.append(sample_path)
    
    return matching_datasets

# Add function to save raw image data without normalization
def save_raw_image(img_data, save_path):
    """
    Save image data in its original format without normalization or conversion.
    
    Args:
        img_data: Image data as numpy array
        save_path: Path to save the image
    """
    # Handle different image formats
    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        # For float images, we need to determine how to save them
        img_min = np.min(img_data)
        img_max = np.max(img_data)
        
        # Convert to the appropriate format based on value range
        if img_min >= 0 and img_max <= 1:
            # Scale to [0, 255] for 8-bit images
            img_data_for_save = (img_data * 255).astype(np.uint8)
        elif img_min >= 0 and img_max <= 255:
            # Already in 8-bit range
            img_data_for_save = img_data.astype(np.uint8)
        else:
            # For other ranges, save as 16-bit
            # Scale to [0, 65535] for 16-bit images
            normalized = (img_data - img_min) / (img_max - img_min)
            img_data_for_save = (normalized * 65535).astype(np.uint16)
    else:
        # For integer types, just use as is
        img_data_for_save = img_data
    
    # Convert RGB to BGR for OpenCV
    if len(img_data.shape) == 3 and img_data.shape[2] == 3:
        img_data_for_save = cv2.cvtColor(img_data_for_save, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(str(save_path), img_data_for_save)
    return img_data_for_save.dtype, img_data_for_save.shape

def process_dataset(dataset, output_dir, crop_margin, args):
    """
    Process a dataset with the handheld super-resolution algorithm.
    
    Args:
        dataset: Instance of SRData or SyntheticBurstVal
        output_dir: Directory to save results
        crop_margin: Number of pixels to crop from borders to avoid edge artifacts
        args: Command-line arguments
        
    Returns:
        Dictionary with metrics or None if processing failed
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the burst images and normalize if needed
        burst = dataset.get_burst()

        if isinstance(burst, torch.Tensor):
            burst = burst.cpu().numpy()
        
        # Ensure burst is within reasonable range [0, 1]
        burst_min = np.min(burst)
        burst_max = np.max(burst)
        if burst_min < 0 or burst_max > 1.5:
            print(f"Warning: Burst data has unusual range: min={burst_min}, max={burst_max}")
            print("Normalizing burst data to [0, 1] range...")
            burst = (burst - burst_min) / (burst_max - burst_min)
            
        # Handle color channels appropriately
        if args.grayscale:
            # For grayscale images, dimensions are in [B, H, W]
            _, height, width = burst.shape
            channels = 1
        elif args.keep_rggb:
            # For RGGB images, dimensions are in [B, H, W, 4]
            _, height, width, channels = burst.shape
        else:
            # For RGB images, dimensions are in [B, H, W, 3]
            _, height, width, channels = burst.shape

        print(f"Burst shape: {burst.shape}, height={height}, width={width}, channels={channels}")

        # Create parameters dictionary for the algorithm
        params = {
            'scale': args.scale,
            'base_detail': True,
            'alignment': 'Fnet',
            'mode': 'grey',
            'merging': {"kernel": "handheld"},
            'post processing': {"on": False}
        }

        # Set up options
        options = {
            'verbose': 1
        }

        print(f"Starting to process burst of shape {burst.shape}")
        print("burst shape: ", burst.shape)
        try:
            # Ensure burst is in the correct format for processing
            if burst.shape[1] != 3:
                print(f"Transposing burst from shape {burst.shape} to [B, C, H, W] format")
                burst = burst.transpose(0, 3, 1, 2)
            
            # Process the images
            processed = process_rgb(burst, None, options, params)
            output = processed['image']
            scale_factor = processed['scale_factor']
            shift_amount = processed['shift_amount']
            output_shape = output.shape
            
            print(f"Successfully processed burst. Output shape: {output_shape}")
            
            # Save the output image in its raw format without normalization
            save_path = output_path / "sr_output.png"
            dtype, saved_shape = save_raw_image(output, save_path)
            print(f"Saved output image to {save_path} with dtype {dtype} and shape {saved_shape}")
            
            # Also save a raw numpy version for perfect preservation
            np_save_path = output_path / "sr_output.npy"
            np.save(str(np_save_path), output)
            print(f"Saved raw numpy array to {np_save_path}")
            
            # Create metrics dictionary
            metrics = {
                'scale_factor': float(scale_factor),
                'shift_amount': float(shift_amount) if shift_amount is not None else None,
                'output_height': int(output_shape[0]),
                'output_width': int(output_shape[1])
            }
            
            # Calculate metrics against high-resolution ground truth, if available
            hr_ground_truth = dataset.get_original_hr()
            
            if hr_ground_truth is not None:
                print("Calculating image quality metrics...")
                
                try:
                    # Get the output and ground truth in the right format
                    output_for_metrics = output.copy()
                    hr_gt = hr_ground_truth
                    
                    # Ensure HR ground truth is properly formatted
                    if isinstance(hr_gt, torch.Tensor):
                        print(f"HR ground truth tensor shape: {hr_gt.shape}, dtype: {hr_gt.dtype}")
                        print(f"HR tensor min: {hr_gt.min()}, max: {hr_gt.max()}")
                        
                        # Convert tensor format if needed
                        if hr_gt.dim() == 3 and hr_gt.shape[0] == 3:  # [C, H, W]
                            print("Converting tensor from [C,H,W] to [H,W,C] format")
                            hr_gt = hr_gt.permute(1, 2, 0)
                        
                        # Check if values are outside [0, 1] range
                        gt_min = hr_gt.min().item()
                        gt_max = hr_gt.max().item()
                        
                        if gt_min < 0 or gt_max > 1:
                            print(f"WARNING: HR ground truth has values outside [0,1] range: min={gt_min}, max={gt_max}")
                            # Clip values to [0, 1] for metrics calculation
                            hr_gt = torch.clamp(hr_gt, 0, 1)
                    else:
                        hr_gt = hr_gt.cpu().numpy()
                    
                    # Save the HR ground truth image in raw format
                    if isinstance(hr_gt, torch.Tensor):
                        hr_np = hr_gt.cpu().numpy()
                    else:
                        hr_np = hr_gt
                        
                    hr_save_path = output_path / "hr_ground_truth.png"
                    dtype, saved_shape = save_raw_image(hr_np, hr_save_path)
                    print(f"Saved HR ground truth to {hr_save_path} with dtype {dtype} and shape {saved_shape}")
                    
                    # Also save a raw numpy version of HR
                    np_hr_save_path = output_path / "hr_ground_truth.npy"
                    np.save(str(np_hr_save_path), hr_np)
                    print(f"Saved raw HR numpy array to {np_hr_save_path}")
                    
                    # Create baseline image (first frame of the burst)
                    baseline_save_path = output_path / "baseline.png"
                    if burst.ndim == 4:  # [B, C, H, W] or [B, H, W, C]
                        if burst.shape[1] == 3:  # [B, C, H, W]
                            baseline = burst[0].transpose(1, 2, 0)  # Convert to [H, W, C]
                        else:  # [B, H, W, C]
                            baseline = burst[0]
                    else:  # Grayscale [B, H, W]
                        baseline = burst[0]

                    # Save the baseline image in raw format 
                    dtype, saved_shape = save_raw_image(baseline, baseline_save_path)
                    print(f"Saved baseline image to {baseline_save_path} with dtype {dtype} and shape {saved_shape}")
                    
                    # Also save a raw numpy version of baseline
                    np_baseline_save_path = output_path / "baseline.npy"
                    np.save(str(np_baseline_save_path), baseline)
                    print(f"Saved raw baseline numpy array to {np_baseline_save_path}")
                    
                    # Resize baseline to match HR ground truth dimensions for metrics calculation
                    if baseline.shape != hr_gt.shape:
                        print(f"Resizing baseline from {baseline.shape} to match HR GT {hr_gt.shape}")
                        if len(hr_gt.shape) == 3:  # [H, W, C]
                            baseline = cv2.resize(baseline, (hr_gt.shape[1], hr_gt.shape[0]))
                        else:  # [H, W]
                            baseline = cv2.resize(baseline, (hr_gt.shape[1], hr_gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                    
                    # Create a side-by-side comparison image (resize all to same dimensions)
                    if len(hr_gt.shape) == 3 and hr_gt.shape[2] == 3:  # RGB
                        # Resize HR ground truth and baseline to match output dimensions
                        hr_gt_resized = cv2.resize(hr_gt.cpu().numpy(), (output_for_metrics.shape[1], output_for_metrics.shape[0]))
                        baseline_resized = cv2.resize(baseline, (output_for_metrics.shape[1], output_for_metrics.shape[0]))
                        comparison = np.hstack([hr_gt_resized, output_for_metrics, baseline_resized])
                        comparison_save_path = output_path / "comparison.png"
                        dtype, saved_shape = save_raw_image(comparison, comparison_save_path)
                        print(f"Saved comparison image to {comparison_save_path} with dtype {dtype} and shape {saved_shape}")
                    
                    # After creating the tensors, move them to the same device
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # Move tensors to device
                    hr_tensor = hr_gt.permute(2, 0, 1).unsqueeze(0).float().to(device)
                    output_tensor = torch.from_numpy(output_for_metrics).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    baseline_tensor = torch.from_numpy(baseline).permute(2, 0, 1).unsqueeze(0).float().to(device)

                    # Initialize LPIPS on the same device
                    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

                    # Align both output and baseline to the HR ground truth
                    output_aligned = align_kornia_brute_force(output_tensor.squeeze(0), hr_tensor.squeeze(0)).unsqueeze(0)
                    baseline_aligned = align_kornia_brute_force(baseline_tensor.squeeze(0), hr_tensor.squeeze(0)).unsqueeze(0)
                    
                    # Apply handheld color matching using match_colors (reference=GT, query=output/baseline)
                    gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
                    gauss_kernel = gauss_kernel.to(device)
                    
                    # Map output -> GT colors
                    aligned_output, _ = match_colors(hr_tensor, output_tensor, output_tensor, ksz, gauss_kernel)
                    # Map baseline -> GT colors
                    aligned_baseline, _ = match_colors(hr_tensor, baseline_tensor, baseline_tensor, ksz, gauss_kernel)

                    
                    # Save color-matched results as raw arrays
                    color_matched_output_np = aligned_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    color_matched_baseline_np = aligned_baseline.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    
                    np.save(str(output_path / "color_matched_output.npy"), color_matched_output_np)
                    np.save(str(output_path / "color_matched_baseline.npy"), color_matched_baseline_np)
                    
                    # Also save as image files
                    save_raw_image(color_matched_output_np, output_path / "color_matched_output.png")
                    save_raw_image(color_matched_baseline_np, output_path / "color_matched_baseline.png")

                    # Calculate metrics using the color-matched tensors
                    baseline_metrics = calculate_metrics(aligned_baseline, hr_tensor, loss_fn_alex=loss_fn_alex, crop_margin=args.crop_margin)
                    model_metrics = calculate_metrics(aligned_output, hr_tensor, loss_fn_alex=loss_fn_alex, crop_margin=args.crop_margin)
                    
                    # Store metrics in the dictionary
                    metrics['psnr'] = float(model_metrics['psnr'])
                    metrics['ssim'] = float(model_metrics['ssim'])
                    metrics['lpips'] = float(model_metrics['lpips'])
                    
                    # Store baseline metrics
                    metrics['baseline_psnr'] = float(baseline_metrics['psnr'])
                    metrics['baseline_ssim'] = float(baseline_metrics['ssim'])
                    metrics['baseline_lpips'] = float(baseline_metrics['lpips'])
                    
                    # Calculate improvements
                    metrics['psnr_improvement'] = float(model_metrics['psnr'] - baseline_metrics['psnr'])
                    metrics['ssim_improvement'] = float(model_metrics['ssim'] - baseline_metrics['ssim'])
                    metrics['lpips_improvement'] = float(baseline_metrics['lpips'] - model_metrics['lpips'])  # LPIPS lower is better
                    
                    print(f"Model metrics - PSNR: {model_metrics['psnr']:.2f}dB, SSIM: {model_metrics['ssim']:.4f}, LPIPS: {model_metrics['lpips']:.4f}")
                    print(f"Baseline metrics - PSNR: {baseline_metrics['psnr']:.2f}dB, SSIM: {baseline_metrics['ssim']:.4f}, LPIPS: {baseline_metrics['lpips']:.4f}")
                    print(f"Improvements - PSNR: {model_metrics['psnr'] - baseline_metrics['psnr']:.2f}dB, SSIM: {model_metrics['ssim'] - baseline_metrics['ssim']:.4f}, LPIPS: {baseline_metrics['lpips'] - model_metrics['lpips']:.4f}")
                    
                    # Create a comparison visualization
                    try:
                        plt.figure(figsize=(15, 5))
                        
                        # Convert to numpy for plotting
                        output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        hr_img = hr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        baseline_img = baseline_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        
                        # Properly handle single-channel images
                        if output_img.shape[-1] == 1:
                            output_img = output_img.squeeze(-1)
                        if hr_img.shape[-1] == 1:
                            hr_img = hr_img.squeeze(-1)
                        if baseline_img.shape[-1] == 1:
                            baseline_img = baseline_img.squeeze(-1)
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(hr_img, cmap='gray' if hr_img.ndim == 2 else None)
                        plt.title('HR Ground Truth')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(output_img, cmap='gray' if output_img.ndim == 2 else None)
                        plt.title(f'SR Output\nPSNR: {model_metrics["psnr"]:.2f}dB\nSSIM: {model_metrics["ssim"]:.4f}\nLPIPS: {model_metrics["lpips"]:.4f}')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(baseline_img, cmap='gray' if baseline_img.ndim == 2 else None)
                        plt.title(f'Bilinear Baseline\nPSNR: {baseline_metrics["psnr"]:.2f}dB\nSSIM: {baseline_metrics["ssim"]:.4f}\nLPIPS: {baseline_metrics["lpips"]:.4f}')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(output_path / "comparison_plot.png")
                        plt.close()
                        print(f"Saved comparison plot to {output_path}/comparison_plot.png")
                    except Exception as e:
                        print(f"Warning: Could not create comparison visualization: {e}")
                except Exception as e:
                    print(f"Error calculating metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    exit("")
            
            return metrics
        except Exception as e:
            print(f"Error processing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"Error setting up dataset processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process multiple datasets through the handheld super-resolution pipeline')
    
    # Dataset selection and location
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to the base data directory for BurstSR datasets')
    parser.add_argument('--dataset-type', type=str, choices=['burstsr', 'synthetic', 'worldstrat_test'], default='burstsr',
                        help='Dataset type: "burstsr" for data in --data-dir, "synthetic" for SyntheticBurstVal, or "worldstrat_test" for WorldStrat test data')
    parser.add_argument('--synth-dir', type=str, default='../SyntheticBurstVal',
                        help='Path to the SyntheticBurstVal directory (only used with --dataset-type synthetic)')
    parser.add_argument('--burst-synth-dir', type=str, default='SyntheticBurstVal',
                        help='Path to the SyntheticBurstVal directory relative to the workspace root')
    parser.add_argument('--worldstrat-test-dir', type=str, default='worldstrat_test_data',
                        help='Path to the worldstrat_test_data directory (only used with --dataset-type worldstrat_test)')
    parser.add_argument('--sample-id', type=str, default=None,
                        help='Process a specific sample ID (e.g., "0" or "Landcover-743192_rgb") instead of all samples')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Maximum number of different scenes/samples to process (default: process all)')
    
    # Processing parameters
    parser.add_argument('--scale', type=int, default=4,
                        help='Scale factor for super-resolution (default: 4)')
    parser.add_argument('--shift', type=float, default=1.0,
                        help='Shift amount in pixels (default: 1.0)')
    parser.add_argument('--aug', type=str, choices=['none', 'light', 'medium', 'heavy'], default='none',
                        help='Augmentation type (default: none)')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of burst samples to use (default: 8)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Process images as grayscale (default: False)')
    parser.add_argument('--crop-margin', type=int, default=16,
                        help='Crop margin from borders to avoid edge effects (default: 16)')
    parser.add_argument('--keep-rggb', action='store_true',
                        help='Keep separate G1 and G2 channels in RGGB format (default: False)')
    parser.add_argument('--no-dataloader', action='store_true',
                        help='Do not use the consistent dataloaders from data.py (default: False)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/handheld',
                        help='Base directory to save results. Will create organized subdirectories for each dataset and sample')
    
    return parser.parse_args()

def _create_summary_tables(df, output_dir, dataset_name):
    """
    Create summary tables for a dataset with all metrics together.
    Optimized for reporting results in a paper, keeping metrics grouped by dataset configuration.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save the summary tables
        dataset_name: Name of the dataset or aggregation group
    """
    import pandas as pd
    import numpy as np
    
    # Make sure the output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the full detailed results first
    df.to_csv(output_dir / f'{dataset_name}_full_results.csv', index=False)
    
    # Get configuration columns that exist in the DataFrame
    config_columns = []
    for col in ['scale', 'shift', 'aug', 'num_samples']:
        if col in df.columns and df[col].notna().any():
            config_columns.append(col)
    
    # Get all metrics columns - group them by type for easier reporting
    model_metrics = []      # Model metrics (psnr, ssim, lpips)
    baseline_metrics = []   # Baseline metrics (baseline_psnr, baseline_ssim, baseline_lpips)
    improvement_metrics = [] # Improvement metrics (psnr_improvement, ssim_improvement, lpips_improvement)
    
    # Find available metrics
    for prefix, metrics_list in [('', model_metrics), ('baseline_', baseline_metrics)]:
        for metric in ['psnr', 'ssim', 'lpips']:
            col_name = f"{prefix}{metric}"
            if col_name in df.columns:
                metrics_list.append(col_name)
    
    # Find improvement metrics
    for metric in ['psnr_improvement', 'ssim_improvement', 'lpips_improvement']:
        if metric in df.columns:
            improvement_metrics.append(metric)
    
    # Combine all metric types
    all_metrics = model_metrics + baseline_metrics + improvement_metrics
    
    if not all_metrics:
        print(f"Warning: No metrics columns found for {dataset_name}")
        return
    
    # Create a simple aggregated table with metrics for each unique configuration
    if config_columns:
        # Group by configuration columns
        try:
            # Calculate mean, std for each metric by configuration
            agg_dict = {}
            for metric in all_metrics:
                agg_dict[metric] = ['mean', 'std', 'count']
            
            # Group by configuration and aggregate metrics
            summary = df.groupby(config_columns).agg(agg_dict)
            
            # Save the raw numerical summary
            summary.to_csv(output_dir / f'{dataset_name}_by_config.csv')
            
            # Create a readable formatted summary
            formatted_rows = []
            
            # Iterate through each configuration group
            for config_values, group_data in summary.groupby(level=list(range(len(config_columns)))):
                row = {}
                
                # Add configuration values
                if isinstance(config_values, tuple):
                    for i, col in enumerate(config_columns):
                        row[col] = config_values[i]
                else:
                    # Single index case
                    row[config_columns[0]] = config_values
                
                # Add sample count
                first_metric = all_metrics[0]
                row['sample_count'] = group_data[first_metric]['count'].iloc[0]
                
                # Add formatted metrics (mean ± std)
                for metric in all_metrics:
                    # Check if we have valid data
                    if pd.notna(group_data[metric]['mean'].iloc[0]) and pd.notna(group_data[metric]['std'].iloc[0]):
                        if 'psnr' in metric:  # PSNR and PSNR improvements
                            # Format with 2 decimal places for PSNR values
                            row[metric] = f"{group_data[metric]['mean'].iloc[0]:.2f} ± {group_data[metric]['std'].iloc[0]:.2f}"
                        else:
                            # Format with 4 decimal places for SSIM and LPIPS values
                            row[metric] = f"{group_data[metric]['mean'].iloc[0]:.4f} ± {group_data[metric]['std'].iloc[0]:.4f}"
                    else:
                        row[metric] = "N/A"
                
                formatted_rows.append(row)
            
            # Convert to DataFrame and save
            formatted_df = pd.DataFrame(formatted_rows)
            
            # Sort by configuration parameters
            if config_columns:
                formatted_df = formatted_df.sort_values(by=config_columns)
            
            formatted_df.to_csv(output_dir / f'{dataset_name}_summary.csv', index=False)
            
            # Also save an Excel version for easier viewing
            try:
                formatted_df.to_excel(output_dir / f'{dataset_name}_summary.xlsx', index=False)
            except Exception as e:
                print(f"Could not save Excel file: {e}")
            
            # Create a paper-ready summary with model vs baseline in formatted tables
            try:
                paper_summary = []
                
                # We want to create a table with the following rows: Configuration | PSNR (Ours) | PSNR (Bilinear) | LPIPS (Ours) | LPIPS (Bilinear) | SSIM (Ours) | SSIM (Bilinear)
                for config_values, group_data in summary.groupby(level=list(range(len(config_columns)))):
                    row = {}
                    
                    # Add configuration as a string
                    if isinstance(config_values, tuple):
                        config_str = " | ".join([f"{col}={val}" for col, val in zip(config_columns, config_values)])
                    else:
                        config_str = f"{config_columns[0]}={config_values}"
                    
                    row['configuration'] = config_str
                    row['sample_count'] = group_data[first_metric]['count'].iloc[0]
                    
                    # Add each metric with model and baseline side by side
                    for base_metric in ['psnr', 'ssim', 'lpips']:
                        model_col = base_metric
                        baseline_col = f"baseline_{base_metric}"
                        
                        # Only add if both metrics exist
                        if model_col in all_metrics and baseline_col in all_metrics:
                            # Format appropriately based on metric type
                            if base_metric == 'psnr':
                                # PSNR with 2 decimal places
                                row[f"{base_metric}_model"] = f"{group_data[model_col]['mean'].iloc[0]:.2f} ± {group_data[model_col]['std'].iloc[0]:.2f}"
                                row[f"{base_metric}_bilinear"] = f"{group_data[baseline_col]['mean'].iloc[0]:.2f} ± {group_data[baseline_col]['std'].iloc[0]:.2f}"
                            else:
                                # SSIM and LPIPS with 4 decimal places
                                row[f"{base_metric}_model"] = f"{group_data[model_col]['mean'].iloc[0]:.4f} ± {group_data[model_col]['std'].iloc[0]:.4f}"
                                row[f"{base_metric}_bilinear"] = f"{group_data[baseline_col]['mean'].iloc[0]:.4f} ± {group_data[baseline_col]['std'].iloc[0]:.4f}"
                            
                            # Also add improvement if available
                            improvement_col = f"{base_metric}_improvement"
                            if improvement_col in all_metrics:
                                if base_metric == 'psnr':
                                    row[f"{base_metric}_improvement"] = f"{group_data[improvement_col]['mean'].iloc[0]:.2f} ± {group_data[improvement_col]['std'].iloc[0]:.2f}"
                                else:
                                    row[f"{base_metric}_improvement"] = f"{group_data[improvement_col]['mean'].iloc[0]:.4f} ± {group_data[improvement_col]['std'].iloc[0]:.4f}"
                    
                    paper_summary.append(row)
                
                paper_df = pd.DataFrame(paper_summary)
                
                # Sort by configuration parameters if possible
                if config_columns:
                    try:
                        # Try to sort based on configuration values extracted from the string
                        if len(config_columns) == 1:
                            # Simple case with single config parameter
                            paper_df[config_columns[0]] = paper_df['configuration'].str.extract(f"{config_columns[0]}=(.*)")
                            paper_df = paper_df.sort_values(by=config_columns[0])
                            paper_df = paper_df.drop(columns=config_columns[0])
                        else:
                            # Just keep the original sorting from earlier
                            pass
                    except:
                        # If sorting fails, just keep the original order
                        pass
                
                paper_df.to_csv(output_dir / f'{dataset_name}_paper_ready.csv', index=False)
                
                try:
                    paper_df.to_excel(output_dir / f'{dataset_name}_paper_ready.xlsx', index=False)
                except Exception as e:
                    print(f"Could not save Excel file: {e}")
                
            except Exception as e:
                print(f"Error creating paper-ready summary: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error creating summary tables for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # If no configuration columns, just aggregate all metrics
        summary_data = {}
        
        for metric in all_metrics:
            summary_data[metric] = {
                'mean': df[metric].mean(),
                'std': df[metric].std(),
                'min': df[metric].min(),
                'max': df[metric].max(),
                'count': df[metric].count()
            }
        
        # Create and save summary
        summary_rows = []
        for metric, stats in summary_data.items():
            row = {
                'metric': metric,
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'count': stats['count']
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / f'{dataset_name}_summary.csv', index=False)
        
        # Create a paper-ready summary with model vs baseline metrics
        paper_rows = []
        
        for base_metric in ['psnr', 'ssim', 'lpips']:
            model_col = base_metric
            baseline_col = f"baseline_{base_metric}"
            
            if model_col in summary_data and baseline_col in summary_data:
                model_stats = summary_data[model_col]
                baseline_stats = summary_data[baseline_col]
                
                row = {
                    'metric': base_metric,
                    'model_mean': f"{model_stats['mean']:.4f}",
                    'model_std': f"{model_stats['std']:.4f}",
                    'baseline_mean': f"{baseline_stats['mean']:.4f}",
                    'baseline_std': f"{baseline_stats['std']:.4f}",
                    'samples': model_stats['count']
                }
                
                # Format the result as you'd write it in a paper
                if base_metric == 'psnr':
                    row['model_formatted'] = f"{model_stats['mean']:.2f} ± {model_stats['std']:.2f}"
                    row['baseline_formatted'] = f"{baseline_stats['mean']:.2f} ± {baseline_stats['std']:.2f}"
                else:
                    row['model_formatted'] = f"{model_stats['mean']:.4f} ± {model_stats['std']:.4f}"
                    row['baseline_formatted'] = f"{baseline_stats['mean']:.4f} ± {baseline_stats['std']:.4f}"
                
                paper_rows.append(row)
        
        if paper_rows:
            paper_df = pd.DataFrame(paper_rows)
            paper_df.to_csv(output_dir / f'{dataset_name}_paper_ready.csv', index=False)
    
    print(f"Created summary tables for {dataset_name} with {len(df)} samples")

def create_complete_dataframe(results_by_sample):
    """
    Create a complete DataFrame from the results dictionary including configuration parameters
    extracted from sample IDs/paths.
    
    Args:
        results_by_sample: Dictionary with sample_id as keys and metrics dictionaries as values
        
    Returns:
        pandas.DataFrame: DataFrame with all results and extracted configuration parameters
    """
    import pandas as pd
    import re
    
    # Pattern for extracting configuration from BurstSR sample paths
    scale_pattern = r'scale_(\d+)'
    shift_pattern = r'shift_([\d\.]+)px'
    aug_pattern = r'aug_(\w+)'
    
    # List to hold all processed entries
    all_entries = []
    
    for sample_id, metrics in results_by_sample.items():
        # Start with the metrics
        entry = metrics.copy()
        
        # Add the sample_id
        entry['sample_id'] = sample_id
        
        # For BurstSR, extract dataset and configuration
        if '/' in sample_id:
            dataset_name, config = sample_id.split('/', 1)
            entry['dataset_name'] = dataset_name
            entry['config'] = config
            
            # Extract scale
            scale_match = re.search(scale_pattern, config)
            if scale_match:
                entry['scale'] = int(scale_match.group(1))
            else:
                entry['scale'] = None
                
            # Extract shift
            shift_match = re.search(shift_pattern, config)
            if shift_match:
                entry['shift'] = float(shift_match.group(1))
            else:
                entry['shift'] = None
                
            # Extract augmentation
            aug_match = re.search(aug_pattern, config)
            if aug_match:
                entry['aug'] = aug_match.group(1)
            else:
                entry['aug'] = 'none'
                
            # Set dataset_type
            entry['dataset_type'] = 'burstsr'
        else:
            # For synthetic datasets
            entry['dataset_name'] = sample_id
            entry['config'] = None
            
            # For SyntheticBurstVal, use scale factor from metrics if it was determined during processing
            # This ensures the scale factor is correctly reported based on the actual dimensions
            # This would have been set in the process_dataset function when computing the actual scale
            if 'actual_scale' in metrics:
                entry['scale'] = metrics['actual_scale']
            else:
                entry['scale'] = 2  # Default scale factor for SyntheticBurstVal is 2x (HR is downscaled by 0.5)
                
            entry['shift'] = 0.0  # No explicit shift in SyntheticBurstVal
            entry['aug'] = 'none'  # No augmentation in SyntheticBurstVal
            entry['dataset_type'] = 'synthetic'
        
        all_entries.append(entry)
    
    # Create DataFrame
    df = pd.DataFrame(all_entries)
    
    return df

def aggregate_results(results_by_sample, output_dir=None):
    """
    Aggregate results across all samples.
    
    Args:
        results_by_sample: Dictionary of results keyed by sample_id
        output_dir: Directory to save aggregated results
        
    Returns:
        DataFrame with aggregated results
    """
    if not results_by_sample:
        print("No results to aggregate.")
        return None
    
    # Convert to DataFrame using the complete dataframe function
    df = create_complete_dataframe(results_by_sample)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create summary tables for this dataset
        dataset_name = output_dir.name if output_dir.name else "aggregated"
        _create_summary_tables(df, output_dir, dataset_name)
        
    return df

def main():
    """
    Main function to process multiple datasets.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create base output directory
    base_output_path = Path(args.output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine the data directory based on the dataset type
    data_dir = Path(args.data_dir)
    synth_dir = None
    worldstrat_test_dir = None
    
    if args.dataset_type == 'synthetic':
        # Find SyntheticBurstVal directory
        possible_paths = [
            Path(args.synth_dir) if hasattr(args, 'synth_dir') and args.synth_dir else None,
            Path(args.burst_synth_dir) if hasattr(args, 'burst_synth_dir') and args.burst_synth_dir else None,
            Path("SyntheticBurstVal"),
            Path("../SyntheticBurstVal"),
            Path("../../SyntheticBurstVal"),
        ]
        
        for path in possible_paths:
            if path is not None and path.exists() and (path / "bursts").exists() and (path / "gt").exists():
                synth_dir = path
                print(f"Found SyntheticBurstVal dataset at: {synth_dir}")
                break
        else:
            print("Warning: Could not find SyntheticBurstVal dataset. Please check the path.")
            print(f"Tried the following paths: {', '.join(str(p) for p in possible_paths if p is not None)}")
    
    elif args.dataset_type == 'worldstrat_test':
        # Find worldstrat_test_data directory
        possible_paths = [
            Path(args.worldstrat_test_dir) if hasattr(args, 'worldstrat_test_dir') and args.worldstrat_test_dir else None,
            Path("worldstrat_test_data"),
            Path("../worldstrat_test_data"),
            Path("../../worldstrat_test_data"),
        ]
        
        for path in possible_paths:
            if path is not None and path.exists():
                # Check if it has the expected structure (at least one sample with hr/lr subdirs)
                sample_dirs = [d for d in path.iterdir() if d.is_dir()]
                if sample_dirs:
                    # Check if the first sample has hr and lr subdirectories
                    first_sample = sample_dirs[0]
                    if (first_sample / "hr").exists() and (first_sample / "lr").exists():
                        worldstrat_test_dir = path
                        print(f"Found worldstrat_test_data dataset at: {worldstrat_test_dir}")
                        break
        else:
            print("Warning: Could not find worldstrat_test_data dataset. Please check the path.")
            print(f"Tried the following paths: {', '.join(str(p) for p in possible_paths if p is not None)}")
        
    print("Processing with the following parameters:")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {base_output_path}")
    print(f"Scale factor: {args.scale}x")
    print(f"Shift amount: {args.shift} pixels")
    print(f"Augmentation: {args.aug}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Grayscale mode: {args.grayscale}")
    print(f"Crop margin: {args.crop_margin} pixels")
    print(f"RGGB mode: {args.keep_rggb}")
    
    # Find matching datasets using the specified criteria
    matching_datasets = find_matching_datasets(
        data_dir, 
        dataset_type=args.dataset_type, 
        scale_factor=args.scale,
        shift_amount=args.shift,
        aug_type=args.aug, 
        sample_id=args.sample_id,
        synth_dir=synth_dir,
        worldstrat_test_dir=worldstrat_test_dir
    )
    
    # Process each matching dataset
    if not matching_datasets:
        print("No matching datasets found")
        return
    
    # Apply max_scenes limit if specified
    if args.max_scenes is not None and args.max_scenes > 0:
        if len(matching_datasets) > args.max_scenes:
            print(f"Limiting to {args.max_scenes} scenes out of {len(matching_datasets)} available")
            matching_datasets = matching_datasets[:args.max_scenes]
    
    print(f"Found {len(matching_datasets)} matching datasets to process")

    # Aggregate metrics for all processed samples
    results_by_sample = {}
    
    for dataset_idx, sample_path in enumerate(matching_datasets):
        # Extract the sample ID properly based on dataset type
        if args.dataset_type == 'burstsr':
            # For burstsr, the sample ID is the top-level folder name (e.g., "Landcover-775262_rgb")
            # sample_path can be either the sample folder, or a subfolder with the processing parameters
            if "scale_" in str(sample_path):
                # This is a full path with the processing parameters, extract the parent as the sample ID
                sample_id = sample_path.parent.name
                config_path = sample_path
            else:
                # This is just the sample folder, use its name as the sample ID
                sample_id = sample_path.name
                # Construct the configuration path based on the parameters if needed
                if args.scale is not None and args.shift is not None and args.aug is not None:
                    config_path = sample_path / f"scale_{args.scale}_shift_{args.shift}px_aug_{args.aug}"
                else:
                    config_path = sample_path
        elif args.dataset_type == 'synthetic':
            # For synthetic, sample_id is already the folder name
            sample_id = sample_path.name
            config_path = sample_path
        else:  # worldstrat_test
            # For worldstrat_test, sample_id is already the folder name
            sample_id = sample_path.name
            config_path = sample_path

        # Create output directory based on dataset type and sample ID
        if args.dataset_type == 'burstsr':
            dataset_name = "BurstSR"
        elif args.dataset_type == 'synthetic':
            dataset_name = "SyntheticBurst"
        elif args.dataset_type == 'worldstrat_test':
            dataset_name = "WorldStratTest"
        else:
            dataset_name = "Unknown"
        sample_output_dir = base_output_path / dataset_name / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing dataset {dataset_idx+1}/{len(matching_datasets)}: {sample_id}")

        try:
            if args.dataset_type == 'burstsr':
                print(f"Loading dataset from: {config_path}")
                dataset = SRData(data_dir=config_path, num_samples=args.num_samples, keep_in_memory=True, grayscale=args.grayscale)
            elif args.dataset_type == 'synthetic':
                print(f"Loading synthetic dataset for sample: {sample_id}")
                dataset = SyntheticBurstVal(
                    data_dir=synth_dir,
                    sample_id=sample_id,
                    keep_in_memory=True,
                    scale_factor=args.scale,  # Pass scale factor here
                    df=1  # Default downsampling factor
                )
            elif args.dataset_type == 'worldstrat_test':
                print(f"Loading worldstrat_test dataset for sample: {sample_id}")
                dataset = WorldStratTestDataset(
                    data_dir=worldstrat_test_dir,
                    sample_id=sample_id,
                    keep_in_memory=True,
                    scale_factor=args.scale
                )
        
            # Process the dataset
            metrics = process_dataset(
                    dataset=dataset,
                    output_dir=sample_output_dir,
                    crop_margin=args.crop_margin,
                    args=args
                )
                
            if metrics is None:
                print(f"Failed to process dataset {sample_id}")
                continue
            
            # Add processing parameters to metrics
            metrics['num_samples'] = args.num_samples
            metrics['grayscale'] = args.grayscale
            metrics['crop_margin'] = args.crop_margin
            metrics['keep_rggb'] = args.keep_rggb
            metrics['sample_id'] = sample_id
            metrics['dataset_name'] = dataset_name
            
            # Store results with appropriate key for aggregation
            results_key = f"{dataset_name}/{sample_id}"
            results_by_sample[results_key] = metrics
            
            # Save individual metrics to JSON for this sample
            with open(sample_output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            
            print(f"Saved results for {sample_id} to {sample_output_dir}")
            
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            traceback.print_exc()
    
    # Aggregate results
    if results_by_sample:
        print("\nAggregating results...")
        
        # Create aggregated results directory
        aggregate_dir = base_output_path / "aggregated"
        aggregate_dir.mkdir(exist_ok=True)
        
        # Save all metrics to a single JSON file
        with open(aggregate_dir / "all_metrics.json", "w") as f:
            json.dump(results_by_sample, f, indent=4)
        
        # Generate aggregated statistics
        aggregate_results(results_by_sample, aggregate_dir)
        
        print(f"All aggregated results saved to {aggregate_dir}/all_metrics.json")
    else:
        print("No valid results to aggregate.")
        
    print("Processing complete.")

if __name__ == '__main__':
    main() 