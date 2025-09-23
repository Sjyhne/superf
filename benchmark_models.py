#!/usr/bin/env python3
"""
Benchmark script that runs multiple model configurations using the same args as optimize.py
Usage: python run_benchmark.py --output_folder results --dataset satburst_synth --sample_id Landcover-743192_rgb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import argparse
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import glob
import json
from datetime import datetime
import pandas as pd
import os
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
import time

from data import get_dataset
from losses import BasicLosses
from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR
from models.nir import NIR, nir_loss


def color_correct_linear3x3(pred_rgb, gt_rgb, mask=None, clip=(0.0, 1.0)):
    """
    Estimate a 3x3 linear color transform A so that A @ pred ≈ gt, then apply it.

    Args:
        pred_rgb: (H, W, 3) float array, prediction in linear RGB, aligned to GT
        gt_rgb:   (H, W, 3) float array, ground truth in linear RGB
        mask:     Optional (H, W) bool array for valid pixels used in fitting A
        clip:     Optional (lo, hi) to clip output

    Returns:
        corrected: (H, W, 3) color-corrected prediction
        A:         (3, 3) color correction matrix (maps pred → gt)
    """
    assert pred_rgb.shape == gt_rgb.shape and pred_rgb.shape[-1] == 3
    H, W, _ = pred_rgb.shape

    valid = np.isfinite(pred_rgb).all(-1) & np.isfinite(gt_rgb).all(-1)
    if mask is not None:
        valid &= mask.astype(bool)

    X = pred_rgb[valid].reshape(-1, 3).astype(np.float64)  # N x 3
    Y = gt_rgb[valid].reshape(-1, 3).astype(np.float64)    # N x 3

    # Solve min_A || X @ A^T - Y ||^2 via least squares per output channel
    A_T, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (3 x 3)
    A = A_T.T

    corrected = (pred_rgb @ A.T)
    if clip is not None:
        lo, hi = clip
        corrected = np.clip(corrected, lo, hi)
    return corrected.astype(pred_rgb.dtype), A


def train_one_iteration(model, optimizer, train_sample, device, downsample_factor):
    """Single training iteration - same as optimize.py"""
    model.train()
    
    recon_criterion = BasicLosses.mse_loss
    trans_criterion = BasicLosses.mae_loss
    
    if model.use_gnll:
        recon_criterion = nn.GaussianNLLLoss()

    input = train_sample['input'].to(device)
    lr_target = train_sample['lr_target'].to(device)
    sample_id = train_sample['sample_id'].to(device)
    scale_factor = train_sample['scale_factor'].to(device)
    
    # Get ground truth shifts
    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    if model.use_gnll:
        output, pred_shifts, pred_variance = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
        recon_loss = recon_criterion(output, lr_target, pred_variance)
    else:
        if isinstance(model, INR):
            output, pred_shifts = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
            recon_loss = recon_criterion(output, lr_target)
        elif isinstance(model, NIR):
            output, pred_shifts = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
            recon_loss = nir_loss(output, lr_target)

    if isinstance(model, (INR, NIR)):
        pred_dx, pred_dy = pred_shifts
        trans_loss = trans_criterion(pred_dx, gt_dx) + trans_criterion(pred_dy, gt_dy)
    else:
        trans_loss = torch.zeros(1, device=device)

    # Only backpropagate the reconstruction loss
    recon_loss.backward()
    optimizer.step()
    
    return {
        'recon_loss': recon_loss.item(),
        'trans_loss': trans_loss.item(),
        'total_loss': recon_loss.item() + trans_loss.item()
    }


def evaluate_model(model, train_data, device, output_folder):
    """Evaluate model on all samples and return comprehensive results."""
    model.eval()
    results = {
        'samples': [],
        'aggregated_metrics': {}
    }
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_bilinear_psnr = []
    all_bilinear_ssim = []
    all_bilinear_lpips = []
    all_alignment_errors = []
    all_final_dx = []
    all_final_dy = []
    
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    print(f"Evaluating model on {len(train_data)} samples...")
    
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        
        # Evaluate on each sample
        for sample_idx in range(len(train_data)):
            sample_id = torch.tensor([sample_idx]).to(device)
            
            # Get ground truth HR for this sample
            hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
            
            # Get model output
            if model.use_gnll:
                output, pred_shifts, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            else:
                output, pred_shifts = model(hr_coords, sample_id, scale_factor=1, training=False)

            # Unstandardize output - use same simple approach as optimize.py
            output = output * train_data.get_lr_std(sample_idx).to(device) + train_data.get_lr_mean(sample_idx).to(device)

            # Get original LR for bilinear comparison - use same simple approach as optimize.py
            lr_original = train_data.get_lr_sample(sample_idx).cpu().numpy()
            
            # Convert GT to numpy first
            gt_np = hr_image.squeeze().cpu().numpy()
            
            # Create bilinear comparison - use same approach as optimize.py
            lr_h, lr_w = lr_original.shape[:2]
            hr_h, hr_w = gt_np.shape[:2]
            lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert to numpy and clip - use same approach as optimize.py
            pred_np = output.squeeze().cpu().numpy()
            pred_np = np.clip(pred_np, 0, 1)
            gt_np = np.clip(gt_np, 0, 1)
            lr_original = np.clip(lr_original, 0, 1)
            lr_bilinear = np.clip(lr_bilinear, 0, 1)

            # Convert to tensors (BCHW) for metrics - use same approach as optimize.py
            pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
            bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
            
            # Calculate metrics - use same approach as optimize.py
            model_psnr = peak_signal_noise_ratio(pred_tensor.cpu(), gt_tensor.cpu(), data_range=1.0).item()
            bilinear_psnr = peak_signal_noise_ratio(bilinear_tensor.cpu(), gt_tensor.cpu(), data_range=1.0).item()
            
            model_ssim = ssim(pred_tensor.cpu(), gt_tensor.cpu(), data_range=1.0).item()
            bilinear_ssim = ssim(bilinear_tensor.cpu(), gt_tensor.cpu(), data_range=1.0).item()
            
            model_lpips = lpips_fn((pred_tensor*2-1).to(device), (gt_tensor*2-1).to(device)).item()
            bilinear_lpips = lpips_fn((bilinear_tensor*2-1).to(device), (gt_tensor*2-1).to(device)).item()
            
            # Calculate alignment error if shifts are available
            if isinstance(model, (INR, NIR)) and pred_shifts is not None:
                pred_dx, pred_dy = pred_shifts
                final_dx = pred_dx.item()
                final_dy = pred_dy.item()
                
                # Get ground truth shifts from the training sample
                # We need to get the training sample to access the ground truth shifts
                train_sample = train_data[sample_idx]
                if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
                    gt_dx = train_sample['shifts']['dx_percent']
                    gt_dy = train_sample['shifts']['dy_percent']
                    # Convert to tensor if needed
                    if torch.is_tensor(gt_dx):
                        gt_dx = gt_dx.item()
                    if torch.is_tensor(gt_dy):
                        gt_dy = gt_dy.item()
                    
                    # Convert predicted shifts from pixel coordinates to percentage coordinates
                    # The model predicts in pixel space, but ground truth is in percentage space
                    lr_h, lr_w = train_data.get_lr_sample(sample_idx).shape[1:3]  # Get LR image dimensions
                    pred_dx_percent = final_dx / lr_w
                    pred_dy_percent = final_dy / lr_h
                    
                    alignment_error = np.sqrt((pred_dx_percent - gt_dx)**2 + (pred_dy_percent - gt_dy)**2)
                    
                else:
                    print(f"Sample {sample_idx}: No ground truth shifts available")
                    alignment_error = 0.0
            else:
                final_dx = 0.0
                final_dy = 0.0
                alignment_error = 0.0
            
            # Save images for visual inspection - for each data sample
            # Get the actual sample ID from the dataset
            if hasattr(train_data, 'sample_id_str'):
                # For SyntheticBurstVal, use the formatted sample ID string
                sample_id = train_data.sample_id_str
            elif hasattr(train_data, 'sample_id'):
                # For other datasets, use the sample_id attribute
                sample_id = str(train_data.sample_id)
            else:
                sample_id = f"sample_{sample_idx:03d}"
            
            images_dir = Path(output_folder) / "images" / sample_id
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the images for saving
            pred_img = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_img = gt_np.copy()
            lr_img = bilinear_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
            
            # Convert to uint8 for saving
            pred_img_uint8 = (pred_img * 255).astype(np.uint8)
            gt_img_uint8 = (gt_img * 255).astype(np.uint8)
            lr_img_uint8 = (lr_img * 255).astype(np.uint8)
            
            # Save individual images
            cv2.imwrite(str(images_dir / f"prediction.png"), 
                       cv2.cvtColor(pred_img_uint8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(images_dir / f"ground_truth.png"), 
                       cv2.cvtColor(gt_img_uint8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(images_dir / f"bilinear.png"), 
                       cv2.cvtColor(lr_img_uint8, cv2.COLOR_RGB2BGR))
            
            # Create side-by-side comparison image with larger layout
            h, w = pred_img.shape[:2]
            # Make images larger for better readability
            scale_factor = 3
            new_h, new_w = h * scale_factor, w * scale_factor
            
            # Resize images for better visibility
            pred_resized = cv2.resize(pred_img_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            gt_resized = cv2.resize(gt_img_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            lr_resized = cv2.resize(lr_img_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Create larger comparison image
            comparison_img = np.zeros((new_h, new_w * 3, 3), dtype=np.uint8)
            comparison_img[:, :new_w] = pred_resized
            comparison_img[:, new_w:2*new_w] = gt_resized
            comparison_img[:, 2*new_w:] = lr_resized
            
            # Add labels with larger font
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0  # Much larger font
            color = (255, 255, 255)
            thickness = 4  # Thicker text
            
            cv2.putText(comparison_img, f"Pred (PSNR: {model_psnr:.2f})", 
                       (20, 60), font, font_scale, color, thickness)
            cv2.putText(comparison_img, f"GT (PSNR: {bilinear_psnr:.2f})", 
                       (new_w + 20, 60), font, font_scale, color, thickness)
            cv2.putText(comparison_img, f"Bilinear", 
                       (2*new_w + 20, 60), font, font_scale, color, thickness)
            
            # Add vertical separator lines
            cv2.line(comparison_img, (new_w, 0), (new_w, new_h), (128, 128, 128), 4)
            cv2.line(comparison_img, (2*new_w, 0), (2*new_w, new_h), (128, 128, 128), 4)
            
            cv2.imwrite(str(images_dir / f"comparison.png"), 
                       cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
            
            # Store sample results
            sample_result = {
                'sample_id': sample_idx,
                'model_psnr': model_psnr,
                'bilinear_psnr': bilinear_psnr,
                'psnr_improvement': model_psnr - bilinear_psnr,
                'model_ssim': model_ssim,
                'bilinear_ssim': bilinear_ssim,
                'ssim_improvement': model_ssim - bilinear_ssim,
                'model_lpips': model_lpips,
                'bilinear_lpips': bilinear_lpips,
                'lpips_improvement': bilinear_lpips - model_lpips,
                'alignment_error': alignment_error,
                'final_dx': final_dx,
                'final_dy': final_dy
            }
            
            results['samples'].append(sample_result)
            
            # Collect for aggregation
            all_psnr.append(model_psnr)
            all_ssim.append(model_ssim)
            all_lpips.append(model_lpips)
            all_bilinear_psnr.append(bilinear_psnr)
            all_bilinear_ssim.append(bilinear_ssim)
            all_bilinear_lpips.append(bilinear_lpips)
            all_alignment_errors.append(alignment_error)
            all_final_dx.append(final_dx)
            all_final_dy.append(final_dy)
            
            if (sample_idx + 1) % 10 == 0:
                print(f"  Evaluated {sample_idx + 1}/{len(train_data)} samples...")
    
    # Calculate aggregated metrics
    results['aggregated_metrics'] = {
        'mean_model_psnr': np.mean(all_psnr),
        'std_model_psnr': np.std(all_psnr),
        'mean_bilinear_psnr': np.mean(all_bilinear_psnr),
        'std_bilinear_psnr': np.std(all_bilinear_psnr),
        'mean_psnr_improvement': np.mean([m - b for m, b in zip(all_psnr, all_bilinear_psnr)]),
        'std_psnr_improvement': np.std([m - b for m, b in zip(all_psnr, all_bilinear_psnr)]),
        
        'mean_model_ssim': np.mean(all_ssim),
        'std_model_ssim': np.std(all_ssim),
        'mean_bilinear_ssim': np.mean(all_bilinear_ssim),
        'std_bilinear_ssim': np.std(all_bilinear_ssim),
        'mean_ssim_improvement': np.mean([m - b for m, b in zip(all_ssim, all_bilinear_ssim)]),
        'std_ssim_improvement': np.std([m - b for m, b in zip(all_ssim, all_bilinear_ssim)]),
        
        'mean_model_lpips': np.mean(all_lpips),
        'std_model_lpips': np.std(all_lpips),
        'mean_bilinear_lpips': np.mean(all_bilinear_lpips),
        'std_bilinear_lpips': np.std(all_bilinear_lpips),
        'mean_lpips_improvement': np.mean([b - m for m, b in zip(all_lpips, all_bilinear_lpips)]),
        'std_lpips_improvement': np.std([b - m for m, b in zip(all_lpips, all_bilinear_lpips)]),
        
        'mean_alignment_error': np.mean(all_alignment_errors),
        'std_alignment_error': np.std(all_alignment_errors),
        'mean_final_dx': np.mean(all_final_dx),
        'std_final_dx': np.std(all_final_dx),
        'mean_final_dy': np.mean(all_final_dy),
        'std_final_dy': np.std(all_final_dy),
        
        'num_samples': len(all_psnr)
    }
    
    print(f"Evaluation complete: {len(all_psnr)} samples processed")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Single Model - Same args as optimize.py")
    
    # Same parameters as optimize.py - EXACT SAME DEFAULTS
    parser.add_argument("--dataset", type=str, default="satburst_synth", 
                       choices=["satburst_synth", "worldstrat", "burst_synth", "worldstrat_test"])
    parser.add_argument("--sample_id", default="Landcover-743192_rgb")
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor, or upsampling factor for the data")
    parser.add_argument("--scale_factor", type=float, default=4, help="scale factor for the input training grid")
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    
    # Model parameters - SAME AS optimize.py
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10", 
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--use_gnll", action="store_true")
    # Transformation control flags (same style as --use_gnll)
    parser.add_argument("--use_base_frame", action="store_true",
                        help="Enable using first frame as base frame (disabled by default)")
    parser.add_argument("--use_direct_param_T", action="store_true",
                        help="Enable directly-parameterized affine T (disabled by default)")
    
    # Training parameters - SAME AS optimize.py
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="7", help="CUDA device number")
    
    # Benchmark-specific parameters
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for benchmark results")
    
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse input projection
    if args.input_projection.startswith("fourier_"):
        args.fourier_scale = float(args.input_projection.split("_")[1])
        args.input_projection = "fourier"

    # Setup dataset - same as optimize.py
    if args.dataset == "satburst_synth":
        # Check if we have an absolute path from the batch script
        if 'DATA_DIR_ABSOLUTE' in os.environ:
            args.root_satburst_synth = os.environ['DATA_DIR_ABSOLUTE']
        else:
            args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
    elif args.dataset == "worldstrat_test":
        # Set the path to worldstrat_test_data
        if 'DATA_DIR_ABSOLUTE' in os.environ:
            args.root_worldstrat_test = os.environ['DATA_DIR_ABSOLUTE']
        else:
            args.root_worldstrat_test = "worldstrat_test_data"
    elif args.dataset == "burst_synth":
        # Set the path to SyntheticBurstVal
        if 'DATA_DIR_ABSOLUTE' in os.environ:
            args.root_burst_synth = os.environ['DATA_DIR_ABSOLUTE']
        else:
            args.root_burst_synth = "SyntheticBurstVal"

    train_data = get_dataset(args=args, name=args.dataset)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)

    print(f"Loaded dataset: {args.dataset}")
    print(f"Sample ID: {args.sample_id}")
    print(f"Downsampling factor: {args.df}")
    print(f"Number of samples in dataset: {len(train_data)}")

    # Setup model - use actual number of samples from dataset
    actual_num_samples = len(train_data)
    input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale)
    decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)
    
    if args.model == 'nir':
        model = NIR(input_projection, decoder, actual_num_samples, use_gnll=args.use_gnll).to(device)
    else:
        # Honor flags exactly like --use_gnll: default False, True only if flag is set
        model = INR(
            input_projection,
            decoder,
            actual_num_samples,
            use_gnll=args.use_gnll,
            use_base_frame=args.use_base_frame,
            use_direct_param_T=args.use_direct_param_T,
        ).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)
    
    print(f"Starting training for {args.iters} iterations...")
    print(f"Training on {len(train_data)} samples")
    
    # Training loop - use DataLoader properly with timing
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    
    # Start timing
    training_start_time = time.time()
    
    while iteration < args.iters:
        # Use DataLoader to get training samples properly
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break
                
            train_losses = train_one_iteration(
                model, optimizer, train_sample, device, args.df
            )
            scheduler.step()
            iteration += 1
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon': f"{train_losses['recon_loss']:.4f}",
                'trans': f"{train_losses['trans_loss']:.4f}"
            })
            
            if iteration % 100 == 0:
                print(f"\nIter {iteration}: Loss: {train_losses['total_loss']:.6f}")
    
    # End timing
    training_end_time = time.time()
    training_time_seconds = training_end_time - training_start_time
    training_time_minutes = training_time_seconds / 60.0
    
    progress_bar.close()
    
    print(f"Training completed in {training_time_minutes:.2f} minutes ({training_time_seconds:.2f} seconds)")
    
    # Evaluate model on all samples
    print("Evaluating model...")
    results = evaluate_model(model, train_data, device, args.output_folder)
    
    # Add timing information to results
    results['timing_metrics'] = {
        'training_time_seconds': training_time_seconds,
        'training_time_minutes': training_time_minutes
    }
    
    # Add model info to results
    results['model_info'] = {
        'model': args.model,
        'network_depth': args.network_depth,
        'network_hidden_dim': args.network_hidden_dim,
        'projection_dim': args.projection_dim,
        'input_projection': args.input_projection,
        'fourier_scale': args.fourier_scale,
        'use_gnll': args.use_gnll,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'iterations': args.iters,
        'dataset': args.dataset,
        'sample_id': args.sample_id,
        'downsample_factor': args.df
    }
    
    # Save results
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model}_{args.input_projection}_{args.network_depth}layers"
    results_file = output_folder / f"benchmark_{model_name}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    metrics = results['aggregated_metrics']
    timing = results['timing_metrics']
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model} with {args.input_projection}")
    print(f"Mean PSNR: {metrics['mean_model_psnr']:.2f} ± {metrics['std_model_psnr']:.2f}")
    print(f"Mean Bilinear PSNR: {metrics['mean_bilinear_psnr']:.2f} ± {metrics['std_bilinear_psnr']:.2f}")
    print(f"PSNR Improvement: {metrics['mean_psnr_improvement']:.2f} ± {metrics['std_psnr_improvement']:.2f}")
    print(f"Mean SSIM: {metrics['mean_model_ssim']:.4f} ± {metrics['std_model_ssim']:.4f}")
    print(f"Mean LPIPS: {metrics['mean_model_lpips']:.4f} ± {metrics['std_model_lpips']:.4f}")
    print(f"Mean Alignment Error: {metrics['mean_alignment_error']:.4f} ± {metrics['std_alignment_error']:.4f}")
    print(f"Training Time: {timing['training_time_minutes']:.2f} minutes")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
