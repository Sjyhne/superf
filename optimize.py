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

from data import get_dataset
from utils import bilinear_resize_torch, align_output_to_target, get_valid_mask
from losses import BasicLosses
from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR
from models.nir import NIR, nir_loss
from handheld.utils import align_kornia_brute_force
from handheld.evals_2 import get_gaussian_kernel, match_colors

import time

import os
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
import pandas as pd

def train_one_iteration(model, optimizer, train_sample, device, downsample_factor):
    model.train()
    
    # Initialize loss functions
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
        
        # Check for NaN values in variance before computing loss
        if torch.isnan(pred_variance).any() or torch.isinf(pred_variance).any():
            print(f"Warning: NaN/Inf detected in pred_variance, replacing with small positive value")
            pred_variance = torch.clamp(pred_variance, min=1e-6, max=1e6)
            pred_variance = torch.where(torch.isnan(pred_variance) | torch.isinf(pred_variance), 
                                      torch.full_like(pred_variance, 1e-6), pred_variance)
        
        recon_loss = recon_criterion(output, lr_target, pred_variance)
        
        # Check for NaN in loss
        if torch.isnan(recon_loss) or torch.isinf(recon_loss):
            print(f"Warning: NaN/Inf detected in recon_loss, replacing with MSE loss")
            recon_loss = F.mse_loss(output, lr_target)
    else:
        output, pred_shifts = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
        recon_loss = recon_criterion(output, lr_target)

    if isinstance(model, INR):
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


def test_one_epoch(model, test_loader, device):
    model.eval()
    
    with torch.no_grad():
        hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)
        
        if model.use_gnll:
            output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            if isinstance(model, INR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            elif isinstance(model, NIR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image)
                output = output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

        # Unstandardize the output
        output = output * test_loader.get_lr_std(0).to(device) + test_loader.get_lr_mean(0).to(device)
        
        loss = F.mse_loss(output, hr_image)
        
        # Calculate PSNR
        psnr = -10 * torch.log10(loss)
        
    return loss.item(), psnr.item()


def optimize_and_evaluate_sample(model, train_data, device, sample_idx, args, output_dir):
    """Optimize model for a single sample and return comprehensive results."""
    print(f"\n{'='*60}")
    print(f"Optimizing sample {sample_idx + 1}")
    print(f"{'='*60}")
    
    # Record start time for timing metrics
    start_time = time.time()
    
    # Setup optimizer for this sample
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    
    # Training loop for this sample
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc=f"Training Sample {sample_idx + 1}")
    
    # Lists to store training metrics
    psnr_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    
    # Track timing for different phases
    training_start_time = time.time()
    
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    
    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break
                
            train_losses = train_one_iteration(model, optimizer, train_sample, device, args.df)
            scheduler.step()
            iteration += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon': f"{train_losses['recon_loss']:.4f}",
                'trans': f"{train_losses['trans_loss']:.4f}"
            })
            
            # Periodic evaluation
            if iteration % 100 == 0:
                test_loss, test_psnr = test_one_epoch(model, train_data, device)
                print(f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                      f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB")

                # Store training metrics
                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses['recon_loss'])
                trans_loss_list.append(train_losses['trans_loss'])
                total_loss_list.append(train_losses['total_loss'])

    progress_bar.close()
    
    # Record training end time
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    # Final evaluation with alignment and color matching
    evaluation_start_time = time.time()
    model.eval()
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)
        
        if model.use_gnll:
            output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)
        
        final_test_loss = F.mse_loss(output, hr_image).item()   
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()
        
        # Convert tensors to numpy for alignment and color matching
        pred_tensor = torch.from_numpy(output.squeeze().cpu().numpy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        gt_tensor = torch.from_numpy(hr_image.squeeze().cpu().numpy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        # Get LR for bilinear comparison – always work in HWC
        if hasattr(train_data, 'get_lr_sample_hwc'):
            lr_standardized_hwc = train_data.get_lr_sample_hwc(0).cpu().numpy()  # H, W, 3 (standardized)
            lr_needs_unstandardize = True
        else:
            lr_any = train_data.get_lr_sample(0).cpu().numpy()  # might be CHW or HWC or multi-frame
            if lr_any.ndim == 3 and lr_any.shape[0] == 3:  # CHW -> HWC
                lr_standardized_hwc = np.transpose(lr_any, (1, 2, 0))
            elif lr_any.ndim == 3 and lr_any.shape[2] > 3:  # H, W, (3*T)
                H, W, C = lr_any.shape
                if C % 3 == 0:
                    T = C // 3
                    lr_standardized_hwc = lr_any.reshape(H, W, T, 3)[:, :, 0, :]
                else:
                    lr_standardized_hwc = lr_any[:, :, :3]
            else:
                lr_standardized_hwc = lr_any  # assume HWC
            # SRData.get_lr_sample returns unstandardized already → do NOT unstandardize again
            lr_needs_unstandardize = False

        # Unstandardize only if the LR we fetched is standardized (e.g., WorldStratTestDataset)
        if lr_needs_unstandardize:
            lr_std = train_data.get_lr_std(0).cpu().numpy()
            lr_mean = train_data.get_lr_mean(0).cpu().numpy()
            if lr_std.ndim == 1:
                lr_std = lr_std.reshape(1, 1, -1)
            if lr_mean.ndim == 1:
                lr_mean = lr_mean.reshape(1, 1, -1)
            lr_original = lr_standardized_hwc * lr_std + lr_mean  # H, W, 3
        else:
            lr_original = lr_standardized_hwc

        lr_h, lr_w = lr_original.shape[:2]
        hr_h, hr_w = hr_image.shape[1], hr_image.shape[2]

        # Resize LR (still HWC) then convert to BCHW for metrics
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        # Align outputs for fair comparison
        print("Aligning outputs for fair comparison")
        gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
        
        # Align model prediction to ground truth
        pred_aligned = align_kornia_brute_force(pred_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
        pred_aligned, _ = match_colors(pred_aligned, gt_tensor, pred_aligned, ksz, gauss_kernel)

        print("bilinear_tensor:", bilinear_tensor.shape)
        print("gt_tensor:", gt_tensor.shape)

        # Align bilinear baseline to ground truth
        bilinear_aligned = align_kornia_brute_force(bilinear_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
        bilinear_aligned, _ = match_colors(bilinear_aligned, gt_tensor, bilinear_aligned, ksz, gauss_kernel)

        # Calculate comprehensive metrics using aligned tensors
        model_psnr = peak_signal_noise_ratio(pred_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        bilinear_psnr = peak_signal_noise_ratio(bilinear_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        
        model_ssim = ssim(pred_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        model_lpips = lpips_fn((pred_aligned*2-1).to(device), (gt_tensor*2-1).to(device)).item()
        bilinear_lpips = lpips_fn((bilinear_aligned*2-1).to(device), (gt_tensor*2-1).to(device)).item()
        
        # Calculate additional metrics
        # MSE (Mean Squared Error)
        model_mse = F.mse_loss(pred_aligned, gt_tensor).item()
        bilinear_mse = F.mse_loss(bilinear_aligned, gt_tensor).item()
        
        # MAE (Mean Absolute Error)
        model_mae = F.l1_loss(pred_aligned, gt_tensor).item()
        bilinear_mae = F.l1_loss(bilinear_aligned, gt_tensor).item()
        
        # Calculate alignment error if shifts are available
        alignment_error = 0.0
        final_dx = 0.0
        final_dy = 0.0
        if hasattr(model, 'transformation_net') and hasattr(model.transformation_net, 'get_final_shifts'):
            try:
                pred_shifts = model.transformation_net.get_final_shifts()
                if pred_shifts is not None:
                    pred_dx, pred_dy = pred_shifts
                    final_dx = pred_dx.item() if torch.is_tensor(pred_dx) else pred_dx
                    final_dy = pred_dy.item() if torch.is_tensor(pred_dy) else pred_dy
                    
                    # Calculate alignment error (assuming ground truth shifts are 0 for now)
                    alignment_error = np.sqrt(final_dx**2 + final_dy**2)
            except:
                pass
        
        # Convert aligned tensors back to numpy for visualization
        pred_aligned_np = pred_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_aligned_np = bilinear_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()
        
        # Ensure images are in valid range
        pred_aligned_np = np.clip(pred_aligned_np, 0, 1)
        bilinear_aligned_np = np.clip(bilinear_aligned_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        
        # Convert lr_original from CHW to HWC for visualization
        if lr_original.ndim == 3 and lr_original.shape[0] == 3:
            lr_original = np.transpose(lr_original, (1, 2, 0))  # Convert from CHW to HWC
        
        # Save individual sample visualization
        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(lr_original)
        axes[0, 0].set_title('Original LR Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(bilinear_aligned_np)
        axes[0, 1].set_title(f'Bilinear (Aligned)\nPSNR: {bilinear_psnr:.2f} dB', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pred_aligned_np)
        axes[1, 0].set_title(f'Model Output (Aligned)\nPSNR: {model_psnr:.2f} dB', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gt_np)
        axes[1, 1].set_title('Ground Truth HR', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout(pad=2.0)
        plt.savefig(sample_dir / "comparison.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        # Save individual images
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_aligned_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(sample_dir / "prediction_aligned.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(sample_dir / "ground_truth.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Plot training curves if we have data
        if len(psnr_list) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(iteration_list, psnr_list, color='blue', linewidth=2, label='PSNR (Test)')
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('PSNR (dB)', fontsize=12)
            ax1.set_title(f'Sample {sample_idx + 1} - Training PSNR Evolution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.plot(iteration_list, recon_loss_list, color='red', linewidth=2, label='Reconstruction Loss')
            ax2.plot(iteration_list, trans_loss_list, color='green', linewidth=2, label='Transformation Loss')
            ax2.plot(iteration_list, total_loss_list, color='purple', linewidth=2, label='Total Loss')
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Sample {sample_idx + 1} - Training Loss Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(sample_dir / "training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
    
    # Record evaluation end time
    evaluation_end_time = time.time()
    evaluation_time = evaluation_end_time - evaluation_start_time
    total_time = evaluation_end_time - start_time
    
    # Return comprehensive results for this sample
    return {
        'sample_idx': sample_idx,
        'sample_info': {
            'dataset': args.dataset,
            'sample_id': getattr(args, 'sample_id', f'sample_{sample_idx}'),
            'num_lr_frames': len(train_data),
            'iterations': args.iters,
            'model_type': args.model,
            'input_projection': args.input_projection,
            'network_depth': args.network_depth,
            'network_hidden_dim': args.network_hidden_dim,
        },
        'image_metrics': {
            'model_psnr': model_psnr,
            'bilinear_psnr': bilinear_psnr,
            'psnr_improvement': model_psnr - bilinear_psnr,
            'model_ssim': model_ssim,
            'bilinear_ssim': bilinear_ssim,
            'ssim_improvement': model_ssim - bilinear_ssim,
            'model_lpips': model_lpips,
            'bilinear_lpips': bilinear_lpips,
            'lpips_improvement': bilinear_lpips - model_lpips,
            'model_mse': model_mse,
            'bilinear_mse': bilinear_mse,
            'mse_improvement': bilinear_mse - model_mse,
            'model_mae': model_mae,
            'bilinear_mae': bilinear_mae,
            'mae_improvement': bilinear_mae - model_mae,
        },
        'alignment_metrics': {
            'alignment_error': alignment_error,
            'final_dx': final_dx,
            'final_dy': final_dy,
            'alignment_used': True,  # Since we used align_kornia_brute_force
        },
        'training_metrics': {
            'final_test_loss': final_test_loss,
            'final_test_psnr': final_psnr,
            'iterations': iteration_list,
            'psnr': psnr_list,
            'recon_loss': recon_loss_list,
            'trans_loss': trans_loss_list,
            'total_loss': total_loss_list,
            'convergence_iteration': len(psnr_list),  # Number of evaluation points
            'final_recon_loss': recon_loss_list[-1] if recon_loss_list else None,
            'final_trans_loss': trans_loss_list[-1] if trans_loss_list else None,
            'final_total_loss': total_loss_list[-1] if total_loss_list else None,
        },
        'timing_metrics': {
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60.0,
            'evaluation_time_seconds': evaluation_time,
            'evaluation_time_minutes': evaluation_time / 60.0,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60.0,
            'time_per_iteration_seconds': training_time / args.iters if args.iters > 0 else 0,
        },
        'image_dimensions': {
            'hr_height': hr_image.shape[1],
            'hr_width': hr_image.shape[2],
            'lr_height': lr_original.shape[0],
            'lr_width': lr_original.shape[1],
            'scale_factor': hr_image.shape[1] / lr_original.shape[0],
        }
    }


def visualize_lr_variance(model, train_data, device, output_dir, sample_id):
    """
    Visualize variance maps for each LR sample when using GNLL.
    
    Args:
        model: Trained model with GNLL enabled
        train_data: Training dataset
        device: Device to run on
        output_dir: Directory to save visualizations
        sample_id: Sample ID being processed
    """
    if not model.use_gnll:
        print("Warning: visualize_lr_variance called but model does not use GNLL")
        return
    
    model.eval()
    with torch.no_grad():
        # Get HR coordinates for inference
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        
        # Create output directory for variance visualizations
        variance_dir = output_dir / "variance_visualizations"
        variance_dir.mkdir(exist_ok=True)
        
        # Get number of LR samples based on dataset type
        if hasattr(train_data, 'num_samples'):
            num_samples = train_data.num_samples
        elif hasattr(train_data, 'lr_paths'):
            num_samples = len(train_data.lr_paths)
        else:
            print("Warning: Cannot determine number of LR samples. Skipping variance visualization.")
            return
            
        print(f"Creating variance visualizations for {num_samples} LR samples...")
        
        # Process each LR sample individually
        for i in range(num_samples):
            sample_id_tensor = torch.tensor([i]).to(device)
            
            # Get the model output with variance for this specific sample
            # Pass an HR-sized frame so GNLL variance head can run at test-time
            output, _, variance = model(hr_coords, sample_id_tensor, scale_factor=1, training=False, lr_frames=hr_image)

            # Ensure variance is a tensor
            if isinstance(variance, list):
                try:
                    variance = torch.stack(variance, dim=0)
                except Exception:
                    variance = None
            if variance is None:
                variance = torch.full_like(output, 1e-6)
            
            # Best-effort unstandardization and variance scaling with dataset stats
            std_i = None
            mean_i = None
            
            try:
                std_i = train_data.get_lr_std(i)
                mean_i = train_data.get_lr_mean(i)
            except (TypeError, IndexError):
                # Some datasets (e.g., worldstrat_test) may not index per-sample; fall back to 0
                try:
                    std_i = train_data.get_lr_std(0)
                    mean_i = train_data.get_lr_mean(0)
                except (TypeError, IndexError, AttributeError):
                    pass
            except AttributeError:
                pass
            
            lr_np = None
            if std_i is not None and mean_i is not None:
                # Convert to numpy first, then to tensor to avoid indexing issues
                if hasattr(std_i, 'cpu'):
                    std_i = std_i.cpu().numpy()
                if hasattr(mean_i, 'cpu'):
                    mean_i = mean_i.cpu().numpy()
                
                # Convert to tensor
                std_i = torch.tensor(std_i, device=device, dtype=torch.float32)
                mean_i = torch.tensor(mean_i, device=device, dtype=torch.float32)
                
                # Ensure shapes broadcast: [1,1,C]
                if std_i.ndim == 1:
                    std_i = std_i.view(1, 1, -1)
                    mean_i = mean_i.view(1, 1, -1)
                output = output * std_i + mean_i
                # Variance scales by std^2
                # variance = variance * (std_i ** 2)

                # Try to fetch and unstandardize the LR sample for display
                try:
                    if hasattr(train_data, 'get_lr_sample_hwc'):
                        lr_sample = train_data.get_lr_sample_hwc(i)
                        if hasattr(lr_sample, 'cpu'):
                            lr_np = lr_sample.cpu().numpy()
                        else:
                            lr_np = np.array(lr_sample)
                    elif hasattr(train_data, 'get_lr_sample'):
                        lr_sample = train_data.get_lr_sample(i)
                        if hasattr(lr_sample, 'cpu'):
                            lr_np = lr_sample.permute(1, 2, 0).cpu().numpy()
                        else:
                            lr_np = np.array(lr_sample).transpose(1, 2, 0)
                    # Unstandardize LR
                    if lr_np is not None:
                        std_np = std_i.squeeze(0).squeeze(0).detach().cpu().numpy()
                        mean_np = mean_i.squeeze(0).squeeze(0).detach().cpu().numpy()
                        lr_np = lr_np * std_np + mean_np
                except Exception:
                    lr_np = None
            
            # Convert to numpy
            output_np = output.squeeze().cpu().numpy()
            variance_np = variance.squeeze().cpu().numpy()
            hr_np = hr_image.squeeze().cpu().numpy()
            
            # Clip values to valid range
            output_np = np.clip(output_np, 0, 1)
            hr_np = np.clip(hr_np, 0, 1)
            # Variance should be non-negative
            if variance_np.min() < 0:
                variance_np = np.maximum(variance_np, 0)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Original images
            axes[0, 0].imshow(hr_np)
            axes[0, 0].set_title(f'Ground Truth HR', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(output_np)
            axes[0, 1].set_title(f'Model Output (Sample {i})', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Show difference between model output and GT
            diff = np.abs(output_np - hr_np)
            im_diff = axes[0, 2].imshow(diff, cmap='hot')
            axes[0, 2].set_title(f'Absolute Error (Sample {i})', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Row 2: Variance analysis
            # Raw variance map
            im_var = axes[1, 0].imshow(variance_np, cmap='viridis')
            axes[1, 0].set_title(f'Variance Map (Sample {i})', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im_var, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Variance overlaid on model output (transparency)
            # Build a 2D variance map (H x W)
            if variance_np.ndim == 3:
                var_map = variance_np.mean(axis=-1)
            else:
                var_map = variance_np
            
            # Prepare a 3-channel base image for overlay
            disp_img = output_np
            if disp_img.ndim == 2:
                disp_img = np.repeat(disp_img[..., None], 3, axis=-1)
            elif disp_img.shape[-1] == 1:
                disp_img = np.repeat(disp_img, 3, axis=-1)
            
            # Ensure display image is in [0,1]
            disp_img = np.clip(disp_img, 0.0, 1.0)
            axes[1, 1].imshow(disp_img)
            
            # High variance mask (2D)
            thresh = np.percentile(var_map, 75)
            high_var_mask = var_map > thresh
            
            # 3-channel overlay
            overlay = np.zeros_like(disp_img)
            if high_var_mask.any():
                overlay[high_var_mask, :] = [1.0, 0.0, 0.0]
                axes[1, 1].imshow(overlay, alpha=0.3)
            axes[1, 1].set_title(f'High Variance Regions (Sample {i})', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            # Show the LR sample alongside
            if lr_np is not None:
                # Resize LR to HR size for visualization
                if lr_np.ndim == 2:
                    lr_np = np.repeat(lr_np[..., None], 3, axis=-1)
                elif lr_np.shape[-1] == 1:
                    lr_np = np.repeat(lr_np, 3, axis=-1)
                lr_vis = cv2.resize(lr_np, (output_np.shape[1], output_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                lr_vis = np.clip(lr_vis, 0.0, 1.0)
                axes[1, 2].imshow(lr_vis)
                axes[1, 2].set_title(f'LR Sample (Sample {i})', fontsize=12, fontweight='bold')
                axes[1, 2].axis('off')
            else:
                # Fallback: show variance stats if LR not available
                axes[1, 2].text(0.1, 0.8, f'Variance Statistics:', fontsize=12, fontweight='bold', transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.7, f'Mean: {np.mean(variance_np):.6f}', fontsize=10, transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.6, f'Std: {np.std(variance_np):.6f}', fontsize=10, transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.5, f'Min: {np.min(variance_np):.6f}', fontsize=10, transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.4, f'Max: {np.max(variance_np):.6f}', fontsize=10, transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.3, f'75th percentile: {np.percentile(variance_np, 75):.6f}', fontsize=10, transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.2, f'95th percentile: {np.percentile(variance_np, 95):.6f}', fontsize=10, transform=axes[1, 2].transAxes)
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].axis('off')
            
            plt.tight_layout(pad=2.0)
            
            # Save individual variance visualization
            variance_path = variance_dir / f"sample_{i:03d}_variance_analysis.png"
            plt.savefig(variance_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
            
            # Save individual variance map as numpy array
            np.save(variance_dir / f"sample_{i:03d}_variance.npy", variance_np)
            np.save(variance_dir / f"sample_{i:03d}_output.npy", output_np)
        
        # Create a summary visualization showing all variance maps side by side
        create_variance_summary(train_data, variance_dir, device)
        
        print(f"Variance visualizations saved to {variance_dir}")

def create_variance_summary(train_data, variance_dir, device):
    """
    Create a summary visualization showing all variance maps in a grid.
    """
    # This would require loading all the saved variance maps and creating a grid
    # For now, we'll create a simple summary
    summary_path = variance_dir / "variance_summary.txt"
    
    # Get number of LR samples based on dataset type
    if hasattr(train_data, 'num_samples'):
        num_samples = train_data.num_samples
    elif hasattr(train_data, 'lr_paths'):
        num_samples = len(train_data.lr_paths)
    else:
        num_samples = "Unknown"
    
    with open(summary_path, 'w') as f:
        f.write("Variance Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of LR samples: {num_samples}\n")
        f.write(f"Each sample has been processed individually to show model uncertainty.\n")
        f.write(f"High variance regions indicate where the model is less confident.\n")
        f.write(f"Check individual sample_XXX_variance_analysis.png files for detailed analysis.\n")
    
    print(f"Variance summary saved to {summary_path}")

def create_summary_visualization(all_results, output_dir):
    """Create summary visualization showing metrics across all samples."""
    if not all_results:
        return
    
    # Extract metrics from nested structure
    sample_indices = [r['sample_idx'] for r in all_results]
    model_psnr = [r['image_metrics']['model_psnr'] for r in all_results]
    bilinear_psnr = [r['image_metrics']['bilinear_psnr'] for r in all_results]
    psnr_improvement = [r['image_metrics']['psnr_improvement'] for r in all_results]
    model_ssim = [r['image_metrics']['model_ssim'] for r in all_results]
    bilinear_ssim = [r['image_metrics']['bilinear_ssim'] for r in all_results]
    ssim_improvement = [r['image_metrics']['ssim_improvement'] for r in all_results]
    model_lpips = [r['image_metrics']['model_lpips'] for r in all_results]
    bilinear_lpips = [r['image_metrics']['bilinear_lpips'] for r in all_results]
    lpips_improvement = [r['image_metrics']['lpips_improvement'] for r in all_results]
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PSNR comparison
    axes[0, 0].bar(sample_indices, model_psnr, alpha=0.7, label='Model', color='blue')
    axes[0, 0].bar(sample_indices, bilinear_psnr, alpha=0.7, label='Bilinear', color='orange')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title('PSNR Comparison Across Samples')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR improvement
    colors = ['green' if x > 0 else 'red' for x in psnr_improvement]
    axes[0, 1].bar(sample_indices, psnr_improvement, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('PSNR Improvement (dB)')
    axes[0, 1].set_title('PSNR Improvement (Model - Bilinear)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM comparison
    axes[1, 0].bar(sample_indices, model_ssim, alpha=0.7, label='Model', color='purple')
    axes[1, 0].bar(sample_indices, bilinear_ssim, alpha=0.7, label='Bilinear', color='orange')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('SSIM Comparison Across Samples')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # LPIPS comparison
    axes[1, 1].bar(sample_indices, model_lpips, alpha=0.7, label='Model', color='brown')
    axes[1, 1].bar(sample_indices, bilinear_lpips, alpha=0.7, label='Bilinear', color='orange')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('LPIPS')
    axes[1, 1].set_title('LPIPS Comparison Across Samples')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
    # Create box plots for aggregated metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PSNR box plot
    axes[0].boxplot([model_psnr, bilinear_psnr], labels=['Model', 'Bilinear'])
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR Distribution Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # SSIM box plot
    axes[1].boxplot([model_ssim, bilinear_ssim], labels=['Model', 'Bilinear'])
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Distribution Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # LPIPS box plot
    axes[2].boxplot([model_lpips, bilinear_lpips], labels=['Model', 'Bilinear'])
    axes[2].set_ylabel('LPIPS')
    axes[2].set_title('LPIPS Distribution Comparison')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_distribution.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
    # Calculate and save aggregated statistics
    summary_stats = {
        'total_samples': len(all_results),
        'psnr': {
            'model_mean': np.mean(model_psnr),
            'model_std': np.std(model_psnr),
            'model_min': np.min(model_psnr),
            'model_max': np.max(model_psnr),
            'bilinear_mean': np.mean(bilinear_psnr),
            'bilinear_std': np.std(bilinear_psnr),
            'bilinear_min': np.min(bilinear_psnr),
            'bilinear_max': np.max(bilinear_psnr),
            'improvement_mean': np.mean(psnr_improvement),
            'improvement_std': np.std(psnr_improvement),
            'improvement_min': np.min(psnr_improvement),
            'improvement_max': np.max(psnr_improvement)
        },
        'ssim': {
            'model_mean': np.mean(model_ssim),
            'model_std': np.std(model_ssim),
            'model_min': np.min(model_ssim),
            'model_max': np.max(model_ssim),
            'bilinear_mean': np.mean(bilinear_ssim),
            'bilinear_std': np.std(bilinear_ssim),
            'bilinear_min': np.min(bilinear_ssim),
            'bilinear_max': np.max(bilinear_ssim),
            'improvement_mean': np.mean(ssim_improvement),
            'improvement_std': np.std(ssim_improvement),
            'improvement_min': np.min(ssim_improvement),
            'improvement_max': np.max(ssim_improvement)
        },
        'lpips': {
            'model_mean': np.mean(model_lpips),
            'model_std': np.std(model_lpips),
            'model_min': np.min(model_lpips),
            'model_max': np.max(model_lpips),
            'bilinear_mean': np.mean(bilinear_lpips),
            'bilinear_std': np.std(bilinear_lpips),
            'bilinear_min': np.min(bilinear_lpips),
            'bilinear_max': np.max(bilinear_lpips),
            'improvement_mean': np.mean(lpips_improvement),
            'improvement_std': np.std(lpips_improvement),
            'improvement_min': np.min(lpips_improvement),
            'improvement_max': np.max(lpips_improvement)
        }
    }
    
    # Save aggregated statistics to JSON
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save human-readable summary
    summary_text = f"""Multi-Sample Super-Resolution Results Summary
================================================

Total Samples Processed: {len(all_results)}

PSNR Results (dB):
------------------
Model Output:
  Mean: {summary_stats['psnr']['model_mean']:.2f} ± {summary_stats['psnr']['model_std']:.2f}
  Range: {summary_stats['psnr']['model_min']:.2f} - {summary_stats['psnr']['model_max']:.2f}

Bilinear Baseline:
  Mean: {summary_stats['psnr']['bilinear_mean']:.2f} ± {summary_stats['psnr']['bilinear_std']:.2f}
  Range: {summary_stats['psnr']['bilinear_min']:.2f} - {summary_stats['psnr']['bilinear_max']:.2f}

PSNR Improvement (Model - Bilinear):
  Mean: {summary_stats['psnr']['improvement_mean']:.2f} ± {summary_stats['psnr']['improvement_std']:.2f}
  Range: {summary_stats['psnr']['improvement_min']:.2f} - {summary_stats['psnr']['improvement_max']:.2f}

SSIM Results:
-------------
Model Output:
  Mean: {summary_stats['ssim']['model_mean']:.4f} ± {summary_stats['ssim']['model_std']:.4f}
  Range: {summary_stats['ssim']['model_min']:.4f} - {summary_stats['ssim']['model_max']:.4f}

Bilinear Baseline:
  Mean: {summary_stats['ssim']['bilinear_mean']:.4f} ± {summary_stats['ssim']['bilinear_std']:.4f}
  Range: {summary_stats['ssim']['bilinear_min']:.4f} - {summary_stats['ssim']['bilinear_max']:.4f}

SSIM Improvement (Model - Bilinear):
  Mean: {summary_stats['ssim']['improvement_mean']:.4f} ± {summary_stats['ssim']['improvement_std']:.4f}
  Range: {summary_stats['ssim']['improvement_min']:.4f} - {summary_stats['ssim']['improvement_max']:.4f}

LPIPS Results:
--------------
Model Output:
  Mean: {summary_stats['lpips']['model_mean']:.4f} ± {summary_stats['lpips']['model_std']:.4f}
  Range: {summary_stats['lpips']['model_min']:.4f} - {summary_stats['lpips']['model_max']:.4f}

Bilinear Baseline:
  Mean: {summary_stats['lpips']['bilinear_mean']:.4f} ± {summary_stats['lpips']['bilinear_std']:.4f}
  Range: {summary_stats['lpips']['bilinear_min']:.4f} - {summary_stats['lpips']['bilinear_max']:.4f}

LPIPS Improvement (Bilinear - Model):
  Mean: {summary_stats['lpips']['improvement_mean']:.4f} ± {summary_stats['lpips']['improvement_std']:.4f}
  Range: {summary_stats['lpips']['improvement_min']:.4f} - {summary_stats['lpips']['improvement_max']:.4f}

Files Generated:
- summary_metrics.png: Bar charts comparing metrics across samples
- metrics_distribution.png: Box plots showing metric distributions
- summary_statistics.json: Detailed numerical statistics
- sample_XXX/: Individual results for each sample
"""
    
    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(summary_text)
    
    print(f"📊 Summary visualizations saved to {output_dir}/summary_metrics.png and {output_dir}/metrics_distribution.png")
    print(f"📈 Aggregated statistics saved to {output_dir}/summary_statistics.json and {output_dir}/summary_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Minimal Satellite Super-Resolution Training")
    
    # Essential parameters only
    parser.add_argument("--dataset", type=str, default="satburst_synth", 
                       choices=["satburst_synth", "worldstrat", "burst_synth", "worldstrat_test"])
    parser.add_argument("--sample_id", default="Landcover-743192_rgb")
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor, or upsampling factor for the data")
    parser.add_argument("--scale_factor", type=float, default=4, help="scale factor for the input training grid")
    
    # Multi-sample optimization parameters
    parser.add_argument("--multi_sample", action="store_true", help="Optimize against all samples in dataset")
    parser.add_argument("--output_folder", type=str, default="multi_sample_results", help="Output folder for multi-sample results")

    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    
    # Model parameters
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10", 
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--use_gnll", action="store_true")
    parser.add_argument("--visualize_variance", action="store_true", help="Visualize variance maps for each LR sample when using GNLL (single sample only)")
    parser.add_argument("--no_base_frame", action="store_true", help="Disable base frame (default: use_base_frame=True)")
    parser.add_argument("--no_direct_param_T", action="store_true", help="Disable direct parameter T (default: use_direct_param_T=True)")
    parser.add_argument("--use_color_shift", action="store_true", help="Use color shift (default: use_color_shift=False)")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="7", help="CUDA device number")
    
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.input_projection = "fourier"

    # Setup dataset
    if args.dataset == "satburst_synth":
        args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
    elif args.dataset == "burst_synth":
        args.root_burst_synth = "SyntheticBurstVal"
        # Convert sample_id to integer for burst_synth dataset
        try:
            args.sample_id = int(args.sample_id)
        except ValueError:
            print(f"Warning: sample_id '{args.sample_id}' cannot be converted to integer for burst_synth dataset. Using 0 instead.")
            args.sample_id = 0

    # Handle multi-sample vs single-sample optimization
    if args.multi_sample:
        # Multi-sample optimization
        print(f"Starting multi-sample optimization for dataset: {args.dataset}")
        
        # Setup output directory
        output_dir = Path(args.output_folder)
        output_dir.mkdir(exist_ok=True)
        
        # Get all samples in the dataset
        if args.dataset == "worldstrat_test":
            # For worldstrat_test, we need to get all sample IDs
            from data import WorldStratTestDataset
            data_root = "worldstrat_test_data"
            sample_dirs = [d for d in Path(data_root).iterdir() if d.is_dir()]
            sample_ids = [d.name for d in sample_dirs]
            print(f"Found {len(sample_ids)} samples: {sample_ids[:5]}...")
        elif args.dataset == "burst_synth":
            # For burst_synth, get all sample IDs from the gt folder
            if 'DATA_DIR_ABSOLUTE' in os.environ:
                data_root = Path(os.environ['DATA_DIR_ABSOLUTE'])
            else:
                data_root = Path("SyntheticBurstVal")
            
            gt_dir = data_root / "gt"
            if gt_dir.exists():
                sample_dirs = [d for d in gt_dir.iterdir() if d.is_dir()]
                sample_ids = [int(d.name) for d in sample_dirs if d.name.isdigit()]
                sample_ids.sort()
                print(f"Found {len(sample_ids)} samples: {sample_ids[:5]}...")
            else:
                print(f"Error: GT directory {gt_dir} not found!")
                return
        elif args.dataset == "satburst_synth":
            # For satburst_synth, each sample is a directory inside data/
            data_root = Path("data")
            if not data_root.exists():
                print(f"Error: data directory {data_root} not found!")
                return
            sample_dirs = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
            sample_ids = [d.name for d in sample_dirs]
            sample_ids.sort()
            print(f"Found {len(sample_ids)} samples: {sample_ids[:5]}...")
        else:
            print(f"Error: Unsupported dataset for multi-sample: {args.dataset}")
            return
        
        # Setup model components (needed for all samples)
        input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale)
        decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)
        
        # Run optimization for each sample
        all_results = []
        for sample_idx, sample_id in enumerate(sample_ids):
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{len(sample_ids)}: {sample_id}")
            print(f"{'='*60}")
            
            # Create a FRESH model for each sample (this is the key fix!)
            print(f"🔄 Creating fresh model for sample {sample_id} (sample {sample_idx + 1}/{len(sample_ids)})")
            model = INR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll, 
                       use_base_frame=not args.no_base_frame, use_direct_param_T=not args.no_direct_param_T, use_color_shift=args.use_color_shift).to(device)
            print(f"✅ Fresh model created and initialized")
            
            # Set the sample_id for this iteration
            args.sample_id = sample_id
            # Recompute dataset-specific roots per sample when needed
            if args.dataset == "satburst_synth":
                args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
            
            # Get dataset for this specific sample
            train_data = get_dataset(args=args, name=args.dataset)
            
            # Run optimization for this sample with the fresh model
            result = optimize_and_evaluate_sample(model, train_data, device, sample_idx, args, output_dir)
            all_results.append(result)
        
        # Create summary visualizations
        create_summary_visualization(all_results, output_dir)
        return
        
    elif args.dataset == "burst_synth":
        # Set the path to SyntheticBurstVal
        if 'DATA_DIR_ABSOLUTE' in os.environ:
            args.root_burst_synth = os.environ['DATA_DIR_ABSOLUTE']
        else:
            args.root_burst_synth = "SyntheticBurstVal"

    train_data = get_dataset(args=args, name=args.dataset)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)

    # Setup model
    input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale)
    decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)
    model = INR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll, 
               use_base_frame=not args.no_base_frame, use_direct_param_T=not args.no_direct_param_T).to(device)
    # model = NIR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)

    print(f"Starting training for {args.iters} iterations...")
    
    # Training loop
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    
    # Lists to store PSNR and losses for plotting
    psnr_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    
    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break
                
            # Train one iteration
            train_losses = train_one_iteration(model, optimizer, train_sample, device, args.df)
            
            # Check for NaN/Inf in losses and break if detected
            if (torch.isnan(torch.tensor(train_losses['recon_loss'])) or 
                torch.isinf(torch.tensor(train_losses['recon_loss'])) or
                torch.isnan(torch.tensor(train_losses['total_loss'])) or 
                torch.isinf(torch.tensor(train_losses['total_loss']))):
                print(f"\nERROR: NaN/Inf detected in losses at iteration {iteration}")
                print(f"Reconstruction loss: {train_losses['recon_loss']}")
                print(f"Total loss: {train_losses['total_loss']}")
                print("Stopping training to prevent further issues.")
                break
            
            scheduler.step()
            iteration += 1

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon': f"{train_losses['recon_loss']:.4f}",
                'trans': f"{train_losses['trans_loss']:.4f}"
            })
            
            # Periodic evaluation
            if iteration % 100 == 0:
                test_loss, test_psnr = test_one_epoch(model, train_data, device)
                print(f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                      f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB")
                
                # Additional debugging for GNLL
                if model.use_gnll and (torch.isnan(torch.tensor(train_losses['recon_loss'])) or 
                                     torch.isinf(torch.tensor(train_losses['recon_loss']))):
                    print(f"WARNING: NaN/Inf detected in reconstruction loss at iteration {iteration}")
                    print(f"Reconstruction loss: {train_losses['recon_loss']}")
                    print(f"Total loss: {train_losses['total_loss']}")

                # Append to lists for plotting
                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses['recon_loss'])
                trans_loss_list.append(train_losses['trans_loss'])
                total_loss_list.append(train_losses['total_loss'])

    progress_bar.close()
    
    # Final evaluation and save output
    model.eval()
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)
        
        if model.use_gnll:
            output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)
        
        final_test_loss = F.mse_loss(output, hr_image).item()   
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()
        
        # Convert tensors to numpy for saving as images
        pred_np = output.squeeze().cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()

        # Build a 3-channel LR baseline image
        if hasattr(train_data, 'get_lr_sample_hwc'):
            lr_original = train_data.get_lr_sample_hwc(0).cpu().numpy()  # H x W x 3
        else:
            lr_original = train_data.get_lr_sample(0).cpu().numpy()      # possibly H x W x (3*T)
            if lr_original.ndim == 3 and lr_original.shape[2] > 3:
                H, W, C = lr_original.shape
                if C % 3 == 0:
                    T = C // 3
                    lr_original = lr_original.reshape(H, W, T, 3)
                    # Use first frame as baseline (or replace with .mean(axis=2) to average)
                    lr_original = lr_original[:, :, 0, :]
                else:
                    # Fallback: take first 3 channels
                    lr_original = lr_original[:, :, :3]

        # Unstandardize LR using dataset stats to get proper colors
        lr_std = train_data.get_lr_std(0).cpu().numpy()
        lr_mean = train_data.get_lr_mean(0).cpu().numpy()
        # Ensure shapes broadcast to HxWx3
        if lr_std.ndim == 1:
            lr_std = lr_std.reshape(1, 1, -1)
        if lr_mean.ndim == 1:
            lr_mean = lr_mean.reshape(1, 1, -1)
        lr_original = lr_original * lr_std + lr_mean

        lr_h, lr_w = lr_original.shape[:2]
        hr_h, hr_w = gt_np.shape[:2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        lr_bilinear = np.clip(lr_bilinear, 0, 1)

        # Convert numpy arrays to torch tensors for alignment and color matching
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # [1, C, H, W]
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # [1, C, H, W]
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # [1, C, H, W]
        
        # Align outputs for fair comparison (following og_main.py approach)
        # Skip alignment for worldstrat_test dataset as images are already aligned
        if args.dataset == "worldstrat_test":
            print("Skipping alignment for worldstrat_test dataset (images are already aligned)")
            pred_aligned = pred_tensor
            bilinear_aligned = bilinear_tensor
            gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
            
            pred_aligned, _ = match_colors(pred_aligned, gt_tensor, pred_aligned, ksz, gauss_kernel)
            bilinear_aligned, _ = match_colors(bilinear_aligned, gt_tensor, bilinear_aligned, ksz, gauss_kernel)
        else:
            gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
            
            # Align model prediction to ground truth
            pred_aligned = align_kornia_brute_force(pred_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
            pred_aligned, _ = match_colors(pred_aligned, gt_tensor, pred_aligned, ksz, gauss_kernel)
            
            # Align bilinear baseline to ground truth
            bilinear_aligned = align_kornia_brute_force(bilinear_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
            bilinear_aligned, _ = match_colors(bilinear_aligned, gt_tensor, bilinear_aligned, ksz, gauss_kernel)


        # PSNR - using aligned tensors for fair comparison
        model_psnr = peak_signal_noise_ratio(pred_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        bilinear_psnr = peak_signal_noise_ratio(bilinear_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()

        # SSIM - using aligned tensors for fair comparison
        model_ssim = ssim(pred_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_aligned.cpu(), gt_tensor.cpu(), data_range=1.0).item()

        # LPIPS (expects [-1,1] range) - using aligned tensors for fair comparison
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        pred_lpips = lpips_fn((pred_aligned*2-1).to(device), (gt_tensor*2-1).to(device)).item()
        bilinear_lpips = lpips_fn((bilinear_aligned*2-1).to(device), (gt_tensor*2-1).to(device)).item()

        # Convert aligned tensors back to numpy for visualization
        pred_aligned_np = pred_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_aligned_np = bilinear_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Ensure aligned images are in valid range
        pred_aligned_np = np.clip(pred_aligned_np, 0, 1)
        bilinear_aligned_np = np.clip(bilinear_aligned_np, 0, 1)
        
        # Create structured output directory for single sample results
        output_base_dir = Path("single_samples")
        dataset_dir = output_base_dir / args.dataset
        sample_dir = dataset_dir / str(args.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison figure with LR, bilinear upsampling (aligned), model output (aligned), and ground truth
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original LR image
        axes[0, 0].imshow(lr_original)
        axes[0, 0].set_title('Original LR Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Bilinear upsampling (color-aligned for fair comparison)
        axes[0, 1].imshow(bilinear_aligned_np)
        axes[0, 1].set_title(f'Bilinear Upsampling (Aligned)\nPSNR: {bilinear_psnr:.2f} dB', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Model output (aligned)
        axes[1, 0].imshow(pred_aligned_np)
        axes[1, 0].set_title(f'Model Output (Aligned)\nPSNR: {model_psnr:.2f} dB', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Ground truth
        axes[1, 1].imshow(gt_np)
        axes[1, 1].set_title('Ground Truth HR', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout(pad=2.0)
        comparison_path = sample_dir / "comparison.png"
        plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        # Save individual images for reference (using aligned images)
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_aligned_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        pred_path = sample_dir / "model_output_aligned.png"
        plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        gt_path = sample_dir / "ground_truth.png"
        plt.savefig(gt_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Save bilinear baseline for reference (aligned version)
        plt.figure(figsize=(8, 8))
        plt.imshow(bilinear_aligned_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        bilinear_path = sample_dir / "bilinear_baseline.png"
        plt.savefig(bilinear_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Save LR original for reference
        plt.figure(figsize=(8, 8))
        plt.imshow(lr_original)
        plt.axis('off')
        plt.tight_layout(pad=0)
        lr_path = sample_dir / "lr_original.png"
        plt.savefig(lr_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        output_path = comparison_path
        
    print(f"\nFinal Results:")
    print(f"Test Loss: {final_test_loss:.6f}")
    print(f"Test PSNR: {final_psnr:.2f} dB")
    print(f"Model PSNR: {model_psnr:.2f} dB")
    print(f"Bilinear PSNR: {bilinear_psnr:.2f} dB")
    print(f"PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB")
    print(f"Model output saved to {output_path}")
    
    # Create structured output directory for single sample results
    output_base_dir = Path("single_samples")
    dataset_dir = output_base_dir / args.dataset
    sample_dir = dataset_dir / str(args.sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PSNR results to a text file in the structured directory
    results_text = f"""Super-Resolution Results
    =======================

    Dataset: {args.dataset}
    Sample ID: {args.sample_id}
    Downsampling Factor: {args.df}
    Model: {args.model}
    Iterations: {args.iters}

    PSNR Results:
    - Model Output: {model_psnr:.2f} dB
    - Bilinear Interpolation: {bilinear_psnr:.2f} dB
    - PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB

    SSIM Results:
    - Model Output: {model_ssim:.4f}
    - Bilinear Interpolation: {bilinear_ssim:.4f}
    - SSIM Improvement: {model_ssim - bilinear_ssim:.4f}

    LPIPS Results:
    - Model Output: {pred_lpips:.4f}
    - Bilinear Interpolation: {bilinear_lpips:.4f}
    - LPIPS Improvement: {bilinear_lpips - pred_lpips:.4f}

    Training Results:
    - Final Test Loss: {final_test_loss:.6f}
    - Final Test PSNR: {final_psnr:.2f} dB
    - Final Reconstruction Loss: {recon_loss_list[-1] if recon_loss_list else 0:.6f}
    - Final Transformation Loss: {trans_loss_list[-1] if trans_loss_list else 0:.6f}
    - Final Total Loss: {total_loss_list[-1] if total_loss_list else 0:.6f}

    Training Metrics History:
    """
    
    if len(psnr_list) > 0:
        results_text += f"- Number of evaluation points: {len(psnr_list)}\n"
        results_text += f"- PSNR range: {min(psnr_list):.2f} - {max(psnr_list):.2f} dB\n"
        results_text += f"- Reconstruction loss range: {min(recon_loss_list):.6f} - {max(recon_loss_list):.6f}\n"
        results_text += f"- Transformation loss range: {min(trans_loss_list):.6f} - {max(trans_loss_list):.6f}\n"
        results_text += f"- Total loss range: {min(total_loss_list):.6f} - {max(total_loss_list):.6f}\n"
        results_text += f"- Final PSNR: {psnr_list[-1]:.2f} dB\n"
        results_text += f"- Final reconstruction loss: {recon_loss_list[-1]:.6f}\n"
        results_text += f"- Final transformation loss: {trans_loss_list[-1]:.6f}\n"
        results_text += f"- Final total loss: {total_loss_list[-1]:.6f}\n"
    else:
        results_text += "- No training metrics recorded (training may have been too short)\n"
    
    # Save to both current directory (for backward compatibility) and structured directory
    with open("psnr_results.txt", "w") as f:
        f.write(results_text)
    
    with open(sample_dir / "metrics.txt", "w") as f:
        f.write(results_text)
    
    # Save metrics as JSON for easier parsing
    metrics_dict = {
        'dataset': args.dataset,
        'sample_id': str(args.sample_id),
        'downsampling_factor': args.df,
        'model': args.model,
        'iterations': args.iters,
        'learning_rate': args.learning_rate,
        'psnr': {
            'model': model_psnr,
            'bilinear': bilinear_psnr,
            'improvement': model_psnr - bilinear_psnr
        },
        'ssim': {
            'model': model_ssim,
            'bilinear': bilinear_ssim,
            'improvement': model_ssim - bilinear_ssim
        },
        'lpips': {
            'model': pred_lpips,
            'bilinear': bilinear_lpips,
            'improvement': bilinear_lpips - pred_lpips
        },
        'training': {
            'final_test_loss': final_test_loss,
            'final_test_psnr': final_psnr,
            'final_recon_loss': recon_loss_list[-1] if recon_loss_list else 0,
            'final_trans_loss': trans_loss_list[-1] if trans_loss_list else 0,
            'final_total_loss': total_loss_list[-1] if total_loss_list else 0
        }
    }
    
    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Results saved to: {sample_dir}")
    print(f"PSNR results also saved to psnr_results.txt (current directory)")

    # Plot PSNR and all losses
    if len(psnr_list) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot PSNR on top subplot
        ax1.plot(iteration_list, psnr_list, color='blue', linewidth=2, label='PSNR (Test)')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Training PSNR Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot all losses on bottom subplot
        ax2.plot(iteration_list, recon_loss_list, color='red', linewidth=2, label='Reconstruction Loss')
        ax2.plot(iteration_list, trans_loss_list, color='green', linewidth=2, label='Transformation Loss')
        ax2.plot(iteration_list, total_loss_list, color='purple', linewidth=2, label='Total Loss')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        # Save to both current directory (for backward compatibility) and structured directory
        plt.savefig("training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.savefig(sample_dir / "training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        print(f"Training metrics plot saved to training_metrics.png and {sample_dir}/training_metrics.png")
    else:
        print("No metrics data available for plotting (training may have been too short)")
    
    # Generate variance visualizations if requested and using GNLL
    if args.visualize_variance and model.use_gnll and not args.multi_sample:
        print("Generating variance visualizations for each LR sample...")
        # Clear GPU memory before variance visualization
        torch.cuda.empty_cache()
        visualize_lr_variance(model, train_data, device, sample_dir, args.sample_id)
    elif args.visualize_variance and not model.use_gnll:
        print("Warning: --visualize_variance requested but model does not use GNLL. Skipping variance visualization.")
    elif args.visualize_variance and args.multi_sample:
        print("Warning: --visualize_variance is only supported for single sample optimization. Skipping variance visualization.")


if __name__ == "__main__":
    main() 