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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)
    
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
        
        # Get LR for bilinear comparison
        lr_original = train_data.get_lr_sample(0).cpu().numpy()
        lr_h, lr_w = lr_original.shape[:2]
        hr_h, hr_w = hr_image.shape[1], hr_image.shape[2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        # Align outputs for fair comparison
        print("Aligning outputs for fair comparison")
        gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
        
        # Align model prediction to ground truth
        pred_aligned = align_kornia_brute_force(pred_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
        pred_aligned, _ = match_colors(gt_tensor, pred_tensor, pred_aligned, ksz, gauss_kernel)
        
        # Align bilinear baseline to ground truth
        bilinear_aligned = align_kornia_brute_force(bilinear_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
        bilinear_aligned, _ = match_colors(gt_tensor, bilinear_tensor, bilinear_aligned, ksz, gauss_kernel)

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
    model_lpips = [r['image_metrics']['model_lpips'] for r in all_results]
    
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
    
    # SSIM scores
    axes[1, 0].bar(sample_indices, model_ssim, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('SSIM Scores Across Samples')
    axes[1, 0].grid(True, alpha=0.3)
    
    # LPIPS scores
    axes[1, 1].bar(sample_indices, model_lpips, alpha=0.7, color='brown')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('LPIPS')
    axes[1, 1].set_title('LPIPS Scores Across Samples')
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
    axes[1].boxplot([model_ssim], labels=['Model'])
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # LPIPS box plot
    axes[2].boxplot([model_lpips], labels=['Model'])
    axes[2].set_ylabel('LPIPS')
    axes[2].set_title('LPIPS Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_distribution.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
    print(f"📊 Summary visualizations saved to {output_dir}/summary_metrics.png and {output_dir}/metrics_distribution.png")


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
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
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

    # Parse input projection
    if args.input_projection.startswith("fourier_"):
        args.fourier_scale = float(args.input_projection.split("_")[1])
        args.input_projection = "fourier"

    # Setup dataset
    if args.dataset == "satburst_synth":
        args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
    elif args.dataset == "burst_synth":
        args.root_burst_synth = "SyntheticBurstVal"

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
        else:
            args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
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
    model = INR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)
    # model = NIR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

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
        print("Aligning outputs for fair comparison")
        gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
        
        # Align model prediction to ground truth
        pred_aligned = align_kornia_brute_force(pred_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
        pred_aligned, _ = match_colors(gt_tensor, pred_tensor, pred_aligned, ksz, gauss_kernel)
        
        # Align bilinear baseline to ground truth
        bilinear_aligned = align_kornia_brute_force(bilinear_tensor.squeeze(0), gt_tensor.squeeze(0)).unsqueeze(0)
        bilinear_aligned, _ = match_colors(gt_tensor, bilinear_tensor, bilinear_aligned, ksz, gauss_kernel)

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
        
        # Save comparison figure with LR, bilinear upsampling, model output, and ground truth
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original LR image
        axes[0, 0].imshow(lr_original)
        axes[0, 0].set_title('Original LR Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Bilinear upsampling (aligned)
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
        comparison_path = Path("super_resolution_comparison.png")
        plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        # Also save individual images for reference (using aligned images)
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_aligned_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        pred_path = Path("super_resolution_output_aligned.png")
        plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        gt_path = Path("ground_truth.png")
        plt.savefig(gt_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        output_path = comparison_path
        
    print(f"\nFinal Results:")
    print(f"Test Loss: {final_test_loss:.6f}")
    print(f"Test PSNR: {final_psnr:.2f} dB")
    print(f"Model PSNR: {model_psnr:.2f} dB")
    print(f"Bilinear PSNR: {bilinear_psnr:.2f} dB")
    print(f"PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB")
    print(f"Model output saved to {output_path}")
    
    # Save PSNR results to a text file
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
    
    with open("psnr_results.txt", "w") as f:
        f.write(results_text)
    
    print(f"PSNR results saved to psnr_results.txt")

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
        plt.savefig("training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        print(f"Training metrics plot saved to training_metrics.png")
    else:
        print("No metrics data available for plotting (training may have been too short)")


if __name__ == "__main__":
    main() 