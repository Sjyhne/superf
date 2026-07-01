import cv2
import numpy as np
import pathlib
import random
import json
import torch
import torch.nn.functional as F
from utils import apply_shift_torch, sentinel2_psf_downsample_torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import os

def generate_random_translations(num_samples, max_pixels_x, max_pixels_y=None):
    if max_pixels_y is None:
        max_pixels_y = max_pixels_x
        
    translations = []
    for i in range(num_samples):
        if i == 0:
            dx = 0.0
            dy = 0.0
        else:
            dx = random.uniform(-max_pixels_x, max_pixels_x)
            dy = random.uniform(-max_pixels_y, max_pixels_y)
        translations.append((dx, dy))
    return translations

def generate_fixed_lr_translations(num_samples, lr_pixel_shifts, factor):
    translations = []
    translations.append((0.0, 0.0))
    for lr_shift in lr_pixel_shifts:
        hr_shift = lr_shift * factor
        samples_per_shift = (num_samples - 1) // len(lr_pixel_shifts)
        for i in range(samples_per_shift):
            angle = random.uniform(0, 2 * np.pi)
            dx = hr_shift * np.cos(angle)
            dy = hr_shift * np.sin(angle)
            translations.append((dx, dy))
    
    return translations

def generate_fixed_magnitude_translations(num_samples, lr_pixel_shift, factor):
    translations = []
    translations.append((0.0, 0.0))
    hr_shift = lr_pixel_shift * factor
    for i in range(num_samples - 1):
        dx = random.uniform(-hr_shift, hr_shift)
        dy = random.uniform(-hr_shift, hr_shift)
        translations.append((dx, dy))
    
    return translations

def apply_atmospheric_augmentations(image, seed, augment_params=None):
    torch.manual_seed(seed)
    
    if augment_params is None:
        augment_params = {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
        }
    aug_image = image.clone()
    saturation_factor = torch.empty(1).uniform_(*augment_params['saturation']).item()
    aug_image = TF.adjust_saturation(aug_image, saturation_factor)
    
    hue_factor = torch.empty(1).uniform_(*augment_params['hue']).item()
    aug_image = TF.adjust_hue(aug_image, hue_factor)
    
    brightness_factor = torch.empty(1).uniform_(*augment_params['brightness']).item()
    aug_image = TF.adjust_brightness(aug_image, brightness_factor)
    
    contrast_factor = torch.empty(1).uniform_(*augment_params['contrast']).item()
    aug_image = TF.adjust_contrast(aug_image, contrast_factor)

    # Add heavy noise to the image
    aug_image = aug_image + torch.randn_like(aug_image) * 0.1
    
    # Ensure values stay in [0, 1] range
    aug_image = torch.clamp(aug_image, 0, 1)
    
    return aug_image

if __name__ == "__main__":
    # Set seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensures deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = "cuda:0"
    
    # Image settings
    hr_patch_size = 256 # Set to None to use full image, or a number (e.g., 256) for patch extraction
    
    # Multiple downsampling factors to generate
    downsampling_factors = [1]
    
    # Translation settings
    lr_pixel_shifts = [0.5, 1.0, 2.0, 4.0]  # Each will get its own dataset
    sample_counts = [64] #[1, 4, 8, 12, 16]  # Different numbers of samples to test
    
    # Cropping settings
    border_crop = 20  # Pixels to remove from border (black edge)
    
    # Create base data directory
    base_save_folder = pathlib.Path("data")
    
    # Input folder containing HR images
    input_folder = pathlib.Path("SynthSatBurst/hr_images_20")
    
    # Get all image files from the input folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Augmentation settings
    augmentation_types = ['none', 'light', 'medium']
    augment_params = {
        'light': {
            'brightness': (0.9, 1.1),
            'contrast': (0.9, 1.1),
            'saturation': (0.9, 1.1),
            'hue': (-0.05, 0.05),
            'noise': 0.005
        },
        'medium': {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
            'noise': 0.01
        },
        'heavy': {
            'brightness': (0.6, 1.4),
            'contrast': (0.6, 1.4),
            'saturation': (0.6, 1.4),
            'hue': (-0.2, 0.2),
            'noise': 0.02
        }
    }
    
    # Process each image
    for img_idx, img_path in enumerate(image_files):
        print(f"\nProcessing image {img_idx+1}/{len(image_files)}: {img_path.name}")
        
        # Load and crop border from image
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = original_img[border_crop:-border_crop, border_crop:-border_crop]
        
        # Convert to torch tensor
        original_img = torch.from_numpy(original_img).float() / 255.0
        original_img = original_img.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Get image dimensions
        height, width = original_img.shape[-2:]
        
        # Determine if we're using a patch or the full image
        if hr_patch_size is None:
            # Use full image
            hr_img = original_img
            print(f"  Using full image with dimensions {width}x{height}")
        else:
            # Calculate center position for the base patch
            center_y = (height // 2) + 100
            center_x = (width // 2) + 100
            half_patch = hr_patch_size // 2
            
            # Make sure the patch fits within the image
            if center_y - half_patch < 0 or center_y + half_patch > height or \
               center_x - half_patch < 0 or center_x + half_patch > width:
                print(f"Warning: Image {img_path.name} is too small for the patch size. Skipping.")
                continue
            
            # Extract the center patch (this will be our HR ground truth)
            hr_img = original_img[
                :,
                :,
                center_y - half_patch:center_y + half_patch,
                center_x - half_patch:center_x + half_patch
            ]
            print(f"  Using patch of size {hr_patch_size}x{hr_patch_size} from center of image")
        
        # Create a subfolder for this image
        image_name = img_path.stem
        
        # Process each combination of factor, shift, and sample count
        for factor in downsampling_factors:
            for lr_shift in lr_pixel_shifts:
                for num_samples in sample_counts:
                    # Generate translations once for all augmentation types
                    random.seed(seed + img_idx)  # Use different seed for each image
                    translations = generate_fixed_magnitude_translations(
                        num_samples=num_samples,
                        lr_pixel_shift=lr_shift,
                        factor=factor
                    )
                    
                    # Now use these same translations for each augmentation type
                    for aug_type in augmentation_types:
                        print(f"  Processing factor {factor}x with {lr_shift}px LR shift, "
                              f"{num_samples} samples, and {aug_type} augmentation")
                        
                        # Create directory for this combination
                        save_folder = base_save_folder / image_name / f"scale_{factor}_shift_{lr_shift:.1f}px_aug_{aug_type}"
                        save_folder.mkdir(parents=True, exist_ok=True)
                        
                        transform_log = {}
                        
                        for i, (dx, dy) in enumerate(translations):
                            if i == 0:
                                dx = 0
                                dy = 0
                            
                            # 1. First apply shift to full image
                            shifted_full = apply_shift_torch(
                                original_img,
                                dx=torch.tensor([dx], device=device),
                                dy=torch.tensor([dy], device=device)
                            )
                            
                            # 2. Extract HR patch or use full image
                            if hr_patch_size is None:
                                shifted_patch = shifted_full
                            else:
                                shifted_patch = shifted_full[
                                    :,
                                    :,
                                    center_y - half_patch:center_y + half_patch,
                                    center_x - half_patch:center_x + half_patch
                                ]
                            
                            # 3. Save HR patch for reference sample without augmentation
                            if i == 0:
                                hr_np = (shifted_patch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                hr_np = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(str(save_folder / "hr_ground_truth.png"), hr_np)
                            # Apply augmentations only to non-reference samples (i > 0)
                            if aug_type != 'none' and i > 0:
                                aug_seed = seed * 1000 + img_idx * 100 + i
                                
                                # Apply atmospheric augmentations
                                if i > 1:
                                    shifted_patch = apply_atmospheric_augmentations(
                                        shifted_patch,
                                        seed=aug_seed,
                                        augment_params=augment_params[aug_type]
                                    )
                                
                                # Add noise with correct shape
                                permuted_patch = shifted_patch[0].permute(1, 2, 0)
                                noise = np.random.randn(*permuted_patch.shape) * augment_params[aug_type]['noise']
                                hr_np = (permuted_patch.cpu().numpy() * 255).astype(np.uint8) + noise
                                hr_np = np.clip(hr_np, 0, 255).astype(np.uint8)
                            
                            # 4. Downsample to LR with the Sentinel-2 PSF simulation from DSen2:
                            #    Gaussian blur with sigma=1/factor pixels, then factor x factor
                            #    window averaging.
                            shifted_lr = sentinel2_psf_downsample_torch(shifted_patch, factor)
                            lr_height, lr_width = shifted_lr.shape[-2:]

                            # Save LR sample
                            lr_np = (shifted_lr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(save_folder / f"sample_{i:02d}.png"), lr_np)
                            
                            # Update transform log
                            transform_log[f"sample_{i:02d}"] = {
                                'dx_pixels_hr': dx,
                                'dy_pixels_hr': dy,
                                'dx_pixels_lr': dx / factor,
                                'dy_pixels_lr': dy / factor,
                                'dx_percent': (dx/factor) / lr_width,
                                'dy_percent': (dy/factor) / lr_height,
                                'magnitude_pixels_hr': np.sqrt(dx**2 + dy**2),
                                'magnitude_pixels_lr': np.sqrt((dx/factor)**2 + (dy/factor)**2),
                                'shape': lr_np.shape,
                                'path': f"sample_{i:02d}.png",
                                'augmentation': aug_type,
                                'downsampling': {
                                    'method': 'sentinel2_psf_gaussian_blur_avg_pool',
                                    'factor': factor,
                                    'gaussian_sigma_pixels': 1.0 / factor if factor > 1 else 0.0,
                                    'window_size_pixels': factor
                                }
                            }
                        
                        # Save transform log
                        with open(save_folder / "transform_log.json", 'w') as f:
                            json.dump(transform_log, f, indent=2)

    print("\nProcessing complete!")
