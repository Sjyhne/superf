import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import glob
import cv2
import json
from datetime import datetime

from data import get_and_standardize_image
from models.utils import get_decoder
from optimize import train_one_iteration
from input_projections.utils import get_input_projection
from models.inr import INR


def make_square(img):
    """Pad image to square by centering it."""
    h, w = img.shape[:2]
    max_dim = max(h, w)

    if len(img.shape) == 3:
        square_img = np.zeros((max_dim, max_dim, img.shape[2]))
    else:
        square_img = np.zeros((max_dim, max_dim))

    start_h = (max_dim - h) // 2
    start_w = (max_dim - w) // 2
    square_img[start_h:start_h+h, start_w:start_w+w] = img

    return square_img


def save_image_multi_size(img, save_dir, base_name):
    """Save image in multiple sizes (web, fullscreen, thumbnail).

    Returns dict with filenames for each size.
    """
    sizes = {
        'web': (8, 8),        # 800x800
        'fullscreen': (12, 12),  # 1200x1200
        'thumb': (2, 2)       # 200x200
    }
    filenames = {}

    for size_name, figsize in sizes.items():
        filename = f"{base_name}_{size_name}.png"
        plt.figure(figsize=figsize, dpi=100)
        plt.imshow(img)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_dir / filename, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        filenames[size_name] = filename

    return filenames


class ImageDataset:
    def __init__(self, images, lr_images, means, stds, hr_coords, device, downsample_factor):
        self.images = images  # List of HR images
        self.lr_images = lr_images  # List of LR images  
        self.means = means  # List of means for each image
        self.stds = stds  # List of stds for each image
        self.hr_coords = hr_coords
        self.device = device
        self.num_samples = len(images)
        self.downsample_factor = downsample_factor
        # Record LR spatial size from first image
        lr_h, lr_w = lr_images[0].shape[:2]
        self.lr_size = (lr_h, lr_w)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Create training sample format using same coordinate system as data.py
        hr_img = self.images[idx]
        lr_img = self.lr_images[idx]
        
        # Use pre-computed HR coordinates (model downsamples during training)
        coords = self.hr_coords.unsqueeze(0)
        
        return {
            'input': coords.to(self.device),
            'lr_target': lr_img.unsqueeze(0).to(self.device),
            'mean': self.means[idx].unsqueeze(0).to(self.device),
            'std': self.stds[idx].unsqueeze(0).to(self.device),
            'sample_id': torch.tensor([idx]).to(self.device),
            'scale_factor': torch.tensor([float(self.downsample_factor)]).to(self.device),
            'shifts': {
                'dx_percent': torch.tensor([0.0]).to(self.device),
                'dy_percent': torch.tensor([0.0]).to(self.device)
            }
        }
    
    def get_hr_coordinates(self):
        """Return HR coordinate grid for inference."""
        return self.hr_coords
    
    def get_original_hr(self):
        """Return first HR image for reference."""
        return self.images[0]
    
    def get_lr_sample(self, index):
        """Return unstandardized LR sample."""
        lr_img = self.lr_images[index]
        mean = self.means[index]
        std = self.stds[index]
        return lr_img * std + mean
    
    def get_lr_mean(self, index):
        """Return mean for index."""
        return self.means[index]
    
    def get_lr_std(self, index):
        """Return std for index."""
        return self.stds[index]

    def get_lr_size(self):
        return self.lr_size


def create_image_dataset(folder_path, downsample_factor, device):
    """Create a dataset from LR images in a folder. The target HR resolution will be LR_size * downsample_factor."""
    # Find all image files
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP', '*.tiff', '*.TIFF']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(Path(folder_path) / ext)))
    
    if not image_paths:
        print(f"No images found in {folder_path}")
        return None
    
    print(f"Found {len(image_paths)} LR images for super-resolution training")
    
    hr_images = []  # These will be placeholder HR images (not used for training supervision)
    lr_images = []  # The actual LR images from the folder
    means = []
    stds = []
    
    for img_path in image_paths:
        try:
            # Load LR image using cv2 (same as data.py)
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lr_tensor = torch.from_numpy(img).float() / 255.0
            
            # Calculate target HR resolution
            lr_h, lr_w = lr_tensor.shape[:2]
            hr_h, hr_w = lr_h * downsample_factor, lr_w * downsample_factor
            
            # Create placeholder HR image (zeros) - we don't have true HR ground truth
            hr_tensor = torch.zeros(hr_h, hr_w, 3)
            
            # Standardize the LR image (same as data.py)
            lr_standardized, mean, std = get_and_standardize_image(lr_tensor)
            
            hr_images.append(hr_tensor)  # Placeholder HR
            lr_images.append(lr_standardized)  # Actual LR from folder
            means.append(mean)
            stds.append(std)
            
            print(f"Loaded {Path(img_path).name}: LR {lr_w}x{lr_h} -> Target HR {hr_w}x{hr_h}")
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not lr_images:
        return None
    
    # Create coordinate grid for the TARGET HR resolution
    hr_h, hr_w = hr_images[0].shape[:2]
    print(f"\nCreating coordinate grid for target HR resolution: {hr_h}x{hr_w}")
    hr_coords_h = np.linspace(0, 1, hr_h, endpoint=False)
    hr_coords_w = np.linspace(0, 1, hr_w, endpoint=False)
    hr_coords = np.stack(np.meshgrid(hr_coords_w, hr_coords_h, indexing='xy'), -1)
    hr_coords = torch.FloatTensor(hr_coords)
    
    print(f"Dataset created with {len(lr_images)} LR images")
    print(f"Training flow: HR coords ({hr_h}x{hr_w}) -> Model -> Downsampled to LR ({lr_h}x{lr_w}) for training -> Full HR for inference")
    
    return ImageDataset(hr_images, lr_images, means, stds, hr_coords, device, downsample_factor)


def save_checkpoint_data(model, train_data, device, iteration, output_dir, args, loss_data):
    """Save checkpoint data for web visualization."""
    model.eval()
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints" / f"iter_{iteration:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'model_type': args.model,
            'network_depth': args.network_depth,
            'network_hidden_dim': args.network_hidden_dim,
            'downsample_factor': args.df,
            'num_samples': len(train_data)
        },
        'training_config': {
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'total_iterations': args.iters
        },
        'losses': loss_data,
        'samples': [],
        'alignment_stats': {}
    }
    
    # Collect shift data for statistics
    all_dx = []
    all_dy = []
    all_magnitudes = []
    
    # Save data for each image sample
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        
        for idx in range(len(train_data)):
            sample_id = torch.tensor([idx]).to(device)
            
            # Get model output
            if model.use_gnll:
                output, pred_shifts, pred_variance = model(hr_coords, sample_id, scale_factor=1, training=False)
                # Handle pred_variance being either a tensor or list
                if isinstance(pred_variance, list):
                    # If it's a list, convert to tensor or take mean of list
                    if len(pred_variance) > 0:
                        if torch.is_tensor(pred_variance[0]):
                            variance_val = torch.stack(pred_variance).mean().cpu().item()
                        else:
                            variance_val = float(np.mean(pred_variance))
                    else:
                        variance_val = 0.0
                else:
                    # If it's already a tensor
                    variance_val = pred_variance.mean().cpu().item()
            else:
                output, pred_shifts = model(hr_coords, sample_id, scale_factor=1, training=False)
                variance_val = None
            
            # Unstandardize the output
            output = output * train_data.get_lr_std(idx).to(device) + train_data.get_lr_mean(idx).to(device)
            sr_output = output.squeeze().cpu().numpy()
            sr_output = np.clip(sr_output, 0, 1)
            
            # Get shift predictions (handle None case when training=False)
            if pred_shifts is not None:
                pred_dx, pred_dy = pred_shifts
                dx_val = pred_dx.cpu().item()
                dy_val = pred_dy.cpu().item()
                magnitude = np.sqrt(dx_val**2 + dy_val**2)
            else:
                # No shifts when training=False (initial checkpoint)
                dx_val = 0.0
                dy_val = 0.0
                magnitude = 0.0
            
            # Collect for statistics
            all_dx.append(dx_val)
            all_dy.append(dy_val)
            all_magnitudes.append(magnitude)
            
            # Get original LR image for comparison
            lr_original = train_data.get_lr_sample(idx).cpu().numpy()
            
            # Create bilinear interpolation of LR image to HR resolution
            lr_h, lr_w = lr_original.shape[:2]
            hr_h, hr_w = sr_output.shape[:2]
            
            # Use OpenCV for bilinear interpolation
            lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)

            # Make all images square
            sr_square = make_square(sr_output)
            lr_square = make_square(lr_original)
            bilinear_square = make_square(lr_bilinear)

            # Save images in multiple sizes using helper function
            sr_files = save_image_multi_size(sr_square, checkpoint_dir, f"sr_sample_{idx}")
            bilinear_files = save_image_multi_size(bilinear_square, checkpoint_dir, f"bilinear_sample_{idx}")

            # Save LR reference images (only once, at first iteration)
            lr_reference_dir = output_dir / "lr_reference"
            lr_reference_dir.mkdir(exist_ok=True)
            lr_ref_web = lr_reference_dir / f"lr_sample_{idx}_web.png"

            if not lr_ref_web.exists():
                lr_files = save_image_multi_size(lr_square, lr_reference_dir, f"lr_sample_{idx}")
            else:
                lr_files = {
                    'web': f"lr_sample_{idx}_web.png",
                    'fullscreen': f"lr_sample_{idx}_fullscreen.png",
                    'thumb': f"lr_sample_{idx}_thumb.png"
                }

            # Store sample metadata with multiple image sizes
            sample_data = {
                'sample_id': idx,
                'images': {
                    'sr_web': sr_files['web'],
                    'sr_fullscreen': sr_files['fullscreen'],
                    'sr_thumbnail': sr_files['thumb'],
                    'lr_web': lr_files['web'],
                    'lr_fullscreen': lr_files['fullscreen'],
                    'lr_thumbnail': lr_files['thumb'],
                    'bilinear_web': bilinear_files['web'],
                    'bilinear_fullscreen': bilinear_files['fullscreen'],
                    'bilinear_thumbnail': bilinear_files['thumb']
                },
                'lr_reference_paths': {
                    'web': f"lr_reference/{lr_files['web']}",
                    'fullscreen': f"lr_reference/{lr_files['fullscreen']}",
                    'thumbnail': f"lr_reference/{lr_files['thumb']}"
                },
                'alignment': {
                    'dx': dx_val,
                    'dy': dy_val,
                    'magnitude': magnitude,
                    'angle_degrees': np.degrees(np.arctan2(dy_val, dx_val)) if dx_val != 0 or dy_val != 0 else 0
                },
                'image_stats': {
                    'original_sr_shape': sr_output.shape,
                    'original_lr_shape': lr_original.shape,
                    'bilinear_shape': lr_bilinear.shape,
                    'square_shape': sr_square.shape,
                    'sr_mean': float(sr_output.mean()),
                    'sr_std': float(sr_output.std()),
                    'sr_min': float(sr_output.min()),
                    'sr_max': float(sr_output.max()),
                    'bilinear_mean': float(lr_bilinear.mean()),
                    'bilinear_std': float(lr_bilinear.std()),
                    'bilinear_min': float(lr_bilinear.min()),
                    'bilinear_max': float(lr_bilinear.max())
                }
            }
            
            if variance_val is not None:
                sample_data['uncertainty'] = variance_val
                
            checkpoint_data['samples'].append(sample_data)
    
    # Add alignment statistics for this iteration
    checkpoint_data['alignment_stats'] = {
        'mean_dx': float(np.mean(all_dx)),
        'mean_dy': float(np.mean(all_dy)),
        'std_dx': float(np.std(all_dx)),
        'std_dy': float(np.std(all_dy)),
        'mean_magnitude': float(np.mean(all_magnitudes)),
        'max_magnitude': float(np.max(all_magnitudes)),
        'min_magnitude': float(np.min(all_magnitudes)),
        'alignment_convergence': float(np.std(all_magnitudes))
    }
    
    # Save checkpoint metadata
    with open(checkpoint_dir / "checkpoint_data.json", 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    model.train()
    return checkpoint_data


def create_api_documentation(all_checkpoints, final_loss_history):
    """Create comprehensive API documentation for data access."""
    return {
        "data_schema": {
            "master_data.json": {
                "description": "Main experiment metadata and checkpoint index",
                "structure": {
                    "experiment_info": {
                        "total_iterations": "int - Total training iterations completed",
                        "num_checkpoints": "int - Number of saved checkpoints",
                        "num_samples": "int - Number of image samples",
                        "created_at": "string - ISO timestamp of generation",
                        "checkpoint_intervals": "array[int] - List of iteration numbers with saved data",
                        "includes_initial_state": "boolean - Whether iteration 0 is included",
                        "image_sizes": "object - Available image size formats"
                    },
                    "checkpoints": "array[object] - Summary of each checkpoint"
                }
            },
            "loss_evolution.json": {
                "description": "Complete training loss history",
                "structure": {
                    "iterations": "array[int] - Iteration numbers",
                    "recon_losses": "array[float|null] - Reconstruction loss values (null for iter 0)",
                    "trans_losses": "array[float|null] - Translation loss values (null for iter 0)",
                    "total_losses": "array[float|null] - Total loss values (null for iter 0)",
                    "has_initial_point": "boolean - Whether iteration 0 is included"
                }
            },
            "alignment_evolution.json": {
                "description": "Alignment shift evolution throughout training",
                "structure": {
                    "iterations": "array[int] - Checkpoint iteration numbers",
                    "global_stats": {
                        "mean_dx": "array[float] - Mean horizontal shift per iteration",
                        "mean_dy": "array[float] - Mean vertical shift per iteration",
                        "mean_magnitude": "array[float] - Mean shift magnitude per iteration",
                        "convergence": "array[float] - Convergence metric (lower = more aligned)"
                    },
                    "individual_samples": {
                        "sample_X": {
                            "dx": "array[float] - Horizontal shifts for this sample",
                            "dy": "array[float] - Vertical shifts for this sample",
                            "magnitude": "array[float] - Shift magnitudes for this sample",
                            "angle_degrees": "array[float] - Shift angles in degrees"
                        }
                    }
                }
            },
            "image_evolution.json": {
                "description": "Image paths and quality metrics evolution",
                "structure": {
                    "iterations": "array[int] - Checkpoint iteration numbers",
                    "available_sizes": "array[string] - Available image sizes ['web', 'fullscreen', 'thumbnail']",
                    "image_paths": {
                        "sample_X": {
                            "lr_reference": {
                                "web": "string - Path to 800x800 LR reference",
                                "fullscreen": "string - Path to 1200x1200 LR reference",
                                "thumbnail": "string - Path to 200x200 LR reference"
                            },
                            "sr_progression": {
                                "web": "array[string] - Paths to 800x800 SR outputs per iteration",
                                "fullscreen": "array[string] - Paths to 1200x1200 SR outputs per iteration",
                                "thumbnail": "array[string] - Paths to 200x200 SR outputs per iteration"
                            }
                        }
                    }
                }
            }
        },
        "access_patterns": {
            "get_image_at_iteration": {
                "description": "Get image path for specific sample at specific iteration",
                "example": "image_evolution.image_paths.sample_0.sr_progression.web[iteration_index]",
                "parameters": {
                    "sample_id": "int - Sample identifier (0 to num_samples-1)",
                    "iteration_index": "int - Index in iterations array (0 to num_checkpoints-1)",
                    "size": "string - Image size ('web', 'fullscreen', 'thumbnail')"
                }
            },
            "get_loss_at_iteration": {
                "description": "Get loss values for specific iteration",
                "example": "loss_evolution.recon_losses[iteration_index]",
                "note": "Returns null for iteration 0, float values for training iterations"
            }
        }
    }


def save_web_visualization_data(output_dir, all_checkpoints, final_loss_history):
    """Save comprehensive data for web visualization including intermediate shifts and images."""
    web_data_dir = output_dir / "web_data"
    web_data_dir.mkdir(exist_ok=True)
    
    # Create master metadata file
    master_data = {
        'experiment_info': {
            'total_iterations': len(final_loss_history),
            'num_checkpoints': len(all_checkpoints),
            'num_samples': len(all_checkpoints[0]['samples']) if all_checkpoints else 0,
            'created_at': datetime.now().isoformat(),
            'checkpoint_intervals': [cp['iteration'] for cp in all_checkpoints],
            'includes_initial_state': True,
            'image_sizes': {
                'web': '800x800px',
                'fullscreen': '1200x1200px', 
                'thumbnail': '200x200px'
            }
        },
        'checkpoints': []
    }
    
    # Process each checkpoint with enhanced data
    for checkpoint in all_checkpoints:
        checkpoint_info = {
            'iteration': checkpoint['iteration'],
            'relative_path': f"checkpoints/iter_{checkpoint['iteration']:04d}",
            'losses': checkpoint['losses'],
            'alignment_stats': checkpoint['alignment_stats'],
            'samples': checkpoint['samples'],
            'is_initial': checkpoint['iteration'] == 0
        }
        master_data['checkpoints'].append(checkpoint_info)
    
    # Enhanced loss evolution data - fix infinity values
    loss_evolution = {
        'iterations': [0] + [entry['iteration'] for entry in final_loss_history],
        'recon_losses': [None] + [entry['recon_loss'] for entry in final_loss_history],  # Use None instead of inf
        'trans_losses': [None] + [entry['trans_loss'] for entry in final_loss_history], 
        'total_losses': [None] + [entry['total_loss'] for entry in final_loss_history],
        'has_initial_point': True,
        'note': 'Initial iteration (0) has null loss values as no training has occurred yet'
    }
    
    # Enhanced alignment evolution data
    alignment_evolution = {
        'iterations': [cp['iteration'] for cp in all_checkpoints],
        'global_stats': {
            'mean_dx': [cp['alignment_stats']['mean_dx'] for cp in all_checkpoints],
            'mean_dy': [cp['alignment_stats']['mean_dy'] for cp in all_checkpoints],
            'mean_magnitude': [cp['alignment_stats']['mean_magnitude'] for cp in all_checkpoints],
            'convergence': [cp['alignment_stats'].get('alignment_convergence', 0) for cp in all_checkpoints]
        },
        'individual_samples': {},
        'includes_initial_state': True
    }
    
    if all_checkpoints:
        num_samples = len(all_checkpoints[0]['samples'])
        for sample_id in range(num_samples):
            alignment_evolution['individual_samples'][f'sample_{sample_id}'] = {
                'dx': [cp['samples'][sample_id]['alignment']['dx'] for cp in all_checkpoints],
                'dy': [cp['samples'][sample_id]['alignment']['dy'] for cp in all_checkpoints],
                'magnitude': [cp['samples'][sample_id]['alignment']['magnitude'] for cp in all_checkpoints],
                'angle_degrees': [cp['samples'][sample_id]['alignment'].get('angle_degrees', 0) for cp in all_checkpoints]
            }
    
    # Image evolution data with multiple sizes including bilinear
    image_evolution = {
        'iterations': [cp['iteration'] for cp in all_checkpoints],
        'image_paths': {},
        'quality_metrics': {},
        'includes_initial_state': True,
        'available_sizes': ['web', 'fullscreen', 'thumbnail'],
        'available_methods': ['sr', 'lr', 'bilinear']
    }
    
    if all_checkpoints:
        for sample_id in range(num_samples):
            sample_key = f'sample_{sample_id}'
            
            # Build paths for each size and method
            image_evolution['image_paths'][sample_key] = {
                'lr_reference': {
                    'web': f"lr_reference/lr_sample_{sample_id}_web.png",
                    'fullscreen': f"lr_reference/lr_sample_{sample_id}_fullscreen.png",
                    'thumbnail': f"lr_reference/lr_sample_{sample_id}_thumb.png"
                },
                'sr_progression': {
                    'web': [f"checkpoints/iter_{cp['iteration']:04d}/sr_sample_{sample_id}_web.png" for cp in all_checkpoints],
                    'fullscreen': [f"checkpoints/iter_{cp['iteration']:04d}/sr_sample_{sample_id}_fullscreen.png" for cp in all_checkpoints],
                    'thumbnail': [f"checkpoints/iter_{cp['iteration']:04d}/sr_sample_{sample_id}_thumb.png" for cp in all_checkpoints]
                },
                'bilinear_progression': {
                    'web': [f"checkpoints/iter_{cp['iteration']:04d}/bilinear_sample_{sample_id}_web.png" for cp in all_checkpoints],
                    'fullscreen': [f"checkpoints/iter_{cp['iteration']:04d}/bilinear_sample_{sample_id}_fullscreen.png" for cp in all_checkpoints],
                    'thumbnail': [f"checkpoints/iter_{cp['iteration']:04d}/bilinear_sample_{sample_id}_thumb.png" for cp in all_checkpoints]
                }
            }
            
            # Track quality metrics over time including bilinear
            image_evolution['quality_metrics'][sample_key] = {
                'sr_mean': [cp['samples'][sample_id]['image_stats']['sr_mean'] for cp in all_checkpoints],
                'sr_std': [cp['samples'][sample_id]['image_stats']['sr_std'] for cp in all_checkpoints],
                'bilinear_mean': [cp['samples'][sample_id]['image_stats']['bilinear_mean'] for cp in all_checkpoints],
                'bilinear_std': [cp['samples'][sample_id]['image_stats']['bilinear_std'] for cp in all_checkpoints]
            }
    
    # Save all web data files
    with open(web_data_dir / "master_data.json", 'w') as f:
        json.dump(master_data, f, indent=2)
    
    with open(web_data_dir / "loss_evolution.json", 'w') as f:
        json.dump(loss_evolution, f, indent=2)
    
    with open(web_data_dir / "alignment_evolution.json", 'w') as f:
        json.dump(alignment_evolution, f, indent=2)
    
    with open(web_data_dir / "image_evolution.json", 'w') as f:
        json.dump(image_evolution, f, indent=2)
    
    # Create comprehensive API documentation
    api_docs = create_api_documentation(all_checkpoints, final_loss_history)
    with open(web_data_dir / "api_documentation.json", 'w') as f:
        json.dump(api_docs, f, indent=2)
    
    # Create README with usage examples
    readme_content = f"""# Super-Resolution Optimization Visualization Data

## 🎯 Quick Start Guide

### Loading Data in JavaScript:
```javascript
// Load all data files
const masterData = await fetch('web_data/master_data.json').then(r => r.json());
const lossData = await fetch('web_data/loss_evolution.json').then(r => r.json());
const alignmentData = await fetch('web_data/alignment_evolution.json').then(r => r.json());
const imageData = await fetch('web_data/image_evolution.json').then(r => r.json());

// Get basic experiment info
const totalIterations = masterData.experiment_info.total_iterations;
const numSamples = masterData.experiment_info.num_samples;
const checkpointIterations = alignmentData.iterations; // [0, 50, 100, 150, ...]
```

## 📊 Data Access Patterns

### 1. Image Display:
```javascript
// Show image for sample 0 at iteration 100 in web size
function showImage(sampleId, iterationNumber, method = 'sr', size = 'web') {{
    const iterationIndex = imageData.iterations.indexOf(iterationNumber);
    if (iterationIndex === -1) {{
        console.error('Iteration not available:', iterationNumber);
        return;
    }}
    
    const imagePath = imageData.image_paths[`sample_${{sampleId}}`][`${{method}}_progression`][size][iterationIndex];
    document.getElementById('main-image').src = imagePath;
}}

// Get LR reference image
function showReference(sampleId, size = 'web') {{
    const imagePath = imageData.image_paths[`sample_${{sampleId}}`].lr_reference[size];
    document.getElementById('reference-image').src = imagePath;
}}

// Show bilinear baseline
function showBilinear(sampleId, iterationNumber, size = 'web') {{
    showImage(sampleId, iterationNumber, 'bilinear', size);
}}
```

### 2. Loss Visualization:
```javascript
// Plot loss curves (handle null values for iteration 0)
function plotLosses() {{
    const iterations = lossData.iterations;
    const reconLosses = lossData.recon_losses.map(v => v === null ? undefined : v);
    const transLosses = lossData.trans_losses.map(v => v === null ? undefined : v);
    
    // Use your preferred charting library with spanGaps: true for null values
}}
```

### 3. Timeline Control:
```javascript
// Create interactive timeline
function createTimeline() {{
    const iterations = alignmentData.iterations;
    const slider = document.getElementById('iteration-slider');
    
    slider.min = 0;
    slider.max = iterations.length - 1;
    slider.value = 0;
    
    slider.addEventListener('input', (e) => {{
        const iterationIndex = parseInt(e.target.value);
        const iterationNumber = iterations[iterationIndex];
        updateAllVisualization(iterationNumber);
    }});
}}
```

## ⚠️ Important Notes:

### Null Values:
- **Iteration 0 losses**: `null` (no training occurred yet)
- Always check for `null` before using loss values
- Use `spanGaps: true` in charts to handle null values

### Image Sizes:
- **Thumbnail (200×200)**: Timeline previews, overview grids
- **Web (800×800)**: Main display, takes up most screen space  
- **Fullscreen (1200×1200)**: Modal/fullscreen viewing

### Available Methods:
- **sr**: Super-resolved output from the model
- **lr**: Original low-resolution reference
- **bilinear**: Bilinear interpolation baseline

### Array Indexing:
- Iteration numbers: `[0, 50, 100, 150, ...]` (actual iteration values)
- Array indices: `[0, 1, 2, 3, ...]` (for accessing arrays)
- Always use `iterations.indexOf(iterationNumber)` to convert

## 📈 Sample Data Summary:
- **Training iterations**: 0 → {all_checkpoints[-1]['iteration'] if all_checkpoints else 0}
- **Checkpoints saved**: {len(all_checkpoints)} (including initial state)
- **Samples per checkpoint**: {len(all_checkpoints[0]['samples']) if all_checkpoints else 0}
- **Total images**: ~{len(all_checkpoints) * len(all_checkpoints[0]['samples']) * 9 if all_checkpoints else 0} (3 methods × 3 sizes × samples × checkpoints)
- **No infinity values**: All numeric data is finite or explicitly null

Generated on: {datetime.now().isoformat()}
"""
    
    with open(web_data_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"Web visualization data saved to: {web_data_dir}")
    print(f"✅ Images saved in 3 sizes: 200×200 (thumb), 800×800 (web), 1200×1200 (fullscreen)")
    print(f"✅ All images are square format for consistent display")
    print(f"✅ No infinity values - initial iteration uses null values")
    print(f"✅ Comprehensive API documentation included")
    print(f"✅ Bilinear interpolation baseline included for comparison")
    return web_data_dir


def main():
    parser = argparse.ArgumentParser(description="Super-Resolution for Folder of Images")

    # Essential parameters
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing LR images")
    parser.add_argument("--output_folder", type=str, default="sr_outputs", help="Output folder for super-resolved images")
    parser.add_argument("--df", type=int, default=2, help="Downsampling factor (target HR = LR * df)")

    # Model parameters
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10",
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--fourier_scale", type=float, default=10)
    parser.add_argument("--use_gnll", action="store_true")

    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="0", help="CUDA device number (e.g., '0', '1') or 'cpu' for CPU")

    # Web visualization parameters
    parser.add_argument("--save_every", type=int, default=100000, help="Save checkpoint data every N iterations")
    parser.add_argument("--export_web_data", action="store_true", default=True, help="Export data for web visualization")

    args = parser.parse_args()

    # Setup device - allow "cpu" as explicit device string
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        print(f"Warning: CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")

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

    # Load images from input folder and create training data
    print(f"Loading training images from {args.input_folder}...")
    train_data = create_image_dataset(args.input_folder, args.df, device)
    
    if train_data is None:
        print("Failed to create training dataset from images")
        return

    # Setup model (use actual number of images as num_samples)
    actual_num_samples = len(train_data)
    print(f"Setting up model for {actual_num_samples} samples")
    
    input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale)
    decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)

    model = INR(
        input_projection, decoder, actual_num_samples,
        use_gnll=args.use_gnll, use_base_frame=True
    ).to(device)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)

    # Create output directory
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)

    print(f"Starting training for {args.iters} iterations...")
    print(f"Saving checkpoint data every {args.save_every} iterations for web visualization")
    
    # Initialize tracking variables
    iteration = 0
    loss_history = []
    all_checkpoints = []
    
    # Save initial checkpoint (iteration 0 - starting point)  
    print("Saving initial checkpoint (iteration 0)...")
    initial_loss_data = {
        'iteration': 0,
        'recon_loss': None,  # Use None instead of float('inf')
        'trans_loss': None,  # Use None instead of float('inf') 
        'total_loss': None   # Use None instead of float('inf')
    }
    initial_checkpoint = save_checkpoint_data(model, train_data, device, 0, 
                                            output_dir, args, initial_loss_data)
    all_checkpoints.append(initial_checkpoint)
    print(f"✅ Initial state captured - shows starting point before any optimization")
    
    # Training loop with checkpoint saves
    progress_bar = tqdm(total=args.iters, desc="Training")
    
    while iteration < args.iters:
        # Cycle through all images in the dataset
        for idx in range(len(train_data)):
            if iteration >= args.iters:
                break
                
            # Get training sample
            train_sample = train_data[idx]
            
            # Train one iteration
            train_losses = train_one_iteration(model, optimizer, train_sample, device)
            scheduler.step()
            iteration += 1
            
            # Store loss history
            current_loss_data = {
                'iteration': iteration,
                'recon_loss': train_losses['recon_loss'],
                'trans_loss': train_losses['trans_loss'],
                'total_loss': train_losses['total_loss']
            }
            loss_history.append(current_loss_data)

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon': f"{train_losses['recon_loss']:.4f}",
                'trans': f"{train_losses['trans_loss']:.4f}"
            })
            
            # Save checkpoint data for web visualization
            if iteration % args.save_every == 0:
                print(f"\nSaving checkpoint data at iteration {iteration}...")
                checkpoint_data = save_checkpoint_data(model, train_data, device, iteration, 
                                                     output_dir, args, current_loss_data)
                all_checkpoints.append(checkpoint_data)
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}")

    progress_bar.close()
    
    # Save final checkpoint if not already saved
    if iteration % args.save_every != 0:
        print(f"Saving final checkpoint data at iteration {iteration}...")
        final_checkpoint = save_checkpoint_data(model, train_data, device, iteration, 
                                              output_dir, args, loss_history[-1])
        all_checkpoints.append(final_checkpoint)
    
    # Export comprehensive web visualization data
    if args.export_web_data:
        print("Exporting data for web visualization...")
        web_data_dir = save_web_visualization_data(output_dir, all_checkpoints, loss_history)
        
        print(f"\n{'='*60}")
        print("WEB VISUALIZATION DATA READY!")
        print(f"{'='*60}")
        print(f"Data location: {web_data_dir}")
        print(f"Total checkpoints saved: {len(all_checkpoints)}")
        print(f"Iterations covered: 0 -> {iteration} (including starting point)")
        print(f"Total iterations tracked: {len(loss_history)}")
        print(f"Samples per checkpoint: {len(train_data)}")
        print(f"Checkpoint timeline: {[cp['iteration'] for cp in all_checkpoints]}")
        print("\nFiles for web development:")
        print(f"  📄 {web_data_dir / 'master_data.json'} - Main experiment metadata")
        print(f"  📈 {web_data_dir / 'loss_evolution.json'} - Loss tracking data") 
        print(f"  🎯 {web_data_dir / 'alignment_evolution.json'} - Alignment tracking data")
        print(f"  🖼️ {web_data_dir / 'image_evolution.json'} - Image paths and metrics")
        print(f"  📋 {web_data_dir / 'api_documentation.json'} - API documentation")
        print(f"  📁 {web_data_dir.parent / 'checkpoints'} - Checkpoint images and data")
        print(f"  📋 {web_data_dir / 'README.md'} - Usage documentation")
        print(f"\n🚀 VISUALIZATION FEATURES:")
        print(f"  • Shows progression from random initialization (iter 0)")
        print(f"  • Tracks alignment evolution throughout training")
        print(f"  • Multiple image sizes: 200×200, 800×800, 1200×1200")
        print(f"  • Square format images for consistent display")
        print(f"  • Complete loss curves from start to finish")
        print(f"  • No infinity values - all data web-ready")
    
    print(f"\nOptimization complete! All data saved to {output_dir}")


if __name__ == "__main__":
    main() 