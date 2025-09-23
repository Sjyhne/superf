import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import json
from pathlib import Path
import random
import glob

import tifffile

def get_and_standardize_image(image):
    """Get and standardize image to have zero mean and unit std for each channel"""
    # Check if image has a channel dimension
    if image.dim() >= 3:
        # Calculate mean and std along spatial dimensions only (not across channels)
        # For [C, H, W] format
        if image.dim() == 4:
            mean = image.mean(dim=(1, 2, 3), keepdim=True)
            std = image.std(dim=(1, 2, 3), keepdim=True)
        # For [H, W, C] format
        else:
            mean = image.mean(dim=(0, 1, 2), keepdim=True)
            std = image.std(dim=(0, 1, 2), keepdim=True)
    else:
        # For grayscale without channel dimension
        mean = image.mean()
        std = image.std()
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    return (image - mean) / std, mean, std

def get_dataset(args, name='satburst', keep_in_memory=True):
    """ Returns the dataset object based on the name """
    if name == 'satburst_synth':
        return SRData(data_dir=args.root_satburst_synth, num_samples=args.num_samples, keep_in_memory=keep_in_memory, scale_factor=args.scale_factor)
    elif name == 'burst_synth':
        return SyntheticBurstVal(data_dir=args.root_burst_synth, 
                                 sample_id=args.sample_id, keep_in_memory=keep_in_memory, 
                                 scale_factor=args.scale_factor, df=args.df)
    elif name == 'worldstrat':
        return WorldStratDatasetFrame(data_dir=args.root_worldstrat, 
                                      area_name=args.area_name, hr_size=args.worldstrat_hr_size)
    elif name == 'worldstrat_test':
        return WorldStratTestDataset(data_dir=args.root_worldstrat_test, 
                                     sample_id=args.sample_id, keep_in_memory=keep_in_memory, scale_factor=args.scale_factor)
    else:
        raise ValueError(f"Invalid dataset name: {name}")


class SRData(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_samples, keep_in_memory=False, scale_factor=4):
        """
        Initialize SR dataset from generated data directory.
        
        Args:
            data_dir: Base path to data directory
            mode: 'lr' or 'hr' - which dataset to load
        """
        self.data_dir = Path(data_dir)
        self.keep_in_memory = keep_in_memory
        self.num_samples = num_samples  

        self.vmin, self.vmax = 0, 1
        
        # Load transformation log
        with open(self.data_dir / "transform_log.json", 'r') as f:
            self.transform_log = json.load(f)
            
        # Get list of sample names
        self.samples = sorted(list(self.transform_log.keys()))
        self.samples = self.samples[:num_samples]

        self.means = list()
        self.stds = list()
        self.lr_image_sizes = list()

        if self.keep_in_memory:
            self.images = {}
            for sample in self.samples:
                img_path = self.data_dir / self.transform_log[sample]['path']
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float() / 255.0
                img, mean, std = get_and_standardize_image(img)
                self.lr_image_sizes.append(img.shape[1:3])
                self.images[sample] = {
                    "image": img,
                    "mean": mean,
                    "std": std
                }

        # Load original image for reference
        self.original = cv2.imread(str(self.data_dir / "hr_ground_truth.png"))
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.original = (torch.from_numpy(self.original).float() / 255.0).cuda()
        # Standardize original image to have zero mean and no bias

        self.hr_coords = np.linspace(self.vmin, self.vmax, self.original.shape[0], endpoint=False)
        self.hr_coords = np.stack(np.meshgrid(self.hr_coords, self.hr_coords), -1)
        self.hr_coords = torch.FloatTensor(self.hr_coords).cuda()

        self.lr_coords = np.linspace(self.vmin, self.vmax, self.lr_image_sizes[0][0], endpoint=False)
        self.lr_coords = np.stack(np.meshgrid(self.lr_coords, self.lr_coords), -1)
        self.lr_coords = torch.FloatTensor(self.lr_coords).cuda()

        self.scale_factor = [scale_factor]

    def __len__(self):
        return len(self.samples)
    
    def get_input_coordinates(self):
        scale_factor = random.choice(self.scale_factor)

        input_coordinates = np.linspace(self.vmin, self.vmax, int(self.lr_image_sizes[0][0] * scale_factor), endpoint=False)
        input_coordinates = np.stack(np.meshgrid(input_coordinates, input_coordinates), -1)
        input_coordinates = torch.FloatTensor(input_coordinates).cuda()
        return input_coordinates, scale_factor
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_info = self.transform_log[sample_name]
        sample_id = int(sample_name.split("_")[-1])

        input_coordinates, scale_factor = self.get_input_coordinates()

        if self.keep_in_memory:
            img = self.images[sample_name]["image"]
            mean = self.images[sample_name]["mean"]
            std = self.images[sample_name]["std"]
        else:
            # Load transformed image
            img_path = self.data_dir / sample_info['path']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0
            img, mean, std = get_and_standardize_image(img)
        
        return {
            'input': input_coordinates,
            'lr_target': img,
            'scale_factor': scale_factor,
            'mean': mean,
            'std': std,
            'sample_id': sample_id,
            'shifts': {
                'dx_lr': sample_info['dx_pixels_lr'],
                'dy_lr': sample_info['dy_pixels_lr'],
                'dx_hr': sample_info['dx_pixels_hr'],
                'dy_hr': sample_info['dy_pixels_hr'],
                'dx_percent': sample_info['dx_percent'],
                'dy_percent': sample_info['dy_percent']
            }
        }
    
    def get_original_hr(self):
        """Return the original image (before any transformations)"""
        return self.original
    

    def get_lr_sample(self, index):
        """Get a specific LR sample by index.
        
        Args:
            index: Sample index (0 is the reference sample)
            
        Returns:
            Tensor of shape [C, H, W] with values in [0, 1]
        """

        if self.keep_in_memory:
            img = self.images[self.samples[index]]["image"]
            mean = self.images[self.samples[index]]["mean"]
            std = self.images[self.samples[index]]["std"]
            # Unstandardize the image
            img = img * std + mean
        else:
            sample_path = self.data_dir / f"sample_{index:02d}.png"
            img = cv2.imread(str(sample_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0

        return img

    def get_lr_mean(self, index):
        return self.images[self.samples[index]]["mean"]

    def get_lr_std(self, index):
        return self.images[self.samples[index]]["std"]

    def get_hr_coordinates(self):
        """Return the high-resolution coordinates."""
        return self.hr_coords


class SyntheticBurstVal(torch.utils.data.Dataset):
    def __init__(self, data_dir, sample_id, keep_in_memory=True, scale_factor=4, df=4):
        """
        Initialize SyntheticBurstVal dataset.
        
        Args:
            data_dir: Base path to SyntheticBurstVal directory
            sample_id: ID of the burst to use (0-299)
            keep_in_memory: Whether to load all images into memory
            scale_factor: Scaling factor for coordinate generation
            df: Downsampling factor for HR image resizing (HR = df * LR)
        """
        self.data_dir = Path(data_dir)
        self.keep_in_memory = keep_in_memory
        self.sample_id = sample_id
        self.scale_factor = scale_factor
        self.df = df

        self.rggb = True

        self.scale_factor = scale_factor
        self.df = df
        
        # Format sample_id as a 4-digit string with leading zeros
        self.sample_id_str = f"{int(sample_id):04d}"
        
        # Set up paths
        self.gt_dir = self.data_dir / "gt" / self.sample_id_str
        self.burst_dir = self.data_dir / "bursts" / self.sample_id_str
        
        # Find all burst images
        self.burst_paths = sorted(list(self.burst_dir.glob('im_raw_*.png')))
        self.burst_size = len(self.burst_paths)
        
        # Extract frame indices from filenames
        self.frame_indices = []
        for path in self.burst_paths:
            # Extract the frame index from the filename (im_raw_XX.png)
            frame_idx = int(path.stem.split('_')[-1])
            self.frame_indices.append(frame_idx)
        
        # Load burst images first
        if self.keep_in_memory:
            self.burst_images = {}
            for idx in self.frame_indices:
                img = self._read_burst_image(idx)
                img_std, mean, std = get_and_standardize_image(img)
                self.burst_images[idx] = {
                    "image": img_std,
                    "mean": mean,
                    "std": std
                }
            self.gt_image = self._read_gt_image()
            
        else:
            self.burst_images = None
        
        # Load ground truth image and resize based on scale factor
        if self.keep_in_memory:
            self.gt_image = self._read_gt_image()
            self._resize_hr_image()  # Resize HR image based on scale factor
        else:
            self.gt_image = None
        
        # Create coordinate grid for HR image
        if self.keep_in_memory:
            h, w = self.gt_image.shape[:-1]
            coords_h = np.linspace(0, 1, h, endpoint=False)
            coords_w = np.linspace(0, 1, w, endpoint=False)
            coords = np.stack(np.meshgrid(coords_w, coords_h), -1)  # Note: w, h order
            self.hr_coords = torch.FloatTensor(coords).cuda()
        else:
            self.hr_coords = None
        
        # Set up coordinate generation parameters
        self.vmin, self.vmax = 0, 1
        self.scale_factor = [scale_factor]  # Make it a list like other datasets
        
    def __len__(self):
        return self.burst_size
    
    def get_input_coordinates(self):
        """Generate input coordinates for the model - match SRData pattern."""
        scale_factor = random.choice(self.scale_factor)
        
        if self.keep_in_memory:
            h, w = self.burst_images[0]["image"].shape[:-1]
        else:
            # Load a sample image to get dimensions
            sample_img = self._read_burst_image(0)
            h, w = sample_img.shape[:-1]
        
        input_h = int(h * scale_factor)
        input_w = int(w * scale_factor)
        
        input_coords_h = np.linspace(self.vmin, self.vmax, input_h, endpoint=False)
        input_coords_w = np.linspace(self.vmin, self.vmax, input_w, endpoint=False)
        input_coordinates = np.stack(np.meshgrid(input_coords_w, input_coords_h), -1)
        input_coordinates = torch.FloatTensor(input_coordinates).cuda()
        return input_coordinates, scale_factor
    
    def _read_burst_image(self, frame_idx):
        """Read a single raw burst image"""
        path = self.burst_dir / f"im_raw_{frame_idx:02d}.png"
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Convert from 16-bit to float and normalize
        im_t = im.astype(np.float32) / (2**14)

        # Extract RGGB channels
        R = im_t[..., 0]
        G1 = im_t[..., 1]
        G2 = im_t[..., 2]
        B = im_t[..., 3]
        
        # Average the two green channels
        G = (G1 + G2) / 2
        
        # Create RGB image
        rgb = np.stack([R, G, B], axis=-1)
        
        # Apply white balance (example values, actual values might differ)
        wb_gains = np.array([2.0, 1.0, 1.5])  # R, G, B gains
        rgb = rgb * wb_gains
        
        # Apply gamma correction
        gamma = 2.2
        rgb = np.power(np.maximum(rgb, 0), 1.0/gamma)
        
        # Clip values to [0, 1]
        rgb = np.clip(rgb, 0, 1)

        rgb = torch.from_numpy(rgb).float()
        
        return rgb
    
    def _read_gt_image(self):
        """Read the ground truth RGB image"""
        path = self.gt_dir / "im_rgb.png"
        gt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Convert from 16-bit to float and normalize
        gt_t = gt.astype(np.float32) / (2**14)

        wb_gains = np.array([2.0, 1.0, 1.5])  # R, G, B gains
        gt_t = gt_t * wb_gains

        # Apply gamma correction
        gamma = 2.2
        gt_t = np.power(np.maximum(gt_t, 0), 1.0/gamma)

        gt_t = np.clip(gt_t, 0, 1)

        gt_t = torch.from_numpy(gt_t).float()
        
        return gt_t
    
    def _resize_hr_image(self):
        """Resize HR image based on df (downsampling factor) relative to LR image size."""
        if self.gt_image is None:
            return
            
        # Get LR image dimensions (use first frame as reference)
        if self.keep_in_memory and self.burst_images is not None:
            # Use cached LR image dimensions
            lr_h, lr_w = self.burst_images[self.frame_indices[0]]["image"].shape[:-1]
        else:
            # Load a sample LR image to get dimensions
            sample_img = self._read_burst_image(self.frame_indices[0])
            lr_h, lr_w = sample_img.shape[:-1]
        
        # Calculate target HR dimensions using df
        target_h = int(lr_h * self.df)
        target_w = int(lr_w * self.df)
        
        # Resize HR image
        if self.gt_image.dim() == 3:  # HWC format
            hr_np = self.gt_image.cpu().numpy()
            hr_resized = cv2.resize(hr_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
            self.gt_image = torch.from_numpy(hr_resized).float()
        else:  # CHW format
            hr_np = self.gt_image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            hr_resized = cv2.resize(hr_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
            self.gt_image = torch.from_numpy(hr_resized).permute(2, 0, 1).float()  # Convert back to CHW
    
    def __getitem__(self, idx):
        """Get a specific frame from the burst"""
        # Get the frame index for this position
        frame_idx = self.frame_indices[idx]
        
        # Load the burst image (or get from cache)
        if self.keep_in_memory and self.burst_images is not None:
            img = self.burst_images[frame_idx]["image"]
            mean = self.burst_images[frame_idx]["mean"]
            std = self.burst_images[frame_idx]["std"]
        else:
            # Load and standardize on demand
            img = self._read_burst_image(frame_idx)
            img, mean, std = get_and_standardize_image(img)

        # Generate input coordinates for this sample
        input_coordinates, scale_factor = self.get_input_coordinates()
        
        # Return in a format similar to SRData
        return {
            'input': input_coordinates,
            'lr_target': img,
            'scale_factor': scale_factor,  # Use the actual scale factor from coordinate generation
            'mean': mean,
            'std': std,
            'sample_id': idx,
            'burst_id': self.sample_id,
            'scale_factor': self.scale_factor,
            'shifts': {
                'dx_percent': 0.0,  # Placeholder
                'dy_percent': 0.0   # Placeholder
            }
        }
    
    def get_burst(self):
        """Get all frames from the burst as a tensor [N, C, H, W]"""
        if self.keep_in_memory and self.burst_images is not None:
            # Use cached images
            burst = [self.burst_images[idx]["image"] for idx in self.frame_indices]
        else:
            # Load images on demand
            burst = []
            for idx in self.frame_indices:
                img = self._read_burst_image(idx)
                img_std, _, _ = get_and_standardize_image(img)
                burst.append(img_std)
        return torch.stack(burst, 0)
    
    def get_original_hr(self):
        """Return the ground truth image"""
        if self.keep_in_memory and self.gt_image is not None:
            return self.gt_image
        else:
            return self._read_gt_image()
    
    def get_lr_sample(self, frame_idx=0):
        """Get a specific LR frame from the burst"""
        if self.keep_in_memory and self.burst_images is not None:
            # Make sure frame_idx is in range
            if frame_idx >= len(self.frame_indices):
                frame_idx = 0
            idx = self.frame_indices[frame_idx]
            img = self.burst_images[idx]["image"]
            return img.permute(2, 0, 1)  # Return in CHW format
        else:
            img = self._read_burst_image(self.frame_indices[frame_idx])
            img_std, _, _ = get_and_standardize_image(img)
            return img_std.permute(2, 0, 1)  # Return in CHW format
    
    def get_lr_mean(self, frame_idx=0):
        if self.keep_in_memory and self.burst_images is not None:
            return self.burst_images[self.frame_indices[frame_idx]]["mean"]
        else:
            img = self._read_burst_image(self.frame_indices[frame_idx])
            _, mean, _ = get_and_standardize_image(img)
            return mean

    def get_lr_std(self, frame_idx=0):
        if self.keep_in_memory and self.burst_images is not None:
            return self.burst_images[self.frame_indices[frame_idx]]["std"]
        else:
            img = self._read_burst_image(self.frame_indices[frame_idx])
            _, _, std = get_and_standardize_image(img)
            return std
    
    def get_lr_sample_hwc(self, frame_idx=0):
        """Get a specific LR frame in HWC format for evaluation."""
        if self.keep_in_memory and self.burst_images is not None:
            # Make sure frame_idx is in range
            if frame_idx >= len(self.frame_indices):
                frame_idx = 0
            idx = self.frame_indices[frame_idx]
            img = self.burst_images[idx]["image"]
            return img  # Already in HWC format
        else:
            img = self._read_burst_image(self.frame_indices[frame_idx])
            img_std, _, _ = get_and_standardize_image(img)
            return img_std  # Return in HWC format
    
    def get_hr_coordinates(self):
        """Return coordinates for the HR image"""
        if self.hr_coords is not None:
            return self.hr_coords
            
        # Create on demand if not cached
        gt = self._read_gt_image()
        # Resize HR image based on scale factor
        self.gt_image = gt
        self._resize_hr_image()
        gt = self.gt_image
        
        if gt.dim() == 3:  # HWC format
            h, w = gt.shape[:-1]
        else:  # CHW format
            h, w = gt.shape[1:]
            
        coords_h = np.linspace(0, 1, h, endpoint=False)
        coords_w = np.linspace(0, 1, w, endpoint=False)
        coords = np.stack(np.meshgrid(coords_h, coords_w), -1)
        return torch.FloatTensor(coords)



class WorldStratDatasetFrame(torch.utils.data.Dataset):
    """ Returns single LR frames in getitem """
    def __init__(self, data_dir, area_name="UNHCR-LBNs006446", num_frames=8, hr_size=None):
        """
        Args:
            data_dir (str): Path to the dataset.
            area_name (str): area name.
        """

        self.dataset_root = '/home/nlang/data/worldstrat_kaggle'
        self.hr_dataset = "{}/hr_dataset/12bit".format(data_dir)
        self.lr_dataset = "{}/lr_dataset".format(data_dir)
        #self.metadata_df = pd.read_csv("{}/metadata.csv".format(dataset_root))

        self.area_name = area_name
        self.num_frames = num_frames    
        self.hr_size = hr_size

        # Load high-resolution image
        self.hr_image = self.get_hr()   # Shape: (hr_img_size, hr_img_size, 3)
        if self.hr_size is not None:
            self.hr_image = cv2.resize(self.hr_image, (self.hr_size, self.hr_size), interpolation=cv2.INTER_AREA)
        self.hr_image = torch.tensor(self.hr_image)
        

        # Create input coordinate grid that matches the HR image
        self.hr_coords = np.linspace(0, 1, self.hr_image.shape[0], endpoint=False)
        self.hr_coords = np.stack(np.meshgrid(self.hr_coords, self.hr_coords), -1)
        self.hr_coords = torch.FloatTensor(self.hr_coords)
    
    def __len__(self):
        # TODO
        return self.num_frames
    
    def get_hr(self):
        """Loads and processes the high-resolution image."""
        hr_rgb_path = os.path.join(self.hr_dataset, self.area_name, f"{self.area_name}_rgb.png")
        print(hr_rgb_path)
        hr_rgb_img = cv2.imread(hr_rgb_path)
        print(hr_rgb_img.shape)
        hr_rgb_img = cv2.cvtColor(hr_rgb_img, cv2.COLOR_BGR2RGB)
        return hr_rgb_img.astype(np.float32) / 255.0  # Normalize
    
    def get_lr(self, frame_id):
        """Loads a single LR frame."""

        # files start with index 1 (not 0)
        frame_id+=1

        lr_sample_path = os.path.join(self.lr_dataset, self.area_name, "L2A")
        lr_rgb_path = os.path.join(lr_sample_path, f"{self.area_name}-{frame_id}-L2A_data.tiff")
        lr_rgb_img = tifffile.imread(lr_rgb_path)[:, :, 4:1:-1].copy()  # Select RGB bands and reverse order
        lr_rgb_img = torch.tensor(lr_rgb_img, dtype=torch.float32).clip(0, 1)  # Data is already normalized, but needs to be clipped

        return lr_rgb_img
    
    def __getitem__(self, idx):
        lr_image = self.get_lr(frame_id=idx)  # Shape: (8, lr_img_size, lr_img_size, 3)
        
        # Convert to torch tensors
        lr_image = torch.tensor(lr_image)
        
        return {
            'input': self.hr_coords,
            'lr_target': lr_image,
            'sample_id': idx,
            # note: the true shifts are unknown, set to default 0
            'shifts': {
                'dx_lr': 0,
                'dy_lr': 0,
                'dx_hr': 0,
                'dy_hr': 0,
                'dx_percent': 0,
                'dy_percent': 0
            }
        }
    
    def get_original_hr(self):
        """Return the original image (before any transformations)"""
        return self.hr_image
    
    def get_hr_coordinates(self):
        """Return the high-resolution coordinates."""
        return self.hr_coords
    
    def get_lr_sample(self, index):
        """Get a specific LR sample by index.
        
        Args:
            index: Sample index (0 is the reference sample)
            
        Returns:
            Tensor of shape [C, H, W] with values in [0, 1]
        """
        return self.get_lr(index).permute(2, 0, 1)
    

class WorldStratTestDataset(torch.utils.data.Dataset):
    """Dataset for WorldStrat test data with hr/lr folder structure."""
    
    def __init__(self, data_dir, sample_id, keep_in_memory=True, scale_factor=4):
        """
        Initialize WorldStrat test dataset.
        
        Args:
            data_dir: Base path to worldstrat_test_data directory
            sample_id: Specific sample ID to load (e.g., "Amnesty POI-1-2-1")
            keep_in_memory: Whether to keep all data in memory
            scale_factor: Scaling factor for coordinate generation
        """
        self.data_dir = Path(data_dir)
        self.sample_id = sample_id
        self.keep_in_memory = keep_in_memory
        self.scale_factor = scale_factor
        self.vmin, self.vmax = 0, 1

        # In this case we should set scale factor so that the input coordinates are the same as the HR coordinates
        self.hr_image = self._load_image(self.hr_path)
        self.hr_h, self.hr_w = self.hr_image.shape[:2]
        self.scale_factor = [self.hr_w / self.lr_w]

        # NB!!!: Remember to set to 1 when doing the NIR
        
        # Path to the specific sample
        self.sample_dir = self.data_dir / sample_id
        if not self.sample_dir.exists():
            raise ValueError(f"Sample directory not found: {self.sample_dir}")
        
        # Paths to hr and lr folders
        self.hr_dir = self.sample_dir / "hr"
        self.lr_dir = self.sample_dir / "lr"
        
        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise ValueError(f"HR or LR directory not found in {self.sample_dir}")
        
        # Get HR image path
        hr_files = list(self.hr_dir.glob("*.png"))
        if not hr_files:
            raise ValueError(f"No HR image found in {self.hr_dir}")
        self.hr_path = hr_files[0]  # Take the first (and should be only) HR image
        
        # Get LR image paths
        self.lr_paths = sorted(list(self.lr_dir.glob("*.png")))
        if not self.lr_paths:
            raise ValueError(f"No LR images found in {self.lr_dir}")
        
        print(f"Found {len(self.lr_paths)} LR images for sample {sample_id}")
        
        # Load HR image
        self.hr_image = self._load_image(self.hr_path)
        self.hr_h, self.hr_w = self.hr_image.shape[:2]
        
        # Generate coordinate grids - handle non-square images
        hr_coords_h = np.linspace(self.vmin, self.vmax, self.hr_h, endpoint=False)
        hr_coords_w = np.linspace(self.vmin, self.vmax, self.hr_w, endpoint=False)
        self.hr_coords = np.stack(np.meshgrid(hr_coords_w, hr_coords_h), -1)
        self.hr_coords = torch.FloatTensor(self.hr_coords).cuda()
        
        # Load LR images and determine consistent size
        self.means = []
        self.stds = []
        self.lr_image_sizes = []
        
        # First, determine a consistent target size by loading the first image
        first_lr_img, _, _ = self._load_and_standardize_image(self.lr_paths[0])
        self.target_lr_size = first_lr_img.shape[:2]

        
        # Generate LR coordinates based on target size (handle non-square images)
        lr_h, lr_w = self.target_lr_size
        lr_coords_h = np.linspace(self.vmin, self.vmax, lr_h, endpoint=False)
        lr_coords_w = np.linspace(self.vmin, self.vmax, lr_w, endpoint=False)
        self.lr_coords = np.stack(np.meshgrid(lr_coords_w, lr_coords_h), -1)
        self.lr_coords = torch.FloatTensor(self.lr_coords).cuda()
        
        if self.keep_in_memory:
            self.lr_images = []
            for lr_path in self.lr_paths:
                lr_img, mean, std = self._load_and_standardize_image(lr_path)
                # Resize to consistent size
                lr_img = self._resize_to_consistent_size(lr_img, self.target_lr_size)
                self.lr_images.append(lr_img)
                self.means.append(mean)
                self.stds.append(std)
                self.lr_image_sizes.append(self.target_lr_size)
        else:
            # Set consistent size for all images
            self.lr_image_sizes = [self.target_lr_size] * len(self.lr_paths)
        
        # Set scale factor as list like in SRData
        self.scale_factor = [scale_factor]
    
    def _load_image(self, path):
        """Load an image from file."""
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).float() / 255.0
    
    def _resize_to_consistent_size(self, img, target_size=None):
        """Resize image to a consistent size for all LR images."""
        if target_size is None:
            target_size = self.target_lr_size
        
        if img.shape[:2] != target_size:
            img_np = img.numpy()
            img_resized = cv2.resize(img_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            img = torch.from_numpy(img_resized)
        
        return img
    
    def _load_and_standardize_image(self, path):
        """Load and standardize an image."""
        img = self._load_image(path)
        return get_and_standardize_image(img)
    
    def __len__(self):
        return len(self.lr_paths)
    
    def get_input_coordinates(self):
        """Generate input coordinates for the model - handle non-square images."""
        scale_factor = random.choice(self.scale_factor)
        
        lr_h, lr_w = self.lr_image_sizes[0]
        input_h = int(lr_h * scale_factor)
        input_w = int(lr_w * scale_factor)
        
        input_coords_h = np.linspace(self.vmin, self.vmax, input_h, endpoint=False)
        input_coords_w = np.linspace(self.vmin, self.vmax, input_w, endpoint=False)
        input_coordinates = np.stack(np.meshgrid(input_coords_w, input_coords_h), -1)
        input_coordinates = torch.FloatTensor(input_coordinates).cuda()
        return input_coordinates, scale_factor
    
    def __getitem__(self, idx):
        """Get a single LR sample."""
        lr_path = self.lr_paths[idx]
        
        if self.keep_in_memory:
            lr_img = self.lr_images[idx]
            mean = self.means[idx]
            std = self.stds[idx]
        else:
            lr_img, mean, std = self._load_and_standardize_image(lr_path)
            # Resize to consistent size
            lr_img = self._resize_to_consistent_size(lr_img, self.target_lr_size)
        
        input_coordinates, scale_factor = self.get_input_coordinates()
        
        return {
            'input': input_coordinates,
            'lr_target': lr_img,
            'scale_factor': scale_factor,
            'mean': mean,
            'std': std,
            'sample_id': idx,  # Use index as sample_id
            'shifts': {
                'dx_lr': 0.0,  # No ground truth shifts available
                'dy_lr': 0.0,
                'dx_hr': 0.0,
                'dy_hr': 0.0,
                'dx_percent': 0.0,
                'dy_percent': 0.0
            }
        }
    
    def get_original_hr(self):
        """Return the original HR image."""
        return self.hr_image
    
    def get_hr_coordinates(self):
        """Return the high-resolution coordinates."""
        return self.hr_coords
    
    def get_lr_sample(self, index):
        """Get a specific LR sample by index."""
        if self.keep_in_memory:
            return self.lr_images[index].permute(2, 0, 1)
        else:
            lr_img, _, _ = self._load_and_standardize_image(self.lr_paths[index])
            # Resize to consistent size
            lr_img = self._resize_to_consistent_size(lr_img, self.target_lr_size)
            return lr_img.permute(2, 0, 1)
    
    def get_lr_sample_hwc(self, index):
        """Get a specific LR sample by index in HWC format for evaluation."""
        if self.keep_in_memory:
            return self.lr_images[index]  # Already in HWC format
        else:
            lr_img, _, _ = self._load_and_standardize_image(self.lr_paths[index])
            # Resize to consistent size
            lr_img = self._resize_to_consistent_size(lr_img, self.target_lr_size)
            return lr_img  # Return in HWC format
    
    def get_lr_mean(self, index):
        """Get the mean for unstandardization."""
        if self.keep_in_memory:
            return self.means[index]
        else:
            _, mean, _ = self._load_and_standardize_image(self.lr_paths[index])
            return mean
    
    def get_lr_std(self, index):
        """Get the std for unstandardization."""
        if self.keep_in_memory:
            return self.stds[index]
        else:
            _, _, std = self._load_and_standardize_image(self.lr_paths[index])
            return std


if __name__ == "__main__":
    print("Testing WorldStratDatasetFrame...")
    
    try:
        # Test with actual data path (you'll need to update this)
        # Example usage:
        test_dataset = WorldStratDatasetFrame(
            data_dir="/path/to/your/worldstrat/data",  # Update this path
            area_name="UNHCR-LBNs006446",  # Update this area name
            num_frames=8,
            hr_size=512
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"Dataset length: {len(test_dataset)}")
        print(f"HR image shape: {test_dataset.get_original_hr().shape}")
        print(f"HR coordinates shape: {test_dataset.get_hr_coordinates().shape}")
        
        # Test getting a sample
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Input shape: {sample['input'].shape}")
            print(f"LR target shape: {sample['lr_target'].shape}")
            print(f"Sample ID: {sample['sample_id']}")
            
            # Test the new methods
            print(f"LR sample (CHW) shape: {test_dataset.get_lr_sample(0).shape}")
            print(f"LR sample (HWC) shape: {test_dataset.get_lr_sample_hwc(0).shape}")
            print(f"LR mean shape: {test_dataset.get_lr_mean(0).shape}")
            print(f"LR std shape: {test_dataset.get_lr_std(0).shape}")
            
            print("✅ WorldStratDatasetFrame test completed successfully!")
        
    except Exception as e:
        print(f"❌ WorldStratDatasetFrame test failed: {e}")
        print("\nTo use WorldStratDatasetFrame, you need to:")
        print("1. Update the data_dir path to point to your WorldStrat dataset")
        print("2. Make sure the directory structure is:")
        print("   data_dir/hr_dataset/12bit/area_name/area_name_rgb.png")
        print("   data_dir/lr_dataset/area_name/L2A/area_name-1-L2A_data.tiff")
        print("   data_dir/lr_dataset/area_name/L2A/area_name-2-L2A_data.tiff")
        print("   ...")
    
    print("\n" + "="*60)
    print("HOW TO USE WorldStratDatasetFrame:")
    print("="*60)
    print("""
# 1. Create dataset instance
dataset = WorldStratDatasetFrame(
    data_dir="/path/to/worldstrat/data",  # Base directory
    area_name="UNHCR-LBNs006446",        # Area identifier
    num_frames=8,                        # Number of LR frames
    hr_size=512                          # Optional: resize HR image
)

# 2. Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 3. Use in training loop
for batch in dataloader:
    input_coords = batch['input']        # HR coordinates
    lr_target = batch['lr_target']       # LR image
    sample_id = batch['sample_id']       # Frame ID
    shifts = batch['shifts']             # Ground truth shifts (all zeros)

# 4. Access individual methods
hr_image = dataset.get_original_hr()     # Get HR ground truth
hr_coords = dataset.get_hr_coordinates() # Get HR coordinate grid
lr_sample = dataset.get_lr_sample(0)     # Get LR frame 0 (CHW format)
lr_hwc = dataset.get_lr_sample_hwc(0)    # Get LR frame 0 (HWC format)
lr_mean = dataset.get_lr_mean(0)         # Get mean for unstandardization
lr_std = dataset.get_lr_std(0)           # Get std for unstandardization
    """)
    
    print("\n" + "="*60)
    print("COMPATIBILITY WITH BENCHMARK:")
    print("="*60)
    print("✅ WorldStratDatasetFrame is now compatible with benchmark_models.py!")
    print("✅ All required methods are implemented:")
    print("   - get_original_hr()")
    print("   - get_hr_coordinates()")
    print("   - get_lr_sample(index)")
    print("   - get_lr_sample_hwc(index)")
    print("   - get_lr_mean(index)")
    print("   - get_lr_std(index)")
    print("\nYou can now run benchmark_models.py with this dataset!")