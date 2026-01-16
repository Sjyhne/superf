#!/usr/bin/env python3
"""
Synthetic L1B Multi-Exposure Dataset Generation for RGB Images
This script generates synthetic burst sequences from RGB images with:
- Multiple exposures
- Random shifts/warps
- Realistic noise
- Exposure variations
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy import ndimage

# Configuration
sigma_blur = 0.3  # Gaussian blur sigma
burst_size = 10   # Number of frames per burst
bit_depth = 4095  # 12-bit image normalization factor

class RGBDatasetGenerator:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and normalize data
        self.original_data = np.load(self.input_path) / bit_depth
        
        # Verify RGB format
        if self.original_data.ndim != 4 or self.original_data.shape[-1] != 3:
            raise ValueError("Input data must be RGB with shape (n_images, height, width, 3)")
        
        self.n_images = self.original_data.shape[0]
        self.height = self.original_data.shape[1]
        self.width = self.original_data.shape[2]
        
        # Initialize containers
        self.blurred_data = np.empty_like(self.original_data)
        self.warps = np.empty((self.n_images, burst_size-1, 2))

    def apply_gaussian_blur(self):
        """Apply Gaussian blur to each color channel separately."""
        print("Applying Gaussian blur...")
        for i in tqdm(range(self.n_images)):
            for c in range(3):  # Process each color channel
                self.blurred_data[i, ..., c] = gaussian_filter(
                    self.original_data[i, ..., c], 
                    sigma=sigma_blur
                )
        return self.blurred_data

    def generate_warps(self, R_max=2):
        """Generate random warps/shifts for each image."""
        def pick_warps(n_warps):
            warps = R_max * (np.random.random((n_warps, 2)) - 0.5)
            while True:
                norms = np.abs(warps[:, 0]) + np.abs(warps[:, 1])
                invalid = warps[norms > R_max]
                n_invalid = invalid.shape[0]
                if n_invalid == 0:
                    break
                invalid = R_max * (np.random.random((n_invalid, 2)) - 0.5)
            return warps

        print("Generating random warps...")
        for i in tqdm(range(self.n_images)):
            self.warps[i] = pick_warps(burst_size-1)
        return self.warps

    def apply_warps(self):
        """Apply warps to create burst sequence, maintaining color consistency."""
        print("Applying warps to create burst sequences...")
        HR_warped = np.empty((self.n_images, burst_size, self.height, self.width, 3))
        
        for i in tqdm(range(self.n_images)):
            # First frame is unmodified
            HR_warped[i, 0] = self.blurred_data[i]
            
            # Apply warps to subsequent frames
            for j in range(self.warps.shape[1]):
                # Add 0 shift for color channel
                shift_params = (*self.warps[i, j], 0)
                shifted = ndimage.shift(self.blurred_data[i], shift_params)
                
                # Maintain average intensity per channel
                for c in range(3):
                    shifted[..., c] = shifted[..., c] * (
                        np.mean(self.blurred_data[i, ..., c]) / 
                        np.mean(shifted[..., c])
                    )
                
                HR_warped[i, j+1] = shifted
        
        return np.clip(HR_warped, 0, 1)

    def generate_exposure_ratios(self):
        """Generate random exposure ratios for burst sequence."""
        print("Generating exposure ratios...")
        # Random values between 1.2 and 1.4
        alphas = np.random.random((self.n_images, burst_size-1)) * (1.4 - 1.2) + 1.2
        
        # Apply random powers for wider exposure range
        choices = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        powers = np.random.choice(choices, size=(self.n_images, burst_size-1))
        ratios = alphas**powers
        
        # Add reference ratio (1.0) for first frame
        ratios = np.concatenate((np.ones(self.n_images)[:, None], ratios), axis=1)
        return ratios

    def apply_noise(self, data, ratios):
        """Apply realistic sensor noise to RGB data."""
        print("Applying sensor noise...")
        # Noise parameters
        a, b = 0.26, 27
        
        # Generate noise for each channel
        noise_shape = data.shape
        normalised_noises = np.random.normal(0, 1, size=noise_shape)
        
        # Calculate noise standard deviation
        noise_std = np.sqrt(
            a * ratios[:, :, None, None, None] * data * bit_depth + b
        ) / bit_depth
        
        # Apply noise and exposure ratios
        noised = data * ratios[:, :, None, None, None] + noise_std * normalised_noises
        return np.clip(noised, 0, 1)

    def generate_dataset(self):
        """Generate complete synthetic dataset."""
        # Apply processing steps
        self.apply_gaussian_blur()
        self.generate_warps()
        HR_warped = self.apply_warps()
        
        # Downsample
        print("Downsampling...")
        downsampled = HR_warped[:, :, ::2, ::2, :]
        
        # Generate and apply exposure variations
        ratios = self.generate_exposure_ratios()
        noised = self.apply_noise(downsampled, ratios)
        
        # Save results
        print("Saving results...")
        np.save(self.output_path / 'noised.npy', noised)
        np.save(self.output_path / 'ratios_gt.npy', ratios)
        
        # Generate jittered ratios
        print("Generating jittered ratios...")
        jitter_rates = [0, 0.05, 0.2]
        for jitter_rate in tqdm(jitter_rates):
            jitter_noise = (np.random.random((self.n_images, burst_size-1)) * 2 - 1) * jitter_rate
            jitter_noise = np.concatenate((np.zeros(self.n_images)[:, None], jitter_noise), axis=1)
            noised_ratios = (1 + jitter_noise) * ratios
            np.save(
                self.output_path / f'ratios_noised_{int(100*jitter_rate)}.npy',
                noised_ratios
            )
        
        # Save dataset info
        with open(self.output_path / 'dataset_info.txt', 'w') as f:
            f.write("RGB Dataset Information\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of samples: {self.n_images}\n")
            f.write(f"Burst size: {burst_size}\n")
            f.write(f"Image dimensions: {self.height}x{self.width}\n")
            f.write(f"Gaussian blur sigma: {sigma_blur}\n")
            f.write(f"Bit depth normalization: {bit_depth}\n")
            f.write(f"Exposure ratio range: 1.2^(-5) to 1.4^5\n")
            f.write(f"Jitter rates: {jitter_rates}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic RGB burst dataset')
    parser.add_argument('input_path', type=str, help='Path to input .npy file (RGB images)')
    parser.add_argument('output_path', type=str, help='Path to save generated dataset')
    args = parser.parse_args()
    
    # Generate dataset
    generator = RGBDatasetGenerator(args.input_path, args.output_path)
    generator.generate_dataset()

if __name__ == "__main__":
    main() 