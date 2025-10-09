#!/usr/bin/env python3
"""
Copy data from the data folder for specific configurations:
- Augmentations: none, light
- LR shift: 1.0px
- Scales: 2, 4, 8
"""

import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def copy_data_subset(source_dir, output_dir, scales=[2, 4, 8], augmentations=['none', 'light'], lr_shift=1.0):
    """
    Copy data for specified configurations.
    
    Args:
        source_dir: Path to the data folder
        output_dir: Path to copy the subset to
        scales: List of scale factors to include
        augmentations: List of augmentation types to include
        lr_shift: LR shift value in pixels
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all sample directories
    sample_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.endswith('_rgb')]
    
    if not sample_dirs:
        print(f"No sample directories found in {source_path}")
        return
    
    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Copying data for scales: {scales}, augmentations: {augmentations}, lr_shift: {lr_shift}px")
    
    total_copied = 0
    total_skipped = 0
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_name = sample_dir.name
        print(f"\nProcessing {sample_name}...")
        
        # Create sample directory in output
        sample_output_dir = output_path / sample_name
        sample_output_dir.mkdir(exist_ok=True)
        
        for scale in scales:
            for aug in augmentations:
                # Construct source directory name
                source_subdir_name = f"scale_{scale}_shift_{lr_shift}px_aug_{aug}"
                source_subdir = sample_dir / source_subdir_name
                
                if not source_subdir.exists():
                    print(f"  Warning: {source_subdir_name} not found in {sample_name}")
                    total_skipped += 1
                    continue
                
                # Create destination directory
                dest_subdir = sample_output_dir / source_subdir_name
                dest_subdir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files from source to destination
                try:
                    for file_path in source_subdir.iterdir():
                        if file_path.is_file():
                            dest_file = dest_subdir / file_path.name
                            shutil.copy2(file_path, dest_file)
                            total_copied += 1
                    
                    print(f"  ✓ Copied {source_subdir_name}")
                    
                except Exception as e:
                    print(f"  Error copying {source_subdir_name}: {e}")
                    total_skipped += 1
    
    print(f"\n{'='*60}")
    print("COPY SUMMARY")
    print(f"{'='*60}")
    print(f"Total files copied: {total_copied}")
    print(f"Total directories skipped: {total_skipped}")
    print(f"Output directory: {output_path}")
    print(f"Expected structure:")
    print(f"  {output_path}/")
    print(f"  ├── sample1_rgb/")
    print(f"  │   ├── scale_2_shift_1.0px_aug_none/")
    print(f"  │   ├── scale_2_shift_1.0px_aug_light/")
    print(f"  │   ├── scale_4_shift_1.0px_aug_none/")
    print(f"  │   ├── scale_4_shift_1.0px_aug_light/")
    print(f"  │   ├── scale_8_shift_1.0px_aug_none/")
    print(f"  │   └── scale_8_shift_1.0px_aug_light/")
    print(f"  └── sample2_rgb/")
    print(f"      └── ...")

def main():
    parser = argparse.ArgumentParser(description="Copy data subset for specific configurations")
    parser.add_argument("--source_dir", type=str, default="/raid/home/sandej17/satellite_sr/data",
                       help="Source data directory")
    parser.add_argument("--output_dir", type=str, default="/raid/home/sandej17/satellite_sr/data_subset",
                       help="Output directory for subset")
    parser.add_argument("--scales", type=int, nargs='+', default=[2, 4, 8],
                       help="Scale factors to include")
    parser.add_argument("--augmentations", type=str, nargs='+', default=['none', 'light'],
                       help="Augmentation types to include")
    parser.add_argument("--lr_shift", type=float, default=1.0,
                       help="LR shift value in pixels")
    
    args = parser.parse_args()
    
    print("Data Subset Copy Tool")
    print("=" * 50)
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Scales: {args.scales}")
    print(f"Augmentations: {args.augmentations}")
    print(f"LR Shift: {args.lr_shift}px")
    print("=" * 50)
    
    copy_data_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        scales=args.scales,
        augmentations=args.augmentations,
        lr_shift=args.lr_shift
    )

if __name__ == "__main__":
    main()
