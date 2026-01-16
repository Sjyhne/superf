#!/bin/bash
# Real World Super-Resolution Script
# Usage: ./run_real_world_sr.sh <lat> <lon> [scale_factor] [iters] [device]

set -e  # Exit on error

# Check if coordinates are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <latitude> <longitude> [scale_factor] [iters] [device]"
    echo "Example: $0 58.721324 9.229434 5 2000 0"
    echo ""
    echo "Arguments:"
    echo "  latitude       : Latitude of the location"
    echo "  longitude      : Longitude of the location"
    echo "  scale_factor   : Super-resolution scale factor (default: 5)"
    echo "  iters          : Number of training iterations (default: 2000)"
    echo "  device         : CUDA device number (default: 0)"
    exit 1
fi

# Parse arguments
LAT=$1
LON=$2
SCALE_FACTOR=${3:-5}
ITERS=${4:-2000}
DEVICE=${5:-4}

echo "============================================================================"
echo "REAL WORLD SUPER-RESOLUTION"
echo "============================================================================"
echo "Location: ($LAT, $LON)"
echo "Scale Factor: $SCALE_FACTOR"
echo "Iterations: $ITERS"
echo "Device: $DEVICE"
echo "============================================================================"

# Create unique location identifier from coordinates
LOC_ID=$(printf "lat%.5f_lon%.5f" "$LAT" "$LON" | tr -d '-' | tr '.' 'p')

# Create output directories
DATA_DIR="real_world_data/${LOC_ID}"
RESULTS_DIR="real_world_results/${LOC_ID}"

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if LR samples already exist
TCI_DIR="${DATA_DIR}/TCI"
if [ -d "$TCI_DIR" ] && [ "$(ls -A $TCI_DIR/*.png 2>/dev/null)" ]; then
    NUM_EXISTING=$(ls -1 $TCI_DIR/*.png 2>/dev/null | wc -l)
    echo "✓ Found existing LR samples ($NUM_EXISTING images)"
    echo "  Skipping download - using existing data"
    echo ""
else
    echo "No existing LR samples found - will download new images"
    echo ""
fi

# Export variables so they're available to the Python script
export LAT LON SCALE_FACTOR ITERS DEVICE DATA_DIR RESULTS_DIR SKIP_DOWNLOAD

# Create a modified version of real_world_test.py with location-specific parameters
DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_DIR" python3 <<PYTHON_EOF
import sys
import os
from pathlib import Path

# Get variables from environment
lat = float(os.environ['LAT'])
lon = float(os.environ['LON'])
data_dir = os.environ['DATA_DIR']
results_dir = os.environ['RESULTS_DIR']
scale_factor = int(os.environ['SCALE_FACTOR'])
iters = int(os.environ['ITERS'])
device_id = os.environ.get('DEVICE', '4')

# Check if we should skip download
tci_dir = f"{data_dir}/TCI"
import glob
existing_images = glob.glob(f"{tci_dir}/*.png")
skip_download = len(existing_images) > 0

# Read the original script
with open('real_world_test.py', 'r') as f:
    script_content = f.read()

# Find and replace the coordinates
script_content = script_content.replace(
    'LAT, LON = 58.72132425373909, 9.229434602964929',
    f'LAT, LON = {lat}, {lon}'
)

# Update output directories - but make create_output_directory safe (no deletion)
safe_create_dir = f'''
def create_output_directory(base_dir="output_cog"):
    """Create output directory without deleting existing data"""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

'''
# Replace the function definition
script_content = script_content.replace(
    'def create_output_directory(base_dir="output_cog"):',
    '# ORIGINAL FUNCTION REPLACED TO PREVENT DATA DELETION\\n' + safe_create_dir + '# ORIGINAL: def create_output_directory(base_dir="output_cog"):'
)

# Update the call to use our data directory
script_content = script_content.replace(
    'output_dir = create_output_directory("results")',
    f'output_dir = create_output_directory("{data_dir}")'
)

# Update scale factor
script_content = script_content.replace(
    'SCALE_FACTOR = 5',
    f'SCALE_FACTOR = {scale_factor}'
)

# Update iterations
script_content = script_content.replace(
    'ITERS = 2000',
    f'ITERS = {iters}'
)

# Update output folder
script_content = script_content.replace(
    'OUTPUT_FOLDER = Path("results/SR")',
    f'OUTPUT_FOLDER = Path("{results_dir}")'
)

# Ensure INPUT_FOLDER is set before it's used
script_content = script_content.replace(
    'INPUT_FOLDER = Path("results/TCI")',
    f'INPUT_FOLDER = Path("{tci_dir}")'
)

# Skip download section if images already exist
if skip_download:
    # More precise skipping - only comment out the actual download loop
    # but keep the setup code
    lines = script_content.split('\\n')
    new_lines = []
    skip_mode = False
    
    for i, line in enumerate(lines):
        # Start skipping at the download loop
        if 'for item in available_items:' in line:
            skip_mode = True
            new_lines.append('# SKIPPED - images already exist: for item in available_items:')
            continue
            
        # Stop skipping after the download loop ends
        if skip_mode and line.strip().startswith('print("Finished downloading'):
            skip_mode = False
            new_lines.append('# SKIPPED download results summary')
            continue
            
        if skip_mode and '# Summarize the results' in line:
            skip_mode = False
        
        if not skip_mode:
            new_lines.append(line)
        else:
            # Comment out skipped lines
            if line.strip() and not line.strip().startswith('#'):
                new_lines.append('# SKIPPED: ' + line)
    
    script_content = '\\n'.join(new_lines)
    print(f"Skipping download section (found {len(existing_images)} existing images)")

# Update device setting to use the provided DEVICE argument
script_content = script_content.replace(
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    f'device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")'
)
print(f"Set CUDA device to {device_id}")

# Fix the NoneType error by adding error handling before len() calls
error_handling = '''
# Add error handling for loading LR images
original_load_lr = load_lr_images
def load_lr_images_with_error_handling(folder_path):
    result = original_load_lr(folder_path)
    if result[0] is None:
        print(f"ERROR: No LR images found in {folder_path}")
        print("Please check the folder path and ensure images exist.")
        sys.exit(1)
    return result

# Replace the function
load_lr_images = load_lr_images_with_error_handling
'''

# Insert error handling before the load_lr_images call
if 'def load_lr_images(folder_path):' in script_content:
    # Find the line with the first load_lr_images call
    load_call = 'lr_images, means, stds = load_lr_images(INPUT_FOLDER)'
    if load_call in script_content:
        # Insert error handling before the first call
        script_content = script_content.replace(
            load_call,
            error_handling + '\\n' + load_call,
            1  # Only replace the first occurrence
        )
        print("Added error handling for LR image loading")

# Save the modified script
modified_script = Path('real_world_test_modified.py')
with open(modified_script, 'w') as f:
    f.write(script_content)

print(f"Created modified script: {modified_script}")
PYTHON_EOF

# Run the modified script
echo "Running super-resolution for location ($LAT, $LON)..."
python3 real_world_test_modified.py

# Clean up
rm -f real_world_test_modified.py

echo ""
echo "============================================================================"
echo "COMPLETED!"
echo "============================================================================"
echo "Location: ($LAT, $LON)"
echo "Data saved to: $DATA_DIR"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================================"
