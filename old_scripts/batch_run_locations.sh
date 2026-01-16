#!/bin/bash
# Batch run super-resolution for multiple locations from locations.txt
# Uses run_real_world_sr.sh which processes already downloaded TCI folders

echo "============================================================================"
echo "BATCH SUPER-RESOLUTION RUN"
echo "============================================================================"

# Default parameters (can be overridden via command line)
SCALE_FACTOR=5
ITERS=2000
DEVICE=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scale_factor|--df)
            SCALE_FACTOR="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Processes locations from locations.txt using run_real_world_sr.sh"
            echo "If TCI folders already exist, download is skipped and existing data is used"
            echo ""
            echo "Options:"
            echo "  --scale_factor <factor>  Super-resolution scale factor (default: 5)"
            echo "  --df <factor>            Alias for --scale_factor"
            echo "  --iters <iterations>     Number of training iterations (default: 2000)"
            echo "  --device <id>            CUDA device number (default: 1)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --scale_factor 5 --iters 2000 --device 1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Read locations from file
LOCATION_FILE="locations.txt"

if [ ! -f "$LOCATION_FILE" ]; then
    echo "Error: $LOCATION_FILE not found!"
    exit 1
fi

echo "Parameters:"
echo "  Scale factor: $SCALE_FACTOR"
echo "  Iterations: $ITERS"
echo "  Device: $DEVICE"
echo ""

# Count total locations (excluding empty lines)
TOTAL=$(grep -v '^[[:space:]]*$' "$LOCATION_FILE" | grep -v '^#' | wc -l)
echo "Found $TOTAL locations to process"
echo ""

# Process each location
LOCATION_NUM=1
while read -r lat lon; do
    # Skip empty lines and comments
    [[ -z "$lat" || -z "$lon" ]] && continue
    [[ "$lat" =~ ^#.*$ ]] && continue
    
    echo "============================================================================"
    echo "Processing location $LOCATION_NUM of $TOTAL"
    echo "Coordinates: ($lat, $lon)"
    echo "============================================================================"
    echo ""
    
    # Run the super-resolution script (will use existing TCI folder if available)
    ./run_real_world_sr.sh "$lat" "$lon" "$SCALE_FACTOR" "$ITERS" "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Completed location $LOCATION_NUM of $TOTAL"
    else
        echo ""
        echo "✗ Failed location $LOCATION_NUM of $TOTAL"
    fi
    echo ""
    
    # Increment counter
    ((LOCATION_NUM++))
    
    # Optional: Add a small delay between runs
    # sleep 2
    
done < "$LOCATION_FILE"

echo "============================================================================"
echo "ALL LOCATIONS COMPLETED!"
echo "============================================================================"
echo "Results saved in:"
echo "  - real_world_data/  (downloaded LR images)"
echo "  - real_world_results/  (SR outputs)"
echo "============================================================================"
