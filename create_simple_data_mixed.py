#!/usr/bin/env python3
"""
Simple script to create mixed dataset comparisons with data folder samples.
"""

import subprocess
import sys
import os

def create_data_mixed_comparison(synthetic_samples, data_samples, output_path):
    """Create a mixed comparison with specified samples."""
    
    print(f"🎯 Creating mixed comparison with data folder samples...")
    print(f"SyntheticBurstVal samples: {synthetic_samples}")
    print(f"Data folder samples: {data_samples}")
    print(f"Output: {output_path}")
    
    # Create the mixed dataset comparison script content
    script_content = f'''#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_data_mixed_comparison import create_mixed_dataset_comparison

# Sample configurations
sample_configs = []
'''

    # Add synthetic burst samples
    for i, sample_name in enumerate(synthetic_samples):
        script_content += f'''
sample_configs.append({{
    'dataset_type': 'synthetic_burst',
    'sample_name': '{sample_name}',
    'handheld_sample': '{sample_name}',
    'nir_sample': 'sample_{i:03d}',
    'ours_sample': 'sample_{i:03d}'
}})
'''

    # Add data folder samples
    for i, sample_name in enumerate(data_samples):
        script_content += f'''
sample_configs.append({{
    'dataset_type': 'data_folder',
    'sample_name': '{sample_name}',
    'handheld_sample': '{synthetic_samples[0] if synthetic_samples else "0006"}',  # Use first synthetic sample for handheld
    'nir_sample': 'sample_{i:03d}',
    'ours_sample': 'sample_{i:03d}'
}})
'''

    script_content += f'''
# Create the comparison
create_mixed_dataset_comparison(sample_configs, '{output_path}')
print(f"✅ Mixed comparison created: {output_path}")
'''

    # Write and run the script
    script_path = "temp_data_mixed_comparison.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print("✅ Mixed comparison created successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Error occurred:")
        print(e.stderr)
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python create_simple_data_mixed.py <synthetic_samples> <data_samples> <output_path>")
        print("Example: python create_simple_data_mixed.py '0006,0013' 'Amnesty POI-7-1-3_rgb,UNHCR-NERs009690_rgb' mixed_comparison.png")
        print("\nAvailable samples:")
        print("SyntheticBurstVal: 0006, 0013, 0014, 0017, 0022, 0024, 0062, 0065, 0084, 0089, ...")
        print("Data folder: ASMSpotter-28-1-2_rgb, ASMSpotter-29-3-1_rgb, Amnesty POI-11-1-1_rgb, Amnesty POI-17-3-1_rgb, ...")
        return
    
    synthetic_samples = [x.strip() for x in sys.argv[1].split(',')]
    data_samples = [x.strip() for x in sys.argv[2].split(',')]
    output_path = sys.argv[3]
    
    create_data_mixed_comparison(synthetic_samples, data_samples, output_path)

if __name__ == "__main__":
    main()
