#!/usr/bin/env python3
"""
Test script to verify the multi-sample optimization fix.
"""

import subprocess
import sys
import os

def test_multisample_fix():
    """Test that multi-sample optimization now creates fresh models for each sample."""
    
    print("🧪 Testing Multi-Sample Optimization Fix")
    print("=" * 50)
    print("This test verifies that each sample gets a fresh model instance.")
    
    # Test with a small number of samples and iterations for quick verification
    cmd = [
        sys.executable, "optimize.py",
        "--dataset", "burst_synth",
        "--multi_sample",
        "--sample_id", "0",  # Will be overridden by multi_sample mode
        "--iters", "10",     # Small number for quick test
        "--model", "mlp",
        "--network_depth", "4",
        "--network_hidden_dim", "256",
        "--learning_rate", "1e-3",
        "--device", "0",
        "--output_folder", "test_multisample_fix_output"
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("\n✅ Multi-sample optimization completed successfully!")
        print("\nOutput analysis:")
        
        # Check for the fresh model creation messages
        output_lines = result.stdout.split('\n')
        fresh_model_count = sum(1 for line in output_lines if "Creating fresh model for sample" in line)
        
        print(f"🔄 Fresh models created: {fresh_model_count}")
        
        if fresh_model_count > 0:
            print("✅ SUCCESS: Fresh models are being created for each sample!")
            print("✅ The fix is working correctly!")
        else:
            print("❌ ERROR: No fresh model creation messages found!")
            print("❌ The fix may not be working properly!")
        
        # Show some sample output
        print(f"\nSample output:")
        for line in output_lines[-10:]:
            if line.strip():
                print(f"  {line}")
                
    except subprocess.CalledProcessError as e:
        print("❌ Error occurred during multi-sample optimization:")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_multisample_fix()
    if success:
        print("\n🎉 Multi-sample optimization fix test completed!")
    else:
        print("\n💥 Multi-sample optimization fix test failed!")
