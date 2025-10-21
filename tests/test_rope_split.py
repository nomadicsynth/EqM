#!/usr/bin/env python
"""Test RoPE dimension splitting"""
import torch
import sys
sys.path.insert(0, 'EqM')

# Test the actual grid sizes
def test_grid_sizes():
    """Check actual grid dimensions for different configurations"""
    
    # EqM-S/4 with 256x256 images
    image_size = 256
    latent_size = image_size // 8  # VAE downsamples 8x
    patch_size = 4
    
    print("="*60)
    print("Grid Size Analysis for EqM-S/4")
    print("="*60)
    
    for num_frames in [4, 8, 16, 32]:
        spatial_patches = latent_size // patch_size  # 32 / 4 = 8
        temporal_patches = num_frames // patch_size    # varies
        
        total_patches = temporal_patches * spatial_patches * spatial_patches
        
        print(f"\nnum_frames={num_frames} frames:")
        print(f"  Latent size: {latent_size}x{latent_size}")
        print(f"  Patch size: {patch_size}x{patch_size}x{patch_size}")
        print(f"  Grid dimensions (T×H×W): {temporal_patches}×{spatial_patches}×{spatial_patches}")
        print(f"  Total patches: {total_patches}")
        print(f"  Temporal range: 0-{temporal_patches-1} ({temporal_patches} values)")
        print(f"  Spatial range: 0-{spatial_patches-1} ({spatial_patches} values)")
        print(f"  T:H:W ratio: 1:{spatial_patches/temporal_patches}:{spatial_patches/temporal_patches}")
    
    print("\n" + "="*60)
    print("Current RoPE Implementation (1/3 each):")
    print("="*60)
    hidden_size = 384  # EqM-S
    dim_t = hidden_size // 3  # 128
    dim_h = hidden_size // 3  # 128
    dim_w = hidden_size - dim_t - dim_h  # 128
    
    print(f"\nHidden size: {hidden_size}")
    print(f"  Temporal dims: {dim_t} (1/3)")
    print(f"  Height dims: {dim_h} (1/3)")
    print(f"  Width dims: {dim_w} (1/3)")
    print(f"\nThis gives EQUAL capacity to each dimension,")
    print(f"even though spatial dimensions have much larger range!")
    
    print("\n" + "="*60)
    print("Alternative: Proportional Split")
    print("="*60)
    
    for num_frames in [4, 8, 16]:
        temporal_patches = num_frames // patch_size
        spatial_patches = 8
        
        # Total range units
        total_range = temporal_patches + spatial_patches + spatial_patches
        
        # Allocate proportionally
        dim_t_prop = int(hidden_size * temporal_patches / total_range)
        dim_h_prop = int(hidden_size * spatial_patches / total_range)
        dim_w_prop = hidden_size - dim_t_prop - dim_h_prop
        
        print(f"\nnum_frames={num_frames} (grid {temporal_patches}×8×8):")
        print(f"  Temporal: {dim_t_prop} dims ({dim_t_prop/hidden_size*100:.1f}%)")
        print(f"  Height: {dim_h_prop} dims ({dim_h_prop/hidden_size*100:.1f}%)")
        print(f"  Width: {dim_w_prop} dims ({dim_w_prop/hidden_size*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Recommendation:")
    print("="*60)
    print("The 1/3 split is CORRECT for typical video use cases because:")
    print("1. It's simple and fixed (works with any num_frames)")
    print("2. Temporal dimension needs capacity for TIME_SCALE encoding")
    print("3. In practice, 1/3 each works well (standard in video models)")
    print("4. The alternative would need rebalancing for different num_frames")
    print("="*60)

if __name__ == "__main__":
    test_grid_sizes()
