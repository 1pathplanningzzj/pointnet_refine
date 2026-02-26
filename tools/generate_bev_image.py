import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys

# Add Valid search path for src
sys.path.append(os.getcwd())

def load_pcd_data_simple(pcd_path):
    """
    Simple loader that handles the specific ASCII/Binary format we have.
    Ideally imports from src.dataset, but independent is safer for tools.
    """
    try:
        with open(pcd_path, 'rb') as f:
            header = []
            while True:
                line = f.readline().strip()
                header.append(line)
                if line.startswith(b'DATA'):
                    break
            
            data_type = header[-1].split()[1] # DATA ascii or DATA binary
            
            if data_type == b'ascii':
                print("Loading ASCII PCD...")
                data = np.loadtxt(pcd_path, skiprows=len(header), dtype=np.float32)
                return data
            else:
                print("Loading Binary PCD...")
                from src.dataset import load_pcd_data
                return load_pcd_data(pcd_path)
    except Exception as e:
        print(f"Error loading custom loader, trying src.dataset: {e}")
        from src.dataset import load_pcd_data
        return load_pcd_data(pcd_path)

def generate_bev_image(pcd_path, resolution=0.03, output_path="bev_intensity_highres.png"):
    print(f"Loading {pcd_path}...")
    data = load_pcd_data_simple(pcd_path)
    
    if data.shape[1] < 4:
        print("Error: PCD data does not have intensity column (x, y, z, intensity)")
        return
    
    # Extract columns (VMA coordinates: Z is usually height)
    # BeV -> X-Y plane
    
    # We filter data to focus on the road (Z range) if necessary
    # Let's inspect Z distribution briefly?
    # z_vals = data[:, 2]
    # print(f"Z stats: min {z_vals.min():.2f} max {z_vals.max():.2f}")
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    intensity = data[:, 3]
    
    # Filter by Z (Height) to remove ground noise or ceiling? 
    # For now, keep all.
    
    # Determine bounds
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    padding = 2.0 # meters
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    width_m = y_max - y_min
    height_m = x_max - x_min
    
    img_w = int(width_m / resolution)
    img_h = int(height_m / resolution)
    
    print(f"Generating Image: {img_w}x{img_h} pixels (Resolution: {resolution}m/px)")
    print(f"Area: X[{x_min:.1f}, {x_max:.1f}] Y[{y_min:.1f}, {y_max:.1f}]")
    
    # 2. Rasterize
    # Initialize buffers
    # Use -1 to indicate empty
    bev_map = np.full((img_h, img_w), -1.0, dtype=np.float32)
    
    # Map X, Y to U, V
    # U (col) -> Y axis (Left-Right)
    # V (row) -> X axis (Top-Bottom). X_max is UP (Top)
    
    u = ((y - y_min) / resolution).astype(np.int32)
    v = ((x_max - x) / resolution).astype(np.int32)
    
    # Clip to be safe
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[mask]
    v = v[mask]
    ints = intensity[mask]
    
    # Sort by intensity so highest intensity draws last (on top)
    sort_idx = np.argsort(ints)
    u_sorted = u[sort_idx]
    v_sorted = v[sort_idx]
    ints_sorted = ints[sort_idx]
    
    # Fill map
    bev_map[v_sorted, u_sorted] = ints_sorted
    
    # 3. Apply Colormap
    # Max intensity stats
    valid_mask = (bev_map >= 0)
    if not np.any(valid_mask):
        print("No valid points in map!")
        return

    valid_vals = bev_map[valid_mask]
    min_val = 0 # np.min(valid_vals)
    max_val = np.percentile(valid_vals, 99.0) # Robust max
    if max_val <= 0: max_val = 255.0

    print(f"Mapping Intensity [{min_val:.1f} - {max_val:.1f}] to Colormap")
    
    # Create final RGBA image (White background)
    final_img = np.ones((img_h, img_w, 4), dtype=np.float32) # RGBA White
    
    # Normalize valid pixels
    norm_vals = np.clip((valid_vals - min_val) / (max_val - min_val), 0, 1)
    
    # Get colormap
    cmap = cm.get_cmap('jet')
    colored_vals = cmap(norm_vals) # Returns (N, 4) floats
    
    # Update image
    final_img[valid_mask] = colored_vals
    
    # Save using plt
    plt.imsave(output_path, final_img)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    
    target_pcd = "data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd"
    
    # Check if user provided argument
    if len(sys.argv) > 1:
        target_pcd = sys.argv[1]
        
    generate_bev_image(target_pcd, resolution=0.03)
