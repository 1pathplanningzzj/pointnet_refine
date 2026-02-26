#!/usr/bin/env python3
"""
可视化 noisy line 和点云，检查坐标对齐情况
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.getcwd())
from src.dataset import load_pcd_data

# Config
DATA_DIR = "./vma_infer_data_v2"  # 可以改为 "./vma_infer_data" 使用旧数据
OUTPUT_DIR = "./noisy_pcd_vis"
NUM_SAMPLES = 20  # 可视化多少个样本

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_bev_map(pcd_points, resolution=0.05, padding=5.0):
    """生成BEV点云图"""
    x = pcd_points[:, 0]
    y = pcd_points[:, 1]
    intensity = pcd_points[:, 3]
    
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding
    
    width_m = y_max - y_min
    height_m = x_max - x_min
    
    img_w = int(width_m / resolution)
    img_h = int(height_m / resolution)
    
    bev_map = np.full((img_h, img_w), 0.0, dtype=np.float32)
    
    # Map X->Row(V, inverted), Y->Col(U)
    u = ((y - y_min) / resolution).astype(np.int32)
    v = ((x_max - x) / resolution).astype(np.int32)
    
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, ints = u[mask], v[mask], intensity[mask]
    
    # Sort by intensity to draw brightest last
    sort_idx = np.argsort(ints)
    bev_map[v[sort_idx], u[sort_idx]] = ints[sort_idx]
    
    return bev_map, [y_min, y_max, x_min, x_max]

def main():
    json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return
    
    print(f"Found {len(json_files)} files, will visualize {min(NUM_SAMPLES, len(json_files))} samples...")
    
    for i, json_file in enumerate(json_files):
        if i >= NUM_SAMPLES:
            break
        
        json_path = os.path.join(DATA_DIR, json_file)
        pcd_path = json_path.replace('.json', '.pcd')
        
        if not os.path.exists(pcd_path):
            print(f"  Skip {json_file}: PCD not found")
            continue
        
        print(f"Processing {json_file}...")
        
        # Load data
        full_pcd = load_pcd_data(pcd_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            items = data.get('items', [])
        
        if not items:
            print(f"  Skip {json_file}: No items")
            continue
        
        # Generate BEV map
        bev_img, extent = generate_bev_map(full_pcd, resolution=0.05)
        
        # Normalize intensity
        if bev_img.max() > 0:
            norm_img = bev_img / np.percentile(bev_img[bev_img>0], 90)
            norm_img = np.clip(norm_img, 0, 1)
            norm_img = np.power(norm_img, 0.6)
        else:
            norm_img = bev_img
        
        masked_img = np.ma.masked_where(bev_img == 0, norm_img)
        
        # Plot each item
        for item_idx, item in enumerate(items):
            noisy_candidates = item.get('noisy_candidates', [])
            if not noisy_candidates:
                continue
            
            # Get bounds
            xs, ys = [], []
            
            # Noisy line
            noisy = noisy_candidates[0]
            noisy_xs = [p['x'] for p in noisy]
            noisy_ys = [p['y'] for p in noisy]
            xs.extend(noisy_xs)
            ys.extend(noisy_ys)
            
            # GT (if exists)
            gt_list = item.get('position', [])
            if gt_list:
                gt_xs = [p['x'] for p in gt_list]
                gt_ys = [p['y'] for p in gt_list]
                xs.extend(gt_xs)
                ys.extend(gt_ys)
            
            if not xs:
                continue
            
            # Determine crop bounds
            pad = 10.0
            min_x, max_x = min(xs) - pad, max(xs) + pad
            min_y, max_y = min(ys) - pad, max(ys) + pad
            
            # Setup figure
            plot_h = max_x - min_x
            plot_w = max_y - min_y
            if plot_w <= 0 or plot_h <= 0:
                continue
            
            aspect = plot_h / plot_w
            if aspect > 5:
                aspect = 5
            if aspect < 0.2:
                aspect = 0.2
            
            fig, ax = plt.subplots(figsize=(10, 10 * aspect), dpi=100)
            ax.set_facecolor('black')
            
            # Show point cloud
            ax.imshow(masked_img, cmap='jet', extent=extent, origin='upper', interpolation='nearest')
            
            # Zoom in
            ax.set_xlim(min_y, max_y)
            ax.set_ylim(min_x, max_x)
            
            # Plot noisy line
            ax.plot(noisy_ys, noisy_xs, color='red', linewidth=2.0, linestyle='--', 
                   label='Noisy (VMA)', alpha=0.9, marker='o', markersize=3)
            
            # Plot GT if exists
            if gt_list:
                ax.plot(gt_ys, gt_xs, color='lime', linewidth=2.5, 
                       label='GT', alpha=0.9, marker='s', markersize=2)
            
            # Add coordinate info
            noisy_arr = np.array([[p['x'], p['y']] for p in noisy])
            info_text = f"X: [{noisy_arr[:, 0].min():.1f}, {noisy_arr[:, 0].max():.1f}]\n"
            info_text += f"Y: [{noisy_arr[:, 1].min():.1f}, {noisy_arr[:, 1].max():.1f}]\n"
            info_text += f"Points: {len(noisy)}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'),
                   color='white', family='monospace')
            
            ax.set_title(f"{json_file} - Item {item_idx}\nNoisy Line vs Point Cloud", 
                        color='white', fontsize=12)
            ax.set_xlabel("Y (m)", color='white')
            ax.set_ylabel("X (m)", color='white')
            ax.tick_params(colors='white')
            
            # Legend
            ax.legend(loc='upper right', facecolor='black', labelcolor='white', 
                     framealpha=0.7, edgecolor='white')
            
            # Grid
            ax.grid(True, alpha=0.3, color='gray', linestyle='--')
            
            out_name = f"{os.path.splitext(json_file)[0]}_item_{item_idx}_noisy_pcd.png"
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, out_name), facecolor='black', dpi=150)
            plt.close(fig)
            
            print(f"  Saved {out_name}")
    
    print(f"\nDone! Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
