import os
import sys
import numpy as np
import json
import glob
from scipy.spatial import KDTree

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import load_pcd_data, resample_polyline

DATA_ROOT = "/homes/zhangzijian/pointnet_refine/data/vma_test_data"
INFER_DATA_ROOT = "/homes/zhangzijian/pointnet_refine/vma_infer_data"
CROP_RADIUS = 0.3  # From Dataset default
SAMPLE_COUNT = 1024

def analyze_scene(pcd_path, json_path):
    print(f"\nAnalyzing: {os.path.basename(pcd_path)}")
    
    # 1. Load PCD
    full_pcd = load_pcd_data(pcd_path)
    total_points = len(full_pcd)
    print(f"  Total Scene Points: {total_points}")
    
    if total_points == 0:
        return

    # 2. Load JSON Lines
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    print(f"  Lane Lines Candidates: {len(items)}")
    
    roi_point_counts = []
    
    for i, item in enumerate(items):
        noisy_list = item.get('noisy_candidates', [])
        for n_idx, noisy_raw in enumerate(noisy_list):
            noisy_pts = np.array([[p['x'], p['y'], p['z']] for p in noisy_raw])
            
            # Resample like Dataset
            noisy_resampled = resample_polyline(noisy_pts, 200) # DENSE for True Tube Analysis
            
            # KDTree Query (Simulate Dataset Crop)
            tree = KDTree(noisy_resampled)
            dists, _ = tree.query(full_pcd[:, :3])
            
            mask = dists < CROP_RADIUS
            roi_points = full_pcd[mask]
            count = len(roi_points)
            roi_point_counts.append(count)
            
            ratio = count / total_points * 100
            sample_ratio = SAMPLE_COUNT / count * 100 if count > 0 else 0
            
            print(f"    Line {i}-{n_idx}: ROI Points ({CROP_RADIUS}m radius) = {count}")
            print(f"      -> {ratio:.2f}% of Total Scene")
            if count < SAMPLE_COUNT:
                print(f"      -> WARNING: Less than {SAMPLE_COUNT} points! (Oversampling needed)")
            else:
                print(f"      -> Sampling Ratio: {SAMPLE_COUNT}/{count} = {sample_ratio:.1f}% kept")

    if roi_point_counts:
        avg_roi = np.mean(roi_point_counts)
        print(f"  Average ROI Points per Line: {avg_roi:.1f}")

def main():
    # Find some PCDs in the inference folder
    pcd_files = sorted(glob.glob(os.path.join(INFER_DATA_ROOT, "*.pcd")))
    
    if not pcd_files:
        print("No PCD files found to analyze.")
        return
        
    # Analyze first 5 scenes
    for pcd_path in pcd_files[:5]:
        json_path = pcd_path.replace('.pcd', '.json')
        if os.path.exists(json_path):
            analyze_scene(pcd_path, json_path)
        else:
            print(f"Skipping {pcd_path} (No JSON found)")

if __name__ == "__main__":
    main()
