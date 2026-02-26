#!/usr/bin/env python3
"""
统计 refinement 的改进效果：
- 有多少样本 refined 后误差减小了
- 平均改进幅度
- 改进/退步的分布
- Z 轴方向的改进情况 tobe test
"""

import os
import json
import torch
import numpy as np
import sys
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Try to import KDTree, fallback to pure numpy if not available
try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using pure numpy for distance calculation")

sys.path.append(os.getcwd())
from src.dataset import resample_polyline, load_pcd_data, weighted_sampling
from src.model import LineRefineNet

# Config - Must match training parameters!
DATA_DIR = "./inference_data"  # 可以改为 "./vma_infer_data" 使用旧数据，或 "./vma_infer_data_v2" 使用新数据
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
NUM_LINE_POINTS = 32          # Match train_dist.py default
NUM_CONTEXT_POINTS = 2048     # Match train_dist.py
CROP_RADIUS = 0.5             # Match train_dist.py
DECAY_SCALE = 2.0             # Match train_dist.py default (decay_scale parameter)

def compute_metrics(pred, gt):
    """计算 Chamfer distance (XY) 和 Z axis metrics"""
    if gt is None or len(gt) == 0 or len(pred) == 0:
        return float('inf'), float('inf')
    
    if HAS_SCIPY:
        # Pred to GT
        tree_gt = KDTree(gt[:, :2])
        dists_p2g, indices_p2g = tree_gt.query(pred[:, :2])
        mean_xy_p2g = np.mean(dists_p2g)
        mean_z_p2g = np.mean(np.abs(pred[:, 2] - gt[indices_p2g, 2]))
        
        # GT to Pred
        tree_pred = KDTree(pred[:, :2])
        dists_g2p, indices_g2p = tree_pred.query(gt[:, :2])
        mean_xy_g2p = np.mean(dists_g2p)
        mean_z_g2p = np.mean(np.abs(gt[:, 2] - pred[indices_g2p, 2]))

    else:
        # Pure numpy implementation
        # Pred to GT
        diff_p2g = pred[:, :2][:, np.newaxis, :] - gt[:, :2][np.newaxis, :, :]
        dist_matrix_p2g = np.linalg.norm(diff_p2g, axis=2)
        indices_p2g = np.argmin(dist_matrix_p2g, axis=1)
        dists_p2g = np.min(dist_matrix_p2g, axis=1)
        mean_xy_p2g = np.mean(dists_p2g)
        mean_z_p2g = np.mean(np.abs(pred[:, 2] - gt[indices_p2g, 2]))
        
        # GT to Pred
        diff_g2p = gt[:, :2][:, np.newaxis, :] - pred[:, :2][np.newaxis, :, :]
        dist_matrix_g2p = np.linalg.norm(diff_g2p, axis=2)
        indices_g2p = np.argmin(dist_matrix_g2p, axis=1)
        dists_g2p = np.min(dist_matrix_g2p, axis=1)
        mean_xy_g2p = np.mean(dists_g2p)
        mean_z_g2p = np.mean(np.abs(gt[:, 2] - pred[indices_g2p, 2]))
    
    return (mean_xy_p2g + mean_xy_g2p) / 2.0, (mean_z_p2g + mean_z_g2p) / 2.0

def process_single_line(model, pcd_points, noisy_line_raw, device):
    """处理一条线，返回 refined line"""
    noisy_points = resample_polyline(noisy_line_raw, NUM_LINE_POINTS)
    
    if len(pcd_points) > 0 and len(noisy_points) > 0:
        noisy_lines_dense = resample_polyline(noisy_line_raw, 200)
        if HAS_SCIPY:
            tree = KDTree(noisy_lines_dense)
            dists, _ = tree.query(pcd_points[:, :3])
        else:
            # Pure numpy: compute distance from each point to nearest line point
            diff = pcd_points[:, :3][:, np.newaxis, :] - noisy_lines_dense[np.newaxis, :, :]
            dists = np.min(np.linalg.norm(diff, axis=2), axis=1)
        mask = dists < CROP_RADIUS
        context_points = pcd_points[mask]
    else:
        context_points = np.zeros((0, 4))
    
    # Reset random seed for consistent sampling
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Use same parameters as training!
    context_points = weighted_sampling(context_points, noisy_points, NUM_CONTEXT_POINTS, decay_scale=DECAY_SCALE)
    if len(context_points) < NUM_CONTEXT_POINTS:
        pad = np.zeros((NUM_CONTEXT_POINTS - len(context_points), 4))
        context_points = np.vstack([context_points, pad])
    
    center = np.mean(noisy_points, axis=0)
    context_xyz = context_points[:, :3] - center
    context_int = context_points[:, 3:4]
    noisy_centered = noisy_points - center
    
    input_pcd = np.hstack([context_xyz, context_int])
    tensor_pcd = torch.from_numpy(input_pcd).float().unsqueeze(0).to(device)
    tensor_noisy = torch.from_numpy(noisy_centered).float().unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        offsets_stack = model(tensor_pcd, tensor_noisy)
        final_offset = offsets_stack[-1]
    
    pred_offset_np = final_offset[0].cpu().numpy()
    refined_line = noisy_points + pred_offset_np
    
    return refined_line, noisy_points

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LineRefineNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
        return
    
    json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    if not json_files:
        print("No data found.")
        return
    
    # Statistics
    all_noisy_errors = []
    all_refined_errors = []
    all_noisy_z_errors = []
    all_refined_z_errors = []
    
    improvements = []  # (noisy_err - refined_err)
    improvements_z = []
    
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0
    total_count = 0
    
    print(f"\nProcessing {len(json_files)} files...")
    
    for json_file in json_files:
        json_path = os.path.join(DATA_DIR, json_file)
        pcd_path = json_path.replace('.json', '.pcd')
        
        if not os.path.exists(pcd_path):
            continue
        
        full_pcd = load_pcd_data(pcd_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            items = data.get('items', [])
        
        for item in items:
            gt_list = item.get('position', [])
            if len(gt_list) < 2:
                continue
            
            gt_3d_raw = np.array([[p['x'], p['y'], p.get('z', 0)] for p in gt_list])
            gt_3d_res = resample_polyline(gt_3d_raw, NUM_LINE_POINTS)
            
            candidates = item.get('noisy_candidates', [])
            if not candidates:
                continue
            
            # Only process first candidate
            cand = candidates[0]
            noisy_arr_3d = np.array([[p['x'], p['y'], p['z']] for p in cand])
            if len(noisy_arr_3d) < 2:
                continue
            
            try:
                refined_3d, noisy_resampled = process_single_line(model, full_pcd, noisy_arr_3d, device)
                
                err_noise, z_err_noise = compute_metrics(noisy_resampled, gt_3d_res)
                err_refine, z_err_refine = compute_metrics(refined_3d, gt_3d_res)
                
                if err_noise == float('inf') or err_refine == float('inf'):
                    continue
                
                all_noisy_errors.append(err_noise)
                all_refined_errors.append(err_refine)
                all_noisy_z_errors.append(z_err_noise)
                all_refined_z_errors.append(z_err_refine)
                
                improvement = err_noise - err_refine
                improvements.append(improvement)
                
                improvement_z = z_err_noise - z_err_refine
                improvements_z.append(improvement_z)
                
                total_count += 1
                
                if improvement > 0.001:  # Improved (threshold to avoid numerical noise)
                    improved_count += 1
                elif improvement < -0.001:  # Degraded
                    degraded_count += 1
                else:  # Unchanged
                    unchanged_count += 1
                    
            except Exception as e:
                print(f"Error processing {json_file} item: {e}")
                continue
    
    if total_count == 0:
        print("No valid samples processed.")
        return
    
    # Calculate statistics
    all_noisy_errors = np.array(all_noisy_errors)
    all_refined_errors = np.array(all_refined_errors)
    improvements = np.array(improvements)
    
    all_noisy_z_errors = np.array(all_noisy_z_errors)
    all_refined_z_errors = np.array(all_refined_z_errors)
    improvements_z = np.array(improvements_z)
    
    print("\n" + "="*60)
    print("Refinement Improvement Statistics")
    print("="*60)
    print(f"\nTotal samples processed: {total_count}")
    print(f"\n--- Improvement Count ---")
    print(f"  Improved:   {improved_count} ({improved_count/total_count*100:.2f}%)")
    print(f"  Degraded:   {degraded_count} ({degraded_count/total_count*100:.2f}%)")
    print(f"  Unchanged:  {unchanged_count} ({unchanged_count/total_count*100:.2f}%)")
    
    print(f"\n--- Error Statistics (Chamfer XY) ---")
    print(f"Noisy (before refinement):")
    print(f"  Mean:   {all_noisy_errors.mean():.4f} m")
    print(f"  Median: {np.median(all_noisy_errors):.4f} m")
    print(f"  Std:    {all_noisy_errors.std():.4f} m")
    
    print(f"\nRefined (after refinement):")
    print(f"  Mean:   {all_refined_errors.mean():.4f} m")
    print(f"  Median: {np.median(all_refined_errors):.4f} m")
    print(f"  Std:    {all_refined_errors.std():.4f} m")
    
    print(f"\n--- Improvement Statistics (Chamfer XY) ---")
    print(f"  Mean improvement:   {improvements.mean():.4f} m")
    print(f"  Median improvement: {np.median(improvements):.4f} m")
    print(f"  Max improvement:    {improvements.max():.4f} m")
    print(f"  Max degradation:    {improvements.min():.4f} m")
    print(f"  Std:                {improvements.std():.4f} m")
    
    print(f"\n" + "-"*30)
    print("--- Z Axis Statistics ---")
    print(f"-"*30)
    
    print(f"Noisy Z Error:")
    print(f"  Mean:   {all_noisy_z_errors.mean():.4f} m")
    print(f"  Median: {np.median(all_noisy_z_errors):.4f} m")
    
    print(f"\nRefined Z Error:")
    print(f"  Mean:   {all_refined_z_errors.mean():.4f} m")
    print(f"  Median: {np.median(all_refined_z_errors):.4f} m")
    
    print(f"\nZ Improvement:")
    print(f"  Mean improvement:   {improvements_z.mean():.4f} m")
    print(f"  Median improvement: {np.median(improvements_z):.4f} m")
    
    # Improvement percentage
    relative_improvements = improvements / (all_noisy_errors + 1e-6) * 100
    relative_improvements_z = improvements_z / (all_noisy_z_errors + 1e-6) * 100
    print(f"\n--- Relative Improvement ---")
    print(f"  XY: Mean {relative_improvements.mean():.2f}%, Median {np.median(relative_improvements):.2f}%")
    print(f"  Z:  Mean {relative_improvements_z.mean():.2f}%, Median {np.median(relative_improvements_z):.2f}%")
    
    # Percentiles
    print(f"\n--- Improvement Distribution (XY) ---")
    print(f"  P25: {np.percentile(improvements, 25):.4f} m")
    print(f"  P50: {np.percentile(improvements, 50):.4f} m")
    print(f"  P75: {np.percentile(improvements, 75):.4f} m")
    print(f"  P90: {np.percentile(improvements, 90):.4f} m")
    print(f"  P95: {np.percentile(improvements, 95):.4f} m")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✅ {improved_count/total_count*100:.1f}% of samples improved")
    print(f"❌ {degraded_count/total_count*100:.1f}% of samples degraded")
    print(f"➡️  {unchanged_count/total_count*100:.1f}% of samples unchanged")
    print(f"\nAverage XY error reduction: {improvements.mean():.4f} m ({relative_improvements.mean():.2f}%)")
    print(f"Average Z error reduction:  {improvements_z.mean():.4f} m ({relative_improvements_z.mean():.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
