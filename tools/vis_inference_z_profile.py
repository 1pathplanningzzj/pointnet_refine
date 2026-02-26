import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Ensure src is importable
sys.path.append(os.getcwd())

from src.dataset import resample_polyline, load_pcd_data, weighted_sampling
from src.model import LineRefineNet

# Config
DATA_DIR = "./inference_data"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_DIR = "./inference_vis_z_profile"
NUM_VIS_SAMPLES = 50
NUM_LINE_POINTS = 32
NUM_CONTEXT_POINTS = 2048 
CROP_RADIUS = 0.5        
RESOLUTION = 0.05 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_line(model, pcd_points, noisy_line_raw, device):
    """
    Runs model inference for a single line.
    Returns the REFINED line in ORIGINAL coordinates.
    """
    # 1. Resample Noisy Line
    noisy_points = resample_polyline(noisy_line_raw, NUM_LINE_POINTS) # (32, 3)
    
    # 2. Crop Context
    if len(pcd_points) > 0 and len(noisy_points) > 0:
        noisy_lines_dense = resample_polyline(noisy_line_raw, 200)
        tree = KDTree(noisy_lines_dense)
        dists, _ = tree.query(pcd_points[:, :3])
        mask = dists < CROP_RADIUS
        context_points = pcd_points[mask]
    else:
        context_points = np.zeros((0, 4))
        
    # 3. Sampling
    context_points = weighted_sampling(context_points, noisy_points, NUM_CONTEXT_POINTS)
    if len(context_points) < NUM_CONTEXT_POINTS:
        pad = np.zeros((NUM_CONTEXT_POINTS - len(context_points), 4))
        context_points = np.vstack([context_points, pad])

    # 4. Normalize
    center = np.mean(noisy_points, axis=0)
    context_xyz = context_points[:, :3] - center
    context_int = context_points[:, 3:4]
    noisy_centered = noisy_points - center
    
    # 5. Prepare Tensor
    input_pcd = np.hstack([context_xyz, context_int])
    tensor_pcd = torch.from_numpy(input_pcd).float().unsqueeze(0).to(device)
    tensor_noisy = torch.from_numpy(noisy_centered).float().unsqueeze(0).to(device)
    
    # 6. Forward
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
        
    for i, json_file in enumerate(json_files):
        if i >= NUM_VIS_SAMPLES: break
        
        print(f"Processing {json_file}...")
        json_path = os.path.join(DATA_DIR, json_file)
        pcd_path = json_path.replace('.json', '.pcd')
        
        if not os.path.exists(pcd_path): continue
        
        # 1. Load Data
        full_pcd = load_pcd_data(pcd_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            items = data.get('items', [])

        def calc_z_metric(gt, pred):
            """
            Calculates Mean Absolute Z-Error
            """
            if gt is None or len(gt) == 0: return -1.0, -1.0
            
            # Simple average Z diff matching points roughly (since resampled to same num points)
            # Assuming points are ordered along the line.
            # A more robust way is closest point distance in 3D composed only of Z component
            
            z_errs = []
            for p in pred:
                # Find closest point in GT (2D projection closest) to compare Z
                dists_2d = np.linalg.norm(gt[:, :2] - p[:2], axis=1)
                best_idx = np.argmin(dists_2d)
                z_err = abs(gt[best_idx, 2] - p[2])
                z_errs.append(z_err)
            
            return np.mean(z_errs), np.max(z_errs)

        for item_idx, item in enumerate(items):
            gt_list = item.get('position', [])
            gt_3d_res = None
            
            if len(gt_list) > 1:
                gt_3d_raw = np.array([[p['x'], p['y'], p.get('z', 0)] for p in gt_list])
                gt_3d_res = resample_polyline(gt_3d_raw, NUM_LINE_POINTS)
            
            candidates = item.get('noisy_candidates', [])
            if not candidates and gt_3d_res is None: continue

            # Only process if we have candidates
            candidates_to_process = candidates[:1] 
            
            for i_c, cand in enumerate(candidates_to_process):
                noisy_arr_3d = np.array([[p['x'], p['y'], p['z']] for p in cand])
                if len(noisy_arr_3d) < 2: continue
                
                # Inference
                refined_3d, noisy_resampled = process_single_line(model, full_pcd, noisy_arr_3d, device)
                
                # Calculate Z Metrics
                mean_err_noise, max_err_noise = calc_z_metric(gt_3d_res, noisy_resampled)
                mean_err_refine, max_err_refine = calc_z_metric(gt_3d_res, refined_3d)

                # --- PLOTTING ---
                # Draw Side View (Z vs X)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
                
                # Plot GT
                if gt_3d_res is not None:
                    ax.plot(gt_3d_res[:, 0], gt_3d_res[:, 2], color='lime', linewidth=3.0, label='Ground Truth Z', alpha=0.8)
                    # Scatter GT points to see density
                    ax.scatter(gt_3d_res[:, 0], gt_3d_res[:, 2], color='lime', s=10)

                # Plot Noisy
                label_noise = f"Noisy (Mean Z-Err={mean_err_noise:.3f}m)"
                ax.plot(noisy_resampled[:, 0], noisy_resampled[:, 2], color='red', linestyle='--', linewidth=1.5, label=label_noise)
                
                # Plot Refined
                label_refine = f"Refined (Mean Z-Err={mean_err_refine:.3f}m)"
                ax.plot(refined_3d[:, 0], refined_3d[:, 2], color='blue', linewidth=2.5, label=label_refine)
                ax.scatter(refined_3d[:, 0], refined_3d[:, 2], color='blue', s=15, marker='x')

                ax.set_title(f"Z-Axis Profile Correction\nFile: {json_file}\nItem {item_idx} | Noise Scale: {0.1 if i_c==0 else 'Unknown'}")
                ax.set_xlabel("Local X (Longitudinal Distance) [m]")
                ax.set_ylabel("Height Z [m]")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()
                
                # Force Z axis to be reasonable scale to see differences
                # Find all Zs
                all_zs = []
                if gt_3d_res is not None: all_zs.extend(gt_3d_res[:, 2])
                all_zs.extend(noisy_resampled[:, 2])
                all_zs.extend(refined_3d[:, 2])
                
                z_min, z_max = min(all_zs), max(all_zs)
                margin = (z_max - z_min) * 0.2 + 0.1
                ax.set_ylim(z_min - margin, z_max + margin)

                out_name = f"{os.path.splitext(json_file)[0]}_item_{item_idx}_Z_profile.png"
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, out_name))
                plt.close(fig)
                print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
