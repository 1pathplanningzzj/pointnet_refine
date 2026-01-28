import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import KDTree

# Ensure src is importable
sys.path.append(os.getcwd())

from src.dataset import resample_polyline, load_pcd_data, weighted_sampling
from src.model import LineRefineNet

# Config
DATA_DIR = "./vma_infer_data"  # Using inference_data (same distribution as training data)
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_DIR = "./inference_vis_bev"  # Changed output dir to avoid confusion
NUM_VIS_SAMPLES = 50
NUM_LINE_POINTS = 32
NUM_CONTEXT_POINTS = 2048 # Increased to match new training config
CROP_RADIUS = 0.5        # Adjusted to match training (0.5m)
RESOLUTION = 0.05 # 5cm/pixel

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

def generate_bev_map(pcd_points, resolution=0.05, padding=5.0):
    # Extract
    x = pcd_points[:, 0]
    y = pcd_points[:, 1]
    intensity = pcd_points[:, 3]
    
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding
    
    width_m = y_max - y_min
    height_m = x_max - x_min
    
    img_w = int(width_m / resolution)
    img_h = int(height_m / resolution)
    
    # Init Map with NaN or -1
    bev_map = np.full((img_h, img_w), 0.0, dtype=np.float32)
    
    # Map X->Row(V, inverted), Y->Col(U)
    # Top of image (V=0) is X_max
    u = ((y - y_min) / resolution).astype(np.int32)
    v = ((x_max - x) / resolution).astype(np.int32)
    
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, ints = u[mask], v[mask], intensity[mask]
    
    # Sort by intensity to draw brightest last
    sort_idx = np.argsort(ints)
    bev_map[v[sort_idx], u[sort_idx]] = ints[sort_idx]
    
    return bev_map, [y_min, y_max, x_min, x_max] 

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
            
        # 2. Generate BEV
        bev_img, extent = generate_bev_map(full_pcd, RESOLUTION)
        # extent = [y_min, y_max, x_min, x_max]
        
        # Normalize Intensity for Vis
        if bev_img.max() > 0:
            norm_img = bev_img / np.percentile(bev_img[bev_img>0], 90)
            norm_img = np.clip(norm_img, 0, 1)
            norm_img = np.power(norm_img, 0.6)
        else:
            norm_img = bev_img
        
        # Mask 0 values
        masked_img = np.ma.masked_where(bev_img == 0, norm_img)

        # 3. Plot per Item
        def calc_metric(gt, pred):
            """
            Calculates Chamfer Distance (symmetric average minimum distance)
            to measure geometric similarity, ignoring index-to-index alignment.
            """
            if gt is None or len(gt) == 0: return -1.0
            
            # GT to Pred
            # For each point in GT, find closest in Pred
            dists_g2p = []
            for p in gt[:, :2]:
                d = np.linalg.norm(pred[:, :2] - p, axis=1).min()
                dists_g2p.append(d)
            mean_g2p = np.mean(dists_g2p)
            
            # Pred to GT
            dists_p2g = []
            for p in pred[:, :2]:
                d = np.linalg.norm(gt[:, :2] - p, axis=1).min()
                dists_p2g.append(d)
            mean_p2g = np.mean(dists_p2g)
            
            return (mean_g2p + mean_p2g) / 2.0

        for item_idx, item in enumerate(items):
            # Gather Bounds for this item
            xs, ys = [], []
            
            # GT
            gt_list = item.get('position', [])
            gt_arr = np.empty((0, 2))
            gt_3d_res = None
            
            if len(gt_list) > 1:
                gt_arr = np.array([[p['x'], p['y']] for p in gt_list])
                xs.extend(gt_arr[:, 0])
                ys.extend(gt_arr[:, 1])
                
                gt_3d_raw = np.array([[p['x'], p['y'], p.get('z', 0)] for p in gt_list])
                gt_3d_res = resample_polyline(gt_3d_raw, NUM_LINE_POINTS)
            
            # Candidates
            candidates = item.get('noisy_candidates', [])
            if not candidates and len(gt_arr) == 0: continue

            for cand in candidates:
                c_pts = np.array([[p['x'], p['y']] for p in cand])
                xs.extend(c_pts[:, 0])
                ys.extend(c_pts[:, 1])
            
            if not xs: continue
            
            # Determine Crop Bounds with padding
            pad = 10.0
            min_x, max_x = min(xs) - pad, max(xs) + pad
            min_y, max_y = min(ys) - pad, max(ys) + pad
            
            # Setup Figure
            plot_h = max_x - min_x
            plot_w = max_y - min_y
            if plot_w <= 0 or plot_h <= 0: continue
            
            aspect = plot_h / plot_w
            if aspect > 5: aspect = 5
            if aspect < 0.2: aspect = 0.2

            fig, ax = plt.subplots(figsize=(8, 8 * aspect), dpi=100)
            ax.set_facecolor('black')
            
            # Show Full Image
            ax.imshow(masked_img, cmap='jet', extent=extent, origin='upper', interpolation='nearest')
            
            # Zoom in
            ax.set_xlim(min_y, max_y)
            ax.set_ylim(min_x, max_x)
            
            # Plot GT
            if len(gt_arr) > 1:
                ax.plot(gt_arr[:, 1], gt_arr[:, 0], color='lime', linewidth=2.0, label='Target GT', alpha=0.8)
                
            # Plot Candidates & Refine
            # Only process the first candidate for clarity (one GT -> one candidate)
            # If you want to see all candidates, you can change this to: candidates = item.get('noisy_candidates', [])
            candidates_to_process = candidates[:1]  # Only first candidate
            
            for i_c, cand in enumerate(candidates_to_process):
                noisy_arr_3d = np.array([[p['x'], p['y'], p['z']] for p in cand])
                if len(noisy_arr_3d) < 2: continue
                
                refined_3d, noisy_resampled = process_single_line(model, full_pcd, noisy_arr_3d, device)
                
                # Metrics
                err_noise = calc_metric(gt_3d_res, noisy_resampled)
                err_refine = calc_metric(gt_3d_res, refined_3d)
                
                noise_label = f"Noisy (E={err_noise:.2f}m)"
                refine_label = f"Refined (E={err_refine:.2f}m)"

                # Noisy
                ax.plot(noisy_arr_3d[:,1], noisy_arr_3d[:,0], color='red', linewidth=1.5, linestyle='--', label=noise_label)
                # Refined
                ax.plot(refined_3d[:,1], refined_3d[:,0], color='cyan', linewidth=2.0, label=refine_label, alpha=0.9)

            ax.set_title(f"{json_file} - Item {item_idx}")
            ax.set_xlabel("Y (m)")
            ax.set_ylabel("X (m)")
            
            # Legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='black', labelcolor='white', framealpha=0.5)

            out_name = f"{os.path.splitext(json_file)[0]}_item_{item_idx}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, out_name))
            plt.close(fig)
            print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
