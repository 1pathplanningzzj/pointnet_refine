import torch
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
from scipy.spatial import KDTree

# Ensure src is importable
sys.path.append(os.getcwd())

from src.dataset import resample_polyline, load_pcd_data, weighted_sampling
from src.model import LineRefineNet

# Config
DATA_DIR = "./vma_infer_data"
MODEL_PATH = "checkpoints/refine_model_epoch_50.pth"
OUTPUT_HTML_DIR = "./vma_inference_vis_whole_scene"
NUM_VIS_SAMPLES = 50  # Number of SCENES to visualize
NUM_LINE_POINTS = 32
NUM_CONTEXT_POINTS = 1024
CROP_RADIUS = 0.3  # Reduced to ignore distant noise

os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)

def crop_gt_to_pred_range(gt_points, pred_points):
    """
    Crops the GT polyline to the segment closest to the start and end of the prediction.
    Uses nearest neighbor search to find corresponding start/end points on GT.
    Also handles direction alignment (flips GT crop if necessary to minimize ADE).
    """
    if len(gt_points) < 2 or len(pred_points) < 2:
        return gt_points
    
    # Prediction start and end
    pred_start = pred_points[0]
    pred_end = pred_points[-1]
    
    # Distances to all GT points
    dists_start = np.linalg.norm(gt_points - pred_start, axis=1)
    dists_end = np.linalg.norm(gt_points - pred_end, axis=1)
    
    idx_start = np.argmin(dists_start)
    idx_end = np.argmin(dists_end)
    
    # Determine crop indices
    idx_min = min(idx_start, idx_end)
    idx_max = max(idx_start, idx_end)
    
    # Ensure we get a valid segment (at least 2 points if possible)
    if idx_min == idx_max:
        idx_min = max(0, idx_min - 1)
        idx_max = min(len(gt_points)-1, idx_max + 1)
        
    cropped_gt = gt_points[idx_min : idx_max + 1]
    
    if len(cropped_gt) < 2:
        return gt_points
        
    # Check direction: Does the crop run Start->End or End->Start?
    # We compare distances of endpoints to ensure alignment with Pred
    # Case 1: GT[0] matches Pred[0] -> Dist is small
    d_normal = np.linalg.norm(cropped_gt[0] - pred_start) + np.linalg.norm(cropped_gt[-1] - pred_end)
    # Case 2: GT[0] matches Pred[-1] -> Dist is small (Reverse)
    d_reverse = np.linalg.norm(cropped_gt[0] - pred_end) + np.linalg.norm(cropped_gt[-1] - pred_start)
    
    if d_reverse < d_normal:
        cropped_gt = cropped_gt[::-1]
        
    return cropped_gt

def process_single_line(model, pcd_points, noisy_line_raw, device):
    """
    Runs model inference for a single line.
    Returns the REFINED line in ORIGINAL coordinates.
    """
    # 1. Resample Noisy Line
    noisy_points = resample_polyline(noisy_line_raw, NUM_LINE_POINTS) # (32, 3)
    
    # 2. Crop Context (Same logic as Dataset for consistency)
    if len(pcd_points) > 0 and len(noisy_points) > 0:
        # Use dense points for cropping to eliminate gaps
        noisy_lines_dense = resample_polyline(noisy_line_raw, 200)
        tree = KDTree(noisy_lines_dense)
        dists, _ = tree.query(pcd_points[:, :3])
        mask = dists < CROP_RADIUS
        context_points = pcd_points[mask]
    else:
        context_points = np.zeros((0, 4))
        
    # 3. Sampling
    context_points = weighted_sampling(context_points, noisy_points, NUM_CONTEXT_POINTS) # (1024, 4)
    if len(context_points) < NUM_CONTEXT_POINTS:
        # Pad with zeros if empty or not enough
        pad = np.zeros((NUM_CONTEXT_POINTS - len(context_points), 4))
        context_points = np.vstack([context_points, pad])

    # 4. Normalize
    center = np.mean(noisy_points, axis=0)
    context_xyz = context_points[:, :3] - center
    context_int = context_points[:, 3:4]
    noisy_centered = noisy_points - center
    
    # 5. Prepare Tensor
    input_pcd = np.hstack([context_xyz, context_int]) # (1024, 4)
    
    # Add batch dim
    tensor_pcd = torch.from_numpy(input_pcd).float().unsqueeze(0).to(device) # (1, 1024, 4)
    tensor_noisy = torch.from_numpy(noisy_centered).float().unsqueeze(0).to(device) # (1, 32, 3)
    
    # 6. Forward
    model.eval()
    with torch.no_grad():
        # Output is (L, B, M, 3)
        offsets_stack = model(tensor_pcd, tensor_noisy) 
        # Take the LAST layer output
        final_offset = offsets_stack[-1] # (1, 32, 3)
        
    # 7. Restore Coordinates
    pred_offset_np = final_offset[0].cpu().numpy()
    
    # Refined = (Noisy - C) + Offset + C = Noisy + Offset
    # The model predicts accumulation offset from initial noisy line
    refined_line = noisy_points + pred_offset_np
    
    return refined_line, noisy_points

def main():
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LineRefineNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Model file {MODEL_PATH} not found!")
        return

    # 2. Find Files
    json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    if len(json_files) == 0:
        print("No JSON files found in data dir.")
        return

    print(f"Found {len(json_files)} scenes. Processing top {NUM_VIS_SAMPLES}...")

    for i, json_file in enumerate(json_files):
        if i >= NUM_VIS_SAMPLES:
            break
            
        json_path = os.path.join(DATA_DIR, json_file)
        pcd_path = json_path.replace('.json', '.pcd')
        
        if not os.path.exists(pcd_path):
            print(f"PCD not found for {json_file}, skipping.")
            continue
            
        print(f"Processing Scene {i}: {json_file}")
        
        # Load Raw Data
        full_pcd = load_pcd_data(pcd_path) # (N, 4)
        with open(json_path, 'r') as f:
            data = json.load(f)
            items = data.get('items', [])
            
        # Initialize Figure
        fig = go.Figure()
        
        # --- Analyze Intensity Stats ---
        intensities = full_pcd[:, 3]
        if len(intensities) > 0:
            p50 = np.percentile(intensities, 50)
            p90 = np.percentile(intensities, 90)
            p99 = np.percentile(intensities, 99)
            print(f"    [Intensity Stats] Min: {intensities.min():.1f}, P50: {p50:.1f}, P90: {p90:.1f}, P99: {p99:.1f}, Max: {intensities.max():.1f}")

        # --- Plot Background PCD ---
        # Subsample for display (e.g. limits to avoid browser crash, but 20k is too sparse)
        # Increased to 200k for denser visualization
        MAX_POINTS = 400000
        if len(full_pcd) > MAX_POINTS:
            indices = np.random.choice(len(full_pcd), MAX_POINTS, replace=False)
            plot_pcd = full_pcd[indices]
        else:
            plot_pcd = full_pcd
            
        fig.add_trace(go.Scatter3d(
            x=plot_pcd[:, 0], y=plot_pcd[:, 1], z=plot_pcd[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=plot_pcd[:, 3], # Intensity
                colorscale='Jet',
                cmin=1, 
                cmax=40,
                showscale=True,
                colorbar=dict(title="Intensity"),
                opacity=0.8
            ),
            name='Scene Point Cloud'
        ))

        # --- Plot Context GT Lines (Once per Scene) ---
        # Extract from the first item if available (all items share the same context lines)
        if len(items) > 0:
            context_gts_all = items[0].get('context_lines', [])
            for c_gt in context_gts_all:
                pts = np.array([[p['x'], p['y'], p['z']] for p in c_gt])
                fig.add_trace(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    mode='lines',
                    line=dict(color='orange', width=2), # Changed to orange
                    showlegend=False, 
                    opacity=0.4,
                    name='Context GT'
                ))
        
        # --- Process Each VMA Line ---
        for item_idx, item in enumerate(items):
            # 1. Get Noisy Input (VMA)
            # We iterate all noisy candidates (usually just 1 for training data gen, but list structure)
            noisy_candidates_list = item.get('noisy_candidates', [])
            
            # 2. Get Target GT (if exists)
            gt_pts_list = item.get('position', [])
            gt_points = np.array([[p['x'], p['y'], p['z']] for p in gt_pts_list]) if gt_pts_list else np.empty((0,3))
            
            # Calculate Resampled GT for Metric
            gt_32 = np.empty((0,3))
            if len(gt_points) > 1:
                gt_32 = resample_polyline(gt_points, NUM_LINE_POINTS)

            # --- Visualization: Target GT (Green) ---
            if len(gt_points) > 1:
                fig.add_trace(go.Scatter3d(
                    x=gt_points[:,0], y=gt_points[:,1], z=gt_points[:,2],
                    mode='lines', # solid
                    line=dict(color='green', width=2),
                    name=f'GT Line {item_idx}'
                ))

            # --- Inference & Visualization: Noisy & Refined ---
            for n_idx, noisy_raw_pts in enumerate(noisy_candidates_list):
                noisy_arr = np.array([[p['x'], p['y'], p['z']] for p in noisy_raw_pts])
                
                if len(noisy_arr) < 2:
                    continue
                    
                # Run Inference
                refined_arr, resampled_noisy = process_single_line(model, full_pcd, noisy_arr, device)
                
                # Calculate Metrics (ADE: Average Displacement Error)
                metric_info = ""
                if len(gt_points) > 1:
                    # Crop GT to match Prediction Range for fair metric comparison
                    cropped_gt_for_metric = crop_gt_to_pred_range(gt_points, resampled_noisy)
                    if len(cropped_gt_for_metric) > 1:
                        gt_32_metric = resample_polyline(cropped_gt_for_metric, NUM_LINE_POINTS)
                        
                        diff_noisy = np.linalg.norm(resampled_noisy - gt_32_metric, axis=1)
                        ade_noisy = np.mean(diff_noisy)
                        
                        diff_refined = np.linalg.norm(refined_arr - gt_32_metric, axis=1)
                        ade_refined = np.mean(diff_refined)
                        
                        improvement = ade_noisy - ade_refined
                        metric_info = f"<br>ADE: {ade_noisy:.2f}m -> {ade_refined:.2f}m (Imp: {improvement:.2f}m)"
                        print(f"    Line {item_idx}-{n_idx}: ADE {ade_noisy:.3f} -> {ade_refined:.3f}")

                # Plot Noisy (Red Dash)
                fig.add_trace(go.Scatter3d(
                    x=resampled_noisy[:,0], y=resampled_noisy[:,1], z=resampled_noisy[:,2],
                    mode='lines',
                    line=dict(color='red', width=1.5, dash='dash'),
                    name=f'Noisy {item_idx}{metric_info}'
                ))
                
                # Plot Refined (Magenta)
                fig.add_trace(go.Scatter3d(
                    x=refined_arr[:,0], y=refined_arr[:,1], z=refined_arr[:,2],
                    mode='lines+markers',
                    marker=dict(size=1.5, color='magenta'),
                    line=dict(color='magenta', width=2),
                    name=f'Refined {item_idx}'
                ))

        # Layout
        fig.update_layout(
            title=f"Whole Scene Refinement - {json_file}",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Save
        out_path = os.path.join(OUTPUT_HTML_DIR, f"scene_{json_file.replace('.json', '.html')}")
        fig.write_html(out_path)
        print(f"Saved {out_path}")

    print("All done.")

if __name__ == "__main__":
    main()
