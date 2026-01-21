import torch
from torch.utils.data import DataLoader
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Ensure src is importable
sys.path.append(os.getcwd())

from src.dataset import LaneRefineDataset
from src.model import LineRefineNet

# Config
DATA_DIR = "./train_data"
MODEL_PATH = "checkpoints/refine_model_epoch_50.pth"
OUTPUT_HTML_DIR = "./inference_vis_epoch10"
NUM_VIS_SAMPLES = 15  # Visualize 10 samples

os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)

def main():
    # 1. Load Data
    print(f"Loading inference dataset from {DATA_DIR}...")
    dataset = LaneRefineDataset(DATA_DIR)
    
    if len(dataset) == 0:
        print("Dataset is empty. Run generate_inference_data.py and augment_inference_data.py first.")
        return

    # Use batch_size=1 so we can process sample by sample simply
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LineRefineNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Model file {MODEL_PATH} not found!")
        return
        
    model.eval()

    print(f"Starting inference visualization for {NUM_VIS_SAMPLES} random samples...")
    
    vis_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if vis_count >= NUM_VIS_SAMPLES:
                break
                
            # Inputs
            pcd = batch['context'].to(device)       # (B, N, 4)
            noisy_line = batch['noisy_line'].to(device) # (B, M, 3)
            gt_offsets = batch['target_offset'].to(device) # (B, M, 3)
            
            # Forward Pass
            pred_offsets = model(pcd, noisy_line) # (B, M, 3)
            
            # Move to CPU for plotting
            pcd_np = pcd[0].cpu().numpy()          # (N, 4)
            noisy_np = noisy_line[0].cpu().numpy() # (M, 3)
            gt_off_np = gt_offsets[0].cpu().numpy()# (M, 3)
            pred_off_np = pred_offsets[0].cpu().numpy()# (M, 3)
            
            # Reconstruct Absolute Lines
            gt_line_np = noisy_np + gt_off_np
            pred_line_np = noisy_np + pred_off_np
            
            # --- Visualization using Plotly ---
            fig = go.Figure()
            
            # 1. Plot Context Point Cloud
            # Subsample if too dense for web viz
            if pcd_np.shape[0] > 5000:
                indices = np.random.choice(pcd_np.shape[0], 5000, replace=False)
                plot_pcd = pcd_np[indices]
            else:
                plot_pcd = pcd_np
                
            fig.add_trace(go.Scatter3d(
                x=plot_pcd[:, 0], y=plot_pcd[:, 1], z=plot_pcd[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=plot_pcd[:, 3], # Intensity
                    colorscale='Viridis',
                    cmin=0, cmax=30, # Intensity scaling matching previous script
                    opacity=0.6
                ),
                name='Context PCD'
            ))
            
            # 2. Plot Noisy Input (Red Dashed)
            fig.add_trace(go.Scatter3d(
                x=noisy_np[:,0], y=noisy_np[:,1], z=noisy_np[:,2],
                mode='lines+markers',
                marker=dict(size=3, color='red'),
                line=dict(color='red', width=3, dash='dash'),
                name='Noisy Input'
            ))
            
            # 3. Plot Ground Truth (Green Solid)
            fig.add_trace(go.Scatter3d(
                x=gt_line_np[:,0], y=gt_line_np[:,1], z=gt_line_np[:,2],
                mode='lines+markers',
                marker=dict(size=3, color='green'),
                line=dict(color='green', width=5),
                name='Ground Truth'
            ))
            
            # 4. Plot Refined Output (Blue Solid)
            fig.add_trace(go.Scatter3d(
                x=pred_line_np[:,0], y=pred_line_np[:,1], z=pred_line_np[:,2],
                mode='lines+markers',
                marker=dict(size=4, color='cyan'),
                line=dict(color='cyan', width=4),
                name='Refined Prediction'
            ))
            
            # Calculate simple metric for title
            l1_loss = np.mean(np.abs(pred_line_np - gt_line_np))
            input_diff = np.mean(np.abs(noisy_np - gt_line_np))
            
            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)'
                ),
                title=f"Sample {vis_count} - Input Err: {input_diff:.3f}m -> Refined Err: {l1_loss:.3f}m",
                margin=dict(r=0, l=0, b=0, t=40)
            )
            
            save_path = os.path.join(OUTPUT_HTML_DIR, f"inference_result_{vis_count}.html")
            fig.write_html(save_path)
            print(f"Saved visualization to {save_path}")
            
            vis_count += 1
            
    print("Inference visualization complete.")

if __name__ == "__main__":
    main()
