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
DATA_DIR = "./vma_infer_data"
MODEL_PATH = "checkpoints/refine_model_epoch_100.pth"
OUTPUT_HTML_DIR = "./vma_inference_vis"
NUM_VIS_SAMPLES = 100  # Visualize all samples

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
                name='Target GT'
            ))

            # 3b. Plot ALL Context GT Lines (Yellow/Grey)
            if 'context_lines' in batch:
                # batch['context_lines'] is a LIST of tensors/arrays
                # But DataLoader collate_fn might have stacked them if they were same length 
                # OR it's a list if using custom collate. 
                # Standard default_collate fails on lists of variable size arrays. 
                # But I added it to the dict. `dataset.py` returns list of numpy arrays.
                # Default collate usually crashes if elements are lists of varying len arrays.
                # If they are just lists, maybe it works?
                # Actually, wait. dataset returns a DICT. 
                # If 'context_lines' is a list of arrays (variable size), default_collate MIGHT fail or turn it into list of tensors (if same size).
                # But here batch size is 1. So it's probably [ [Line1, Line2] ]
                
                # Let's inspect variable
                ctx_lines = batch['context_lines']
                # Since batch_size=1, ctx_lines should be a list of length 1 (the batch).
                # Inside that, it depends on how collate handled the list of arrays.
                # If collate didn't touch it, it is [ sample0_lines ]
                
                # Safely trying to iterate
                try:
                    # Depending on PyTorch version and collate:
                    # If it's a list of lists of tensors:
                    sample_lines = ctx_lines
                    if isinstance(sample_lines, list):
                        # Flatten? No, structure is [Batch_Sample0_List, Batch_Sample1_List...]
                        # But here batch is 1.
                        # Actually default_collate is smart about lists.
                        # It transposes. [ [L1a, L1b], [L2a, L2b] ] -> [ [L1a, L2a], [L1b, L2b] ]
                        # BUT lines are VARIABLE length list. So it might have errored out or left it?
                        # Let's just assume we can get it from the dataset directly if needed, 
                        # but let's try to parse what we have.
                        
                        # Assuming batch_size=1, it might be just the list for sample 0
                        # But let's look at visualization logic.
                        # center = batch['center'][0] -> we need to ADD center back for visualization? 
                        # No, noisy_np is already "centered" by dataset?
                        # Wait, dataset returns `noisy_line_centered`. 
                        # So `noisy_np` IS centered. 
                        # But `gt_line_np` = `noisy` + `gt_offset`, so it is also centered.
                        # `pred_line_np` is also centered.
                        # So to plot Context Lines, they must be centered too.
                        # Dataset returns `context_lines_norm`. Good.
                        
                        # Getting sample 0 lines:
                        # Iterate everything in ctx_lines and try to plot
                        for idx, line_tensor in enumerate(ctx_lines):
                            # In DataLoader with batch_size=1, sometimes it unzips logic
                            # It's safer to access the original dataset if we want consistent behavior
                            # but let's try.
                            pass
                except:
                    pass
                
                # The robust way with batch_size=1 is:
                current_lines = []
                # Check based on type
                if isinstance(ctx_lines, list):
                     # Likely [Line1_Batch, Line2_Batch...] if lines were consistent count?
                     # No, if variable count, PyTorch 1.x usually leaves it as list of inputs?
                     pass
                
                # Let's simplify: Just iterate `ctx_lines` and see if they look like tensors
                flat_lines = []
                
                # Recursively flatten lists
                def extract_tensors(item):
                    if torch.is_tensor(item):
                        if item.dim() >= 2: flat_lines.append(item[0].cpu().numpy()) # Take batch 0
                    elif isinstance(item, list):
                        for sub in item: extract_tensors(sub)
                
                extract_tensors(ctx_lines)
                
                for ln_idx, ln_np in enumerate(flat_lines):
                     if ln_np.shape[-1] != 3: continue
                     
                     fig.add_trace(go.Scatter3d(
                        x=ln_np[:,0], y=ln_np[:,1], z=ln_np[:,2],
                        mode='lines',
                        line=dict(color='yellow', width=2),
                        name=f'Context GT {ln_idx}'
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
