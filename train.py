import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import LaneRefineDataset
from src.model import LineRefineNet
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    CHECKPOINTS_DIR = args.checkpoints_dir
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Settings
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    DATA_ROOT = "train_data"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Data
    # Increased crop_radius to 4.0m to capture true lane markings under large noise
    dataset = LaneRefineDataset(DATA_ROOT, crop_radius=4.0, num_context_points=2048)
    dataloader = DataLoader(
        dataset,  
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=16, # Reduced to 8 for stability
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # 2. Model
    model = LineRefineNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.L1Loss() # MAE usually better for geometry than MSE
    
    print(f"Start training on {DEVICE} with {len(dataset)} samples...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_init_err = 0.0
        total_refine_err = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            context = batch['context'].to(DEVICE, non_blocking=True)
            noisy_line = batch['noisy_line'].to(DEVICE, non_blocking=True)
            target_offset = batch['target_offset'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward: Returns list [layer_1, layer_2, ..., layer_final]
            # Output shape: (L, B, M, 3)
            pred_offsets_stack = model(context, noisy_line)
            
            # Deep Supervision Loss
            loss = 0.0
            num_layers = pred_offsets_stack.shape[0]
            
            for l in range(num_layers):
                 loss += criterion(pred_offsets_stack[l], target_offset)
            
            loss = loss / num_layers
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            # ----- Geometry metrics: initial vs refined error -----
            with torch.no_grad():
                # All in centered coordinates; relative distances不受平移影响
                line_gt = noisy_line + target_offset               # (B, M, 3)
                line_pred = noisy_line + pred_offsets_stack[-1]    # (B, M, 3)

                init_err = (noisy_line - line_gt).norm(dim=-1).mean()   # 平均点到点距离
                refine_err = (line_pred - line_gt).norm(dim=-1).mean()

                total_init_err += init_err.item()
                total_refine_err += refine_err.item()
            
            if batch_idx % 5 == 0:
                 print(f"Epoch {epoch+1}, Batch {batch_idx}, "
                       f"Loss: {loss.item():.6f}, "
                       f"InitErr: {init_err.item():.4f}m, "
                       f"RefineErr: {refine_err.item():.4f}m")

        avg_loss = total_loss / num_batches
        avg_init_err = total_init_err / num_batches
        avg_refine_err = total_refine_err / num_batches
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Loss: {avg_loss:.6f}, "
              f"InitErr: {avg_init_err:.4f}m, "
              f"RefineErr: {avg_refine_err:.4f}m")
        
        # Save checkpoint occasionally
        if (epoch+1) % 5 == 0:
            save_path = os.path.join(CHECKPOINTS_DIR, f"refine_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

    print("Training Complete.")
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_model.pth"))

if __name__ == "__main__":
    main()
