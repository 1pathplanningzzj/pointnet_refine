import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import LaneRefineDataset
from src.model import LineRefineNet
import torch.nn.functional as F

def main():
    # Settings
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    DATA_ROOT = "train_data"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Data
    dataset = LaneRefineDataset(DATA_ROOT, crop_radius=1.0)
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
    
    print(f"Start training on {DEVICE} with {len(dataset)} samples...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            context = batch['context'].to(DEVICE, non_blocking=True)
            noisy_line = batch['noisy_line'].to(DEVICE, non_blocking=True)
            target_offset = batch['target_offset'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward
            pred_offset = model(context, noisy_line)
            
            # Loss: MSE between predicted offset and target offset
            loss = F.mse_loss(pred_offset, target_offset)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                 print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint occasionally
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/refine_model_epoch_{epoch+1}.pth")

    print("Training Complete.")
    torch.save(model.state_dict(), "checkpoints/best_model.pth")

if __name__ == "__main__":
    main()
