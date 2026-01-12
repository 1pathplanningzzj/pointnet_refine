import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import PointNetRefine
from src.dataset import RefineDataset, generate_mock_training_data
import numpy as np

def train():
    # --- Config ---
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data Generation (Using the Trick) ---
    print("Generating mock data using 'GT + Noise' strategy...")
    # 使用我们在 dataset.py 里写的逻辑
    train_raw = generate_mock_training_data(1000)
    val_raw = generate_mock_training_data(200)

    train_ds = RefineDataset(train_raw)
    val_ds = RefineDataset(val_raw)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model & Opt ---
    # output_dim = 1 (Lateral Offset)
    model = PointNetRefine(input_dim=3, output_dim=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss: SmoothL1Loss 
    criterion = nn.SmoothL1Loss() 

    print("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for points, target in train_loader:
            points, target = points.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(points)
            
            # target shape [B, 1], output shape [B, 1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss = 0
        val_l1_error = 0
        model.eval()
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(DEVICE), target.to(DEVICE)
                output = model(points)
                val_loss += criterion(output, target).item()
                val_l1_error += torch.abs(output - target).mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_l1 = val_l1_error / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | L1 Error: {avg_l1:.4f} m")

    # Save
    torch.save(model.state_dict(), "pointnet_refine.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
