import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import PointNetRefine
from src.dataset import RefineDataset, generate_mock_training_data

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train():
    # --- Config ---
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WORK_DIR = "work_dirs/pointnet_refine_v1"
    os.makedirs(WORK_DIR, exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=WORK_DIR)
    
    logger.info(f"Training on {DEVICE}")

    # --- Data Generation (Using the Trick) ---
    logger.info("Generating mock data using 'GT + Noise' strategy...")
    train_raw = generate_mock_training_data(2000)
    val_raw = generate_mock_training_data(400)

    train_ds = RefineDataset(train_raw)
    val_ds = RefineDataset(val_raw)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Model & Opt ---
    model = PointNetRefine(input_dim=3, output_dim=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ========== Loss Function (回归头) ==========
    # 这不是"检测头"，而是回归头（Regression Head）
    # 任务：回归一个连续的标量值（offset），不是分类或检测框
    # SmoothL1Loss: 结合了 L1 和 L2 的优点，在 0 附近更平滑，利于收敛
    # 也可以换成 nn.L1Loss() 或 nn.MSELoss()
    criterion = nn.SmoothL1Loss() 

    best_val_loss = float('inf')

    logger.info("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss_meter = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for points, target in pbar:
            # points: [B, 3, N] - 局部坐标系下的点云（已经过 canonicalization）
            # target: [B, 1] - GT offset（标量，单位：米）
            points, target = points.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            # output: [B, 1] - 网络预测的 offset
            output = model(points)
            
            # ========== Loss 计算位置 ==========
            # 这里计算的是回归损失：预测 offset vs GT offset
            # loss shape: scalar (单个数值)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss_meter.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = np.mean(train_loss_meter)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss_meter = []
        l1_error_meter = []
        
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(DEVICE), target.to(DEVICE)
                output = model(points)
                
                loss = criterion(output, target)
                val_loss_meter.append(loss.item())
                l1_error_meter.append(torch.abs(output - target).mean().item())
        
        avg_val_loss = np.mean(val_loss_meter)
        avg_l1 = np.mean(l1_error_meter)

        # Logging
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | L1 Error: {avg_l1:.4f} m")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Metric/L1_Error', avg_l1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(WORK_DIR, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved best model to {ckpt_path}")
            
        # Save latst
        torch.save(model.state_dict(), os.path.join(WORK_DIR, 'latest.pth'))

    # Save
    logger.info("Training Finished.")
    writer.close()

if __name__ == "__main__":
    train()
