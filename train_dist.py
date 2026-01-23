import os
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from src.dataset import LaneRefineDataset
from src.model import LineRefineNet
import torch.nn.functional as F
import logging
import datetime

def setup():
    # Initialize the process group
    # nccl is recommended for NVIDIA GPUs
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def get_logger(rank, save_dir="work_dirs/train_logs"):
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Create formatters
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    
    # Stream Handler (Print to console) - All ranks or just rank 0? 
    # Usually just rank 0 to keep console clean, or different prefixes.
    # We will let Rank 0 print to console.
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler - Rank 0 Only
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(save_dir, f"train_{timestamp}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    CHECKPOINTS_DIR = args.checkpoints_dir

    # 0. DDP Setup
    # These environment variables are set by torchrun
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Please run this script with 'torchrun'.")
        
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Logger setup
    logger = get_logger(global_rank)

    # Settings
    # Batch size here is PER GPU. Total batch size = 32 * number_of_gpus
    BATCH_SIZE_PER_GPU = 32 
    EPOCHS = 100
    LR = 0.001 # Can scale with world_size if needed
    DATA_ROOT = "train_data"
    
    # 1. Data
    # crop_radius needs to match single GPU training -> 0.3
    dataset = LaneRefineDataset(DATA_ROOT, crop_radius=0.3)
    
    # DistributedSampler handles data splitting across GPUs
    sampler = DistributedSampler(dataset, shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE_PER_GPU, 
        shuffle=False, # Important: shuffle must be False when using DistributedSampler
        num_workers=4, # Workers per GPU
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2,
        sampler=sampler
    )
    
    # 2. Model
    model = LineRefineNet().to(device)
    
    # Wrap model with DDP
    # device_ids tells DDP which GPU this process uses
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.L1Loss()
    
    if global_rank == 0:
        logger.info(f"Start DDP training on {world_size} GPUs. Total Batch Size: {BATCH_SIZE_PER_GPU * world_size}")
        if not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    model.train()
    for epoch in range(EPOCHS):
        # Important: set epoch for sampler to ensure different shuffles each epoch
        sampler.set_epoch(epoch)
        
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            context = batch['context'].to(device, non_blocking=True)
            noisy_line = batch['noisy_line'].to(device, non_blocking=True)
            target_offset = batch['target_offset'].to(device, non_blocking=True)
            
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
            
            if global_rank == 0 and batch_idx % 10 == 0:
                 logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        
        # Only print and save on main process
        if global_rank == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
        
            # Save checkpoint occasionally
            if (epoch+1) % 10 == 0:
                # Use model.module when saving DDP model to get the underlying model weights
                save_path = os.path.join(CHECKPOINTS_DIR, f"refine_model_epoch_{epoch+1}.pth")
                torch.save(model.module.state_dict(), save_path)

    if global_rank == 0:
        logger.info("Training Complete.")
        torch.save(model.module.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
    
    cleanup()

if __name__ == "__main__":
    main()
