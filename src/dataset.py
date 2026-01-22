import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree

def resample_polyline(points, num_points=32):
    """
    Resample a polyline to a fixed number of points using linear interpolation.
    points: (N, 3)
    """
    if len(points) < 2:
        return np.zeros((num_points, 3))

    # Calculate cumulative distance
    dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    
    # Generate target distances
    target_dists = np.linspace(0, total_len, num_points)
    
    # Interpolate
    new_points = np.zeros((num_points, 3))
    for i in range(3):
        new_points[:, i] = np.interp(target_dists, cum_dists, points[:, i])
        
    return new_points

def load_pcd_data(pcd_path):
    # Optimized loader using numpy
    try:
        # Assuming 10 lines of header as per generate_train_data.py
        # Check first line to be safe or just try skiprows=11
        # To be robust, we read until DATA ascii
        with open(pcd_path, 'rb') as f:
            for i, line in enumerate(f):
                if line.strip().startswith(b'DATA'):
                    skip = i + 1
                    break
        
        data = np.loadtxt(pcd_path, skiprows=skip, dtype=np.float32)
        return data
    except Exception as e:
        print(f"Error loading {pcd_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)

def weighted_sampling(context_points, noisy_points, num_samples=1024):
    """
    Weighted sampling based on distance to the lane line.
    Points closer to the line have higher probability of being sampled.
    """
    if len(context_points) <= num_samples:
        if len(context_points) == 0:
            return np.zeros((num_samples, 4))
        # If fewer points than needed, sample with replacement to fill up
        choice = np.random.choice(len(context_points), num_samples, replace=True)
        return context_points[choice]

    # Calculate distance from each context point to the nearest noisy line point
    tree = KDTree(noisy_points)
    distances, _ = tree.query(context_points[:, :3])

    # Weights decay exponentially with distance
    # Scale 0.3 means weight drops to ~36% at 0.3m. Tighter focus for accurate refinement.
    weights = np.exp(-distances / 0.3)
    
    w_sum = weights.sum()
    if w_sum < 1e-6:
        # Fallback to uniform if weights are too small
        weights = None
    else:
        weights = weights / w_sum
        # Fix precision issues: ensure sum is exactly 1.0
        # By re-normalizing, we usually get close enough, but numpy can be picky.
        # A robust way is to subtract the diff from the largest element or re-normalize again.
        # However, passing p to choice requires strict sum 1.
        # Let's use a safe normalization:
        weights = weights / np.sum(weights)
        # Even safer: clip to range and re-norm
        
    choice = np.random.choice(len(context_points), num_samples, replace=False, p=weights)
    return context_points[choice]

class LaneRefineDataset(Dataset):
    def __init__(self, data_root, num_line_points=32, num_context_points=1024, crop_radius=0.6, split='train'):
        self.data_root = data_root
        self.files = [f for f in os.listdir(data_root) if f.endswith('.json')]
        self.num_line_points = num_line_points
        self.num_context_points = num_context_points
        self.crop_radius = crop_radius
        
        # Cache for PCD data to avoid repeated slow IO
        self.pcd_cache = {}
        
        self.samples = []
        self._prepare_index()
        
    def _prepare_index(self):
        print("Indexing dataset...")
        for json_file in self.files:
            json_path = os.path.join(self.data_root, json_file)
            pcd_path = json_path.replace('.json', '.pcd')
            
            if not os.path.exists(pcd_path):
                continue
                
            # We don't load heavy data here, just store metadata to load lazily
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            for i, item in enumerate(data.get('items', [])):
                if 'noisy_candidates' not in item or 'position' not in item:
                    continue
                
                # Each noisy candidate is a training sample
                for noise_idx, _ in enumerate(item['noisy_candidates']):
                    self.samples.append({
                        'pcd_path': pcd_path,
                        'json_path': json_path,
                        'item_idx': i,
                        'noise_idx': noise_idx
                    })
        print(f"Indexed {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 1. Load Full PCD (with caching)
        pcd_path = sample_info['pcd_path']
        if pcd_path in self.pcd_cache:
            pcd_points = self.pcd_cache[pcd_path]
        else:
            pcd_points = load_pcd_data(pcd_path)
            # Limit cache size to avoid OOM (e.g. 50 files ~ 1GB)
            if len(self.pcd_cache) < 20: 
                self.pcd_cache[pcd_path] = pcd_points
        
        # 2. Load Item Data
        with open(sample_info['json_path'], 'r') as f:
            data = json.load(f)
            item = data['items'][sample_info['item_idx']]
            
        gt_points_raw = np.array([[p['x'], p['y'], p['z']] for p in item['position']])
        noisy_points_raw = np.array([[p['x'], p['y'], p['z']] for p in item['noisy_candidates'][sample_info['noise_idx']]])

        # Create context lines list
        context_gt_lines = []
        if 'context_lines' in item:
            for l in item['context_lines']:
                 if len(l) > 0:
                     context_gt_lines.append(np.array([[p['x'], p['y'], p['z']] for p in l]))

        # 3. Resample Lines to fixed size
        gt_points = resample_polyline(gt_points_raw, self.num_line_points)
        noisy_points = resample_polyline(noisy_points_raw, self.num_line_points)
        
        # 4. Crop Context Points (Cylindrical / Distance-based)
        # Instead of simple sphere check around centroid, check distance to the polyline itself
        if len(pcd_points) > 0 and len(noisy_points) > 0:
            # Build KDTree for the noisy line points
            tree = KDTree(noisy_points)
            
            # Query the distance from every PCD point to the nearest line point
            # pcd_points[:, :3] shape: (N, 3)
            # tree contains M points (e.g. 32)
            dists, _ = tree.query(pcd_points[:, :3])
            
            # Keep points within crop_radius of the *line*
            mask = dists < self.crop_radius
            context_points = pcd_points[mask]
        else:
            context_points = np.zeros((0, 4))
            
        # 5. Weighted Sampling
        context_points = weighted_sampling(context_points, noisy_points, self.num_context_points)
            
        # 6. Normalization
        # Center everything to the noisy line centroid
        # This makes the network translation invariant
        center = np.mean(noisy_points, axis=0) # Re-calculate center for normalization
        context_xyz = context_points[:, :3] - center
        context_intensity = context_points[:, 3:4]
        
        noisy_line_centered = noisy_points - center
        gt_line_centered = gt_points - center
        
        # Input: [Context(N, 4), NoisyLine(M, 3)]
        # Target: Offset(M, 3) -> GT - Noisy
        target_offset = gt_line_centered - noisy_line_centered
        
        # Normalize context lines too
        context_lines_norm = []
        for l in context_gt_lines:
            context_lines_norm.append(l - center)

        return {
            'context': torch.from_numpy(np.hstack([context_xyz, context_intensity])).float(), # (N, 4)
            'noisy_line': torch.from_numpy(noisy_line_centered).float(), # (M, 3)
            'target_offset': torch.from_numpy(target_offset).float(), # (M, 3)
            'context_lines': context_lines_norm # List of arrays
        }
