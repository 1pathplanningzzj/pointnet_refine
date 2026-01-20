import os
import json
import glob
import numpy as np
from tqdm import tqdm

TRAIN_DATA_DIR = "/homes/zhangzijian/pointnet_refine/inference_data"

def generate_noisy_line(points_list, noise_scale=1.0):
    """
    Apply random translation, rotation (yaw), and small jitter to a polyline.
    points_list: list of dict {'x', 'y', 'z'}
    """
    pts = np.array([[p['x'], p['y'], p['z']] for p in points_list])
    if len(pts) == 0:
        return []

    # Calculate centroid for rotation
    centroid = np.mean(pts, axis=0)

    # 1. Random Rotation (Yaw) - Non-parallel noise
    # Scale rotation amount with noise_scale
    yaw_range_deg = 5.0 * noise_scale # e.g. 2.5 deg for small noise, 15 deg for large
    angle_rad = np.random.uniform(-yaw_range_deg, yaw_range_deg) * (np.pi / 180.0)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    # Rotation matrix around Z
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # 2. Global offset translation
    dx = np.random.uniform(-noise_scale, noise_scale)
    dy = np.random.uniform(-noise_scale, noise_scale)
    dz = np.random.uniform(-0.1, 0.1)
    translation = np.array([dx, dy, dz])

    # 3. Apply Transform + Per-point jitter
    jitter_scale = 0.05
    
    noisy_points = []
    
    # Vectorized rotation
    centered_pts = pts - centroid
    rotated_pts = centered_pts @ R.T
    final_pts = rotated_pts + centroid + translation

    for p in final_pts:
        jx = np.random.normal(0, jitter_scale)
        jy = np.random.normal(0, jitter_scale)
        jz = np.random.normal(0, jitter_scale/2)
        
        noisy_points.append({
            'x': p[0] + jx,
            'y': p[1] + jy,
            'z': p[2] + jz
        })
    
    return noisy_points

def process_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    for item in items:
        # Ensure we have the GT position
        gt_position = item.get('position', [])
        if not gt_position:
            continue
            
        # Generate 3 noisy candidates
        # Candidate 1: Very Small noise (approx 5-10cm)
        c1 = generate_noisy_line(gt_position, noise_scale=0.1)
        # Candidate 2: Small noise (approx 20cm)
        c2 = generate_noisy_line(gt_position, noise_scale=0.25)
        # Candidate 3: Modest noise (approx 40cm, max requested)
        c3 = generate_noisy_line(gt_position, noise_scale=0.4)
        
        item['noisy_candidates'] = [c1, c2, c3]
        
    # Overwrite the file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.json"))
    print(f"Found {len(files)} JSON files in {TRAIN_DATA_DIR}")
    
    count = 0
    for f in files:
        process_file(f)
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(files)}...", end='\r')
            
    print("\nDone augmenting data.")

if __name__ == "__main__":
    main()
