import os
import json
import glob
import numpy as np
from tqdm import tqdm

TRAIN_DATA_DIR = "/homes/zhangzijian/pointnet_refine/train_data"

def generate_noisy_line(points_list, noise_scale=1.0):
    """
    Apply random translation, rotation (yaw), and small jitter to a polyline.
    points_list: list of dict {'x', 'y', 'z'}
    noise_scale: 噪声量级（米），控制平移和旋转的幅度
    """
    pts = np.array([[p['x'], p['y'], p['z']] for p in points_list])
    if len(pts) == 0:
        return []

    # Calculate centroid for rotation
    centroid = np.mean(pts, axis=0)

    # 1. Random Rotation (Yaw) - 减小旋转角度，更符合实际VMA检测误差
    # 对于小噪声（2-10cm），旋转角度应该很小（<1度）
    yaw_range_deg = 2.0 * noise_scale  # 2cm->0.04度, 10cm->0.2度
    angle_rad = np.random.uniform(-yaw_range_deg, yaw_range_deg) * (np.pi / 180.0)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    # Rotation matrix around Z
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # 2. Global offset translation - 主要噪声来源
    dx = np.random.uniform(-noise_scale, noise_scale)
    dy = np.random.uniform(-noise_scale, noise_scale)
    dz = np.random.uniform(-0.02, 0.02)  # Z方向噪声减小到±2cm
    translation = np.array([dx, dy, dz])

    # 3. Apply Transform + Per-point jitter - 减小逐点抖动
    # 逐点抖动应该比整体噪声小，设为noise_scale的1/4
    jitter_scale = noise_scale * 0.25

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
            
        # Generate 4 noisy candidates with smaller noise levels
        # Candidate 1: 2cm noise
        c1 = generate_noisy_line(gt_position, noise_scale=0.02)
        # Candidate 2: 3cm noise
        c2 = generate_noisy_line(gt_position, noise_scale=0.03)
        # Candidate 3: 5cm noise
        c3 = generate_noisy_line(gt_position, noise_scale=0.05)
        # Candidate 4: 10cm noise
        c4 = generate_noisy_line(gt_position, noise_scale=0.10)

        item['noisy_candidates'] = [c1, c2, c3, c4]
        
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
