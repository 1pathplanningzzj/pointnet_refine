import os
import json
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import struct

# --- Configuration ---
SOURCE_ROOT = "/homes/zhangzijian/pointnet_refine/data/new_train_data"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/train_data"
SEGMENT_LEN = 50.0  # meters
STRIDE = 25.0       # meters

# --- Loaders ---

def load_poses(pose_dir):
    print(f"Loading poses from {pose_dir}...")
    poses = []
    files = glob.glob(os.path.join(pose_dir, "*.json"))
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
                # Robust timestamp extraction
                if 'ts' in data:
                    ts = str(data['ts'])
                else:
                    # Fallback to filename
                    ts = os.path.splitext(os.path.basename(f))[0]

                # Use filename as unique ID for output files
                filename_ts = os.path.splitext(os.path.basename(f))[0]
                
                poses.append({
                    'ts': ts, 
                    'filename_ts': filename_ts,
                    'x': data['x'],
                    'y': data['y'],
                    'z': data['z'],
                    'q': [data['qx'], data['qy'], data['qz'], data['qw']] 
                })
        except Exception as e:
            print(f"Error loading pose {f}: {e}")
            continue
    
    # Sort by timestamp (string) or x
    # Sorting by filename_ts usually works for sequential timestamps
    poses.sort(key=lambda p: p['filename_ts'])
    print(f"Loaded {len(poses)} poses.")
    return poses

def load_pcd_fast(pcd_path):
    print(f"Loading {pcd_path}...")
    if not os.path.exists(pcd_path):
        print(f"PCD not found: {pcd_path}")
        return np.zeros((0, 4), dtype=np.float32)

    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().strip()
            header.append(line)
            if line.startswith(b'DATA'):
                break
        
        # Load binary
        try:
            buffer = f.read()
            # Standard PCD often uses x y z intensity (4 floats or 3 floats + 1 int)
            # We assume consistency with project history (4 floats or mix)
            # If default read fails, we try fallback.
            
            # Legacy logic: 
            dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u2')])
            
            num_points_bytes = len(buffer)
            if num_points_bytes % 14 == 0:
                 data = np.frombuffer(buffer, dtype=dt)
                 arr = np.column_stack((data['x'], data['y'], data['z'], data['intensity'].astype(np.float32)))
            else:
                 # Backup: maybe it's all floats (xyzi) -> 16 bytes
                 dt_f = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
                 if num_points_bytes % 16 == 0:
                    data = np.frombuffer(buffer, dtype=dt_f)
                    arr = np.column_stack((data['x'], data['y'], data['z'], data['intensity']))
                 else:
                    # Last resort: just read as flat floats and reshape? Risky.
                    # Return empty to avoid crash
                    print(f"Warning: PCD buffer size {num_points_bytes} not divisible by 14 or 16.")
                    return np.zeros((0, 4), dtype=np.float32)
                 
            return arr
            
        except Exception as e:
            print(f"Error parsing binary PCD: {e}")
            return np.zeros((0, 4), dtype=np.float32)

def load_gt_items(json_path):
    print(f"Loading GT {json_path}...")
    if not os.path.exists(json_path):
        print(f"GT file not found: {json_path}")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    items = []
    
    # Check if 'items' key exists or if root is list
    item_list = data.get('items', []) if isinstance(data, dict) else data
    
    for item in item_list:
        category = item.get('category', 'unknown')
        raw_pts = []
        
        if 'position' in item and item['position']:
            raw_pts = item['position']
        elif 'semantic_line' in item and item['semantic_line'] and 'position' in item['semantic_line']:
            raw_pts = item['semantic_line']['position']

        pts = []
        for p in raw_pts:
            pts.append([p['x'], p['y'], p['z']])
        if pts:
            items.append({
                'category': category,
                'points': np.array(pts),
                'attributes': item.get('attributes', {})
            })
    return items

# --- Processing Helpers ---

def transform_to_local(global_pts, pose):
    if len(global_pts) == 0: return global_pts
    centered = global_pts[:, :3] - np.array([pose['x'], pose['y'], pose['z']])
    rot = R.from_quat(pose['q'])
    inv_rot = rot.inv()
    local_xyz = inv_rot.apply(centered)
    
    if global_pts.shape[1] > 3:
        return np.column_stack((local_xyz, global_pts[:, 3:]))
    return local_xyz

def clip_polyline_by_x(points, x_min, x_max):
    if len(points) < 2:
        if len(points) == 1:
            if x_min <= points[0][0] <= x_max:
                return points
        return np.empty((0, 3))

    new_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        
        # Clip logic 
        t0, t1 = 0.0, 1.0
        dx = p2[0] - p1[0]
        
        if abs(dx) < 1e-6:
            if p1[0] < x_min or p1[0] > x_max: continue 
        else:
            t_min = (x_min - p1[0]) / dx
            t_max = (x_max - p1[0]) / dx
            
            if dx > 0:
                t0 = max(t0, t_min)
                t1 = min(t1, t_max)
            else:
                t0 = max(t0, t_max)
                t1 = min(t1, t_min)
        
        if t0 <= t1:
            pt0 = p1 + t0 * (p2 - p1)
            pt1 = p1 + t1 * (p2 - p1)
            
            if len(new_points) == 0 or np.linalg.norm(new_points[-1] - pt0) > 1e-6:
                new_points.append(pt0)
            new_points.append(pt1)

    return np.array(new_points)

def save_pcd(path, points):
    num_points = len(points)
    with open(path, 'w') as f:
        f.write("VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {num_points}\nDATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[3])}\n")

def save_json(path, items, ref_ts):
    output = {
        "timestamp": ref_ts,
        "items": []
    }
    for item in items:
        # convert numpy to list
        pos = []
        for p in item['points']:
            pos.append({'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])})
        
        output['items'].append({
            'category': item['category'],
            'attributes': item['attributes'],
            'position': pos
        })
    with open(path, 'w') as f:
        json.dump(output, f, indent=4)

def process_dataset(annotation_dir, gt_json_path, output_dir):
    print(f"\nProcessing Dataset:\n  Raw: {annotation_dir}\n  GT: {gt_json_path}")
    
    pose_dir = os.path.join(annotation_dir, "pose")
    pcd_path = os.path.join(annotation_dir, "merged.pcd")
    
    poses = load_poses(pose_dir)
    if not poses:
        print("  No poses found. Skipping.")
        return

    all_points = load_pcd_fast(pcd_path)
    if len(all_points) == 0:
        print("  No points found. Skipping.")
        return

    gt_items = load_gt_items(gt_json_path)
    
    # Path range
    x_min = min(p['x'] for p in poses)
    x_max = max(p['x'] for p in poses)
    total_len = x_max - x_min
    print(f"  Path X range: {x_min:.1f} to {x_max:.1f} ({total_len:.1f}m)")
    
    current_x = x_min + SEGMENT_LEN / 2
    count = 0
    
    while current_x < x_max:
        # Find closest pose
        closest_pose = min(poses, key=lambda p: abs(p['x'] - current_x))
        if abs(closest_pose['x'] - current_x) > 10.0:
            current_x += STRIDE
            continue
            
        ts_name = closest_pose['filename_ts']
        
        # optimize: filter points by radius 60m
        dx = all_points[:, 0] - closest_pose['x']
        dy = all_points[:, 1] - closest_pose['y']
        
        # Careful with large arrays, pre-filter reduces memory
        subset_mask = (dx**2 + dy**2) < 3600
        subset_points = all_points[subset_mask]
        
        if len(subset_points) > 0:
             # Transform
            local_points = transform_to_local(subset_points, closest_pose)
            
            # Crop X
            mask = (local_points[:, 0] >= -SEGMENT_LEN/2) & (local_points[:, 0] <= SEGMENT_LEN/2)
            final_points = local_points[mask]
            
            # Process GT
            final_gt_items = []
            for item in gt_items:
                local_line = transform_to_local(item['points'], closest_pose)
                clipped = clip_polyline_by_x(local_line, -SEGMENT_LEN/2, SEGMENT_LEN/2)
                if len(clipped) > 1:
                    final_gt_items.append({
                        'category': item['category'],
                        'points': clipped,
                        'attributes': item['attributes']
                    })
            
            # Save
            save_pcd(os.path.join(output_dir, f"{ts_name}.pcd"), final_points)
            save_json(os.path.join(output_dir, f"{ts_name}.json"), final_gt_items, ts_name)
            count += 1
            
        current_x += STRIDE
    
    print(f"  Generated {count} samples.")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Search for all _annotation_raw_data folders
    raw_dirs = glob.glob(os.path.join(SOURCE_ROOT, "*_annotation_raw_data"))
    
    print(f"Found {len(raw_dirs)} datasets in {SOURCE_ROOT}")
    
    for d in raw_dirs:
        # Infer GT json path
        # Directory: .../NAME_annotation_raw_data
        # Expected GT: .../NAME.bag.json
        
        dir_name = os.path.basename(d)
        if dir_name.endswith("_annotation_raw_data"):
            base_name = dir_name[:-20] # remove _annotation_raw_data
            
            # Try probable GT paths
            gt_candidate_1 = os.path.join(SOURCE_ROOT, f"{base_name}.bag.json")
            gt_candidate_2 = os.path.join(SOURCE_ROOT, f"{base_name}.json")
            
            if os.path.exists(gt_candidate_1):
                process_dataset(d, gt_candidate_1, OUTPUT_DIR)
            elif os.path.exists(gt_candidate_2):
                process_dataset(d, gt_candidate_2, OUTPUT_DIR)
            else:
                print(f"Warning: Could not find GT json for {base_name}")
                print(f"Tried: {gt_candidate_1}\n       {gt_candidate_2}")

if __name__ == "__main__":
    main()
