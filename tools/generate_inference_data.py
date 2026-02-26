import os
import json
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import struct

# --- 1. Helper Config ---
# Global config changed to batch processing
DATA_ROOT = "/homes/zhangzijian/pointnet_refine/data/vma_infer_data_test"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/inference_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENT_LEN = 50.0  # meters
STRIDE = 25.0       # meters

# --- 2. Loaders ---

def load_poses(pose_dir):
    # print("Loading poses...")
    poses = []
    if not os.path.exists(pose_dir):
        print(f"Pose dir does not exist: {pose_dir}")
        return []
        
    files = glob.glob(os.path.join(pose_dir, "*.json"))
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            ts = data['ts']
            filename_ts = os.path.splitext(os.path.basename(f))[0]
            
            poses.append({
                'ts': str(ts),
                'filename_ts': filename_ts,
                'x': data['x'],
                'y': data['y'],
                'z': data['z'],
                'q': [data['qx'], data['qy'], data['qz'], data['qw']] # scipy scalar definition
            })
    
    # Sort by x
    poses.sort(key=lambda p: p['x'])
    return poses

def load_pcd_fast(pcd_path):
    print(f"Loading {pcd_path}...")
    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().strip()
            header.append(line)
            if line.startswith(b'DATA'):
                break
        
        points = 0
        for line in header:
            if line.startswith(b'POINTS'):
                points = int(line.split()[1])
        
        # Load binary
        dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u2')])
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=dt)
        
        # Convert to simple Nx4 array [x, y, z, intensity]
        arr = np.column_stack((data['x'], data['y'], data['z'], data['intensity'].astype(np.float32)))
        return arr

def load_gt_items(json_path):
    # print(f"Loading GT {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    items = []
    
    # Handle "lane_lines" format
    if 'lane_lines' in data:
         for lane in data['lane_lines']:
            pts = lane.get('pts', [])
            if not pts or len(pts) < 2:
                continue
            
            xs, ys = pts[0], pts[1]
            zs = pts[2] if len(pts) > 2 else [0.0] * len(xs)
            
            points = np.array([[xs[i], ys[i], zs[i]] for i in range(len(xs))])
            
            items.append({
                'category': 'lane_line',
                'points': points,
                'attributes': {}
            })
    # Handle "items" format
    elif 'items' in data:
        for item in data.get('items', []):
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

# --- 3. Processing ---

def save_pcd(path, points):
    num_points = len(points)
    with open(path, 'w') as f:
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")
        
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

def transform_to_local(global_pts, pose):
    """
    global_pts: Nx3 or Nx4
    pose: dict with x,y,z,q
    """
    # 1. Translate
    centered = global_pts[:, :3] - np.array([pose['x'], pose['y'], pose['z']])
    
    # 2. Rotate (Global -> Local = Inverse of Body -> Global)
    rot = R.from_quat(pose['q'])
    inv_rot = rot.inv()
    local_xyz = inv_rot.apply(centered)
    
    if global_pts.shape[1] > 3:
        return np.column_stack((local_xyz, global_pts[:, 3:]))
    return local_xyz

def clip_polyline_by_x(points, x_min, x_max):
    """
    Clip a checklist of points (polyline) to a valid X range.
    """
    if len(points) < 2:
        if len(points) == 1:
            if x_min <= points[0][0] <= x_max:
                return points
        return np.empty((0, 3))

    new_points = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        
        dx = p2[0] - p1[0]
        t_enter = 0.0
        t_exit = 1.0
        
        if abs(dx) < 1e-6:
            if p1[0] < x_min: t_enter = 2.0
        else:
            t = (x_min - p1[0]) / dx
            if dx > 0:
                t_enter = max(t_enter, t)
            else:
                t_exit = min(t_exit, t)
                
        if abs(dx) < 1e-6:
            if p1[0] > x_max: t_enter = 2.0
        else:
            t = (x_max - p1[0]) / dx
            if dx > 0:
                t_exit = min(t_exit, t)
            else:
                t_enter = max(t_enter, t)
                
        if t_enter <= t_exit:
            t_enter = max(0.0, t_enter)
            t_exit = min(1.0, t_exit)
            
            p_start = p1 + t_enter * (p2 - p1)
            p_end = p1 + t_exit * (p2 - p1)
            
            if len(new_points) == 0 or np.linalg.norm(new_points[-1] - p_start) > 1e-6:
                new_points.append(p_start)
            new_points.append(p_end)

    return np.array(new_points)

def process_one_bag(bag_name, pcd_path, pose_dir, gt_json_path, output_dir):
    # Loaders
    poses = load_poses(pose_dir)
    if not poses:
        print(f"No poses found for {bag_name}.")
        return

    try:
        all_points = load_pcd_fast(pcd_path)
    except Exception as e:
        print(f"Failed to load PCD for {bag_name}: {e}")
        return

    if len(all_points) == 0:
        print(f"No points found for {bag_name}.")
        return
        
    gt_items = load_gt_items(gt_json_path)
    
    # Process
    x_min = poses[0]['x']
    x_max = poses[-1]['x']
    total_len = x_max - x_min
    
    # print(f"  Path covers X: {x_min:.1f} to {x_max:.1f} (Length: {total_len:.1f}m)")
    
    # Sliding window
    current_x = x_min + SEGMENT_LEN / 2 
    seg_idx = 0
    save_count = 0
    
    while current_x < x_max:
        # 1. Find the pose closest to current_x
        closest_pose = min(poses, key=lambda p: abs(p['x'] - current_x))
        dist_to_center = abs(closest_pose['x'] - current_x)
        if dist_to_center > 10.0:
            current_x += STRIDE
            continue
            
        ts_name = closest_pose['filename_ts']
        
        # 2. Pre-filter global points within +/- 60m radius
        dx = all_points[:, 0] - closest_pose['x']
        dy = all_points[:, 1] - closest_pose['y']
        
        dist_sq = dx**2 + dy**2
        mask_radius = dist_sq < (60**2) 
        
        subset_points = all_points[mask_radius].copy()
        
        if len(subset_points) == 0:
            current_x += STRIDE
            continue
            
        # 3. Transform to Local Frame
        local_points = transform_to_local(subset_points, closest_pose)
        
        # 4. Final Crop: Local X in [-25, 25]
        mask_final = (local_points[:, 0] >= -SEGMENT_LEN/2) & (local_points[:, 0] <= SEGMENT_LEN/2)
        final_points = local_points[mask_final]
        
        # 5. Process GT (Clip lines)
        final_gt_items = []
        for item in gt_items:
            local_line = transform_to_local(item['points'], closest_pose)
            clipped_line = clip_polyline_by_x(local_line, -SEGMENT_LEN/2, SEGMENT_LEN/2)
            
            if len(clipped_line) > 1:
                final_gt_items.append({
                    'category': item['category'],
                    'points': clipped_line,
                    'attributes': item['attributes']
                })
        
        # 6. Save
        # Make filename JUST THE TIMESTAMP
        unique_name = f"{ts_name}"
        pcd_out = os.path.join(output_dir, f"{unique_name}.pcd")
        json_out = os.path.join(output_dir, f"{unique_name}.json")
        
        save_pcd(pcd_out, final_points)
        save_json(json_out, final_gt_items, ts_name)
        save_count += 1
        
        current_x += STRIDE
        seg_idx += 1
    
    print(f"  Generated {save_count} segments for {bag_name}")


def main():
    print(f"Scanning for bags in {DATA_ROOT}...")
    
    # Find all .bag.json files
    bag_json_files = glob.glob(os.path.join(DATA_ROOT, "*.bag.json"))
    bag_base_names = []
    
    for f in bag_json_files:
        # Filename example: "TAD_..._.bag.json" -> basename "TAD_..."
        basename = os.path.basename(f).replace(".bag.json", "")
        # Check for raw data dir: "TAD_..._annotation_raw_data"
        raw_data_dir = os.path.join(DATA_ROOT, f"{basename}_annotation_raw_data")
        if os.path.exists(raw_data_dir):
            bag_base_names.append(basename)
            
    print(f"Found {len(bag_base_names)} valid bags to process.")
    
    total_bags = len(bag_base_names)
    for i, bag_name in enumerate(bag_base_names):
        print(f"\n[{i+1}/{total_bags}] Processing Bag: {bag_name}")
        
        pcd_path = os.path.join(DATA_ROOT, f"{bag_name}_annotation_raw_data/merged.pcd")
        pose_dir = os.path.join(DATA_ROOT, f"{bag_name}_annotation_raw_data/pose")
        gt_json_path = os.path.join(DATA_ROOT, f"{bag_name}.bag.json")
        
        process_one_bag(bag_name, pcd_path, pose_dir, gt_json_path, OUTPUT_DIR)

if __name__ == "__main__":
    main()
