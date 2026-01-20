import os
import json
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import struct
import open3d as o3d # Not available, use custom saver

# --- 1. Helper Config ---
BAG_NAME = "TAD_front_lidar_2025-12-01-11-36-53_14_all.bag"
DATA_ROOT = f"/homes/zhangzijian/pointnet_refine/data/train/{BAG_NAME}"
PCD_PATH = os.path.join(DATA_ROOT, "annotation_raw_data/merged.pcd")
POSE_DIR = os.path.join(DATA_ROOT, "annotation_raw_data/pose")
GT_JSON_PATH = os.path.join(DATA_ROOT, f"{BAG_NAME}.json")
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/inference_data"
SEGMENT_LEN = 50.0  # meters
STRIDE = 25.0       # meters

# --- 2. Loaders ---

def load_poses(pose_dir):
    print("Loading poses...")
    poses = []
    files = glob.glob(os.path.join(pose_dir, "*.json"))
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            # Use 'ts' from json content, or filename? Usually filename is strictly timestamp
            # But content has 'ts'. Let's use content 'ts' for consistency with filename if possible
            ts = data['ts']
            # Also store filename base for exact matching if needed, but ts is key
            filename_ts = os.path.splitext(os.path.basename(f))[0]
            
            poses.append({
                'ts': str(ts), # Keep as string to match image names often
                'filename_ts': filename_ts,
                'x': data['x'],
                'y': data['y'],
                'z': data['z'],
                'q': [data['qx'], data['qy'], data['qz'], data['qw']] # scipy scalar definition
            })
    
    # Sort by x (assuming mostly linear forward motion) or ts
    # Sorting by x helps in distance-based sampling
    poses.sort(key=lambda p: p['x'])
    print(f"Loaded {len(poses)} poses.")
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
        
        # Convert to simple Nx4 array for easier manipulation
        # [x, y, z, intensity]
        # intensity cast to float for homogenous array
        arr = np.column_stack((data['x'], data['y'], data['z'], data['intensity'].astype(np.float32)))
        return arr

def load_gt_items(json_path):
    print(f"Loading GT {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    items = []
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
    # points: Nx4 (x, y, z, intensity)
    # Write ASCII or Binary PCD. Binary is faster/smaller.
    
    num_points = len(points)
    with open(path, 'w') as f:
        # Header
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n") # Saving all as float32 for simplicity (intensity cast back to int? or keep float)
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")
        
        # Write data (slow in python loop, but safe for ascii)
        # For speed in bulk generation, maybe binary is essential?
        # Let's write ascii for now, if too slow, switch to binary structure pack
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
    
    # 2. Rotate
    # Global to Local = Inverse of Body to Global
    # Pose orientation usually means Body -> Global
    # So we need inverse rotation
    rot = R.from_quat(pose['q'])
    inv_rot = rot.inv()
    local_xyz = inv_rot.apply(centered)
    
    if global_pts.shape[1] > 3:
        return np.column_stack((local_xyz, global_pts[:, 3:]))
    return local_xyz

def clip_polyline_by_x(points, x_min, x_max):
    """
    Clip a checklist of points (polyline) to a valid X range.
    Interpolates new vertices at the boundaries.
    Assumes Z and Y are linearly interpolated.
    points: (N, 3) numpy array
    """
    if len(points) < 2:
        # Check if single point is in range
        if len(points) == 1:
            if x_min <= points[0][0] <= x_max:
                return points
        return np.empty((0, 3))

    new_points = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        
        # Segment parameterization P(t) = P1 + t*(P2-P1), t in [0, 1]
        dx = p2[0] - p1[0]
        
        t_enter = 0.0
        t_exit = 1.0
        
        # Clip against x_min
        # p1[0] + t * dx >= x_min
        if abs(dx) < 1e-6:
            # Vertical line (const x)
            if p1[0] < x_min: t_enter = 2.0 # Invalid
        else:
            t = (x_min - p1[0]) / dx
            if dx > 0:
                # entering from left
                t_enter = max(t_enter, t)
            else:
                # entering from right?? No, dx < 0 means moving left.
                # condition x >= x_min.
                # start at large x, go to small x.
                # must be less than t for intersection
                t_exit = min(t_exit, t)
                
        # Clip against x_max
        # p1[0] + t * dx <= x_max
        if abs(dx) < 1e-6:
            if p1[0] > x_max: t_enter = 2.0
        else:
            t = (x_max - p1[0]) / dx
            if dx > 0:
                # moving right. must be <= x_max
                t_exit = min(t_exit, t)
            else:
                # moving left. must be <= x_max (always true if starting < max?)
                # condition x <= x_max.
                # if start > max, we need to enter.
                t_enter = max(t_enter, t)
                
        if t_enter <= t_exit:
            # Valid segment exists
            # Calculate points
            # Caution: floating point errors
            t_enter = max(0.0, t_enter)
            t_exit = min(1.0, t_exit)
            
            p_start = p1 + t_enter * (p2 - p1)
            p_end = p1 + t_exit * (p2 - p1)
            
            # Add p_start if it is the first point or disconnected from previous
            if len(new_points) == 0 or np.linalg.norm(new_points[-1] - p_start) > 1e-6:
                new_points.append(p_start)
            
            # Add p_end
            new_points.append(p_end)

    return np.array(new_points)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    poses = load_poses(POSE_DIR)
    all_points = load_pcd_fast(PCD_PATH)
    gt_items = load_gt_items(GT_JSON_PATH)
    
    # Determine the path length
    x_min = poses[0]['x']
    x_max = poses[-1]['x']
    total_len = x_max - x_min
    
    print(f"Path covers X: {x_min:.1f} to {x_max:.1f} (Length: {total_len:.1f}m)")
    
    # Sliding window
    current_x = x_min + SEGMENT_LEN / 2 
    # Start center at first 25m mark (window 0-50)
    
    seg_idx = 0
    
    while current_x < x_max:
        # 1. Find the pose closest to current_x
        # Using simple min distance sort (could be optimized with bisect but poses len is small < 1000)
        closest_pose = min(poses, key=lambda p: abs(p['x'] - current_x))
        
        # Check if pose is reasonably close (e.g. within 5m of target center)
        # If gaps in data, might be far.
        dist_to_center = abs(closest_pose['x'] - current_x)
        if dist_to_center > 10.0:
            print(f"Warning: No pose close to x={current_x:.1f} (closest {dist_to_center:.1f}m away). Skipping.")
            current_x += STRIDE
            continue
            
        ts_name = closest_pose['filename_ts']
        print(f"Processing Segment {seg_idx}: center_x={current_x:.1f}, matches pose {ts_name}")
        
        # 2. Pre-filter global points to avoid transforming everything
        # Just grab points within +/- 50m bounding box of the pose global position
        # This is a safe superset of the final 50m ( +/- 25m ) local crop
        # Note: This assumes X-aligned path roughly. 
        # For curves, 50m radius search is better.
        dx = all_points[:, 0] - closest_pose['x']
        dy = all_points[:, 1] - closest_pose['y']
        
        # 2D radius filter squared
        dist_sq = dx**2 + dy**2
        mask_radius = dist_sq < (60**2) # 60m radius
        
        subset_points = all_points[mask_radius].copy()
        
        if len(subset_points) == 0:
            print("  Empty point cloud in this region.")
            current_x += STRIDE
            continue
            
        # 3. Transform to Local Frame of the closest_pose
        local_points = transform_to_local(subset_points, closest_pose)
        
        # 4. Final Crop: Local X in [-25, 25]
        # X axis in local frame usually is Forward.
        mask_final = (local_points[:, 0] >= -SEGMENT_LEN/2) & (local_points[:, 0] <= SEGMENT_LEN/2)
        final_points = local_points[mask_final]
        
        # 5. Process GT
        final_gt_items = []
        for item in gt_items:
            # Transform line points
            local_line = transform_to_local(item['points'], closest_pose)
            
            # Clip polyline to X range [-25, 25]
            clipped_line = clip_polyline_by_x(local_line, -SEGMENT_LEN/2, SEGMENT_LEN/2)
            
            if len(clipped_line) > 1:
                final_gt_items.append({
                    'category': item['category'],
                    'points': clipped_line,
                    'attributes': item['attributes']
                })
        
        # 6. Save
        pcd_out = os.path.join(OUTPUT_DIR, f"{ts_name}.pcd")
        json_out = os.path.join(OUTPUT_DIR, f"{ts_name}.json")
        
        save_pcd(pcd_out, final_points)
        save_json(json_out, final_gt_items, ts_name)
        
        print(f"  Saved {len(final_points)} points, {len(final_gt_items)} lines.")
        
        current_x += STRIDE
        seg_idx += 1

if __name__ == "__main__":
    main()
