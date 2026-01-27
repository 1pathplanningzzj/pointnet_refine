import os
import json
import numpy as np
import glob
import math
from scipy.optimize import linear_sum_assignment

# --- 1. Helper Config ---
BAG_BASE_NAME = "TAD_front_vision_2025-08-20-11-52-36_85_5to25_0"
DATA_ROOT = "/homes/zhangzijian/pointnet_refine/data/vma_test_data"
RAW_DATA_DIR = os.path.join(DATA_ROOT, f"{BAG_BASE_NAME}_annotation_raw_data")

PCD_PATH = os.path.join(RAW_DATA_DIR, "merged.pcd")
POSE_DIR = os.path.join(RAW_DATA_DIR, "pose")
RESULTS_JSON_PATH = os.path.join(DATA_ROOT, "results_fuse175_test316.json")

OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/vma_infer_data"
SEGMENT_LEN = 50.0  # meters

# --- 2. Math Helpers (No SciPy) ---

def quat_to_matrix(q):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    
    r00 = 1 - 2*qy*qy - 2*qz*qz
    r01 = 2*qx*qy - 2*qz*qw
    r02 = 2*qx*qz + 2*qy*qw
    
    r10 = 2*qx*qy + 2*qz*qw
    r11 = 1 - 2*qx*qx - 2*qz*qz
    r12 = 2*qy*qz - 2*qx*qw
    
    r20 = 2*qx*qz - 2*qy*qw
    r21 = 2*qy*qz + 2*qx*qw
    r22 = 1 - 2*qx*qx - 2*qy*qy
    
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])

GT_JSON_PATH = os.path.join(DATA_ROOT, f"{BAG_BASE_NAME}.bag.json")

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
            points_np = np.array(pts)
            items.append({
                'category': category,
                'points': points_np,
                'attributes': item.get('attributes', {})
            })
    print(f"Loaded {len(items)} GT lines.")
    return items

def transform_to_local(global_pts, pose):
    """
    Transform global points to local frame using pose (x, y, z, q=[x,y,z,w]).
    """
    if len(global_pts) == 0:
        return global_pts
    
    # Translation
    centered = global_pts[:, :3] - np.array([pose['x'], pose['y'], pose['z']])
    
    # Rotation (Inverse of Pose Rotation)
    # R_local_to_global = R_pose
    # R_global_to_local = R_pose^T
    rot_matrix = quat_to_matrix(pose['q'])
    
    # Mathematical derivation for Row Vectors (numpy default):
    # v_local = R^T * v_global
    # v_local.T = (R^T * v_global)^T = v_global.T * R
    # So we should multiply by rot_matrix without transposing.
    # PREVIOUSLY: inv_rot = rot_matrix.T (This was causing the VMA/PCD mismatch)
    inv_rot = rot_matrix 
    
    local_xyz = np.dot(centered, inv_rot) 
    
    return local_xyz

# --- 3. Loaders ---

def load_poses(pose_dir):
    print("Loading poses...")
    poses = []
    files = glob.glob(os.path.join(pose_dir, "*.json"))
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            ts = data['ts']
            try:
                ts_int = int(os.path.splitext(os.path.basename(f))[0])
            except (ValueError, TypeError):
                ts_int = int(ts)
                
            poses.append({
                'ts': ts_int,
                'x': data['x'],
                'y': data['y'],
                'z': data['z'],
                'q': [data['qx'], data['qy'], data['qz'], data['qw']]
            })
    
    poses.sort(key=lambda p: p['ts'])
    print(f"Loaded {len(poses)} poses.")
    return poses

def load_pcd_fast(pcd_path):
    print(f"Loading {pcd_path}...")
    if not os.path.exists(pcd_path):
        print(f"Error: PCD file not found at {pcd_path}")
        return np.empty((0, 4))
        
    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().strip()
            header.append(line)
            if line.startswith(b'DATA'):
                break
        
        buffer = f.read()
        try:
            dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u2')])
            data = np.frombuffer(buffer, dtype=dt)
            arr = np.column_stack((data['x'], data['y'], data['z'], data['intensity'].astype(np.float32)))
        except:
             return np.empty((0, 4))
    return arr

# --- 4. Processing ---

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
            intensity = int(p[3]) if len(p) > 3 else 0
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {intensity}\n")

def save_json_vma_direct(path, vma_items, ref_ts, res_ts):
    output = {
        "timestamp": str(ref_ts),
        "result_timestamp": str(res_ts),
        "items": vma_items
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=4)

def pixel_to_ego(pixel_points, img_shape=(1000, 1000), res=0.05):
    """
    Convert BEV pixel coordinates to Ego vehicle coordinates.
    Based on VMA config:
    - bev_res = 0.05
    - crop_size / input_shape = (1000, 1000)
    - Ego center is typically at the bottom center or center of the image.
    
    Standard VMA/MapTR convention:
    Image (0,0) is Top-Left.
    X is Forward (Vertical in Image), Y is Left (Horizontal in Image).
    
    Range Coverage: 1000px * 0.05m/px = 50.0m
    
    Usually:
    X_Range: [-15.0, 35.0] (Total 50m) or [0, 50]?
    Y_Range: [-25.0, 25.0] (Total 50m)
    
    Let's infer from standard top-down view mappings:
    Typically, vehicle is at (W/2, H/2) or (W/2, H_bottom).
    
    Hypothesis 1: Center-Center (Vehicle at 25m, 0m relative to map center)
    x = (H/2 - v) * res
    y = (W/2 - u) * res
    
    Hypothesis 2: Bottom-Center (Vehicle X=0 is some offest from bottom)
    
    Wait, `transform_method='minmax'` and `code_size=2` suggests normalization to [0,1].
    But here we receive raw pixels or normalized? 
    'pred_instances'['data'] usually contains points in image coordinates (u, v).
    
    Let's start with standard Center-Ego assumption often used in generated maps:
    If the map covers 50m x 50m.
    
    Looking at your previous VMA config: None explicit.
    But let's look at `pixel_to_ego` logic used before:
    x = (H - v) * res  => v=1000 -> x=0; v=0 -> x=50. (Bottom is 0, Top is 50)
    y = (W/2 - u) * res => u=500 -> y=0; u=0 -> y=25; u=1000 -> y=-25.
    
    This matches the visual output where X (Forward) 0-50m.
    And Y (Left) -25 to 25m.
    
    However, if the VMA config implies a different range (e.g., LiDAR range), we need to adjust.
    The config says `bev_res = 0.05` and `input_shape=(1000,1000)`.
    So the coverge is indeed 50m x 50m.
    
    IF the GT and Points are aligned with this (0 to 50m), then X offset is correct.
    But if the GT is effectively [-15, 35] in X, then we need to subtract 15.
    
    Let's check the GT bounds from the visualization.
    The GT lines seem to stretch from bottom to top.
    
    Let's KEEP the previous logic but ensure it aligns with the 50x50m window:
    X: [0, 50], Y: [-25, 25].
    """
    H, W = img_shape
    ego_points = []
    
    # MapTR / VMA Standard for nuScenes often uses:
    # Point Cloud Range: [-15.0, -30.0, -5.0, 35.0, 30.0, 3.0] 
    # But here crop_size=1000*0.05 = 50m.
    # X Range: [-15.0, 35.0] (Span 50m) matches perfectly.
    # Y Range: [-25.0, 25.0] (Span 50m) matches perfectly.
    
    X_MAX = 35.0
    Y_MAX = 25.0
    # Calibration offsets from scene alignment
    X_OFFSET = 15.0
    Y_OFFSET = 0.0
    
    for u, v in pixel_points:
        # v is row (0 at top, H at bottom)
        # Top (v=0) -> X_MAX (35.0)
        # Bottom (v=H) -> X_MIN (-15.0)
        x = X_MAX - v * res + X_OFFSET
        
        # u is col (0 at left, W at right)
        # Left (u=0) -> Y_MAX (25.0)
        # Right (u=W) -> Y_MIN (-25.0)
        y = Y_MAX - u * res + Y_OFFSET
        
        ego_points.append({'x': float(x), 'y': float(y), 'z': 0.0})
    return ego_points

def clip_polygon_against_plane(points, plane_val, is_max=False):
    if len(points) == 0: return points
    output = []
    
    def is_inside(p):
        return p[0] <= plane_val if is_max else p[0] >= plane_val
        
    def intersection(a, b):
        # Avoid division by zero
        denom = b[0] - a[0]
        if abs(denom) < 1e-6:
            return a # Should not happen ideally if crossing
        t = (plane_val - a[0]) / denom
        return a + t * (b - a)

    # Handle first point
    if is_inside(points[0]):
        output.append(points[0])
        
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        
        in1 = is_inside(p1)
        in2 = is_inside(p2)
        
        if in1 and in2:
            output.append(p2)
        elif in1 and not in2:
            output.append(intersection(p1, p2))
        elif not in1 and in2:
            output.append(intersection(p1, p2))
            output.append(p2)
        # If both outside, do nothing (for this plane)
            
    return np.array(output)

def clip_polyline_by_x(points, x_min, x_max):
    """
    Clips polyline against x_min and x_max planes.
    Assumes points are ordered.
    """
    if len(points) == 0: return points
    
    # 1. Clip against min
    p_ge_min = clip_polygon_against_plane(points, x_min, is_max=False)
    if len(p_ge_min) == 0: return np.empty((0, 3))
    
    # 2. Clip against max
    p_final = clip_polygon_against_plane(p_ge_min, x_max, is_max=True)
    return p_final

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    poses = load_poses(POSE_DIR)
    all_points = load_pcd_fast(PCD_PATH)
    gt_items = load_gt_items(GT_JSON_PATH) # Load GT
    
    if len(poses) == 0:
        print("No poses found.")
        return
        
    print(f"Loading VMA Results from {RESULTS_JSON_PATH}...")
    with open(RESULTS_JSON_PATH, 'r') as f:
        results_data = json.load(f)
        
    # Create valid timestamp list
    result_timestamps = []
    # Map back to keys for easy access
    ts_to_key = {}
    
    for k in results_data.keys():
        try:
            ts = int(os.path.basename(k).split('.jpg')[0])
            result_timestamps.append(ts)
            ts_to_key[ts] = k
        except:
            pass
            
    result_timestamps = sorted(result_timestamps)
    print(f"Found {len(result_timestamps)} result timestamps to process.")
    
    generated_count = 0
    
    for res_ts in result_timestamps:
        # Match Pose
        closest_pose = min(poses, key=lambda p: abs(p['ts'] - res_ts))
        diff = abs(closest_pose['ts'] - res_ts)
        
        # 250ms diff threshold
        if diff > 250000000: 
            print(f"  Skipping {res_ts}: Closest pose {closest_pose['ts']} is too far ({diff/1e6:.1f}ms)")
            continue
            
        pose_ts = closest_pose['ts']
        
        # 1. Filter and Transform PCD
        dx = all_points[:, 0] - closest_pose['x']
        dy = all_points[:, 1] - closest_pose['y']
        mask_radius = (dx**2 + dy**2) < (60**2)
        subset_points = all_points[mask_radius].copy()
        
        if len(subset_points) == 0:
            continue
            
        # Transform PCD to Local Ego Frame
        local_xyz = transform_to_local(subset_points, closest_pose)
        
        # Re-attach intensity (Column 3)
        # local_xyz is (N, 3), subset_points is (N, 4)
        if subset_points.shape[1] >= 4:
            local_points = np.hstack([local_xyz, subset_points[:, 3:4]])
        else:
            local_points = local_xyz

        # Crop to strict ROI (Forward 0 to 50m)
        mask_final = (local_points[:, 0] >= 0) & (local_points[:, 0] <= SEGMENT_LEN)
        final_points = local_points[mask_final]
        
        if len(final_points) == 0:
             continue
             

        # 3. Get GT Lines for context
        # Use our transform which SWAPS XY for PCD/GT
        local_gt_lines_for_vis = []
        for item in gt_items:
            # Transform global GT to local frame (XY-Swapped)
            gt_local = transform_to_local(item['points'], closest_pose)
            
            # Simple Bounds check
            if np.any((gt_local[:, 0] > 0) & (gt_local[:, 0] < SEGMENT_LEN)):
                 clipped = clip_polyline_by_x(gt_local, 0, SEGMENT_LEN)
                 if len(clipped) > 1:
                     # Convert back to list of dicts for JSON
                     pos_list = []
                     for p in clipped:
                         pos_list.append({'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])})
                     local_gt_lines_for_vis.append(pos_list)

        # 2. Get VMA Lines (NO TRANSFORMATION, NO Z)
        key = ts_to_key[res_ts]
        vma_entry = results_data[key]
        pred_instances = vma_entry.get('pred_instances', [])
        
        # --- Pre-Process: Collect all Preds ---
        pred_lines_3d = []
        pred_scores = []
        for inst in pred_instances:
            poly_2d = inst['data']
            poly_3d = pixel_to_ego(poly_2d)
            if len(poly_3d) > 1:
                pred_lines_3d.append(poly_3d)
                pred_scores.append(inst.get('score', 0.0))
        
        # --- Hungarian Matching (Global Assignment) ---
        # Cost Matrix: Rows=Preds, Cols=GTs
        num_preds = len(pred_lines_3d)
        num_gts = len(local_gt_lines_for_vis)
        
        matched_gt_indices = {} # PredIdx -> GTIdx
        
        if num_preds > 0 and num_gts > 0:
            cost_matrix = np.full((num_preds, num_gts), 1000.0) # Init with high cost
            
            for i, p_line in enumerate(pred_lines_3d):
                p_pts = np.array([[p['x'], p['y']] for p in p_line])
                
                for j, g_line in enumerate(local_gt_lines_for_vis):
                    g_pts = np.array([[p['x'], p['y']] for p in g_line])
                    if len(g_pts) < 1: continue
                    
                    # Chamfer Distance (One-way: Pred -> GT)
                    # For each point in Pred, find closest in GT, then average
                    diff = p_pts[:, None, :] - g_pts[None, :, :]
                    dists_matrix = np.linalg.norm(diff, axis=2) # (M, N)
                    min_dists = np.min(dists_matrix, axis=1) # (M,)
                    dist = np.mean(min_dists)
                    
                    cost_matrix[i, j] = dist
            
            # Solve Assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Filter by Threshold
            MATCH_THRESHOLD = 15.0
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < MATCH_THRESHOLD:
                    matched_gt_indices[r] = c
                else:
                    pass # Too far, leave unmatched

        # --- Construct Items ---
        vma_items = []
        for i, poly_3d in enumerate(pred_lines_3d):
            best_gt = []
            if i in matched_gt_indices:
                gt_idx = matched_gt_indices[i]
                best_gt = local_gt_lines_for_vis[gt_idx]
                
            item_dict = {
                'category': 'lane_line', 
                'attributes': {'score': pred_scores[i]},
                'position': best_gt, # Matched GT or empty
                'noisy_candidates': [poly_3d],
                'context_lines': local_gt_lines_for_vis
            }
            vma_items.append(item_dict)
        
        if len(vma_items) > 0:
            pcd_out = os.path.join(OUTPUT_DIR, f"{res_ts}.pcd")
            json_out = os.path.join(OUTPUT_DIR, f"{res_ts}.json")
            
            save_pcd(pcd_out, final_points)
            save_json_vma_direct(json_out, vma_items, pose_ts, res_ts)
            generated_count += 1
            if generated_count % 5 == 0:
                print(f"  Generated {generated_count} samples...")
        else:
             print(f"  Skipping {res_ts}: No predictions found in JSON.")

    print(f"Done. Generated {generated_count} samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
