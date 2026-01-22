
import os
import json
import numpy as np
import plotly.graph_objects as go
import glob

# Config
DATA_ROOT = "/homes/zhangzijian/pointnet_refine/data/vma_test_data"
BAG_BASE_NAME = "TAD_front_vision_2025-08-20-11-52-36_85_5to25_0"
PCD_PATH = os.path.join(DATA_ROOT, f"{BAG_BASE_NAME}_annotation_raw_data/merged.pcd")
GT_JSON_PATH = os.path.join(DATA_ROOT, f"{BAG_BASE_NAME}.bag.json")
OUTPUT_HTML = "global_check.html"

def load_pcd_sample(pcd_path, max_points=20000):
    print(f"Loading {pcd_path}...")
    with open(pcd_path, 'rb') as f:
        while True:
            line = f.readline().strip()
            if line.startswith(b'DATA'):
                break
        buffer = f.read()
        try:
            dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u2')])
            data = np.frombuffer(buffer, dtype=dt)
            arr = np.column_stack((data['x'], data['y'], data['z'], data['intensity'].astype(np.float32)))
            
            # Subsample
            if len(arr) > max_points:
                indices = np.random.choice(len(arr), max_points, replace=False)
                arr = arr[indices]
            return arr
        except Exception as e:
            print(e)
            return np.empty((0, 4))

def load_gt_lines(json_path):
    print(f"Loading GT {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    lines = []
    for item in data.get('items', []):
        if 'position' in item:
            pts = np.array([[p['x'], p['y'], p['z']] for p in item['position']])
            if len(pts) > 1:
                lines.append(pts)
    return lines

def main():
    pcd = load_pcd_sample(PCD_PATH)
    gt_lines = load_gt_lines(GT_JSON_PATH)
    
    fig = go.Figure()
    
    # Plot PCD
    fig.add_trace(go.Scatter3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2],
        mode='markers',
        marker=dict(size=1, color=pcd[:, 3], colorscale='Viridis'),
        name='Global PCD'
    ))
    
    # Plot GT
    for i, line in enumerate(gt_lines):
        fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'GT {i}'
        ))
        
    fig.update_layout(title="Global PCD vs GT Alignment Check")
    fig.write_html(OUTPUT_HTML)
    print(f"Saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
