import os
import glob
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def load_pcd_ascii(pcd_path):
    print(f"Loading {pcd_path}...")
    points = []
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
        
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('DATA'):
            data_start = i + 1
            break
            
    for i in range(data_start, len(lines)):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            # x, y, z, intensity
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            intensity = float(parts[3]) if len(parts) > 3 else 0
            points.append([x, y, z, intensity])
            
    return np.array(points)

def load_gt_snippet(json_path):
    print(f"Loading GT {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    items = []
    for item in data.get('items', []):
        points = []
        for p in item.get('position', []):
            points.append([p['x'], p['y'], p['z']])
            
        noisy_lines = []
        if 'noisy_candidates' in item:
            for cand in item['noisy_candidates']:
                cand_pts = []
                for p in cand:
                    cand_pts.append([p['x'], p['y'], p['z']])
                noisy_lines.append(np.array(cand_pts))
                
        items.append({
            'category': item.get('category', 'unknown'),
            'points': np.array(points),
            'noisy_lines': noisy_lines
        })
    return items

def visualize_3d_plotly(pcd_path, json_path, output_html):
    points = load_pcd_ascii(pcd_path)
    gt_items = load_gt_snippet(json_path)
    
    # Subsample for web if too large (plotly can be slow with >50k points)
    if len(points) > 30000:
        indices = np.random.choice(len(points), 30000, replace=False)
        points = points[indices]
        print(f"Subsampled to 30k points for smooth web visualization")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    
    fig = go.Figure()
    
    # Add Point Cloud
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1.5,
            color=intensity,
            colorscale='Viridis',
            cmin=0,
            cmax=30,
            opacity=0.6,
            colorbar=dict(title="Intensity", x=0.8)
        ),
        name='Point Cloud'
    ))
    
    # Add GT Lines
    for i, item in enumerate(gt_items):
        pts = item['points']
        cat = item['category']
        
        if cat == 'lane_line':
            color = 'red'
            width = 4
        elif cat == 'curb':
            color = 'orange'
            width = 6
        else:
            color = 'blue'
            width = 4
            
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='lines+markers',
            line=dict(color=color, width=width),
            marker=dict(size=3, color=color),
            name=f"{cat}_{i}"
        ))
        
        # Add Noisy Candidates
        for j, noisy_pts in enumerate(item.get('noisy_lines', [])):
            # Distinct colors for different noise levels roughly
            # Noise order in list is [Small, Medium, Large]
            noise_colors = ['#00FFFF', '#FFFF00', '#FF00FF'] # Cyan, Yellow, Magenta
            c = noise_colors[j % len(noise_colors)]
            
            fig.add_trace(go.Scatter3d(
                x=noisy_pts[:, 0], y=noisy_pts[:, 1], z=noisy_pts[:, 2],
                mode='lines+markers', # Show markers to see jitter
                line=dict(color=c, width=3, dash='solid'), # Solid line easier to see than dash sometimes? Or keep dash
                marker=dict(size=2, color=c),
                opacity=0.9, # Higher opacity
                name=f"{cat}_{i}_noise_{j}"
            ))

    # Layout config
    fig.update_layout(
        title=f"3D View: {os.path.basename(pcd_path)}",
        scene=dict(
            xaxis_title='X (Forward)',
            yaxis_title='Y (Lateral)',
            zaxis_title='Z (Up)',
            aspectmode='data' # Keep aspect ratio correct
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        legend=dict(x=0.05, y=0.9)
    )
    
    print(f"Saving {output_html}...")
    fig.write_html(output_html)

if __name__ == "__main__":
    snippets = sorted(glob.glob("train_data/*.pcd"))[:3]
    for i, pcd_path in enumerate(snippets):
        json_path = pcd_path.replace('.pcd', '.json')
        if os.path.exists(json_path):
            output_html = f"vis_3d_snippet_{i}.html"
            visualize_3d_plotly(pcd_path, json_path, output_html)
        else:
            print(f"JSON not available for {pcd_path}")
