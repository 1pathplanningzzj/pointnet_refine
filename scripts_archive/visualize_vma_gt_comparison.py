#!/usr/bin/env python3
"""
Visualize VMA detection results and GT annotations on point cloud.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Load point cloud
pcd_file = 'data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd'
pcd = o3d.io.read_point_cloud(pcd_file)
points = np.asarray(pcd.points)

# Load VMA results
vma_file = 'data/vma_test_data/results_fuse175_test316.json'
with open(vma_file, 'r') as f:
    vma_data = json.load(f)

# Load transformation matrices
annot_dir = 'data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/cropped_data/annots'
first_timestamp = list(vma_data.keys())[0].split('/')[-1].replace('.jpg', '')
annot_file = f"{annot_dir}/{first_timestamp}.json"

with open(annot_file, 'r') as f:
    annot = json.load(f)

M_cropped_reverse = np.array(annot['M_cropped_reverse'])
M_bev2world = np.array(annot['M_bev2world'])

def transform_line_coord_2d(line, matrix):
    line = np.array(line)
    x, y = line[:, 0], line[:, 1]
    pts = np.vstack([x, y, np.ones_like(x)])
    pts_trans = matrix @ pts
    return pts_trans[:2].T

def image_to_world(points_2d, M_cropped_reverse, M_bev2world):
    points_bev = transform_line_coord_2d(points_2d, M_cropped_reverse)
    points_world = transform_line_coord_2d(points_bev, M_bev2world)
    return points_world

# Transform VMA lines to world coordinates
vma_lines = []
for timestamp, data in vma_data.items():
    pred_instances = data['pred_instances']
    for line_data in pred_instances:
        points_2d = np.array(line_data['data'])
        points_world = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
        vma_lines.append(points_world)

print(f"Loaded {len(vma_lines)} VMA lines")

# Load GT from per-frame annotations
gt_lines = []
for timestamp in vma_data.keys():
    timestamp_key = timestamp.split('/')[-1].replace('.jpg', '')
    annot_file_path = f"{annot_dir}/{timestamp_key}.json"

    try:
        with open(annot_file_path, 'r') as f:
            annot_data = json.load(f)

        if 'instances' in annot_data:
            for inst in annot_data['instances']:
                if 'semantic_line' in inst and inst['semantic_line'] is not None:
                    if 'position' in inst['semantic_line']:
                        line_points = []
                        for pt in inst['semantic_line']['position']:
                            line_points.append([pt['x'], pt['y']])
                        if len(line_points) >= 2:
                            gt_lines.append(np.array(line_points))
    except FileNotFoundError:
        continue

print(f"Loaded {len(gt_lines)} GT lines")

# Create visualization
fig, ax = plt.subplots(figsize=(20, 16))

# Plot point cloud (colored by Z)
z_colors = points[:, 2]
scatter = ax.scatter(points[:, 0], points[:, 1], c=z_colors, cmap='viridis',
                    s=0.1, alpha=0.3, label='Point Cloud')

# Plot VMA lines (red)
for line in vma_lines:
    ax.plot(line[:, 0], line[:, 1], 'r-', linewidth=1.5, alpha=0.7)

# Plot GT lines (blue)
for line in gt_lines:
    ax.plot(line[:, 0], line[:, 1], 'b-', linewidth=2, alpha=0.8)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='r', linewidth=2, label=f'VMA Detection ({len(vma_lines)} lines)'),
    Line2D([0], [0], color='b', linewidth=2, label=f'GT Annotation ({len(gt_lines)} lines)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

ax.set_xlabel('X (m)', fontsize=14)
ax.set_ylabel('Y (m)', fontsize=14)
ax.set_title('VMA Detection vs GT Annotation on Point Cloud', fontsize=16)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=ax, label='Z (m)')
plt.tight_layout()
plt.savefig('vma_gt_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to vma_gt_comparison.png")

# Print spatial ranges
vma_all = np.vstack(vma_lines)
gt_all = np.vstack(gt_lines)
print(f"\nVMA spatial range: X=[{vma_all[:, 0].min():.1f}, {vma_all[:, 0].max():.1f}], Y=[{vma_all[:, 1].min():.1f}, {vma_all[:, 1].max():.1f}]")
print(f"GT spatial range: X=[{gt_all[:, 0].min():.1f}, {gt_all[:, 0].max():.1f}], Y=[{gt_all[:, 1].min():.1f}, {gt_all[:, 1].max():.1f}]")
