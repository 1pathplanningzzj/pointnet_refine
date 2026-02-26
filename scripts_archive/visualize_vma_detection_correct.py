import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_vma_lines_correct(vma_json_path, annot_dir):
    """Load VMA lines and transform to world coordinates using per-frame matrices"""
    with open(vma_json_path, 'r') as f:
        vma_data = json.load(f)

    all_lines_world = []

    for img_key, frame_data in vma_data.items():
        # Extract timestamp from key: "bag_name/timestamp.jpg"
        timestamp = img_key.split('/')[-1].replace('.jpg', '')
        annot_file = f"{annot_dir}/{timestamp}.json"

        # Load per-frame transformation matrices
        with open(annot_file, 'r') as f:
            annot = json.load(f)

        M_cropped_reverse = np.array(annot['M_cropped_reverse'])
        M_bev2world = np.array(annot['M_bev2world'])

        # Process each detection in this frame
        pred_instances = frame_data['pred_instances']
        for instance in pred_instances:
            line_2d = np.array(instance['data'])  # Shape: (N, 2)

            # Transform: image -> BEV -> world
            line_2d_homo = np.concatenate([line_2d, np.ones((len(line_2d), 1))], axis=1)
            line_bev = (M_cropped_reverse @ line_2d_homo.T).T
            line_bev = line_bev[:, :2] / line_bev[:, 2:3]

            line_bev_homo = np.concatenate([line_bev, np.ones((len(line_bev), 1))], axis=1)
            line_world = (M_bev2world @ line_bev_homo.T).T
            line_world = line_world[:, :2] / line_world[:, 2:3]

            all_lines_world.append(line_world)

    return all_lines_world

def load_point_cloud(pcd_dir):
    """Load and merge point cloud from all frames"""
    pcd_files = sorted(Path(pcd_dir).glob('*.npy'))
    all_points = []

    for pcd_file in pcd_files:
        points = np.load(pcd_file)
        all_points.append(points)

    merged_pcd = np.concatenate(all_points, axis=0)
    return merged_pcd

def main():
    # Paths
    vma_json = '/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json'
    annot_dir = '/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/cropped_data/annots'
    pcd_dir = '/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/cropped_data/pcd'

    print("Loading VMA lines with per-frame transformation matrices...")
    vma_lines = load_vma_lines_correct(vma_json, annot_dir)
    print(f"Loaded {len(vma_lines)} VMA lines")

    print("Loading point cloud...")
    pcd = load_point_cloud(pcd_dir)
    print(f"Loaded {len(pcd)} points")

    # Create BEV visualization
    fig, ax = plt.subplots(figsize=(20, 8))

    # Plot point cloud
    ax.scatter(pcd[:, 0], pcd[:, 1], c='gray', s=0.1, alpha=0.3, label='Point Cloud')

    # Plot VMA lines
    for line in vma_lines:
        ax.plot(line[:, 0], line[:, 1], 'r-', linewidth=1.5, alpha=0.7)

    # Calculate and print spatial range
    all_vma_points = np.concatenate(vma_lines, axis=0)
    print(f"\nVMA spatial range:")
    print(f"  X: [{all_vma_points[:, 0].min():.2f}, {all_vma_points[:, 0].max():.2f}]")
    print(f"  Y: [{all_vma_points[:, 1].min():.2f}, {all_vma_points[:, 1].max():.2f}]")

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('VMA Detection Results in World Coordinates', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('vma_detection_world_coords_recreated.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: vma_detection_world_coords_recreated.png")
    plt.close()

if __name__ == '__main__':
    main()
