import os
import json
import glob
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

# 复用你原来脚本里的工具函数（同目录下的 generate_inference_data_vma.py）
from generate_inference_data_vma import (
    load_poses,
    load_pcd_fast,
    transform_to_local,
    clip_polyline_by_x,
    save_pcd,
    save_json_vma_direct,
    SEGMENT_LEN,
)

DATA_ROOT = "/homes/zhangzijian/pointnet_refine/data/vma_infer_data_test"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/vma_infer_data_v2"

# 允许的 pose 与推理结果时间差（纳秒）
MAX_TS_DIFF_NS = 250_000_000  # 250ms


def load_gt_items(json_path):
    """
    加载 GT 数据，兼容两种格式：
    1. 旧格式: {"items": [{"position": [{"x":..., "y":..., "z":...}]}]}
    2. 新格式: {"lane_lines": [{"pts": [[xs], [ys], [zs]]}]}
    """
    print(f"Loading GT {json_path}...")
    if not os.path.exists(json_path):
        print(f"GT file not found: {json_path}")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    items = []
    
    # 新格式：lane_lines
    if 'lane_lines' in data:
        for lane in data['lane_lines']:
            pts = lane.get('pts', [])
            if not pts or len(pts) < 2:
                continue
            
            # pts 格式: [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
            xs = pts[0] if len(pts) > 0 else []
            ys = pts[1] if len(pts) > 1 else []
            zs = pts[2] if len(pts) > 2 else [0.0] * len(xs)
            
            n = min(len(xs), len(ys), len(zs) if zs else len(xs))
            if n < 2:
                continue
            
            # 转换为点数组
            points = np.array([[xs[i], ys[i], zs[i] if zs else 0.0] for i in range(n)], dtype=np.float32)
            
            items.append({
                'category': 'lane_line',
                'points': points,
                'attributes': {
                    'line_type': lane.get('line_type', 0),
                    'score': lane.get('score', 1.0)
                }
            })
    
    # 旧格式：items
    elif 'items' in data:
        for item in data['items']:
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
    
    print(f"Loaded {len(items)} GT lines.")
    return items


def load_vma_pred_lines_one_frame(json_path, pose):
    """
    从 VMA 的每帧 json（vma_infer/*.json）里取出 lane_lines。
    VMA 的 'pts' 格式是 [[xs], [ys], [zs]]，已经是局部坐标系（ego坐标系），
    但坐标范围是 X: [0, 50m]，需要转换到训练数据使用的范围 X: [-25, 25m]。
    返回：
      - pred_lines_3d: List[List[{'x','y','z'}]]
      - pred_scores:   List[float]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    pred_lines_3d = []
    pred_scores = []

    lane_lines = data.get("lane_lines", [])
    for lane in lane_lines:
        pts = lane.get("pts", [])
        if not pts or len(pts) < 2:
            continue

        xs = pts[0]
        ys = pts[1]
        zs = pts[2] if len(pts) > 2 else [0.0] * len(xs)
        n = min(len(xs), len(ys), len(zs) if zs else len(xs))
        if n < 2:
            continue

        # VMA 坐标已经是局部坐标系，但范围是 X: [0, 50m]
        # 需要转换到训练数据范围 X: [-25, 25m]
        # 转换公式: x_new = x_old - 25.0
        local_xs = np.array(xs[:n], dtype=np.float32) - 25.0
        local_ys = np.array(ys[:n], dtype=np.float32)
        local_zs = np.array(zs[:n], dtype=np.float32) if zs else np.zeros(n, dtype=np.float32)

        poly_3d = [
            {"x": float(local_xs[i]), "y": float(local_ys[i]), "z": float(local_zs[i])}
            for i in range(n)
        ]
        if len(poly_3d) > 1:
            pred_lines_3d.append(poly_3d)
            pred_scores.append(float(lane.get("score", 0.0)))

    return pred_lines_3d, pred_scores


def process_one_bag(bag_base_name: str):
    """
    处理一个包，例如:
      bag_base_name = 'TAD_front_lidar_2025-09-20-10-12-26_43_20to60'
    目录结构假设为：
      - DATA_ROOT / <bag>_annotation_raw_data / merged.pcd, pose/*.json
      - DATA_ROOT / <bag>_lane_autolabel / vma_infer/*.json
      - DATA_ROOT / <bag>.bag.json   (GT)
    """
    print(f"\n=== Processing bag: {bag_base_name} ===")

    raw_data_dir = os.path.join(DATA_ROOT, f"{bag_base_name}_annotation_raw_data")
    pcd_path = os.path.join(raw_data_dir, "merged.pcd")
    pose_dir = os.path.join(raw_data_dir, "pose")
    gt_json_path = os.path.join(DATA_ROOT, f"{bag_base_name}.bag.json")
    results_dir = os.path.join(
        DATA_ROOT, f"{bag_base_name}_lane_autolabel", "vma_infer"
    )

    if not os.path.exists(results_dir):
        print(f"  No vma_infer dir found: {results_dir}, skip.")
        return

    if not os.path.exists(pcd_path):
        print(f"  No PCD found: {pcd_path}, skip.")
        return

    if not os.path.exists(pose_dir):
        print(f"  No pose dir found: {pose_dir}, skip.")
        return

    if not os.path.exists(gt_json_path):
        print(f"  No GT bag json found: {gt_json_path}, skip.")
        return

    # 加载姿态、点云和 GT
    poses = load_poses(pose_dir)
    if len(poses) == 0:
        print("  No poses loaded, skip bag.")
        return

    all_points = load_pcd_fast(pcd_path)
    if len(all_points) == 0:
        print("  Empty PCD, skip bag.")
        return

    gt_items = load_gt_items(gt_json_path)
    print(f"  Loaded {len(gt_items)} GT items from {gt_json_path}")

    # 收集所有帧结果 json
    result_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not result_files:
        print(f"  No VMA infer jsons in {results_dir}")
        return

    print(f"  Found {len(result_files)} infer frames in {results_dir}")

    generated_count = 0

    for jf in result_files:
        ts_str = os.path.splitext(os.path.basename(jf))[0]
        try:
            res_ts = int(ts_str)
        except ValueError:
            print(f"  Skip file (ts parse failed): {jf}")
            continue

        # 匹配最近姿态
        closest_pose = min(poses, key=lambda p: abs(int(p["ts"]) - res_ts))
        diff = abs(int(closest_pose["ts"]) - res_ts)

        if diff > MAX_TS_DIFF_NS:
            print(
                f"  Skipping {res_ts}: pose {closest_pose['ts']} too far "
                f"({diff/1e6:.1f} ms)"
            )
            continue

        pose_ts = int(closest_pose["ts"])

        # 1) 点云过滤 + 转到 local 坐标
        dx = all_points[:, 0] - closest_pose["x"]
        dy = all_points[:, 1] - closest_pose["y"]
        mask_radius = (dx ** 2 + dy ** 2) < (60 ** 2)
        subset_points = all_points[mask_radius].copy()
        if len(subset_points) == 0:
            continue

        local_xyz = transform_to_local(subset_points, closest_pose)
        if subset_points.shape[1] >= 4:
            local_points = np.hstack([local_xyz, subset_points[:, 3:4]])
        else:
            local_points = local_xyz

        # 和训练数据一样裁剪到 [-SEGMENT_LEN/2, SEGMENT_LEN/2] (即 [-25,25])
        mask_final = (local_points[:, 0] >= -SEGMENT_LEN / 2) & (
            local_points[:, 0] <= SEGMENT_LEN / 2
        )
        final_points = local_points[mask_final]
        if len(final_points) == 0:
            continue

        # 2) 将 GT 转到 local 并裁剪，作为 context_lines
        local_gt_lines_for_vis = []
        for item in gt_items:
            gt_local = transform_to_local(item["points"], closest_pose)
            if np.any(
                (gt_local[:, 0] > -SEGMENT_LEN / 2)
                & (gt_local[:, 0] < SEGMENT_LEN / 2)
            ):
                clipped = clip_polyline_by_x(
                    gt_local, -SEGMENT_LEN / 2, SEGMENT_LEN / 2
                )
                if len(clipped) > 1:
                    pos_list = [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                        for p in clipped
                    ]
                    local_gt_lines_for_vis.append(pos_list)

        # 3) 加载当前帧的 VMA 预测（全局 -> local）
        pred_lines_3d, pred_scores = load_vma_pred_lines_one_frame(jf, closest_pose)

        num_preds = len(pred_lines_3d)
        num_gts = len(local_gt_lines_for_vis)
        matched_gt_indices = {}

        # 保持和旧脚本一样的 Hungarian 匹配逻辑
        if num_preds > 0 and num_gts > 0:
            cost_matrix = np.full((num_preds, num_gts), 1000.0, dtype=np.float32)
            for i, p_line in enumerate(pred_lines_3d):
                p_pts = np.array([[p["x"], p["y"]] for p in p_line], dtype=np.float32)
                for j, g_line in enumerate(local_gt_lines_for_vis):
                    g_pts = np.array(
                        [[p["x"], p["y"]] for p in g_line], dtype=np.float32
                    )
                    if len(g_pts) == 0:
                        continue
                    diff_mat = p_pts[:, None, :] - g_pts[None, :, :]
                    dists_matrix = np.linalg.norm(diff_mat, axis=2)
                    min_dists = np.min(dists_matrix, axis=1)
                    dist = float(np.mean(min_dists))
                    cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # 降低匹配阈值：从 15.0m 降到 3.0m，避免匹配到错误的 GT
            # 如果 VMA 预测和 GT 距离超过 3m，很可能是误检或匹配错误
            MATCH_THRESHOLD = 3.0
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < MATCH_THRESHOLD:
                    matched_gt_indices[r] = c
                # 如果距离太大，不匹配（让 position 为空，表示没有匹配的 GT）

        # 4) 组装输出 items
        vma_items = []
        for i, poly_3d in enumerate(pred_lines_3d):
            best_gt = []
            if i in matched_gt_indices:
                gt_idx = matched_gt_indices[i]
                best_gt = local_gt_lines_for_vis[gt_idx]

            item_dict = {
                "category": "lane_line",
                "attributes": {"score": pred_scores[i]},
                "position": best_gt,  # 匹配到的 GT (local) 或空
                "noisy_candidates": [poly_3d],  # 当前预测线 (local)
                "context_lines": local_gt_lines_for_vis,
            }
            vma_items.append(item_dict)

        if len(vma_items) == 0:
            print(f"  {res_ts}: no valid predictions, skip.")
            continue

        # 5) 保存 PCD + JSON
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        pcd_out = os.path.join(OUTPUT_DIR, f"{res_ts}.pcd")
        json_out = os.path.join(OUTPUT_DIR, f"{res_ts}.json")
        save_pcd(pcd_out, final_points)
        save_json_vma_direct(json_out, vma_items, pose_ts, res_ts)

        generated_count += 1
        if generated_count % 10 == 0:
            print(f"  Generated {generated_count} samples so far...")

    print(f"Bag {bag_base_name}: generated {generated_count} samples.")


def main():
    # 自动找所有 *_lane_autolabel 的 bag
    lane_dirs = glob.glob(os.path.join(DATA_ROOT, "*_lane_autolabel"))
    if not lane_dirs:
        print(f"No *_lane_autolabel dirs found under {DATA_ROOT}")
        return

    bag_names = [
        os.path.basename(d).replace("_lane_autolabel", "") for d in lane_dirs
    ]
    bag_names = sorted(set(bag_names))

    print("Bags to process:")
    for name in bag_names:
        print("  ", name)

    for name in bag_names:
        process_one_bag(name)


if __name__ == "__main__":
    main()