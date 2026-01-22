import json
import os
import numpy as np

file1 = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
file2 = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag.json"

with open(file1, 'r') as f:
    ts1 = sorted([int(os.path.basename(k).split('.jpg')[0]) for k in json.load(f).keys()])

with open(file2, 'r') as f:
    images = json.load(f).get("images", [])
    ts2 = sorted(list(set([int(img.split('.jpg')[0]) for img in images])))

print(f"Count F1: {len(ts1)}, F2: {len(ts2)}")

for t1 in ts1:
    # Find closest in ts2
    diffs = [abs(t1 - t2) for t2 in ts2]
    min_diff = min(diffs)
    idx = diffs.index(min_diff)
    closest_t2 = ts2[idx]
    
    # Convert nsec to ms
    diff_ms = min_diff / 1e6
    print(f"F1: {t1} -> F2: {closest_t2} | Diff: {diff_ms:.2f} ms")

