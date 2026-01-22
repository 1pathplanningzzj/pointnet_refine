import json
import os

file1 = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
file2 = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag.json"

try:
    with open(file1, 'r') as f:
        data1 = json.load(f)
        keys1 = list(data1.keys())
        # keys like "TAD...bag/timestamp.jpg"
        ts1 = set()
        for k in keys1:
            try:
                base = os.path.basename(k)
                ts = base.split('.jpg')[0]
                ts1.add(ts)
            except:
                pass
except Exception as e:
    print(f"Error reading file 1: {e}")
    ts1 = set()

try:
    with open(file2, 'r') as f:
        data2 = json.load(f)
        # data2 has "images" list
        images2 = data2.get("images", [])
        ts2 = set()
        for img in images2:
            try:
                ts = img.split('.jpg')[0]
                ts2.add(ts)
            except:
                pass
except Exception as e:
    print(f"Error reading file 2: {e}")
    ts2 = set()

print(f"File 1 (results) has {len(ts1)} timestamps.")
print(f"File 2 (bag json) has {len(ts2)} timestamps.")

intersection = ts1.intersection(ts2)
print(f"Intersection count: {len(intersection)}")

if len(ts1) > 0 and len(ts2) > 0:
    sorted_ts1 = sorted(list(ts1))
    sorted_ts2 = sorted(list(ts2))
    print(f"File 1 range: {sorted_ts1[0]} - {sorted_ts1[-1]}")
    print(f"File 2 range: {sorted_ts2[0]} - {sorted_ts2[-1]}")

    # Check for near matches?
    # Convert to int?
    try:
        int_ts1 = [int(t) for t in sorted_ts1]
        int_ts2 = [int(t) for t in sorted_ts2]
        
        diff = int_ts1[0] - int_ts2[0]
        print(f"Difference in start timestamp: {diff}")
    except:
        pass

