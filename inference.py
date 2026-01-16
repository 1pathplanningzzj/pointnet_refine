import torch
import numpy as np
from src.model import PointNetRefine


def get_canonical_transform(line_segment):
    """
    计算从 Global 到 Local (以线段为中心) 的变换矩阵与平移量
    """
    p_start = line_segment[0]
    p_end = line_segment[1]
    mid_point = (p_start + p_end) / 2.0
    
    tangent = p_end - p_start
    tangent[2] = 0 
    tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
    
    z_axis = np.array([0, 0, 1.0])
    normal = np.cross(z_axis, tangent) 
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    
    # R_global_to_local 的行向量分别是新的 x, y, z 轴
    R_global_to_local = np.vstack([tangent, normal, z_axis]) 
    
    return mid_point, R_global_to_local, normal

def inference_single_line(model, line_segment, context_points, device, num_points=512):
    """
    Args:
        model: Trained PointNetRefine
        line_segment: (2, 3) VMA预测的线段
        context_points: (N, 3) 原始点云
    Returns:
        refined_segment: (2, 3)
    """
    # 1. Canonicalization
    mid_point, R, normal_vec = get_canonical_transform(line_segment)
    
    # Transform points to local
    points_centered = context_points - mid_point
    points_local = points_centered @ R.T # (N, 3)
    
    # 2. Sampling (Simple Random or Farthest Point)
    curr_n = points_local.shape[0]
    if curr_n == 0:
        return line_segment # No points, skip
        
    if curr_n >= num_points:
        choice = np.random.choice(curr_n, num_points, replace=False)
        points_local = points_local[choice, :]
    else:
        choice = np.random.choice(curr_n, num_points, replace=True)
        points_local = points_local[choice, :]
        
    # 3. Model Prediction
    points_tensor = torch.from_numpy(points_local).float().transpose(1, 0).unsqueeze(0) # (1, 3, N)
    points_tensor = points_tensor.to(device)
    
    with torch.no_grad():
        # model output scaled offset
        pred_offset_scalar = model(points_tensor) # (1, 1)
        pred_offset = pred_offset_scalar.item()
        
    # 4. Apply Offset back to Global
    # Offset 是沿着 Normal 方向的距离
    # 我们之前的定义: Label 是 "GT相对于Line的距离"。
    # 所以如果 Label 是 +0.2，表示 GT 在 Line 左边 0.2m。
    # 我们要把 Line 往左移 0.2m 才能贴合 GT。
    # 所以 Refined = Original + Normal * Offset
    
    offset_vec = normal_vec * pred_offset
    
    refined_start = line_segment[0] + offset_vec
    refined_end = line_segment[1] + offset_vec
    
    return np.array([refined_start, refined_end])

if __name__ == "__main__":
    # Load Model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetRefine(input_dim=3, output_dim=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load("pointnet_refine.pth"))
        print("Model loaded.")
    except:
        print("Warning: Model weight not found, using random init.")
    model.eval()
    
    # Mock VMA Output (A line segment)
    vma_pred_line = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]])
    
    # Mock Truth (Offset by +0.3m along normal)
    # Normal of (1,1,0) is (-0.7, 0.7, 0)
    normal = np.array([-0.707, 0.707, 0])
    gt_offset = 0.3
    
    # Mock Points around GT line
    mock_points = []
    for i in range(100):
        t = np.random.rand()
        pt_on_line = vma_pred_line[0] + (vma_pred_line[1] - vma_pred_line[0]) * t
        pt_gt = pt_on_line + normal * gt_offset
        noise = (np.random.rand(3) - 0.5) * 0.05
        mock_points.append(pt_gt + noise)
    mock_points = np.array(mock_points)
    
    print("Original Line Start:", vma_pred_line[0])
    
    # Run Inference
    refined_line = inference_single_line(model, vma_pred_line, mock_points, DEVICE)
    
    print("Refined Line Start: ", refined_line[0])
    print("Expected Shift Dir: ", normal)
    
    # Check if moved in correct direction
    diff = refined_line[0] - vma_pred_line[0]
    print("Actual Adjustment:  ", diff)
