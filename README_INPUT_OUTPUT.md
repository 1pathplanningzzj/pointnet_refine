# PointNet Refine 网络：Input/Output 与 Loss 说明

## 📋 整体流程概览

```
VMA 预测的 Coarse 线 (折线，25个点)
    ↓
[预处理阶段] ROI Crop + 坐标变换 (Canonicalization)
    ↓
局部点云 (N个点，已转换到局部坐标系)
    ↓
[PointNetRefine 网络]
    ↓
预测的 Offset (标量，单位：米)
    ↓
[后处理] 将 Offset 加回 Coarse 线
    ↓
Refined 线 (更精确)
```

---

## 🔍 Input（输入）

### 1. **网络直接接收的输入**

**形状**: `[Batch, Channel, NumPoints]` = `[B, 3, N]`

- **B**: Batch size（训练时通常 32）
- **Channel = 3**: 点的坐标 `(x_local, y_local, z_local)`
  - 这些坐标已经过 **canonicalization**（局部坐标系转换）
  - `x_local`: 沿着 coarse 线的方向（切线方向）
  - `y_local`: 垂直于 coarse 线的方向（**法线方向，这是我们要回归 offset 的方向**）
  - `z_local`: 垂直地面方向
- **N**: 固定点数（如 512），通过采样/补全得到

**代码位置**:
- `src/dataset.py` 的 `__getitem__` 方法返回 `points_tensor`，形状 `[3, N]`
- `train.py` 第 58-59 行：`points, target = points.to(DEVICE), target.to(DEVICE)`

### 2. **Coarse 线的作用（间接输入）**

**重要**: Coarse 线（VMA 预测）**并不直接作为网络的输入特征**，而是用于：

1. **ROI 裁剪**: 确定从原始点云中 crop 哪些点（以 coarse 线为中心，周围 2-5 米）
2. **坐标变换**: 
   - 将点云从全局坐标系转换到局部坐标系
   - 以 coarse 线的中点为原点
   - 将 coarse 线的方向对齐到 X 轴
   - 这样 Y 轴天然就是法线方向

**代码位置**:
- `src/dataset.py` 的 `normalize_pc` 方法（第 26-54 行）
- 使用 `pred_point`（coarse 线中点）和 `line_direction`（coarse 线方向）进行变换

---

## 📤 Output（输出）

**形状**: `[Batch, 1]` = `[B, 1]`

- **标量 offset**（单位：米）
- **物理含义**: 沿法线方向（`y_local` 方向），coarse 线需要移动多少距离才能贴合 GT
- **正负号约定**: 
  - 如果 `offset > 0`：需要沿法线正方向移动
  - 如果 `offset < 0`：需要沿法线负方向移动
  - 具体约定取决于数据构造时的定义（见 `generate_mock_training_data`）

**代码位置**:
- `src/model.py` 第 55 行：`offset = self.fc3(x)`，返回 `[B, 1]`
- `train.py` 第 62 行：`output = model(points)`，`output` 就是预测的 offset

---

## 🎯 Loss Function（损失函数）

### 位置

**文件**: `train.py`

**定义位置**（第 48 行）:
```python
criterion = nn.SmoothL1Loss()
```

**计算位置**（第 64 行）:
```python
loss = criterion(output, target)
```

其中：
- `output`: `[B, 1]` - 网络预测的 offset
- `target`: `[B, 1]` - GT offset（标签）
- `loss`: `scalar` - 单个损失值

### 损失函数类型

**这不是"检测头"（Detection Head），而是回归头（Regression Head）**

- **任务类型**: 回归任务（预测连续值），不是分类或目标检测
- **SmoothL1Loss**: 
  - 公式：`loss = 0.5 * (x - y)^2 / beta` if `|x - y| < beta`，否则 `|x - y| - 0.5 * beta`
  - 优点：结合了 L1（对异常值鲁棒）和 L2（在 0 附近平滑）的优点
  - 默认 `beta=1.0`

### 可选的损失函数

你也可以替换为：
- `nn.L1Loss()`: `|pred - gt|`，对异常值更鲁棒
- `nn.MSELoss()`: `(pred - gt)^2`，对大误差惩罚更重
- `nn.SmoothL1Loss()`: 当前使用，平衡两者

---

## 🔄 完整的数据流示例

### 训练时

```python
# 1. Dataset 返回
points_tensor: [3, 512]  # 局部坐标系点云
label_tensor: [1]        # GT offset，例如 [0.15] 表示需要移动 0.15 米

# 2. Batch 组织（DataLoader）
points: [32, 3, 512]     # Batch size = 32
target: [32, 1]          # GT offsets

# 3. 网络前向
output = model(points)   # [32, 1] - 预测的 offsets

# 4. Loss 计算
loss = criterion(output, target)  # scalar
```

### 推理时

```python
# 1. VMA 输出 coarse 线
vma_line = [[x1, y1, z1], [x2, y2, z2], ...]  # 25 个点

# 2. 对每个关键点/线段：
#    - ROI crop 点云
#    - Canonicalization
#    - 输入网络
offset = model(points_local)  # [1] - 例如 [0.12]

# 3. 应用 offset
refined_point = coarse_point + normal_vec * offset
```

---

## 💡 改进方向（可选）

如果你想让网络**同时看到 coarse 线的特征**，可以考虑：

1. **Point-to-Line Attention**（你提到的 2.3 节）:
   - 输入：点云特征 + coarse 线的特征（如线段的方向向量、长度等）
   - 使用 Cross-Attention：Query 是线段特征，Key/Value 是点云特征

2. **多关键点联合回归**:
   - 输入：一条线上多个关键点的点云 patches
   - 输出：多个 offsets `[K, 1]`
   - Loss：回归损失 + 平滑项（约束相邻 offsets 不要突变）

---

## 📝 总结

| 项目 | 内容 |
|------|------|
| **Input** | 局部坐标系下的点云 `[B, 3, N]`（coarse 线只用于预处理） |
| **Output** | 标量 offset `[B, 1]`（单位：米） |
| **Loss** | `SmoothL1Loss`（回归损失），位置在 `train.py` 第 64 行 |
| **任务类型** | 回归任务（不是检测任务） |
