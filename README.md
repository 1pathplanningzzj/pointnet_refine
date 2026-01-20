# PointNet Lane Refinement

This project implements a PointNet-based neural network to refine noisy lane lines using local point cloud context.

## Project Structure

```text
pointnet_refine/
├── src/                # Core library
│   ├── dataset.py      # Dataset loading and processing
│   └── model.py        # LineRefineNet (PointNet + MLP) architecture
├── tools/              # Helper scripts
│   ├── generate_train_data.py      # Process bag -> training data slices
│   ├── generate_inference_data.py  # Process bag -> inference data slices
│   ├── augment_train_data.py       # Add noise to training data
│   ├── augment_inference_data.py   # Add noise to inference data
│   └── visualize_data.py           # Interactive 3D visualization tool
├── train_data/         # Training dataset (generated)
├── inference_data/     # Inference/Validation dataset (generated)
├── checkpoints/        # Saved model weights
├── train.py            # Main training script
├── inference.py        # Main inference validation script
└── backup/             # Archived old scripts
```

## Quick Start

### 1. Data Preparation

First, slice the raw rosbag data into segments:

```bash
# For Training Data
python tools/generate_train_data.py

# For Inference Data
python tools/generate_inference_data.py
```

Then, augment the data with synthetic noise (needed for training input):

```bash
# Augment Training Data
python tools/augment_train_data.py

# Augment Inference Data (creates noisy inputs to test model)
python tools/augment_inference_data.py
```

### 2. Training

Train the PointNet Refinement model:

```bash
python train.py
```

The model will be saved to `checkpoints/best_model.pth`.

### 3. Inference & Visualization

Run inference on the new dataset:

```bash
python inference.py
```

- This script loads `checkpoints/best_model.pth`.
- Generates 3D visualization results in `inference_vis/`.
- Open the generated `.html` files in a browser to view the **Context (Points)**, **Noisy Input (Red)**, and **Refined Output (Cyan)**.

## Environment

- PyTorch
- NumPy, SciPy
- Open3D / Plotly (for visualization)
