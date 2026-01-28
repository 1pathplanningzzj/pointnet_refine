import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScalePointNetEncoder(nn.Module):
    """Multi-scale PointNet encoder with dual pooling"""
    def __init__(self, in_channel=4, out_dim=1024):
        super(MultiScalePointNetEncoder, self).__init__()
        # Multi-scale feature extraction
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, out_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(out_dim)

        # Fusion layer for multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 128 + 256 + 512 + out_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        # Intensity-aware gating: use intensity channel to softly re-weight fused features.
        # 输入的第 4 维是 intensity，我们用一个小的 1x1 Conv MLP 将其映射到与 out_dim 一致的通道数，
        # 再通过 Sigmoid 得到 [0,1] 的权重，对 fused 做逐点乘法。
        self.intensity_gate = nn.Sequential(
            nn.Conv1d(1, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, N), C = 4 -> [x, y, z, intensity]
        # Separate intensity for gating
        intensity = x[:, 3:4, :]  # (B, 1, N)
        feat1 = F.relu(self.bn1(self.conv1(x)))      # (B, 64, N)
        feat2 = F.relu(self.bn2(self.conv2(feat1)))  # (B, 128, N)
        feat3 = F.relu(self.bn3(self.conv3(feat2)))  # (B, 256, N)
        feat4 = F.relu(self.bn4(self.conv4(feat3)))  # (B, 512, N)
        feat5 = F.relu(self.bn5(self.conv5(feat4)))  # (B, 1024, N)

        # Multi-scale fusion
        multi_scale = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        fused = self.fusion(multi_scale)  # (B, 1024, N)

        # Intensity-aware gating: high-intensity points (lane/curb cores) have larger weights
        gate = self.intensity_gate(intensity)  # (B, 1024, N)
        fused = fused * (0.5 + 0.5 * gate)     # 保持尺度在 [0.5, 1.0] 左右，避免过度放大

        # Dual pooling: max + avg
        max_pool = torch.max(fused, 2, keepdim=False)[0]  # (B, 1024)
        avg_pool = torch.mean(fused, 2, keepdim=False)    # (B, 1024)
        global_feat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2048)

        return global_feat, fused

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates (or MLP based)"""
    def __init__(self, in_dim=3, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, xyz):
        # xyz: (B, N, 3)
        return self.mlp(xyz)

class DetrTransformerDecoderLayer(nn.Module):
    """
    Standard DETR-style Transformer Decoder Layer.
    Decouples Content (tgt/memory) and Position (query_pos/pos).
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos=None, pos=None):
        """
        tgt: (B, M, C) - Query Features (Line points)
        memory: (B, N, C) - Key/Value Features (Context)
        query_pos: (B, M, C) - Query Positional Encoding
        pos: (B, N, C) - Memory Positional Encoding
        """
        # 1. Self Attention (Query-Query)
        # Q = K = tgt + query_pos
        # V = tgt
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross Attention (Query-Memory)
        # Q = tgt + query_pos
        # K = memory + pos
        # V = memory
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        # Note: If memory is a mask, we need to pass it here. But we don't use mask for now.
        tgt2 = self.cross_attn(q, k, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class LineRefineNet(nn.Module):
    def __init__(self, num_line_points=32, feature_dim=1024):
        super(LineRefineNet, self).__init__()

        # Config
        self.d_model = 256
        self.num_decoder_layers = 6 # Increased to 6 for iterative refinement

        # 1. Context Encoder (PointNet)
        self.context_encoder = MultiScalePointNetEncoder(in_channel=4, out_dim=feature_dim)
        self.context_proj = nn.Linear(feature_dim, self.d_model)

        # 2. Line Encoder (Initial Query Features)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.d_model, 1), 
            nn.BatchNorm1d(self.d_model)
        )
        
        # 3. Positional Encoding
        self.pos_emb = PositionalEncoding(in_dim=3, out_dim=self.d_model)

        # 4. Decoder Layers (Iterative)
        self.decoder_layers = nn.ModuleList([
            DetrTransformerDecoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=1024)
            for _ in range(self.num_decoder_layers)
        ])

        # 5. Regression Heads (One per layer, or shared)
        # Here we use separate heads for flexibility
        self.reg_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, 128),
                nn.ReLU(),
                nn.Linear(128, 3) # dx, dy, dz
            )
            for _ in range(self.num_decoder_layers)
        ])

    def forward(self, context, noisy_line):
        """
        context: (B, N, 4) - Point Cloud [x,y,z,i]
        noisy_line: (B, M, 3) - Noisy Polyline [x,y,z]
        Returns: 
           all_offsets_stack: (num_layers, B, M, 3) - Cumulative offsets at each layer
        """
        B, N, C = context.shape
        M = noisy_line.shape[1]

        # --- 1. Encode Context (Memory) ---
        ctx_trans = context.transpose(2, 1)
        _, ctx_pointwise = self.context_encoder(ctx_trans) 
        memory = self.context_proj(ctx_pointwise.transpose(2, 1)) # (B, N, 256)
        
        # Memory Position Embedding (Constant)
        pos_mem = self.pos_emb(context[:, :, :3]) # (B, N, 256)
        
        # --- 2. Encode Line (Initial Query) ---
        line_feat = self.point_mlp(noisy_line.transpose(2, 1))
        tgt = line_feat.transpose(2, 1) # (B, M, 256)

        # Initialize Refined Line Coordinates (for iterative updates)
        current_line_coords = noisy_line.clone() # (B, M, 3)
        
        all_pred_offsets = []

        # --- 3. Iterative Refinement Loop ---
        for i, (decoder_layer, reg_branch) in enumerate(zip(self.decoder_layers, self.reg_branches)):
            
            # Dynamic Positional Encoding (Based on CURRENT refined coordinates)
            pos_tgt = self.pos_emb(current_line_coords) # (B, M, 256)

            # Transformer Decoder Layer
            # tgt: updates across layers (content)
            # pos_tgt: updates across layers (geometry)
            tgt = decoder_layer(tgt, memory, query_pos=pos_tgt, pos=pos_mem)

            # Regress Delta Offset from features
            delta_offset = reg_branch(tgt) # (B, M, 3)
            
            # Update Coordinates
            # Predicted line = current_ref_line + delta
            # So the cumulative offset from original noisy_line is: (current - noisy) + delta
             
            # Update for next layer
            current_line_coords = current_line_coords + delta_offset
            
            # Store cumulative offset for Deep Supervision
            cum_offset = current_line_coords - noisy_line
            all_pred_offsets.append(cum_offset)

        # Stack outputs: (L, B, M, 3)
        return torch.stack(all_pred_offsets) 

if __name__ == '__main__':
    # Test
    fake_ctx = torch.randn(2, 1024, 4)
    fake_line = torch.randn(2, 32, 3)
    model = LineRefineNet()
    out = model(fake_ctx, fake_line)
    print("Output shape:", out.shape)  # Expect (6, 2, 32, 3)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
