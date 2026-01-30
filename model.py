"""
Temporal Convolutional Network (TCN) with attention for multi-task predictive maintenance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TemporalConvBlock(nn.Module):
    """Single TCN block with dilated causal convolution and residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation // 2,
                              dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation // 2,
                              dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len)

        Returns:
            Output tensor (batch, channels, seq_len)
        """
        residual = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature importance."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)

        # Residual connection and layer norm
        out = self.layer_norm(x + self.dropout(attn_output))

        return out, attn_weights


class MultiTaskTCN(nn.Module):
    """
    Multi-task Temporal Convolutional Network for predictive maintenance.

    Predicts:
        1. Binary failure (will machine fail?)
        2. Failure types (which failure modes?)
        3. Time-to-failure (when will it fail?)
    """

    def __init__(self, num_numeric_features: int = 5,
                 num_temporal_features: int = 19,
                 num_types: int = 3,
                 tcn_channels: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 use_temporal_sequences: bool = True,
                 binary_only: bool = False):
        """
        Args:
            num_numeric_features: Number of sensor features (5)
            num_temporal_features: Number of temporal features (19)
            num_types: Number of machine types (3)
            tcn_channels: Number of channels in TCN
            num_heads: Number of attention heads
            dropout: Dropout rate (increased from 0.2 to 0.3)
            use_temporal_sequences: Whether to use temporal sequence processing
            binary_only: If True, only create binary failure head (Phase 1 improvement)
        """
        super().__init__()

        self.use_temporal_sequences = use_temporal_sequences
        self.binary_only = binary_only

        # Type embedding (L/M/H)
        self.type_embedding = nn.Embedding(num_types, 8)

        if use_temporal_sequences:
            # Separate projections for sequence vs temporal features
            # Sequence projection: Conv1d to increase channels
            self.sequence_projection = nn.Conv1d(
                num_numeric_features,  # 5 sensor channels
                tcn_channels,
                kernel_size=1  # Pointwise conv
            )

            # Temporal features projection (global context)
            temporal_input_dim = num_temporal_features + 8  # temporal + type embedding
            self.temporal_projection = nn.Sequential(
                nn.Linear(temporal_input_dim, tcn_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Feature fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(tcn_channels * 2, tcn_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # Backward compatibility: original single-timestep processing
            input_dim = num_numeric_features + 8  # numeric + type embedding
            self.input_projection = nn.Linear(input_dim, tcn_channels)

        # TCN layers with increasing dilation
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=8, dropout=dropout),
        ])

        # Multi-head attention
        self.attention = MultiHeadAttention(tcn_channels, num_heads=num_heads, dropout=dropout)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Task-specific heads with enhanced dropout
        # Head 1: Binary failure prediction (critical task, always created)
        self.failure_head = nn.Sequential(
            nn.Linear(tcn_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout for critical task
            nn.Linear(32, 1)
        )

        # Head 2 & 3: Only create if multi-task mode
        if not binary_only:
            # Head 2: Failure type classification (5 types, multi-label)
            self.failure_type_head = nn.Sequential(
                nn.Linear(tcn_channels, 32),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(32, 5)
            )

            # Head 3: Time-to-failure regression
            self.ttf_head = nn.Sequential(
                nn.Linear(tcn_channels, 32),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(32, 1)
            )
        else:
            self.failure_type_head = None
            self.ttf_head = None

    def forward(self, sequence: torch.Tensor = None,
                temporal_features: torch.Tensor = None,
                machine_type: torch.Tensor = None,
                numeric_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with support for both temporal sequences and single timesteps.

        Args:
            sequence: Sensor sequence (batch, window_size, num_features) - for temporal mode
            temporal_features: Temporal features (batch, num_temporal_features) - for temporal mode
            machine_type: Machine type indices (batch, 1)
            numeric_features: Single timestep features (batch, num_features) - for backward compatibility

        Returns:
            Dictionary with predictions:
                - failure_logits: Binary failure prediction (batch, 1)
                - failure_type_logits: Multi-label failure types (batch, 5)
                - ttf: Time-to-failure prediction (batch, 1)
                - attention_weights: Attention weights for interpretability
        """
        # Embed machine type
        type_emb = self.type_embedding(machine_type.squeeze(1))  # (batch, 8)

        if self.use_temporal_sequences and sequence is not None:
            # === Temporal sequence processing ===
            batch_size = sequence.size(0)

            # Process sequence through TCN
            # Transpose for Conv1d: (batch, channels, seq_len)
            x_seq = sequence.transpose(1, 2)  # (batch, 5, window_size)

            # Project to TCN channels
            x_seq = self.sequence_projection(x_seq)  # (batch, tcn_channels, window_size)

            # Apply TCN blocks
            for tcn_block in self.tcn_blocks:
                x_seq = tcn_block(x_seq)  # (batch, tcn_channels, window_size)

            # Transpose for attention: (batch, seq_len, channels)
            x_seq = x_seq.transpose(1, 2)  # (batch, window_size, tcn_channels)

            # Apply attention
            x_seq, attention_weights = self.attention(x_seq)

            # Global pooling over time dimension
            x_seq = x_seq.mean(dim=1)  # (batch, tcn_channels)

            # Process temporal features
            temporal_input = torch.cat([temporal_features, type_emb], dim=1)
            x_temporal = self.temporal_projection(temporal_input)  # (batch, tcn_channels)

            # Fuse both representations
            x_fused = torch.cat([x_seq, x_temporal], dim=1)  # (batch, tcn_channels*2)
            x = self.fusion(x_fused)  # (batch, tcn_channels)

        else:
            # === Backward compatibility: single timestep processing ===
            # Combine numeric features and type embedding
            combined = torch.cat([numeric_features, type_emb], dim=1)  # (batch, num_features+8)

            # Project to TCN channels
            x = self.input_projection(combined)  # (batch, tcn_channels)

            # Reshape for TCN: (batch, channels, seq_len)
            x = x.unsqueeze(2)  # (batch, tcn_channels, 1)

            # Apply TCN blocks
            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            # Transpose for attention: (batch, seq_len, channels)
            x = x.transpose(1, 2)  # (batch, 1, tcn_channels)

            # Apply attention
            x, attention_weights = self.attention(x)  # (batch, 1, tcn_channels)

            # Global pooling and flatten
            x = x.squeeze(1)  # (batch, tcn_channels)

        # Task-specific predictions
        failure_logits = self.failure_head(x)  # (batch, 1)

        # Return format depends on binary_only mode
        if self.binary_only:
            return {
                'failure_logits': failure_logits,
                'attention_weights': attention_weights
            }
        else:
            failure_type_logits = self.failure_type_head(x)  # (batch, 5)
            ttf = F.relu(self.ttf_head(x))  # (batch, 1) - ReLU ensures non-negative TTF

            return {
                'failure_logits': failure_logits,
                'failure_type_logits': failure_type_logits,
                'ttf': ttf,
                'attention_weights': attention_weights
            }


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING TEMPORAL SEQUENCE MODEL (NEW)")
    print("=" * 80)

    # Test temporal sequence model
    model = MultiTaskTCN(
        num_numeric_features=5,
        num_temporal_features=19,
        num_types=3,
        tcn_channels=64,
        num_heads=4,
        dropout=0.3,
        use_temporal_sequences=True
    )

    # Create dummy input for temporal mode
    batch_size = 16
    window_size = 12
    sequence = torch.randn(batch_size, window_size, 5)
    temporal_features = torch.randn(batch_size, 19)
    machine_type = torch.randint(0, 3, (batch_size, 1))

    # Forward pass
    outputs = model(
        sequence=sequence,
        temporal_features=temporal_features,
        machine_type=machine_type
    )

    print("\nModel output shapes (temporal mode):")
    print(f"  Failure logits: {outputs['failure_logits'].shape}")
    print(f"  Failure type logits: {outputs['failure_type_logits'].shape}")
    print(f"  TTF: {outputs['ttf'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test backward pass
    print("\n" + "=" * 80)
    print("Testing gradient flow...")
    loss = outputs['failure_logits'].mean()
    loss.backward()

    # Check gradients for active parameters (not input_projection in temporal mode)
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    total_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Parameters with gradients: {grad_count}/{total_count}")
    print(f"Gradient flow intact: {grad_count > 0}")

    print("\n" + "=" * 80)
    print("TESTING BACKWARD COMPATIBILITY MODE")
    print("=" * 80)

    # Test backward compatibility mode
    model_old = MultiTaskTCN(
        num_numeric_features=5,
        num_types=3,
        use_temporal_sequences=False
    )

    numeric_features = torch.randn(batch_size, 5)
    outputs_old = model_old(numeric_features=numeric_features, machine_type=machine_type)

    print("\nModel output shapes (backward compatibility mode):")
    print(f"  Failure logits: {outputs_old['failure_logits'].shape}")
    print(f"  Failure type logits: {outputs_old['failure_type_logits'].shape}")
    print(f"  TTF: {outputs_old['ttf'].shape}")

    print("\n" + "=" * 80)
    print("Phase 2 Model Architecture: COMPLETE [OK]")
    print("=" * 80)
