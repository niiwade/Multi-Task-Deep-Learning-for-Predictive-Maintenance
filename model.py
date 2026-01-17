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
                 dilation: int, dropout: float = 0.2):
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

    def __init__(self, num_numeric_features: int = 5, num_types: int = 3,
                 tcn_channels: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        # Type embedding (L/M/H)
        self.type_embedding = nn.Embedding(num_types, 8)

        # Initial feature projection
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

        # Task-specific heads
        # Head 1: Binary failure prediction
        self.failure_head = nn.Sequential(
            nn.Linear(tcn_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Head 2: Failure type classification (5 types, multi-label)
        self.failure_type_head = nn.Sequential(
            nn.Linear(tcn_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 5)
        )

        # Head 3: Time-to-failure regression
        self.ttf_head = nn.Sequential(
            nn.Linear(tcn_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, numeric_features: torch.Tensor, machine_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            numeric_features: Sensor readings (batch, num_features)
            machine_type: Machine type indices (batch, 1)

        Returns:
            Dictionary with predictions:
                - failure_logits: Binary failure prediction (batch, 1)
                - failure_type_logits: Multi-label failure types (batch, 5)
                - ttf: Time-to-failure prediction (batch, 1)
                - attention_weights: Attention weights for interpretability
        """
        batch_size = numeric_features.size(0)

        # Embed machine type
        type_emb = self.type_embedding(machine_type.squeeze(1))  # (batch, 8)

        # Combine numeric features and type embedding
        combined = torch.cat([numeric_features, type_emb], dim=1)  # (batch, num_features+8)

        # Project to TCN channels
        x = self.input_projection(combined)  # (batch, tcn_channels)

        # Reshape for TCN: (batch, channels, seq_len)
        # Since we have single time step, seq_len=1
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
        failure_type_logits = self.failure_type_head(x)  # (batch, 5)
        ttf = F.relu(self.ttf_head(x))  # (batch, 1) - ReLU ensures non-negative TTF

        return {
            'failure_logits': failure_logits,
            'failure_type_logits': failure_type_logits,
            'ttf': ttf,
            'attention_weights': attention_weights
        }


if __name__ == '__main__':
    # Test the model
    model = MultiTaskTCN(num_numeric_features=5, num_types=3)

    # Create dummy input
    batch_size = 16
    numeric_features = torch.randn(batch_size, 5)
    machine_type = torch.randint(0, 3, (batch_size, 1))

    # Forward pass
    outputs = model(numeric_features, machine_type)

    print("Model output shapes:")
    print(f"  Failure logits: {outputs['failure_logits'].shape}")
    print(f"  Failure type logits: {outputs['failure_type_logits'].shape}")
    print(f"  TTF: {outputs['ttf'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
