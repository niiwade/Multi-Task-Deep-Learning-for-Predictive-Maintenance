"""
Advanced model architectures for Phases 2-4 of IMPROVEMENT_PLAN_V2.md.
Includes: Two-Stage Classifier, High-Capacity models, and advanced attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from model import TemporalConvBlock, MultiHeadAttention


class TwoStageClassifier(nn.Module):
    """
    Two-stage classifier for improved precision-recall trade-off.
    Phase 3 improvement from IMPROVEMENT_PLAN_V2.md.

    Stage 1: Conservative detector (high recall, catches most failures)
    Stage 2: Refinement classifier (improves precision, filters false positives)
    """

    def __init__(self, num_numeric_features: int = 5,
                 num_temporal_features: int = 19,
                 num_types: int = 3,
                 tcn_channels: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 use_temporal_sequences: bool = True):
        """
        Args:
            num_numeric_features: Number of sensor features (5)
            num_temporal_features: Number of temporal features (19)
            num_types: Number of machine types (3)
            tcn_channels: Number of channels in TCN
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_temporal_sequences: Whether to use temporal sequence processing
        """
        super().__init__()

        self.use_temporal_sequences = use_temporal_sequences

        # Shared encoder
        # Type embedding
        self.type_embedding = nn.Embedding(num_types, 8)

        if use_temporal_sequences:
            # Sequence projection
            self.sequence_projection = nn.Conv1d(
                num_numeric_features,
                tcn_channels,
                kernel_size=1
            )

            # Temporal features projection
            temporal_input_dim = num_temporal_features + 8
            self.temporal_projection = nn.Sequential(
                nn.Linear(temporal_input_dim, tcn_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Feature fusion
            self.fusion = nn.Sequential(
                nn.Linear(tcn_channels * 2, tcn_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # Backward compatibility
            input_dim = num_numeric_features + 8
            self.input_projection = nn.Linear(input_dim, tcn_channels)

        # TCN layers
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=8, dropout=dropout),
        ])

        # Attention
        self.attention = MultiHeadAttention(tcn_channels, num_heads=num_heads, dropout=dropout)

        # ===== STAGE 1: Conservative Detector =====
        # High recall threshold, catches most failures
        self.stage1_detector = nn.Sequential(
            nn.Linear(tcn_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Binary prediction
        )

        # ===== STAGE 2: Refinement Classifier =====
        # Improves precision on stage1 detections
        # Takes concatenated features: [shared_features, stage1_prob]
        self.stage2_refiner = nn.Sequential(
            nn.Linear(tcn_channels + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # Higher dropout for refinement
            nn.Linear(32, 1)  # Refined binary prediction
        )

    def forward(self, sequence: torch.Tensor = None,
                temporal_features: torch.Tensor = None,
                machine_type: torch.Tensor = None,
                numeric_features: torch.Tensor = None,
                return_stage1_only: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with two-stage prediction.

        Args:
            sequence: Sensor sequence (batch, window_size, num_features)
            temporal_features: Temporal features (batch, num_temporal_features)
            machine_type: Machine type indices (batch, 1)
            numeric_features: Single timestep features (batch, num_features) - backward compatibility
            return_stage1_only: If True, only return stage1 detection (for conservative mode)

        Returns:
            Dictionary with predictions:
                - stage1_logits: Conservative detector output (batch, 1)
                - stage2_logits: Refined classifier output (batch, 1)
                - failure_logits: Final prediction (stage2 by default)
                - attention_weights: Attention weights
        """
        # Embed machine type
        type_emb = self.type_embedding(machine_type.squeeze(1))

        if self.use_temporal_sequences and sequence is not None:
            # Process sequence through TCN
            batch_size = sequence.size(0)
            x_seq = sequence.transpose(1, 2)  # (batch, features, window)
            x_seq = self.sequence_projection(x_seq)

            # Apply TCN blocks
            for tcn_block in self.tcn_blocks:
                x_seq = tcn_block(x_seq)

            # Transpose for attention
            x_seq = x_seq.transpose(1, 2)

            # Apply attention
            x_seq, attention_weights = self.attention(x_seq)

            # Global pooling
            x_seq = x_seq.mean(dim=1)

            # Process temporal features
            temporal_input = torch.cat([temporal_features, type_emb], dim=1)
            x_temporal = self.temporal_projection(temporal_input)

            # Fuse
            x_fused = torch.cat([x_seq, x_temporal], dim=1)
            shared_features = self.fusion(x_fused)

        else:
            # Backward compatibility: single timestep
            combined = torch.cat([numeric_features, type_emb], dim=1)
            x = self.input_projection(combined)
            x = x.unsqueeze(2)

            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            x = x.transpose(1, 2)
            x, attention_weights = self.attention(x)
            shared_features = x.squeeze(1)

        # ===== STAGE 1: Conservative Detection =====
        stage1_logits = self.stage1_detector(shared_features)

        if return_stage1_only:
            return {
                'stage1_logits': stage1_logits,
                'failure_logits': stage1_logits,  # Use stage1 as final
                'attention_weights': attention_weights
            }

        # ===== STAGE 2: Refinement =====
        # Concatenate shared features with stage1 probability
        stage1_prob = torch.sigmoid(stage1_logits)
        refinement_input = torch.cat([shared_features, stage1_prob], dim=1)
        stage2_logits = self.stage2_refiner(refinement_input)

        return {
            'stage1_logits': stage1_logits,
            'stage2_logits': stage2_logits,
            'failure_logits': stage2_logits,  # Use stage2 as final prediction
            'attention_weights': attention_weights
        }


class HighCapacityTCN(nn.Module):
    """
    High-capacity TCN model with 128 channels and deeper task heads.
    Phase 3 improvement from IMPROVEMENT_PLAN_V2.md.
    """

    def __init__(self, num_numeric_features: int = 5,
                 num_temporal_features: int = 19,
                 num_types: int = 3,
                 tcn_channels: int = 128,  # Increased from 64
                 num_heads: int = 8,  # Increased from 4
                 dropout: float = 0.3,
                 use_temporal_sequences: bool = True,
                 binary_only: bool = False):
        """
        Args:
            tcn_channels: 128 (increased capacity)
            num_heads: 8 (more attention heads)
            Other args same as base model
        """
        super().__init__()

        self.use_temporal_sequences = use_temporal_sequences
        self.binary_only = binary_only

        # Type embedding
        self.type_embedding = nn.Embedding(num_types, 16)  # Larger embedding

        if use_temporal_sequences:
            # Sequence projection
            self.sequence_projection = nn.Conv1d(
                num_numeric_features,
                tcn_channels,
                kernel_size=1
            )

            # Temporal features projection
            temporal_input_dim = num_temporal_features + 16
            self.temporal_projection = nn.Sequential(
                nn.Linear(temporal_input_dim, tcn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tcn_channels, tcn_channels),  # Extra layer
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Feature fusion
            self.fusion = nn.Sequential(
                nn.Linear(tcn_channels * 2, tcn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tcn_channels, tcn_channels),  # Extra layer
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            input_dim = num_numeric_features + 16
            self.input_projection = nn.Linear(input_dim, tcn_channels)

        # Deeper TCN with more layers
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=8, dropout=dropout),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=16, dropout=dropout),  # Extra
        ])

        # Multi-head attention
        self.attention = MultiHeadAttention(tcn_channels, num_heads=num_heads, dropout=dropout)

        # Deeper task head (3 layers instead of 2)
        self.failure_head = nn.Sequential(
            nn.Linear(tcn_channels, 64),  # Larger
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

        if not binary_only:
            self.failure_type_head = nn.Sequential(
                nn.Linear(tcn_channels, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(32, 5)
            )

            self.ttf_head = nn.Sequential(
                nn.Linear(tcn_channels, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 32),
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
        """Forward pass - same interface as base model."""
        # Embed machine type
        type_emb = self.type_embedding(machine_type.squeeze(1))

        if self.use_temporal_sequences and sequence is not None:
            # Process sequence
            batch_size = sequence.size(0)
            x_seq = sequence.transpose(1, 2)
            x_seq = self.sequence_projection(x_seq)

            # Apply TCN blocks
            for tcn_block in self.tcn_blocks:
                x_seq = tcn_block(x_seq)

            # Attention
            x_seq = x_seq.transpose(1, 2)
            x_seq, attention_weights = self.attention(x_seq)
            x_seq = x_seq.mean(dim=1)

            # Process temporal features
            temporal_input = torch.cat([temporal_features, type_emb], dim=1)
            x_temporal = self.temporal_projection(temporal_input)

            # Fuse
            x_fused = torch.cat([x_seq, x_temporal], dim=1)
            x = self.fusion(x_fused)

        else:
            # Backward compatibility
            combined = torch.cat([numeric_features, type_emb], dim=1)
            x = self.input_projection(combined)
            x = x.unsqueeze(2)

            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            x = x.transpose(1, 2)
            x, attention_weights = self.attention(x)
            x = x.squeeze(1)

        # Task predictions
        failure_logits = self.failure_head(x)

        if self.binary_only:
            return {
                'failure_logits': failure_logits,
                'attention_weights': attention_weights
            }
        else:
            failure_type_logits = self.failure_type_head(x)
            ttf = F.relu(self.ttf_head(x))

            return {
                'failure_logits': failure_logits,
                'failure_type_logits': failure_type_logits,
                'ttf': ttf,
                'attention_weights': attention_weights
            }


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING TWO-STAGE CLASSIFIER")
    print("=" * 80)

    # Test two-stage classifier
    model = TwoStageClassifier(
        num_numeric_features=5,
        num_temporal_features=19,
        num_types=3,
        tcn_channels=64,
        num_heads=4,
        dropout=0.3,
        use_temporal_sequences=True
    )

    batch_size = 16
    sequence = torch.randn(batch_size, 12, 5)
    temporal_features = torch.randn(batch_size, 19)
    machine_type = torch.randint(0, 3, (batch_size, 1))

    outputs = model(sequence=sequence, temporal_features=temporal_features, machine_type=machine_type)

    print("\nTwo-stage outputs:")
    print(f"  Stage 1 logits: {outputs['stage1_logits'].shape}")
    print(f"  Stage 2 logits: {outputs['stage2_logits'].shape}")
    print(f"  Final logits: {outputs['failure_logits'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n" + "=" * 80)
    print("TESTING HIGH-CAPACITY TCN")
    print("=" * 80)

    # Test high-capacity model
    model_hc = HighCapacityTCN(
        num_numeric_features=5,
        num_temporal_features=19,
        num_types=3,
        tcn_channels=128,
        num_heads=8,
        dropout=0.3,
        use_temporal_sequences=True,
        binary_only=True
    )

    outputs_hc = model_hc(sequence=sequence, temporal_features=temporal_features, machine_type=machine_type)

    print("\nHigh-capacity outputs:")
    print(f"  Failure logits: {outputs_hc['failure_logits'].shape}")

    # Count parameters
    total_params_hc = sum(p.numel() for p in model_hc.parameters())
    print(f"\nTotal parameters: {total_params_hc:,}")
    print(f"Increase over baseline: {(total_params_hc / 129177 - 1) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("ADVANCED MODELS: READY")
    print("=" * 80)
