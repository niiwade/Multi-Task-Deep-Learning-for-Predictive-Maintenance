"""
Phase 3 Training Script: Architecture Improvements
Uses Two-Stage Classifier + High Capacity Model + All Phase 2 improvements

Expected improvement: F1 = 0.85-0.90 (from Phase 2: 0.75-0.85)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model_advanced import TwoStageClassifier, HighCapacityTCN
from data_preprocessing import load_temporal_datasets, SMOTEAugmenter, ClassBalancedBatchSampler
from config import Phase3Config
from train import TverskyLoss, evaluate_epoch


def train_epoch_phase3(model: nn.Module, dataloader, optimizer: torch.optim.Optimizer,
                       criterion, device: torch.device, grad_clip_norm: float = 0.5,
                       use_two_stage: bool = True) -> Dict[str, float]:
    """Train for one epoch with Phase 3 improvements."""
    model.train()
    epoch_losses = {'total': 0, 'failure': 0}
    if use_two_stage:
        epoch_losses['stage1'] = 0
        epoch_losses['stage2'] = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Move data to device
        machine_type = batch['type'].to(device)
        targets = {'failure': batch['failure'].to(device)}

        # Forward pass
        optimizer.zero_grad()

        sequence = batch['sequence'].to(device)
        features = batch['features'].to(device)
        outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)

        # Compute loss
        loss, loss_dict = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        # Accumulate losses
        for key, value in loss_dict.items():
            if key in epoch_losses:
                epoch_losses[key] += value
            else:
                epoch_losses[key] = value
        num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def train_phase3(seed: int = 42, save_dir: Path = None, use_two_stage: bool = None,
                 use_high_capacity: bool = None):
    """
    Train Phase 3 model with advanced architecture.

    Args:
        seed: Random seed
        save_dir: Directory to save checkpoints
        use_two_stage: Use TwoStageClassifier (default: from config)
        use_high_capacity: Use HighCapacityTCN instead of two-stage (default: from config)
    """
    print("=" * 80)
    print("PHASE 3 TRAINING: ARCHITECTURE IMPROVEMENTS")
    print("=" * 80)
    print("\nImprovements from Phase 2:")
    print("  ✓ SMOTE augmentation")
    print("  ✓ Class-balanced batches")
    print("  ✓ Tversky loss")
    print("  ✓ 35% failure ratio")
    print("\nPhase 3 New Features:")

    # Use config defaults if not specified
    if use_two_stage is None:
        use_two_stage = Phase3Config.USE_TWO_STAGE
    if use_high_capacity is None:
        use_high_capacity = Phase3Config.USE_HIGH_CAPACITY

    if use_high_capacity:
        print("  1. High-Capacity TCN (128 channels, 8 heads, 5 blocks)")
        print("  2. Deeper task heads (3 layers)")
        print("  3. 5x more parameters (~643K)")
        model_type = "HighCapacity"
    elif use_two_stage:
        print("  1. Two-Stage Classifier (Detection + Refinement)")
        print("  2. Stage 1: Conservative detector (high recall)")
        print("  3. Stage 2: Precision refinement (filter FP)")
        model_type = "TwoStage"
    else:
        print("  1. Enhanced base model with Phase 3 config")
        model_type = "Enhanced"

    print("\nExpected F1: 0.85-0.90")
    print("=" * 80)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    if save_dir is None:
        save_dir = Path(f'./checkpoints/phase3_{model_type.lower()}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and Phase3Config.USE_CUDA else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data WITHOUT augmentation first (we'll apply SMOTE manually)
    print(f"\nLoading datasets...")
    train_dataset, dev_dataset, test_dataset = load_temporal_datasets(
        str(Phase3Config.TRAIN_PATH),
        str(Phase3Config.DEV_PATH),
        str(Phase3Config.TEST_PATH),
        window_size=Phase3Config.WINDOW_SIZE,
        stride=Phase3Config.STRIDE,
        augment_train=False,
        target_ratio=Phase3Config.TARGET_RATIO
    )

    print(f"Dataset sizes (before SMOTE): Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")

    # Apply SMOTE to training set
    if Phase3Config.USE_SMOTE:
        print(f"\nApplying SMOTE with k={Phase3Config.SMOTE_K_NEIGHBORS} neighbors...")
        smote = SMOTEAugmenter(k_neighbors=Phase3Config.SMOTE_K_NEIGHBORS, seed=seed)

        X_sequences = train_dataset.X_sequences
        y_failure = train_dataset.y_failure
        y_failure_types = train_dataset.y_failure_types
        y_ttf = train_dataset.y_ttf
        X_type = train_dataset.X_type

        X_aug, y_aug, y_types_aug, y_ttf_aug, X_type_aug = smote.fit_resample(
            X_sequences, y_failure, y_failure_types, y_ttf, X_type,
            target_ratio=Phase3Config.TARGET_RATIO
        )

        train_dataset.X_sequences = X_aug
        train_dataset.y_failure = y_aug
        train_dataset.y_failure_types = y_types_aug
        train_dataset.y_ttf = y_ttf_aug
        train_dataset.X_type = X_type_aug

        print(f"Dataset size after SMOTE: {len(train_dataset)}")
        print(f"Failure ratio: {y_aug.mean():.1%}")

    # Create dataloaders with class-balanced sampler
    if Phase3Config.USE_CLASS_BALANCED_BATCHES:
        print(f"\nUsing class-balanced batch sampler...")
        sampler = ClassBalancedBatchSampler(
            labels=train_dataset.y_failure,
            batch_size=Phase3Config.BATCH_SIZE,
            seed=seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=Phase3Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=Phase3Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create model based on configuration
    print(f"\nCreating {model_type} model...")

    if use_high_capacity:
        model = HighCapacityTCN(
            num_numeric_features=Phase3Config.NUM_NUMERIC_FEATURES,
            num_temporal_features=Phase3Config.NUM_TEMPORAL_FEATURES,
            num_types=Phase3Config.NUM_TYPES,
            tcn_channels=Phase3Config.TCN_CHANNELS,  # 128
            num_heads=Phase3Config.NUM_HEADS,  # 8
            dropout=Phase3Config.DROPOUT,
            use_temporal_sequences=Phase3Config.USE_TEMPORAL_SEQUENCES,
            binary_only=True
        ).to(device)
    elif use_two_stage:
        model = TwoStageClassifier(
            num_numeric_features=Phase3Config.NUM_NUMERIC_FEATURES,
            num_temporal_features=Phase3Config.NUM_TEMPORAL_FEATURES,
            num_types=Phase3Config.NUM_TYPES,
            tcn_channels=Phase3Config.TCN_CHANNELS,  # Can be 64 or 128
            num_heads=Phase3Config.NUM_HEADS,
            dropout=Phase3Config.DROPOUT,
            use_temporal_sequences=Phase3Config.USE_TEMPORAL_SEQUENCES
        ).to(device)
    else:
        from model import MultiTaskTCN
        model = MultiTaskTCN(
            num_numeric_features=Phase3Config.NUM_NUMERIC_FEATURES,
            num_temporal_features=Phase3Config.NUM_TEMPORAL_FEATURES,
            num_types=Phase3Config.NUM_TYPES,
            tcn_channels=Phase3Config.TCN_CHANNELS,
            num_heads=Phase3Config.NUM_HEADS,
            dropout=Phase3Config.DROPOUT,
            use_temporal_sequences=Phase3Config.USE_TEMPORAL_SEQUENCES,
            binary_only=True
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create Tversky loss
    if Phase3Config.USE_TVERSKY_LOSS:
        print(f"\nUsing Tversky loss (alpha={Phase3Config.TVERSKY_ALPHA}, beta={Phase3Config.TVERSKY_BETA})...")
        criterion_base = TverskyLoss(
            alpha=Phase3Config.TVERSKY_ALPHA,
            beta=Phase3Config.TVERSKY_BETA
        )
    else:
        from train import AsymmetricFocalLoss
        criterion_base = AsymmetricFocalLoss(
            alpha=Phase3Config.FOCAL_ALPHA,
            gamma=Phase3Config.FOCAL_GAMMA,
            beta=Phase3Config.ASYMMETRIC_BETA
        )

    # Wrapper for multi-task loss interface
    class SimpleBinaryLoss(nn.Module):
        def __init__(self, base_loss):
            super().__init__()
            self.base_loss = base_loss

        def forward(self, outputs, targets):
            loss = self.base_loss(outputs['failure_logits'], targets['failure'])
            loss_dict = {'total': loss.item(), 'failure': loss.item()}

            # Add stage losses if available
            if 'stage1_logits' in outputs:
                stage1_loss = self.base_loss(outputs['stage1_logits'], targets['failure'])
                loss_dict['stage1'] = stage1_loss.item()
            if 'stage2_logits' in outputs:
                stage2_loss = self.base_loss(outputs['stage2_logits'], targets['failure'])
                loss_dict['stage2'] = stage2_loss.item()

            return loss, loss_dict

    criterion = SimpleBinaryLoss(criterion_base)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Phase3Config.LEARNING_RATE,
        weight_decay=Phase3Config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    if Phase3Config.USE_WARMUP:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / Phase3Config.WARMUP_EPOCHS)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=Phase3Config.COSINE_T0,
            T_mult=Phase3Config.COSINE_T_MULT,
            eta_min=Phase3Config.COSINE_ETA_MIN
        )

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    train_history = []

    print(f"\nStarting training for {Phase3Config.NUM_EPOCHS} epochs (patience={Phase3Config.PATIENCE})...")

    for epoch in range(Phase3Config.NUM_EPOCHS):
        # Train
        train_losses = train_epoch_phase3(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=Phase3Config.GRAD_CLIP_NORM,
            use_two_stage=use_two_stage
        )

        # Evaluate
        dev_losses, dev_metrics = evaluate_epoch(
            model, dev_loader, criterion, device,
            use_temporal=Phase3Config.USE_TEMPORAL_SEQUENCES
        )

        # Learning rate scheduling
        if Phase3Config.USE_WARMUP:
            if epoch < Phase3Config.WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Log progress
        f1_score_val = dev_metrics.get('f1_score', 0.0)
        log_msg = (f"Epoch {epoch+1}/{Phase3Config.NUM_EPOCHS} | "
                   f"Train Loss: {train_losses['total']:.4f} | "
                   f"Dev Loss: {dev_losses['total']:.4f} | "
                   f"Dev F1: {f1_score_val:.4f} | "
                   f"Dev Prec: {dev_metrics.get('precision', 0.0):.4f} | "
                   f"Dev Rec: {dev_metrics.get('recall', 0.0):.4f} | "
                   f"LR: {current_lr:.6f}")
        print(log_msg)

        # Track history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_losses['total'],
            'dev_loss': dev_losses['total'],
            'dev_f1': f1_score_val,
            'dev_precision': dev_metrics.get('precision', 0.0),
            'dev_recall': dev_metrics.get('recall', 0.0),
            'lr': current_lr
        })

        # Save best model
        if f1_score_val > best_f1:
            best_f1 = f1_score_val
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'seed': seed,
                'best_f1': best_f1,
                'dev_metrics': dev_metrics,
                'config': 'Phase3Config',
                'model_type': model_type
            }

            checkpoint_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved new best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Phase3Config.PATIENCE:
                print(f"\nEarly stopping triggered (patience={Phase3Config.PATIENCE})")
                break

    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"\n" + "=" * 80)
    print("PHASE 3 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")

    # Compare to Phase 2 expectation
    phase2_expected = 0.80  # Conservative Phase 2 estimate
    improvement = (best_f1 - phase2_expected) / phase2_expected * 100
    print(f"\nImprovement over Phase 2 (expected {phase2_expected:.2f}): {improvement:+.1f}%")

    if best_f1 >= 0.85:
        print("✓ Phase 3 target achieved (F1 >= 0.85)!")
    else:
        print(f"⚠ Phase 3 target not met. Current: {best_f1:.4f}, Target: 0.85")
        print("  Consider: Trying the other model type or increasing training epochs")

    return {
        'best_f1': best_f1,
        'final_epoch': len(train_history),
        'checkpoint_path': str(checkpoint_path),
        'history_path': str(history_path),
        'model_type': model_type
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3 Training: Architecture Improvements')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints')
    parser.add_argument('--two_stage', action='store_true',
                       help='Use two-stage classifier')
    parser.add_argument('--high_capacity', action='store_true',
                       help='Use high-capacity TCN')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        if args.high_capacity:
            output_dir = './checkpoints/phase3_highcapacity'
        elif args.two_stage:
            output_dir = './checkpoints/phase3_twostage'
        else:
            output_dir = './checkpoints/phase3'
    else:
        output_dir = args.output_dir

    result = train_phase3(
        seed=args.seed,
        save_dir=Path(output_dir),
        use_two_stage=args.two_stage if args.two_stage else None,
        use_high_capacity=args.high_capacity if args.high_capacity else None
    )

    print(f"\n✓ Training complete. Best F1: {result['best_f1']:.4f}")
