"""
Phase 2 Training Script: Advanced Class Balancing
Uses SMOTE + Class-Balanced Batches + Tversky Loss

Expected improvement: F1 = 0.75-0.85 (from Phase 1: 0.65-0.75)
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

from model import MultiTaskTCN
from data_preprocessing import load_temporal_datasets, SMOTEAugmenter, ClassBalancedBatchSampler
from config import Phase2Config
from train import TverskyLoss, evaluate_epoch


def train_epoch_phase2(model: nn.Module, dataloader, optimizer: torch.optim.Optimizer,
                       criterion, device: torch.device, grad_clip_norm: float = 0.5) -> Dict[str, float]:
    """Train for one epoch with Phase 2 improvements."""
    model.train()
    epoch_losses = {'total': 0, 'failure': 0}
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
            epoch_losses[key] += value
        num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def train_phase2(seed: int = 42, save_dir: Path = None):
    """
    Train Phase 2 model with SMOTE and class-balanced batches.

    Args:
        seed: Random seed
        save_dir: Directory to save checkpoints
    """
    print("=" * 80)
    print("PHASE 2 TRAINING: ADVANCED CLASS BALANCING")
    print("=" * 80)
    print("\nImprovements:")
    print("  1. SMOTE augmentation (k=5 neighbors)")
    print("  2. Class-balanced batch sampler (50-50 split)")
    print("  3. Tversky loss (alpha=0.3, beta=0.7)")
    print("  4. 35% failure ratio (from 30% in Phase 1)")
    print("\nExpected F1: 0.75-0.85")
    print("=" * 80)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    if save_dir is None:
        save_dir = Path('./checkpoints/phase2')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and Phase2Config.USE_CUDA else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data WITHOUT augmentation first (we'll apply SMOTE manually)
    print(f"\nLoading datasets...")
    train_dataset, dev_dataset, test_dataset = load_temporal_datasets(
        str(Phase2Config.TRAIN_PATH),
        str(Phase2Config.DEV_PATH),
        str(Phase2Config.TEST_PATH),
        window_size=Phase2Config.WINDOW_SIZE,
        stride=Phase2Config.STRIDE,
        augment_train=False,  # We'll use SMOTE instead
        target_ratio=Phase2Config.TARGET_RATIO
    )

    print(f"Dataset sizes (before SMOTE): Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")

    # Apply SMOTE to training set
    if Phase2Config.USE_SMOTE:
        print(f"\nApplying SMOTE with k={Phase2Config.SMOTE_K_NEIGHBORS} neighbors...")
        smote = SMOTEAugmenter(k_neighbors=Phase2Config.SMOTE_K_NEIGHBORS, seed=seed)

        # Get data from dataset
        X_sequences = train_dataset.X_sequences
        y_failure = train_dataset.y_failure
        y_failure_types = train_dataset.y_failure_types
        y_ttf = train_dataset.y_ttf
        X_type = train_dataset.X_type

        # Apply SMOTE
        X_aug, y_aug, y_types_aug, y_ttf_aug, X_type_aug = smote.fit_resample(
            X_sequences, y_failure, y_failure_types, y_ttf, X_type,
            target_ratio=Phase2Config.TARGET_RATIO
        )

        # Update dataset
        train_dataset.X_sequences = X_aug
        train_dataset.y_failure = y_aug
        train_dataset.y_failure_types = y_types_aug
        train_dataset.y_ttf = y_ttf_aug
        train_dataset.X_type = X_type_aug

        print(f"Dataset size after SMOTE: {len(train_dataset)}")
        print(f"Failure ratio: {y_aug.mean():.1%}")

    # Create dataloaders with class-balanced sampler
    if Phase2Config.USE_CLASS_BALANCED_BATCHES:
        print(f"\nUsing class-balanced batch sampler...")
        sampler = ClassBalancedBatchSampler(
            labels=train_dataset.y_failure,
            batch_size=Phase2Config.BATCH_SIZE,
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
            batch_size=Phase2Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=Phase2Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create binary-only model
    print(f"\nCreating binary-only model...")
    model = MultiTaskTCN(
        num_numeric_features=Phase2Config.NUM_NUMERIC_FEATURES,
        num_temporal_features=Phase2Config.NUM_TEMPORAL_FEATURES,
        num_types=Phase2Config.NUM_TYPES,
        tcn_channels=Phase2Config.TCN_CHANNELS,
        num_heads=Phase2Config.NUM_HEADS,
        dropout=Phase2Config.DROPOUT,
        use_temporal_sequences=Phase2Config.USE_TEMPORAL_SEQUENCES,
        binary_only=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create Tversky loss
    if Phase2Config.USE_TVERSKY_LOSS:
        print(f"\nUsing Tversky loss (alpha={Phase2Config.TVERSKY_ALPHA}, beta={Phase2Config.TVERSKY_BETA})...")
        criterion_base = TverskyLoss(
            alpha=Phase2Config.TVERSKY_ALPHA,
            beta=Phase2Config.TVERSKY_BETA
        )
    else:
        from train import AsymmetricFocalLoss
        criterion_base = AsymmetricFocalLoss(
            alpha=Phase2Config.FOCAL_ALPHA,
            gamma=Phase2Config.FOCAL_GAMMA,
            beta=Phase2Config.ASYMMETRIC_BETA
        )

    # Wrapper for multi-task loss interface
    class SimpleBinaryLoss(nn.Module):
        def __init__(self, base_loss):
            super().__init__()
            self.base_loss = base_loss

        def forward(self, outputs, targets):
            loss = self.base_loss(outputs['failure_logits'], targets['failure'])
            return loss, {'total': loss.item(), 'failure': loss.item()}

    criterion = SimpleBinaryLoss(criterion_base)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Phase2Config.LEARNING_RATE,
        weight_decay=Phase2Config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    if Phase2Config.USE_WARMUP:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / Phase2Config.WARMUP_EPOCHS)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=Phase2Config.COSINE_T0,
            T_mult=Phase2Config.COSINE_T_MULT,
            eta_min=Phase2Config.COSINE_ETA_MIN
        )

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    train_history = []

    print(f"\nStarting training for {Phase2Config.NUM_EPOCHS} epochs (patience={Phase2Config.PATIENCE})...")

    for epoch in range(Phase2Config.NUM_EPOCHS):
        # Train
        train_losses = train_epoch_phase2(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=Phase2Config.GRAD_CLIP_NORM
        )

        # Evaluate
        dev_losses, dev_metrics = evaluate_epoch(
            model, dev_loader, criterion, device,
            use_temporal=Phase2Config.USE_TEMPORAL_SEQUENCES
        )

        # Learning rate scheduling
        if Phase2Config.USE_WARMUP:
            if epoch < Phase2Config.WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Log progress
        f1_score_val = dev_metrics.get('f1_score', 0.0)
        print(f"Epoch {epoch+1}/{Phase2Config.NUM_EPOCHS} | "
              f"Train Loss: {train_losses['total']:.4f} | "
              f"Dev Loss: {dev_losses['total']:.4f} | "
              f"Dev F1: {f1_score_val:.4f} | "
              f"Dev Prec: {dev_metrics.get('precision', 0.0):.4f} | "
              f"Dev Rec: {dev_metrics.get('recall', 0.0):.4f} | "
              f"LR: {current_lr:.6f}")

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
                'config': 'Phase2Config'
            }

            checkpoint_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved new best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Phase2Config.PATIENCE:
                print(f"\nEarly stopping triggered (patience={Phase2Config.PATIENCE})")
                break

    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"\n" + "=" * 80)
    print("PHASE 2 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")

    # Compare to Phase 1 expectation
    phase1_expected = 0.70  # Conservative Phase 1 estimate
    improvement = (best_f1 - phase1_expected) / phase1_expected * 100
    print(f"\nImprovement over Phase 1 (expected {phase1_expected:.2f}): {improvement:+.1f}%")

    if best_f1 >= 0.75:
        print("✓ Phase 2 target achieved (F1 >= 0.75)!")
    else:
        print(f"⚠ Phase 2 target not met. Current: {best_f1:.4f}, Target: 0.75")
        print("  Consider: Increasing augmentation ratio, tuning Tversky alpha/beta, or longer training")

    return {
        'best_f1': best_f1,
        'final_epoch': len(train_history),
        'checkpoint_path': str(checkpoint_path),
        'history_path': str(history_path)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 2 Training: Advanced Class Balancing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/phase2',
                       help='Output directory for checkpoints')

    args = parser.parse_args()

    result = train_phase2(
        seed=args.seed,
        save_dir=Path(args.output_dir)
    )

    print(f"\n✓ Training complete. Best F1: {result['best_f1']:.4f}")
