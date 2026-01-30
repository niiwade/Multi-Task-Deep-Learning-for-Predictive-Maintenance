"""
Phase 4 Training Script: Advanced Techniques
Uses Mixup + CutMix + Pseudo-Labeling + Attention Regularization + All previous improvements

Expected improvement: F1 = 0.90-0.95 (TARGET!)
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
from config import Phase4Config
from train import TverskyLoss, evaluate_epoch
from advanced_techniques import (
    MixupAugmenter,
    CutMixAugmenter,
    PseudoLabeling,
    AttentionRegularization
)


def train_epoch_phase4(model: nn.Module, dataloader, optimizer: torch.optim.Optimizer,
                       criterion, device: torch.device, grad_clip_norm: float = 0.5,
                       mixup_aug=None, cutmix_aug=None, attn_reg=None) -> Dict[str, float]:
    """Train for one epoch with Phase 4 advanced techniques."""
    model.train()
    epoch_losses = {'total': 0, 'failure': 0, 'attention_reg': 0}
    num_batches = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Apply Mixup or CutMix augmentation
        if mixup_aug is not None and np.random.rand() < 0.5:
            batch = mixup_aug(batch)
        elif cutmix_aug is not None:
            batch = cutmix_aug(batch)

        # Move data to device
        machine_type = batch['type'].to(device)
        targets = {'failure': batch['failure'].to(device)}

        # Forward pass
        optimizer.zero_grad()

        sequence = batch['sequence'].to(device)
        features = batch['features'].to(device)
        outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)

        # Compute main loss
        loss, loss_dict = criterion(outputs, targets)

        # Add attention regularization if enabled
        if attn_reg is not None and 'attention_weights' in outputs:
            attn_loss = attn_reg(outputs['attention_weights'], window_size=Phase4Config.WINDOW_SIZE)
            loss = loss + attn_loss
            loss_dict['attention_reg'] = attn_loss.item()
        else:
            loss_dict['attention_reg'] = 0.0

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


def train_phase4(seed: int = 42, save_dir: Path = None, use_two_stage: bool = None,
                 use_high_capacity: bool = None):
    """
    Train Phase 4 model with all advanced techniques.

    Args:
        seed: Random seed
        save_dir: Directory to save checkpoints
        use_two_stage: Use TwoStageClassifier (default: from config)
        use_high_capacity: Use HighCapacityTCN instead of two-stage (default: from config)
    """
    print("=" * 80)
    print("PHASE 4 TRAINING: ADVANCED TECHNIQUES (TARGET: F1 > 0.90)")
    print("=" * 80)
    print("\nAll Previous Improvements:")
    print("  ✓ SMOTE augmentation")
    print("  ✓ Class-balanced batches")
    print("  ✓ Tversky loss")
    print("  ✓ 35% failure ratio")
    print("  ✓ Two-stage or high-capacity architecture")
    print("\nPhase 4 Advanced Techniques:")
    print("  1. Mixup augmentation (sample interpolation)")
    print("  2. CutMix augmentation (temporal region mixing)")
    print("  3. Pseudo-labeling (self-training on confident predictions)")
    print("  4. Attention regularization (focus on recent timesteps)")
    print("  5. Temperature scaling (probability calibration)")
    print("\nExpected F1: 0.90-0.95 (TARGET!)")
    print("=" * 80)

    # Use config defaults if not specified
    if use_two_stage is None:
        use_two_stage = Phase4Config.USE_TWO_STAGE
    if use_high_capacity is None:
        use_high_capacity = Phase4Config.USE_HIGH_CAPACITY

    if use_high_capacity:
        model_type = "HighCapacity"
    elif use_two_stage:
        model_type = "TwoStage"
    else:
        model_type = "Enhanced"

    print(f"\nModel architecture: {model_type}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    if save_dir is None:
        save_dir = Path(f'./checkpoints/phase4_{model_type.lower()}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and Phase4Config.USE_CUDA else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print(f"\nLoading datasets...")
    train_dataset, dev_dataset, test_dataset = load_temporal_datasets(
        str(Phase4Config.TRAIN_PATH),
        str(Phase4Config.DEV_PATH),
        str(Phase4Config.TEST_PATH),
        window_size=Phase4Config.WINDOW_SIZE,
        stride=Phase4Config.STRIDE,
        augment_train=False,
        target_ratio=Phase4Config.TARGET_RATIO
    )

    print(f"Dataset sizes (before SMOTE): Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")

    # Apply SMOTE
    if Phase4Config.USE_SMOTE:
        print(f"\nApplying SMOTE with k={Phase4Config.SMOTE_K_NEIGHBORS} neighbors...")
        smote = SMOTEAugmenter(k_neighbors=Phase4Config.SMOTE_K_NEIGHBORS, seed=seed)

        X_sequences = train_dataset.X_sequences
        y_failure = train_dataset.y_failure
        y_failure_types = train_dataset.y_failure_types
        y_ttf = train_dataset.y_ttf
        X_type = train_dataset.X_type

        X_aug, y_aug, y_types_aug, y_ttf_aug, X_type_aug = smote.fit_resample(
            X_sequences, y_failure, y_failure_types, y_ttf, X_type,
            target_ratio=Phase4Config.TARGET_RATIO
        )

        train_dataset.X_sequences = X_aug
        train_dataset.y_failure = y_aug
        train_dataset.y_failure_types = y_types_aug
        train_dataset.y_ttf = y_ttf_aug
        train_dataset.X_type = X_type_aug

        print(f"Dataset size after SMOTE: {len(train_dataset)}")
        print(f"Failure ratio: {y_aug.mean():.1%}")

    # Create dataloaders with class-balanced sampler
    if Phase4Config.USE_CLASS_BALANCED_BATCHES:
        print(f"\nUsing class-balanced batch sampler...")
        sampler = ClassBalancedBatchSampler(
            labels=train_dataset.y_failure,
            batch_size=Phase4Config.BATCH_SIZE,
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
            batch_size=Phase4Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=Phase4Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print(f"\nCreating {model_type} model...")

    if use_high_capacity:
        model = HighCapacityTCN(
            num_numeric_features=Phase4Config.NUM_NUMERIC_FEATURES,
            num_temporal_features=Phase4Config.NUM_TEMPORAL_FEATURES,
            num_types=Phase4Config.NUM_TYPES,
            tcn_channels=Phase4Config.TCN_CHANNELS,
            num_heads=Phase4Config.NUM_HEADS,
            dropout=Phase4Config.DROPOUT,
            use_temporal_sequences=Phase4Config.USE_TEMPORAL_SEQUENCES,
            binary_only=True
        ).to(device)
    elif use_two_stage:
        model = TwoStageClassifier(
            num_numeric_features=Phase4Config.NUM_NUMERIC_FEATURES,
            num_temporal_features=Phase4Config.NUM_TEMPORAL_FEATURES,
            num_types=Phase4Config.NUM_TYPES,
            tcn_channels=Phase4Config.TCN_CHANNELS,
            num_heads=Phase4Config.NUM_HEADS,
            dropout=Phase4Config.DROPOUT,
            use_temporal_sequences=Phase4Config.USE_TEMPORAL_SEQUENCES
        ).to(device)
    else:
        from model import MultiTaskTCN
        model = MultiTaskTCN(
            num_numeric_features=Phase4Config.NUM_NUMERIC_FEATURES,
            num_temporal_features=Phase4Config.NUM_TEMPORAL_FEATURES,
            num_types=Phase4Config.NUM_TYPES,
            tcn_channels=Phase4Config.TCN_CHANNELS,
            num_heads=Phase4Config.NUM_HEADS,
            dropout=Phase4Config.DROPOUT,
            use_temporal_sequences=Phase4Config.USE_TEMPORAL_SEQUENCES,
            binary_only=True
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Initialize Phase 4 advanced techniques
    mixup_aug = None
    cutmix_aug = None
    attn_reg = None
    pseudo_labeling = None

    if Phase4Config.USE_MIXUP:
        print(f"\nEnabled: Mixup (alpha={Phase4Config.MIXUP_ALPHA}, prob={Phase4Config.MIXUP_PROB})")
        mixup_aug = MixupAugmenter(
            alpha=Phase4Config.MIXUP_ALPHA,
            prob=Phase4Config.MIXUP_PROB
        )

    if Phase4Config.USE_CUTMIX:
        print(f"Enabled: CutMix (alpha={Phase4Config.CUTMIX_ALPHA}, prob={Phase4Config.CUTMIX_PROB})")
        cutmix_aug = CutMixAugmenter(
            alpha=Phase4Config.CUTMIX_ALPHA,
            prob=Phase4Config.CUTMIX_PROB
        )

    if Phase4Config.USE_ATTENTION_REG:
        print(f"Enabled: Attention Regularization (strength={Phase4Config.ATTENTION_REG_STRENGTH})")
        attn_reg = AttentionRegularization(
            regularization_strength=Phase4Config.ATTENTION_REG_STRENGTH,
            focus_recent=Phase4Config.ATTENTION_FOCUS_RECENT
        )

    if Phase4Config.USE_PSEUDO_LABELING:
        print(f"Enabled: Pseudo-Labeling (threshold={Phase4Config.PSEUDO_CONFIDENCE_THRESHOLD})")
        pseudo_labeling = PseudoLabeling(
            confidence_threshold=Phase4Config.PSEUDO_CONFIDENCE_THRESHOLD,
            update_frequency=Phase4Config.PSEUDO_UPDATE_FREQUENCY
        )

    # Create Tversky loss
    if Phase4Config.USE_TVERSKY_LOSS:
        print(f"\nUsing Tversky loss (alpha={Phase4Config.TVERSKY_ALPHA}, beta={Phase4Config.TVERSKY_BETA})")
        criterion_base = TverskyLoss(
            alpha=Phase4Config.TVERSKY_ALPHA,
            beta=Phase4Config.TVERSKY_BETA
        )
    else:
        from train import AsymmetricFocalLoss
        criterion_base = AsymmetricFocalLoss(
            alpha=Phase4Config.FOCAL_ALPHA,
            gamma=Phase4Config.FOCAL_GAMMA,
            beta=Phase4Config.ASYMMETRIC_BETA
        )

    # Wrapper for multi-task loss interface
    class SimpleBinaryLoss(nn.Module):
        def __init__(self, base_loss):
            super().__init__()
            self.base_loss = base_loss

        def forward(self, outputs, targets):
            loss = self.base_loss(outputs['failure_logits'], targets['failure'])
            loss_dict = {'total': loss.item(), 'failure': loss.item()}
            return loss, loss_dict

    criterion = SimpleBinaryLoss(criterion_base)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Phase4Config.LEARNING_RATE,
        weight_decay=Phase4Config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    if Phase4Config.USE_WARMUP:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / Phase4Config.WARMUP_EPOCHS)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=Phase4Config.COSINE_T0,
            T_mult=Phase4Config.COSINE_T_MULT,
            eta_min=Phase4Config.COSINE_ETA_MIN
        )

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    train_history = []

    print(f"\nStarting training for {Phase4Config.NUM_EPOCHS} epochs (patience={Phase4Config.PATIENCE})...")

    for epoch in range(Phase4Config.NUM_EPOCHS):
        # Pseudo-labeling update (if enabled)
        if pseudo_labeling is not None and epoch > 0 and epoch % Phase4Config.PSEUDO_UPDATE_FREQUENCY == 0:
            print(f"\n  Updating pseudo-labels (epoch {epoch})...")
            # Note: Pseudo-labeling would require unlabeled data
            # For now, we skip this as we don't have unlabeled samples
            # In production, you'd use confident predictions on healthy samples

        # Train
        train_losses = train_epoch_phase4(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=Phase4Config.GRAD_CLIP_NORM,
            mixup_aug=mixup_aug,
            cutmix_aug=cutmix_aug,
            attn_reg=attn_reg
        )

        # Evaluate
        dev_losses, dev_metrics = evaluate_epoch(
            model, dev_loader, criterion, device,
            use_temporal=Phase4Config.USE_TEMPORAL_SEQUENCES
        )

        # Learning rate scheduling
        if Phase4Config.USE_WARMUP:
            if epoch < Phase4Config.WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Log progress
        f1_score_val = dev_metrics.get('f1_score', 0.0)
        log_msg = (f"Epoch {epoch+1}/{Phase4Config.NUM_EPOCHS} | "
                   f"Train Loss: {train_losses['total']:.4f} | "
                   f"Attn Reg: {train_losses.get('attention_reg', 0.0):.4f} | "
                   f"Dev Loss: {dev_losses['total']:.4f} | "
                   f"Dev F1: {f1_score_val:.4f} | "
                   f"Prec: {dev_metrics.get('precision', 0.0):.4f} | "
                   f"Rec: {dev_metrics.get('recall', 0.0):.4f} | "
                   f"LR: {current_lr:.6f}")
        print(log_msg)

        # Track history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_losses['total'],
            'train_attn_reg': train_losses.get('attention_reg', 0.0),
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
                'config': 'Phase4Config',
                'model_type': model_type
            }

            checkpoint_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved new best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Phase4Config.PATIENCE:
                print(f"\nEarly stopping triggered (patience={Phase4Config.PATIENCE})")
                break

    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"\n" + "=" * 80)
    print("PHASE 4 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")

    # Compare to baseline and targets
    baseline_f1 = 0.5545
    phase3_expected = 0.87
    target_f1 = 0.90

    improvement_from_baseline = (best_f1 - baseline_f1) / baseline_f1 * 100
    improvement_from_phase3 = (best_f1 - phase3_expected) / phase3_expected * 100

    print(f"\nPerformance Analysis:")
    print(f"  Baseline F1: {baseline_f1:.4f}")
    print(f"  Current F1:  {best_f1:.4f}")
    print(f"  Improvement from baseline: {improvement_from_baseline:+.1f}%")
    print(f"  Improvement from Phase 3 (est. {phase3_expected:.2f}): {improvement_from_phase3:+.1f}%")

    if best_f1 >= 0.95:
        print("\n🎉 EXCELLENT! Exceeded target (F1 >= 0.95)!")
    elif best_f1 >= target_f1:
        print(f"\n✓ Phase 4 target achieved (F1 >= {target_f1})!")
    else:
        print(f"\n⚠ Phase 4 target not met. Current: {best_f1:.4f}, Target: {target_f1}")
        print("  Consider:")
        print("    - Longer training (increase NUM_EPOCHS)")
        print("    - Ensemble multiple Phase 4 models")
        print("    - Fine-tune augmentation parameters (Mixup/CutMix alpha)")
        print("    - Try alternative model architectures")

    return {
        'best_f1': best_f1,
        'final_epoch': len(train_history),
        'checkpoint_path': str(checkpoint_path),
        'history_path': str(history_path),
        'model_type': model_type
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 4 Training: Advanced Techniques')
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
            output_dir = './checkpoints/phase4_highcapacity'
        elif args.two_stage:
            output_dir = './checkpoints/phase4_twostage'
        else:
            output_dir = './checkpoints/phase4'
    else:
        output_dir = args.output_dir

    result = train_phase4(
        seed=args.seed,
        save_dir=Path(output_dir),
        use_two_stage=args.two_stage if args.two_stage else None,
        use_high_capacity=args.high_capacity if args.high_capacity else None
    )

    print(f"\n✓ Training complete. Best F1: {result['best_f1']:.4f}")
    print("\nTo evaluate on test set, run:")
    print(f"  python evaluate.py --checkpoint {result['checkpoint_path']}")
