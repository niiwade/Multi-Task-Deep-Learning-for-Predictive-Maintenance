"""
Ensemble training script for Phase 1 of IMPROVEMENT_PLAN_V2.md.
Trains 5 binary failure models with different random seeds for ensemble prediction.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import List, Dict

from model import MultiTaskTCN
from data_preprocessing import load_temporal_datasets
from config import BinaryOnlyConfig
from train import AsymmetricFocalLoss, MultiTaskLoss, train_epoch, evaluate_epoch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: We don't set torch.backends.cudnn.deterministic=True for performance


def train_single_model(seed: int, config_class=BinaryOnlyConfig, save_dir: Path = None) -> Dict:
    """
    Train a single binary failure model with given seed.

    Args:
        seed: Random seed for this model
        config_class: Configuration class to use
        save_dir: Directory to save checkpoints

    Returns:
        Dictionary with training results and best metrics
    """
    print("=" * 80)
    print(f"TRAINING MODEL WITH SEED {seed}")
    print("=" * 80)

    # Set seed
    set_seed(seed)

    # Create save directory
    if save_dir is None:
        save_dir = Path(f'./checkpoints/ensemble_seed_{seed}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config_class.USE_CUDA else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print(f"\nLoading datasets with {config_class.TARGET_RATIO:.0%} failure ratio target...")
    train_dataset, dev_dataset, test_dataset = load_temporal_datasets(
        str(config_class.TRAIN_PATH),
        str(config_class.DEV_PATH),
        str(config_class.TEST_PATH),
        window_size=config_class.WINDOW_SIZE,
        stride=config_class.STRIDE,
        augment_train=config_class.AUGMENT_TRAIN,
        target_ratio=config_class.TARGET_RATIO
    )

    print(f"Dataset sizes: Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")
    print(f"Train failure ratio: {train_dataset.y_failure.mean():.1%}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_class.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config_class.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create binary-only model
    print(f"\nCreating binary-only model...")
    model = MultiTaskTCN(
        num_numeric_features=config_class.NUM_NUMERIC_FEATURES,
        num_temporal_features=config_class.NUM_TEMPORAL_FEATURES,
        num_types=config_class.NUM_TYPES,
        tcn_channels=config_class.TCN_CHANNELS,
        num_heads=config_class.NUM_HEADS,
        dropout=config_class.DROPOUT,
        use_temporal_sequences=config_class.USE_TEMPORAL_SEQUENCES,
        binary_only=True  # Phase 1: Single-task model
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create loss function (asymmetric focal loss)
    criterion = MultiTaskLoss(
        alpha=config_class.FOCAL_ALPHA,
        gamma=config_class.FOCAL_GAMMA,
        task_weights=config_class.TASK_WEIGHTS,
        use_asymmetric=getattr(config_class, 'USE_ASYMMETRIC_LOSS', False),
        beta=getattr(config_class, 'ASYMMETRIC_BETA', 2.0),
        binary_only=True  # Only compute failure loss
    )

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_class.LEARNING_RATE,
        weight_decay=config_class.WEIGHT_DECAY
    )

    # Learning rate scheduler (warmup + cosine annealing)
    if config_class.USE_WARMUP:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / config_class.WARMUP_EPOCHS)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config_class.COSINE_T0,
            T_mult=config_class.COSINE_T_MULT,
            eta_min=config_class.COSINE_ETA_MIN
        )

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    train_history = []

    print(f"\nStarting training for {config_class.NUM_EPOCHS} epochs (patience={config_class.PATIENCE})...")

    for epoch in range(config_class.NUM_EPOCHS):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=config_class.GRAD_CLIP_NORM,
            use_temporal=config_class.USE_TEMPORAL_SEQUENCES
        )

        # Evaluate
        dev_losses, dev_metrics = evaluate_epoch(
            model, dev_loader, criterion, device,
            use_temporal=config_class.USE_TEMPORAL_SEQUENCES
        )

        # Learning rate scheduling
        if config_class.USE_WARMUP:
            if epoch < config_class.WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Log progress
        f1_score = dev_metrics.get('f1_score', 0.0)
        print(f"Epoch {epoch+1}/{config_class.NUM_EPOCHS} | "
              f"Train Loss: {train_losses['total']:.4f} | "
              f"Dev Loss: {dev_losses['total']:.4f} | "
              f"Dev F1: {f1_score:.4f} | "
              f"LR: {current_lr:.6f}")

        # Track history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_losses['total'],
            'dev_loss': dev_losses['total'],
            'dev_f1': f1_score,
            'dev_precision': dev_metrics.get('precision', 0.0),
            'dev_recall': dev_metrics.get('recall', 0.0),
            'lr': current_lr
        })

        # Save best model
        if f1_score > best_f1:
            best_f1 = f1_score
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'seed': seed,
                'best_f1': best_f1,
                'dev_metrics': dev_metrics,
                'config': config_class.to_dict()
            }

            checkpoint_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved new best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config_class.PATIENCE:
                print(f"\nEarly stopping triggered (patience={config_class.PATIENCE})")
                break

    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"\nTraining complete for seed {seed}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return {
        'seed': seed,
        'best_f1': best_f1,
        'final_epoch': len(train_history),
        'checkpoint_path': str(checkpoint_path),
        'history_path': str(history_path)
    }


def train_ensemble(config_class=BinaryOnlyConfig, output_dir: str = './checkpoints/ensemble'):
    """
    Train ensemble of 5 models with different random seeds.

    Args:
        config_class: Configuration class to use
        output_dir: Directory to save all ensemble models
    """
    print("=" * 80)
    print("PHASE 1 ENSEMBLE TRAINING")
    print("=" * 80)
    print(f"\nConfiguration: {config_class.__name__}")
    print(f"Seeds: {config_class.ENSEMBLE_SEEDS}")
    print(f"Target failure ratio: {config_class.TARGET_RATIO:.0%}")
    print(f"Asymmetric loss: {getattr(config_class, 'USE_ASYMMETRIC_LOSS', False)}")
    print(f"Binary only: True")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train each model
    results = []
    for seed in config_class.ENSEMBLE_SEEDS:
        save_dir = output_path / f'seed_{seed}'
        result = train_single_model(seed, config_class, save_dir)
        results.append(result)
        print("\n")

    # Summary
    print("=" * 80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 80)

    avg_f1 = np.mean([r['best_f1'] for r in results])
    std_f1 = np.std([r['best_f1'] for r in results])

    print(f"\nResults summary:")
    for result in results:
        print(f"  Seed {result['seed']}: F1={result['best_f1']:.4f} (epoch {result['final_epoch']})")

    print(f"\nEnsemble statistics:")
    print(f"  Mean F1: {avg_f1:.4f}")
    print(f"  Std F1: {std_f1:.4f}")
    print(f"  Min F1: {min(r['best_f1'] for r in results):.4f}")
    print(f"  Max F1: {max(r['best_f1'] for r in results):.4f}")

    # Save ensemble summary
    summary_path = output_path / 'ensemble_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': config_class.__name__,
            'models': results,
            'statistics': {
                'mean_f1': float(avg_f1),
                'std_f1': float(std_f1),
                'min_f1': float(min(r['best_f1'] for r in results)),
                'max_f1': float(max(r['best_f1'] for r in results))
            }
        }, f, indent=2)

    print(f"\nEnsemble summary saved to: {summary_path}")

    return results


if __name__ == '__main__':
    # Train ensemble with BinaryOnlyConfig
    results = train_ensemble(
        config_class=BinaryOnlyConfig,
        output_dir='./checkpoints/ensemble_phase1'
    )
