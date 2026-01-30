"""
Multi-task training script with focal loss for class imbalance.
Updated with improved data preprocessing, AdamW optimizer, and cosine annealing LR schedule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, Tuple

from model import MultiTaskTCN
from data_preprocessing import load_datasets, load_temporal_datasets
from config import Config


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in binary classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for safety-critical applications.
    Penalizes false negatives (missing failures) more than false positives (false alarms).
    """

    def __init__(self, alpha: float = 0.80, gamma: float = 3.0, beta: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class (default: 0.80)
            gamma: Focusing parameter (default: 3.0)
            beta: Asymmetric penalty for false negatives (beta > 1.0 means FN costs more than FP)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            Asymmetric focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)

        # Compute focal term
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_term = alpha_t * (1 - p_t) ** self.gamma

        # Asymmetric penalty: multiply by beta for actual failures (targets=1)
        # This makes false negatives (target=1, prob low) cost more
        asymmetric_weight = targets * self.beta + (1 - targets) * 1.0

        focal_loss = focal_term * asymmetric_weight * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for binary classification.
    Phase 3 improvement from IMPROVEMENT_PLAN_V2.md.
    Good for imbalanced datasets, directly optimizes F1-like metric.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            Dice loss value (1 - Dice coefficient)
        """
        probs = torch.sigmoid(logits)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # Return 1 - Dice (loss should be minimized)
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss with separate FP/FN penalties.
    Phase 3 improvement from IMPROVEMENT_PLAN_V2.md.

    alpha controls FP penalty, beta controls FN penalty.
    When alpha=beta=0.5, reduces to Dice Loss.
    For safety-critical: set beta > alpha (penalize FN more).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        """
        Args:
            alpha: False positive penalty (lower = less penalty)
            beta: False negative penalty (higher = more penalty)
            smooth: Smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            Tversky loss value
        """
        probs = torch.sigmoid(logits)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        # True positives, false positives, false negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        # Tversky index = TP / (TP + α*FP + β*FN)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Return 1 - Tversky (loss should be minimized)
        return 1 - tversky


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 task_weights: Dict[str, float] = None,
                 use_asymmetric: bool = False, beta: float = 2.0,
                 binary_only: bool = False):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            task_weights: Dictionary of task weights (failure, failure_types, ttf)
            use_asymmetric: Use asymmetric focal loss (penalizes FN more than FP)
            beta: Asymmetric penalty factor (only used if use_asymmetric=True)
            binary_only: If True, only compute failure loss (Phase 1 improvement)
        """
        super().__init__()

        # Task-specific loss functions
        if use_asymmetric:
            self.focal_loss = AsymmetricFocalLoss(alpha=alpha, gamma=gamma, beta=beta)
        else:
            self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

        self.binary_only = binary_only

        if not binary_only:
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.mse_loss = nn.MSELoss()

        # Task weights (default: equal weighting)
        if task_weights is None:
            if binary_only:
                task_weights = {'failure': 1.0}
            else:
                task_weights = {'failure': 1.0, 'failure_types': 1.0, 'ttf': 1.0}
        self.task_weights = task_weights

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model predictions
            targets: Ground truth labels

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Task 1: Binary failure prediction (Focal Loss)
        failure_loss = self.focal_loss(outputs['failure_logits'], targets['failure'])

        if self.binary_only:
            # Single-task mode
            total_loss = self.task_weights['failure'] * failure_loss
            loss_dict = {
                'total': total_loss.item(),
                'failure': failure_loss.item()
            }
        else:
            # Multi-task mode
            # Task 2: Failure type classification (BCE for multi-label)
            failure_type_loss = self.bce_loss(outputs['failure_type_logits'], targets['failure_types'])

            # Task 3: Time-to-failure regression (MSE)
            ttf_loss = self.mse_loss(outputs['ttf'], targets['ttf'])

            # Weighted combination
            total_loss = (self.task_weights['failure'] * failure_loss +
                         self.task_weights['failure_types'] * failure_type_loss +
                         self.task_weights['ttf'] * ttf_loss)

            loss_dict = {
                'total': total_loss.item(),
                'failure': failure_loss.item(),
                'failure_types': failure_type_loss.item(),
                'ttf': ttf_loss.item()
            }

        return total_loss, loss_dict


def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create weighted sampler to handle class imbalance."""
    failures = dataset.y_failure
    class_counts = np.bincount(failures.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[failures.astype(int)]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: MultiTaskLoss, device: torch.device, grad_clip_norm: float = 0.5,
                use_temporal: bool = True) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {'total': 0, 'failure': 0, 'failure_types': 0, 'ttf': 0}
    num_batches = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Move data to device
        machine_type = batch['type'].to(device)
        targets = {
            'failure': batch['failure'].to(device),
            'failure_types': batch['failure_types'].to(device),
            'ttf': batch['ttf'].to(device)
        }

        # Forward pass
        optimizer.zero_grad()

        if use_temporal and 'sequence' in batch:
            # Temporal sequence mode
            sequence = batch['sequence'].to(device)
            features = batch['features'].to(device)
            outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)
        else:
            # Backward compatibility mode
            numeric_features = batch['numeric_features'].to(device)
            outputs = model(numeric_features=numeric_features, machine_type=machine_type)

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


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: MultiTaskLoss,
             device: torch.device, use_temporal: bool = True) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    epoch_losses = {'total': 0, 'failure': 0, 'failure_types': 0, 'ttf': 0}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            # Move data to device
            machine_type = batch['type'].to(device)
            targets = {
                'failure': batch['failure'].to(device),
                'failure_types': batch['failure_types'].to(device),
                'ttf': batch['ttf'].to(device)
            }

            # Forward pass
            if use_temporal and 'sequence' in batch:
                # Temporal sequence mode
                sequence = batch['sequence'].to(device)
                features = batch['features'].to(device)
                outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)
            else:
                # Backward compatibility mode
                numeric_features = batch['numeric_features'].to(device)
                outputs = model(numeric_features=numeric_features, machine_type=machine_type)

            # Compute loss
            _, loss_dict = criterion(outputs, targets)

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def evaluate_epoch(model: nn.Module, dataloader: DataLoader, criterion: MultiTaskLoss,
                   device: torch.device, use_temporal: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate model on validation/test set with both losses and metrics.

    Returns:
        Tuple of (losses_dict, metrics_dict)
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    model.eval()
    epoch_losses = {'total': 0, 'failure': 0}
    num_batches = 0

    # Collect predictions and targets for metrics
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            machine_type = batch['type'].to(device)
            targets = {
                'failure': batch['failure'].to(device),
            }

            # Also get failure_types and ttf if available (for multi-task mode)
            if 'failure_types' in batch:
                targets['failure_types'] = batch['failure_types'].to(device)
            if 'ttf' in batch:
                targets['ttf'] = batch['ttf'].to(device)

            # Forward pass
            if use_temporal and 'sequence' in batch:
                # Temporal sequence mode
                sequence = batch['sequence'].to(device)
                features = batch['features'].to(device)
                outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)
            else:
                # Backward compatibility mode
                numeric_features = batch['numeric_features'].to(device)
                outputs = model(numeric_features=numeric_features, machine_type=machine_type)

            # Compute loss
            _, loss_dict = criterion(outputs, targets)

            # Accumulate losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value
                else:
                    epoch_losses[key] = value
            num_batches += 1

            # Collect predictions for binary failure
            failure_probs = torch.sigmoid(outputs['failure_logits']).cpu().numpy()
            failure_preds = (failure_probs > 0.5).astype(int)
            failure_targets = targets['failure'].cpu().numpy()

            all_preds.extend(failure_preds.flatten())
            all_targets.extend(failure_targets.flatten())

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    # Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        'accuracy': float(accuracy_score(all_targets, all_preds)),
        'precision': float(precision_score(all_targets, all_preds, zero_division=0)),
        'recall': float(recall_score(all_targets, all_preds, zero_division=0)),
        'f1_score': float(f1_score(all_targets, all_preds, zero_division=0))
    }

    return epoch_losses, metrics


def train(config: Dict):
    """Main training function with improved data preprocessing and LR scheduling."""
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
    print(f"Using device: {device}")

    use_temporal = config.get('use_temporal_sequences', False)

    # Load datasets
    print("Loading datasets...")
    if use_temporal:
        train_dataset, dev_dataset, test_dataset = load_temporal_datasets(
            str(config['train_path']),
            str(config['dev_path']),
            str(config['test_path']),
            window_size=config.get('window_size', 12),
            stride=config.get('stride', 1),
            augment_train=config.get('augment_train', True),
            target_ratio=config.get('target_ratio', 0.15)
        )
    else:
        train_dataset, dev_dataset, test_dataset = load_datasets(
            str(config['train_path']),
            str(config['dev_path']),
            str(config['test_path'])
        )

    print(f"\nDataset sizes: Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")

    # Create data loaders
    if config.get('use_weighted_sampling', True) and not use_temporal:
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                 sampler=train_sampler, num_workers=0)
    else:
        # For temporal datasets, augmentation already balances classes
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=0)

    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)

    # Initialize model
    if use_temporal:
        model = MultiTaskTCN(
            num_numeric_features=config.get('num_numeric_features', 5),
            num_temporal_features=config.get('num_temporal_features', 19),
            num_types=config.get('num_types', 3),
            tcn_channels=config['tcn_channels'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            use_temporal_sequences=True
        ).to(device)
    else:
        model = MultiTaskTCN(
            num_numeric_features=config.get('num_numeric_features', 5),
            num_types=config.get('num_types', 3),
            tcn_channels=config['tcn_channels'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            use_temporal_sequences=False
        ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = MultiTaskLoss(
        alpha=config['focal_alpha'],
        gamma=config['focal_gamma'],
        task_weights=config['task_weights']
    )

    # Use AdamW instead of Adam (better regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Warmup + Cosine Annealing LR scheduler
    if config.get('use_warmup', False):
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

        warmup_epochs = config.get('warmup_epochs', 5)

        # Warmup scheduler
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda)

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('cosine_t0', 15),
            T_mult=config.get('cosine_t_mult', 2),
            eta_min=config.get('cosine_eta_min', 1e-6)
        )
    else:
        # Fallback to ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

    # Training loop
    best_dev_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'dev': []}

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=config.get('grad_clip_norm', 0.5),
            use_temporal=use_temporal
        )
        print(f"Train - Total: {train_losses['total']:.4f}, Failure: {train_losses['failure']:.4f}, "
              f"Types: {train_losses['failure_types']:.4f}, TTF: {train_losses['ttf']:.4f}")

        # Evaluate
        dev_losses = evaluate(model, dev_loader, criterion, device, use_temporal=use_temporal)
        print(f"Dev   - Total: {dev_losses['total']:.4f}, Failure: {dev_losses['failure']:.4f}, "
              f"Types: {dev_losses['failure_types']:.4f}, TTF: {dev_losses['ttf']:.4f}")

        # Learning rate scheduling
        if config.get('use_warmup', False):
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
        else:
            scheduler.step(dev_losses['total'])

        # Save history
        history['train'].append(train_losses)
        history['dev'].append(dev_losses)

        # Early stopping
        if dev_losses['total'] < best_dev_loss:
            best_dev_loss = dev_losses['total']
            patience_counter = 0

            # Save best model
            checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_loss': best_dev_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save training history
    history_path = Path(config['checkpoint_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best dev loss: {best_dev_loss:.4f}")


if __name__ == '__main__':
    # Use improved configuration
    print("=" * 80)
    print("TRAINING MULTI-TASK PREDICTIVE MAINTENANCE MODEL")
    print("=" * 80)

    Config.print_config()

    # Train with improved configuration
    config = Config.to_dict()

    train(config)

    print("\n" + "=" * 80)
    print("Training script completed successfully!")
    print("=" * 80)
