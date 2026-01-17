"""
Multi-task training script with focal loss for class imbalance.
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
from data_preprocessing import load_datasets


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


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 task_weights: Dict[str, float] = None):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            task_weights: Dictionary of task weights (failure, failure_types, ttf)
        """
        super().__init__()

        # Task-specific loss functions
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        # Task weights (default: equal weighting)
        if task_weights is None:
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
                criterion: MultiTaskLoss, device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {'total': 0, 'failure': 0, 'failure_types': 0, 'ttf': 0}
    num_batches = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Move data to device
        numeric_features = batch['numeric_features'].to(device)
        machine_type = batch['type'].to(device)
        targets = {
            'failure': batch['failure'].to(device),
            'failure_types': batch['failure_types'].to(device),
            'ttf': batch['ttf'].to(device)
        }

        # Forward pass
        optimizer.zero_grad()
        outputs = model(numeric_features, machine_type)

        # Compute loss
        loss, loss_dict = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
             device: torch.device) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    epoch_losses = {'total': 0, 'failure': 0, 'failure_types': 0, 'ttf': 0}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            # Move data to device
            numeric_features = batch['numeric_features'].to(device)
            machine_type = batch['type'].to(device)
            targets = {
                'failure': batch['failure'].to(device),
                'failure_types': batch['failure_types'].to(device),
                'ttf': batch['ttf'].to(device)
            }

            # Forward pass
            outputs = model(numeric_features, machine_type)

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


def train(config: Dict):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_dataset, dev_dataset, test_dataset = load_datasets(
        config['train_path'],
        config['dev_path'],
        config['test_path']
    )

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             sampler=train_sampler, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)

    # Initialize model
    model = MultiTaskTCN(
        num_numeric_features=5,
        num_types=3,
        tcn_channels=config['tcn_channels'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = MultiTaskLoss(
        alpha=config['focal_alpha'],
        gamma=config['focal_gamma'],
        task_weights=config['task_weights']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    best_dev_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'dev': []}

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Total: {train_losses['total']:.4f}, Failure: {train_losses['failure']:.4f}, "
              f"Types: {train_losses['failure_types']:.4f}, TTF: {train_losses['ttf']:.4f}")

        # Evaluate
        dev_losses = evaluate(model, dev_loader, criterion, device)
        print(f"Dev   - Total: {dev_losses['total']:.4f}, Failure: {dev_losses['failure']:.4f}, "
              f"Types: {dev_losses['failure_types']:.4f}, TTF: {dev_losses['ttf']:.4f}")

        # Learning rate scheduling
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
    # Configuration
    config = {
        'train_path': '../dataset/train/train.csv',
        'dev_path': '../dataset/dev/dev.csv',
        'test_path': '../dataset/test/test.csv',
        'checkpoint_dir': './checkpoints',

        # Model hyperparameters
        'tcn_channels': 64,
        'num_heads': 4,
        'dropout': 0.2,

        # Training hyperparameters
        'batch_size': 32,
        'num_epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,

        # Focal loss parameters
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,

        # Task weights
        'task_weights': {
            'failure': 1.0,
            'failure_types': 1.0,
            'ttf': 0.5  # Lower weight for TTF since it's synthesized
        }
    }

    train(config)
