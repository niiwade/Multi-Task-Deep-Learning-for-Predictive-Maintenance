"""
Evaluation script for multi-task predictive maintenance model.
Computes metrics for all three tasks: failure prediction, failure type classification, and TTF regression.
Enhanced with threshold optimization and probability calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, Optional

from model import MultiTaskTCN
from data_preprocessing import load_datasets, load_temporal_datasets


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.

    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50):
        """
        Optimize temperature on validation set.

        Args:
            logits: Model logits (before sigmoid)
            labels: True binary labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                self.forward(logits), labels
            )
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return self.temperature.item()


def optimize_threshold(predictions: Dict, targets: Dict) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold to maximize F1-score.

    Args:
        predictions: Model predictions dictionary
        targets: Ground truth labels dictionary

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Get probabilities and true labels
    y_true = targets['failure'].flatten()
    y_probs = predictions['failure_probs'].flatten()

    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Compute F1-score for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find threshold that maximizes F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]

    # Get predictions at optimal threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

    # Compute metrics at optimal threshold
    metrics = {
        'optimal_threshold': float(optimal_threshold),
        'precision': float(precisions[optimal_idx]),
        'recall': float(recalls[optimal_idx]),
        'f1_score': float(optimal_f1),
        'accuracy': float(accuracy_score(y_true, y_pred_optimal))
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_optimal)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        })

    return optimal_threshold, metrics


def visualize_precision_recall_curve(predictions: Dict, targets: Dict,
                                     save_path: Optional[str] = None):
    """
    Visualize precision-recall trade-off and optimal threshold.

    Args:
        predictions: Model predictions dictionary
        targets: Ground truth labels dictionary
        save_path: Path to save the plot (optional)
    """
    y_true = targets['failure'].flatten()
    y_probs = predictions['failure_probs'].flatten()

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Find optimal point
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, linewidth=2, label='PR Curve', color='blue')
    plt.scatter(recalls[optimal_idx], precisions[optimal_idx],
               color='red', s=100, zorder=5,
               label=f'Optimal (F1={f1_scores[optimal_idx]:.3f}, ' +
                     f'Threshold={thresholds[optimal_idx]:.3f})')

    # Add iso-F1 curves
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, linestyle='--')
        plt.annotate(f'F1={f_score:.1f}', xy=(0.9, y[45] + 0.02), fontsize=8, color='gray')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve for Failure Prediction', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(loc='best')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    else:
        plt.show()
    plt.close()


def calibrate_model(model: nn.Module, dev_loader: DataLoader, device: torch.device,
                   use_temporal: bool = True) -> Tuple[float, TemperatureScaling]:
    """
    Calibrate model probabilities using validation set.

    Args:
        model: Trained model
        dev_loader: Validation data loader
        device: Device to run on
        use_temporal: Whether using temporal sequences

    Returns:
        Tuple of (optimal_temperature, temperature_scaler)
    """
    # Collect logits and labels from validation set
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dev_loader:
            machine_type = batch['type'].to(device)

            if use_temporal and 'sequence' in batch:
                sequence = batch['sequence'].to(device)
                features = batch['features'].to(device)
                outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)
            else:
                numeric_features = batch['numeric_features'].to(device)
                outputs = model(numeric_features=numeric_features, machine_type=machine_type)

            all_logits.append(outputs['failure_logits'].cpu())
            all_labels.append(batch['failure'])

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits, labels)

    print(f"Optimal temperature: {optimal_temp:.4f}")

    return optimal_temp, temp_scaler


def collect_predictions(model: nn.Module, dataloader: DataLoader,
                       device: torch.device, use_temporal: bool = False) -> Tuple[Dict, Dict]:
    """
    Collect all predictions and ground truth labels.

    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Device to run inference on

    Returns:
        Tuple of (predictions_dict, targets_dict)
    """
    model.eval()

    all_predictions = {
        'failure_logits': [],
        'failure_probs': [],
        'failure_preds': [],
        'failure_type_logits': [],
        'failure_type_probs': [],
        'failure_type_preds': [],
        'ttf': []
    }

    all_targets = {
        'failure': [],
        'failure_types': [],
        'ttf': []
    }

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            machine_type = batch['type'].to(device)

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

            # Binary failure predictions
            failure_logits = outputs['failure_logits'].cpu()
            failure_probs = torch.sigmoid(failure_logits)
            failure_preds = (failure_probs > 0.5).float()

            all_predictions['failure_logits'].append(failure_logits)
            all_predictions['failure_probs'].append(failure_probs)
            all_predictions['failure_preds'].append(failure_preds)

            # Failure type predictions (multi-label) - only if available
            if 'failure_type_logits' in outputs:
                failure_type_logits = outputs['failure_type_logits'].cpu()
                failure_type_probs = torch.sigmoid(failure_type_logits)
                failure_type_preds = (failure_type_probs > 0.5).float()

                all_predictions['failure_type_logits'].append(failure_type_logits)
                all_predictions['failure_type_probs'].append(failure_type_probs)
                all_predictions['failure_type_preds'].append(failure_type_preds)

            # TTF predictions - only if available
            if 'ttf' in outputs:
                all_predictions['ttf'].append(outputs['ttf'].cpu())

            # Targets
            all_targets['failure'].append(batch['failure'])
            if 'failure_types' in batch:
                all_targets['failure_types'].append(batch['failure_types'])
            if 'ttf' in batch:
                all_targets['ttf'].append(batch['ttf'])

    # Concatenate all batches (only for keys that have data)
    for key in list(all_predictions.keys()):
        if len(all_predictions[key]) > 0:
            all_predictions[key] = torch.cat(all_predictions[key], dim=0).numpy()
        else:
            # Remove empty keys (not predicted by this model)
            del all_predictions[key]

    for key in list(all_targets.keys()):
        if len(all_targets[key]) > 0:
            all_targets[key] = torch.cat(all_targets[key], dim=0).numpy()
        else:
            # Remove empty keys
            del all_targets[key]

    return all_predictions, all_targets


def evaluate_binary_failure(predictions: np.ndarray, targets: np.ndarray,
                           probs: np.ndarray) -> Dict[str, float]:
    """
    Evaluate binary failure prediction task.

    Args:
        predictions: Binary predictions (0/1)
        targets: Ground truth labels (0/1)
        probs: Predicted probabilities

    Returns:
        Dictionary of metrics (JSON serializable)
    """
    metrics = {
        'accuracy': float(accuracy_score(targets, predictions)),
        'precision': float(precision_score(targets, predictions, zero_division=0)),
        'recall': float(recall_score(targets, predictions, zero_division=0)),
        'f1_score': float(f1_score(targets, predictions, zero_division=0)),
        'auc_roc': float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.0
    }

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        })

    return metrics


def evaluate_failure_types(predictions: np.ndarray, targets: np.ndarray,
                          probs: np.ndarray) -> Dict[str, float]:
    """
    Evaluate failure type classification (multi-label).

    Args:
        predictions: Multi-label predictions (N, 5)
        targets: Ground truth labels (N, 5)
        probs: Predicted probabilities (N, 5)

    Returns:
        Dictionary of metrics
    """
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    # Overall metrics (micro/macro averaging)
    metrics = {
        'accuracy': float(accuracy_score(targets, predictions)),
        'precision_micro': float(precision_score(targets, predictions, average='micro', zero_division=0)),
        'precision_macro': float(precision_score(targets, predictions, average='macro', zero_division=0)),
        'recall_micro': float(recall_score(targets, predictions, average='micro', zero_division=0)),
        'recall_macro': float(recall_score(targets, predictions, average='macro', zero_division=0)),
        'f1_micro': float(f1_score(targets, predictions, average='micro', zero_division=0)),
        'f1_macro': float(f1_score(targets, predictions, average='macro', zero_division=0))
    }

    # Per-class metrics
    for i, failure_type in enumerate(failure_types):
        metrics[f'{failure_type}_precision'] = float(precision_score(
            targets[:, i], predictions[:, i], zero_division=0
        ))
        metrics[f'{failure_type}_recall'] = float(recall_score(
            targets[:, i], predictions[:, i], zero_division=0
        ))
        metrics[f'{failure_type}_f1'] = float(f1_score(
            targets[:, i], predictions[:, i], zero_division=0
        ))

    return metrics


def evaluate_ttf(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Evaluate time-to-failure regression task.

    Args:
        predictions: Predicted TTF values
        targets: Ground truth TTF values

    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }

    # Separate metrics for failed vs healthy machines
    failed_mask = targets == 0
    healthy_mask = targets > 0

    if np.sum(failed_mask) > 0:
        metrics['mae_failed'] = float(mean_absolute_error(
            targets[failed_mask], predictions[failed_mask]
        ))

    if np.sum(healthy_mask) > 0:
        metrics['mae_healthy'] = float(mean_absolute_error(
            targets[healthy_mask], predictions[healthy_mask]
        ))

    return metrics


def evaluate_model(checkpoint_path: str, test_path: str, dev_path: str = None,
                  batch_size: int = 32, optimize_threshold_flag: bool = True,
                  calibrate: bool = False, output_dir: str = './checkpoints') -> Dict:
    """
    Comprehensive model evaluation with threshold optimization and calibration.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_path: Path to test CSV
        dev_path: Optional path to dev CSV for calibration
        batch_size: Batch size for evaluation
        optimize_threshold_flag: Whether to optimize decision threshold
        calibrate: Whether to calibrate probabilities
        output_dir: Directory to save outputs

    Returns:
        Dictionary of all evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Handle config as string (config class name) vs dictionary
    if isinstance(config, str):
        # Config is a string like 'Phase2Config', 'Phase3Config', etc.
        print(f"Config stored as class name: {config}")

        # Import the appropriate config class
        from config import (Config, BinaryOnlyConfig, Phase2Config,
                           Phase3Config, Phase4Config)

        config_classes = {
            'Config': Config,
            'BinaryOnlyConfig': BinaryOnlyConfig,
            'Phase2Config': Phase2Config,
            'Phase3Config': Phase3Config,
            'Phase4Config': Phase4Config
        }

        if config in config_classes:
            config_class = config_classes[config]
            # Convert to dictionary
            config = config_class.to_dict()
        else:
            # Default to BinaryOnlyConfig if unknown
            print(f"Warning: Unknown config class '{config}', using BinaryOnlyConfig")
            config = BinaryOnlyConfig.to_dict()

    use_temporal = config.get('use_temporal_sequences', False)

    # Check for model type (Phase 3/4 may use advanced architectures)
    model_type = checkpoint.get('model_type', 'Standard')
    print(f"Model type: {model_type}")

    # Load model based on type
    if model_type == 'TwoStage':
        from model_advanced import TwoStageClassifier
        model = TwoStageClassifier(
            num_numeric_features=config.get('num_numeric_features', 5),
            num_temporal_features=config.get('num_temporal_features', 19),
            num_types=config.get('num_types', 3),
            tcn_channels=config.get('tcn_channels', 64),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.3),
            use_temporal_sequences=use_temporal
        ).to(device)
    elif model_type == 'HighCapacity':
        from model_advanced import HighCapacityTCN
        model = HighCapacityTCN(
            num_numeric_features=config.get('num_numeric_features', 5),
            num_temporal_features=config.get('num_temporal_features', 19),
            num_types=config.get('num_types', 3),
            tcn_channels=config.get('tcn_channels', 128),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.3),
            use_temporal_sequences=use_temporal,
            binary_only=True
        ).to(device)
    elif use_temporal:
        # Standard MultiTaskTCN with temporal sequences
        # Check if binary_only (Phase 1-4 all use binary_only)
        binary_only = config.get('task_weights', {}).get('failure', 0) == 1.0 and \
                     len(config.get('task_weights', {})) == 1

        model = MultiTaskTCN(
            num_numeric_features=config.get('num_numeric_features', 5),
            num_temporal_features=config.get('num_temporal_features', 19),
            num_types=config.get('num_types', 3),
            tcn_channels=config.get('tcn_channels', 64),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.3),
            use_temporal_sequences=True,
            binary_only=binary_only
        ).to(device)
    else:
        # Standard MultiTaskTCN without temporal sequences
        model = MultiTaskTCN(
            num_numeric_features=5,
            num_types=3,
            tcn_channels=config.get('tcn_channels', 64),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.3),
            use_temporal_sequences=False
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Print checkpoint info
    epoch_info = checkpoint.get('epoch', 'unknown')
    if 'best_f1' in checkpoint:
        print(f"Loaded model from epoch {epoch_info} with best F1: {checkpoint['best_f1']:.4f}")
    elif 'dev_loss' in checkpoint:
        print(f"Loaded model from epoch {epoch_info} with dev loss: {checkpoint['dev_loss']:.4f}")
    else:
        print(f"Loaded model from epoch {epoch_info}")

    # Load datasets
    if use_temporal:
        train_dataset, dev_dataset, test_dataset = load_temporal_datasets(
            config['train_path'],
            config['dev_path'],
            test_path,
            window_size=config.get('window_size', 12),
            stride=config.get('stride', 1),
            augment_train=False,
            target_ratio=0.15
        )
    else:
        from data_preprocessing import MultiTaskDataset
        train_dataset, dev_dataset, _ = load_datasets(
            './dataset/train/train.csv',
            './dataset/dev/dev.csv',
            './dataset/test/test.csv'
        )
        test_dataset = MultiTaskDataset(test_path, scaler=train_dataset.scaler)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test set size: {len(test_dataset)}")

    # Calibration (if requested and dev set available)
    temperature = 1.0
    if calibrate and dev_path:
        print("\nCalibrating model probabilities...")
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        temperature, temp_scaler = calibrate_model(model, dev_loader, device, use_temporal)

    # Collect predictions
    print("\nCollecting predictions...")
    predictions, targets = collect_predictions(model, test_loader, device, use_temporal)

    # Apply temperature scaling if calibrated
    if calibrate and temperature != 1.0:
        print(f"Applying temperature scaling (T={temperature:.4f})...")
        predictions['failure_probs'] = torch.sigmoid(
            torch.tensor(predictions['failure_logits']) / temperature
        ).numpy()

    # Threshold optimization
    if optimize_threshold_flag:
        print("\nOptimizing decision threshold...")
        optimal_threshold, optimized_metrics = optimize_threshold(predictions, targets)
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"F1-Score at optimal threshold: {optimized_metrics['f1_score']:.4f}")

        # Update predictions with optimal threshold
        predictions['failure_preds'] = (
            predictions['failure_probs'] >= optimal_threshold
        ).astype(float)

        # Visualize PR curve
        pr_curve_path = Path(output_dir) / 'precision_recall_curve.png'
        pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_precision_recall_curve(predictions, targets, save_path=str(pr_curve_path))

    # Evaluate each task
    print("\nEvaluating binary failure prediction...")
    failure_metrics = evaluate_binary_failure(
        predictions['failure_preds'].flatten(),
        targets['failure'].flatten(),
        predictions['failure_probs'].flatten()
    )

    # Add optimization results to metrics
    if optimize_threshold_flag:
        failure_metrics['optimal_threshold'] = float(optimal_threshold)
        failure_metrics['optimized'] = True

    if calibrate:
        failure_metrics['temperature'] = float(temperature)
        failure_metrics['calibrated'] = True

    # Evaluate failure type classification (if available)
    failure_type_metrics = None
    if 'failure_type_preds' in predictions and 'failure_types' in targets:
        print("\nEvaluating failure type classification...")
        failure_type_metrics = evaluate_failure_types(
            predictions['failure_type_preds'],
            targets['failure_types'],
            predictions['failure_type_probs']
        )
    else:
        print("\nSkipping failure type evaluation (binary-only model)")

    # Evaluate time-to-failure regression (if available)
    ttf_metrics = None
    if 'ttf' in predictions and 'ttf' in targets:
        print("\nEvaluating time-to-failure regression...")
        # Denormalize TTF if using temporal dataset
        if use_temporal and hasattr(test_dataset, 'denormalize_ttf'):
            ttf_pred_denorm = test_dataset.denormalize_ttf(predictions['ttf'].flatten())
            ttf_target_denorm = test_dataset.denormalize_ttf(targets['ttf'].flatten())
            ttf_metrics = evaluate_ttf(ttf_pred_denorm, ttf_target_denorm)
            ttf_metrics['denormalized'] = True
        else:
            ttf_metrics = evaluate_ttf(
                predictions['ttf'].flatten(),
                targets['ttf'].flatten()
            )
    else:
        print("\nSkipping TTF evaluation (binary-only model)")

    # Combine all metrics
    results = {
        'binary_failure': failure_metrics,
        'config': {
            'use_temporal_sequences': use_temporal,
            'threshold_optimized': optimize_threshold_flag,
            'calibrated': calibrate
        }
    }

    # Add optional metrics if available
    if failure_type_metrics is not None:
        results['failure_types'] = failure_type_metrics
    if ttf_metrics is not None:
        results['time_to_failure'] = ttf_metrics

    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Binary failure prediction
    print("\n1. BINARY FAILURE PREDICTION")
    print("-" * 60)
    failure_metrics = results['binary_failure']
    print(f"Accuracy:    {failure_metrics['accuracy']:.4f}")
    print(f"Precision:   {failure_metrics['precision']:.4f}")
    print(f"Recall:      {failure_metrics['recall']:.4f}")
    print(f"F1-Score:    {failure_metrics['f1_score']:.4f} (Target: > 0.95)")
    print(f"AUC-ROC:     {failure_metrics['auc_roc']:.4f}")

    if 'true_positives' in failure_metrics:
        print(f"\nConfusion Matrix:")
        print(f"  TN: {failure_metrics['true_negatives']:4d}  FP: {failure_metrics['false_positives']:4d}")
        print(f"  FN: {failure_metrics['false_negatives']:4d}  TP: {failure_metrics['true_positives']:4d}")

    # Failure type classification (if available)
    if 'failure_types' in results:
        print("\n2. FAILURE TYPE CLASSIFICATION")
        print("-" * 60)
        type_metrics = results['failure_types']
        print(f"Overall Accuracy: {type_metrics['accuracy']:.4f} (Target: > 0.95)")
        print(f"Macro F1-Score:   {type_metrics['f1_macro']:.4f}")
        print(f"Micro F1-Score:   {type_metrics['f1_micro']:.4f}")

        print("\nPer-Class Metrics:")
        for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            prec = type_metrics[f'{failure_type}_precision']
            rec = type_metrics[f'{failure_type}_recall']
            f1 = type_metrics[f'{failure_type}_f1']
            print(f"  {failure_type}: Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

    # Time-to-failure regression (if available)
    if 'time_to_failure' in results:
        print("\n3. TIME-TO-FAILURE REGRESSION")
        print("-" * 60)
        ttf_metrics = results['time_to_failure']
        print(f"MAE:   {ttf_metrics['mae']:.2f} hours (Target: < 2 hours)")
        print(f"RMSE:  {ttf_metrics['rmse']:.2f} hours")
        print(f"R²:    {ttf_metrics['r2']:.4f}")

        if 'mae_failed' in ttf_metrics:
            print(f"\nMAE (Failed):  {ttf_metrics['mae_failed']:.2f} hours")
        if 'mae_healthy' in ttf_metrics:
            print(f"MAE (Healthy): {ttf_metrics['mae_healthy']:.2f} hours")

    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate multi-task predictive maintenance model with threshold optimization and calibration'
    )
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='./dataset/test/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--dev_path', type=str, default='./dataset/dev/dev.csv',
                       help='Path to dev CSV (for calibration)')
    parser.add_argument('--output', type=str, default='./checkpoints/evaluation_results.json',
                       help='Path to save results JSON')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save outputs (PR curve, etc.)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--optimize_threshold', action='store_true', default=True,
                       help='Optimize decision threshold (default: True)')
    parser.add_argument('--no_optimize_threshold', action='store_false', dest='optimize_threshold',
                       help='Disable threshold optimization')
    parser.add_argument('--calibrate', action='store_true', default=False,
                       help='Calibrate probabilities using temperature scaling')

    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATING MULTI-TASK PREDICTIVE MAINTENANCE MODEL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test path: {args.test_path}")
    print(f"  Optimize threshold: {args.optimize_threshold}")
    print(f"  Calibrate: {args.calibrate}")
    print()

    # Run evaluation
    results = evaluate_model(
        args.checkpoint,
        args.test_path,
        dev_path=args.dev_path if args.calibrate else None,
        batch_size=args.batch_size,
        optimize_threshold_flag=args.optimize_threshold,
        calibrate=args.calibrate,
        output_dir=args.output_dir
    )

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
