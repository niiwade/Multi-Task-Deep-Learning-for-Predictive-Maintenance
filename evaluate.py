"""
Evaluation script for multi-task predictive maintenance model.
Computes metrics for all three tasks: failure prediction, failure type classification, and TTF regression.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
import json
from typing import Dict, Tuple

from model import MultiTaskTCN
from data_preprocessing import load_datasets


def collect_predictions(model: nn.Module, dataloader: DataLoader,
                       device: torch.device) -> Tuple[Dict, Dict]:
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
            numeric_features = batch['numeric_features'].to(device)
            machine_type = batch['type'].to(device)

            # Forward pass
            outputs = model(numeric_features, machine_type)

            # Binary failure predictions
            failure_logits = outputs['failure_logits'].cpu()
            failure_probs = torch.sigmoid(failure_logits)
            failure_preds = (failure_probs > 0.5).float()

            all_predictions['failure_logits'].append(failure_logits)
            all_predictions['failure_probs'].append(failure_probs)
            all_predictions['failure_preds'].append(failure_preds)

            # Failure type predictions (multi-label)
            failure_type_logits = outputs['failure_type_logits'].cpu()
            failure_type_probs = torch.sigmoid(failure_type_logits)
            failure_type_preds = (failure_type_probs > 0.5).float()

            all_predictions['failure_type_logits'].append(failure_type_logits)
            all_predictions['failure_type_probs'].append(failure_type_probs)
            all_predictions['failure_type_preds'].append(failure_type_preds)

            # TTF predictions
            all_predictions['ttf'].append(outputs['ttf'].cpu())

            # Targets
            all_targets['failure'].append(batch['failure'])
            all_targets['failure_types'].append(batch['failure_types'])
            all_targets['ttf'].append(batch['ttf'])

    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = torch.cat(all_predictions[key], dim=0).numpy()

    for key in all_targets:
        all_targets[key] = torch.cat(all_targets[key], dim=0).numpy()

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
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, zero_division=0),
        'recall': recall_score(targets, predictions, zero_division=0),
        'f1_score': f1_score(targets, predictions, zero_division=0),
        'auc_roc': roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0
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
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
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
        'accuracy': accuracy_score(targets, predictions),
        'precision_micro': precision_score(targets, predictions, average='micro', zero_division=0),
        'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall_micro': recall_score(targets, predictions, average='micro', zero_division=0),
        'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
        'f1_micro': f1_score(targets, predictions, average='micro', zero_division=0),
        'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0)
    }

    # Per-class metrics
    for i, failure_type in enumerate(failure_types):
        metrics[f'{failure_type}_precision'] = precision_score(
            targets[:, i], predictions[:, i], zero_division=0
        )
        metrics[f'{failure_type}_recall'] = recall_score(
            targets[:, i], predictions[:, i], zero_division=0
        )
        metrics[f'{failure_type}_f1'] = f1_score(
            targets[:, i], predictions[:, i], zero_division=0
        )

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
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    # Separate metrics for failed vs healthy machines
    failed_mask = targets == 0
    healthy_mask = targets > 0

    if np.sum(failed_mask) > 0:
        metrics['mae_failed'] = mean_absolute_error(
            targets[failed_mask], predictions[failed_mask]
        )

    if np.sum(healthy_mask) > 0:
        metrics['mae_healthy'] = mean_absolute_error(
            targets[healthy_mask], predictions[healthy_mask]
        )

    return metrics


def evaluate_model(checkpoint_path: str, test_path: str, dev_path: str = None,
                  batch_size: int = 32) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_path: Path to test CSV
        dev_path: Optional path to dev CSV
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of all evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Load model
    model = MultiTaskTCN(
        num_numeric_features=5,
        num_types=3,
        tcn_channels=config['tcn_channels'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with dev loss {checkpoint['dev_loss']:.4f}")

    # Load datasets
    from data_preprocessing import MultiTaskDataset
    train_dataset, _, _ = load_datasets(
        config['train_path'],
        config['dev_path'],
        config['test_path']
    )

    test_dataset = MultiTaskDataset(test_path, scaler=train_dataset.scaler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test set size: {len(test_dataset)}")

    # Collect predictions
    print("\nCollecting predictions...")
    predictions, targets = collect_predictions(model, test_loader, device)

    # Evaluate each task
    print("\nEvaluating binary failure prediction...")
    failure_metrics = evaluate_binary_failure(
        predictions['failure_preds'].flatten(),
        targets['failure'].flatten(),
        predictions['failure_probs'].flatten()
    )

    print("\nEvaluating failure type classification...")
    failure_type_metrics = evaluate_failure_types(
        predictions['failure_type_preds'],
        targets['failure_types'],
        predictions['failure_type_probs']
    )

    print("\nEvaluating time-to-failure regression...")
    ttf_metrics = evaluate_ttf(
        predictions['ttf'].flatten(),
        targets['ttf'].flatten()
    )

    # Combine all metrics
    results = {
        'binary_failure': failure_metrics,
        'failure_types': failure_type_metrics,
        'time_to_failure': ttf_metrics
    }

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

    # Failure type classification
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

    # Time-to-failure regression
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

    parser = argparse.ArgumentParser(description='Evaluate multi-task predictive maintenance model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='../dataset/test/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--output', type=str, default='./checkpoints/evaluation_results.json',
                       help='Path to save results JSON')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(args.checkpoint, args.test_path, batch_size=args.batch_size)

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
