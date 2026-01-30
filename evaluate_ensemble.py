"""
Ensemble evaluation script for Phase 1 of IMPROVEMENT_PLAN_V2.md.
Evaluates ensemble of 5 binary failure models using majority voting.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                            confusion_matrix, classification_report)

from model import MultiTaskTCN
from data_preprocessing import load_temporal_datasets
from config import BinaryOnlyConfig


def load_ensemble_models(ensemble_dir: Path, config_class=BinaryOnlyConfig, device=None) -> List[torch.nn.Module]:
    """
    Load all models from ensemble directory.

    Args:
        ensemble_dir: Directory containing ensemble models
        config_class: Configuration class used for training
        device: Device to load models to

    Returns:
        List of loaded models
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    seeds = config_class.ENSEMBLE_SEEDS

    print(f"Loading {len(seeds)} ensemble models from {ensemble_dir}...")

    for seed in seeds:
        checkpoint_path = ensemble_dir / f'seed_{seed}' / 'best_model.pt'

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for seed {seed}: {checkpoint_path}")
            continue

        # Create model
        model = MultiTaskTCN(
            num_numeric_features=config_class.NUM_NUMERIC_FEATURES,
            num_temporal_features=config_class.NUM_TEMPORAL_FEATURES,
            num_types=config_class.NUM_TYPES,
            tcn_channels=config_class.TCN_CHANNELS,
            num_heads=config_class.NUM_HEADS,
            dropout=config_class.DROPOUT,
            use_temporal_sequences=config_class.USE_TEMPORAL_SEQUENCES,
            binary_only=True
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models.append({
            'model': model,
            'seed': seed,
            'best_f1': checkpoint.get('best_f1', 0.0)
        })

        print(f"  Loaded seed {seed}: F1={checkpoint.get('best_f1', 0.0):.4f}")

    print(f"\nSuccessfully loaded {len(models)} models")
    return models


def predict_ensemble(models: List[Dict], dataloader, device, use_temporal: bool = True,
                     voting: str = 'hard') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make ensemble predictions using voting.

    Args:
        models: List of model dictionaries
        dataloader: Data loader for predictions
        device: Device to use
        use_temporal: Whether to use temporal sequences
        voting: 'hard' for majority voting, 'soft' for probability averaging

    Returns:
        Tuple of (predictions, probabilities, targets)
    """
    all_model_probs = []
    all_targets = []

    # Get predictions from each model
    for model_dict in models:
        model = model_dict['model']
        model.eval()

        model_probs = []
        model_targets = []

        with torch.no_grad():
            for batch in dataloader:
                machine_type = batch['type'].to(device)
                targets = batch['failure'].to(device)

                # Forward pass
                if use_temporal and 'sequence' in batch:
                    sequence = batch['sequence'].to(device)
                    features = batch['features'].to(device)
                    outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)
                else:
                    numeric_features = batch['numeric_features'].to(device)
                    outputs = model(numeric_features=numeric_features, machine_type=machine_type)

                # Get probabilities
                probs = torch.sigmoid(outputs['failure_logits']).cpu().numpy()
                model_probs.extend(probs.flatten())
                model_targets.extend(targets.cpu().numpy().flatten())

        all_model_probs.append(model_probs)
        if len(all_targets) == 0:
            all_targets = model_targets

    all_model_probs = np.array(all_model_probs)  # (num_models, num_samples)
    all_targets = np.array(all_targets)  # (num_samples,)

    if voting == 'hard':
        # Majority voting: each model votes 0 or 1
        model_votes = (all_model_probs > 0.5).astype(int)
        ensemble_preds = (model_votes.sum(axis=0) >= len(models) / 2).astype(int)
        ensemble_probs = all_model_probs.mean(axis=0)
    else:  # soft voting
        # Average probabilities then threshold
        ensemble_probs = all_model_probs.mean(axis=0)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

    return ensemble_preds, ensemble_probs, all_targets


def evaluate_ensemble(ensemble_dir: str = './checkpoints/ensemble_phase1',
                      config_class=BinaryOnlyConfig,
                      voting: str = 'hard',
                      output_file: str = None):
    """
    Evaluate ensemble performance on test set.

    Args:
        ensemble_dir: Directory containing ensemble models
        config_class: Configuration class used for training
        voting: 'hard' for majority voting, 'soft' for probability averaging
        output_file: Optional file to save results
    """
    print("=" * 80)
    print("PHASE 1 ENSEMBLE EVALUATION")
    print("=" * 80)

    ensemble_path = Path(ensemble_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Voting strategy: {voting}")

    # Load models
    models = load_ensemble_models(ensemble_path, config_class, device)

    if len(models) == 0:
        print("Error: No models loaded!")
        return

    # Load test data
    print(f"\nLoading test data...")
    _, _, test_dataset = load_temporal_datasets(
        str(config_class.TRAIN_PATH),
        str(config_class.DEV_PATH),
        str(config_class.TEST_PATH),
        window_size=config_class.WINDOW_SIZE,
        stride=config_class.STRIDE,
        augment_train=False,  # No augmentation for test
        target_ratio=config_class.TARGET_RATIO
    )

    print(f"Test set size: {len(test_dataset)}")
    print(f"Test failure ratio: {test_dataset.y_failure.mean():.1%}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config_class.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Make ensemble predictions
    print(f"\nMaking ensemble predictions...")
    ensemble_preds, ensemble_probs, targets = predict_ensemble(
        models, test_loader, device,
        use_temporal=config_class.USE_TEMPORAL_SEQUENCES,
        voting=voting
    )

    # Evaluate ensemble
    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS")
    print("=" * 80)

    ensemble_metrics = {
        'accuracy': accuracy_score(targets, ensemble_preds),
        'precision': precision_score(targets, ensemble_preds, zero_division=0),
        'recall': recall_score(targets, ensemble_preds, zero_division=0),
        'f1_score': f1_score(targets, ensemble_preds, zero_division=0)
    }

    print(f"\nEnsemble Performance ({voting} voting):")
    print(f"  Accuracy:  {ensemble_metrics['accuracy']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"  F1-Score:  {ensemble_metrics['f1_score']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(targets, ensemble_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d} | FP: {fp:4d}")
    print(f"  FN: {fn:4d} | TP: {tp:4d}")

    # Compare to individual models
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 80)

    individual_results = []

    for model_dict in models:
        model = model_dict['model']
        seed = model_dict['seed']

        model_probs = []

        with torch.no_grad():
            for batch in test_loader:
                machine_type = batch['type'].to(device)

                if config_class.USE_TEMPORAL_SEQUENCES:
                    sequence = batch['sequence'].to(device)
                    features = batch['features'].to(device)
                    outputs = model(sequence=sequence, temporal_features=features, machine_type=machine_type)
                else:
                    numeric_features = batch['numeric_features'].to(device)
                    outputs = model(numeric_features=numeric_features, machine_type=machine_type)

                probs = torch.sigmoid(outputs['failure_logits']).cpu().numpy()
                model_probs.extend(probs.flatten())

        model_preds = (np.array(model_probs) > 0.5).astype(int)

        model_metrics = {
            'seed': seed,
            'accuracy': float(accuracy_score(targets, model_preds)),
            'precision': float(precision_score(targets, model_preds, zero_division=0)),
            'recall': float(recall_score(targets, model_preds, zero_division=0)),
            'f1_score': float(f1_score(targets, model_preds, zero_division=0))
        }

        individual_results.append(model_metrics)

        print(f"\nSeed {seed}:")
        print(f"  F1: {model_metrics['f1_score']:.4f} | "
              f"Precision: {model_metrics['precision']:.4f} | "
              f"Recall: {model_metrics['recall']:.4f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_f1 = np.mean([r['f1_score'] for r in individual_results])
    std_f1 = np.std([r['f1_score'] for r in individual_results])
    min_f1 = min(r['f1_score'] for r in individual_results)
    max_f1 = max(r['f1_score'] for r in individual_results)

    print(f"\nIndividual Models:")
    print(f"  Mean F1: {avg_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Range:   [{min_f1:.4f}, {max_f1:.4f}]")

    print(f"\nEnsemble ({voting} voting):")
    print(f"  F1: {ensemble_metrics['f1_score']:.4f}")
    print(f"  Improvement over mean: {ensemble_metrics['f1_score'] - avg_f1:+.4f}")
    print(f"  Improvement over best: {ensemble_metrics['f1_score'] - max_f1:+.4f}")

    # Save results
    results = {
        'ensemble': {
            'voting': voting,
            'metrics': {k: float(v) for k, v in ensemble_metrics.items()},
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }
        },
        'individual_models': individual_results,
        'summary': {
            'mean_f1': float(avg_f1),
            'std_f1': float(std_f1),
            'min_f1': float(min_f1),
            'max_f1': float(max_f1),
            'ensemble_f1': float(ensemble_metrics['f1_score']),
            'improvement_over_mean': float(ensemble_metrics['f1_score'] - avg_f1),
            'improvement_over_best': float(ensemble_metrics['f1_score'] - max_f1)
        }
    }

    if output_file is None:
        output_file = ensemble_path / 'evaluation_results.json'
    else:
        output_file = Path(output_file)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    # Evaluate ensemble with hard voting
    results = evaluate_ensemble(
        ensemble_dir='./checkpoints/ensemble_phase1',
        config_class=BinaryOnlyConfig,
        voting='hard'
    )
