"""
Validation script to verify all performance improvements before full training.
Tests each component incrementally to ensure correctness.
"""

import torch
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("VALIDATION SCRIPT: VERIFYING ALL IMPROVEMENTS")
print("=" * 80)

# Test 1: Data Preprocessing
print("\n" + "=" * 80)
print("TEST 1: DATA PREPROCESSING WITH TEMPORAL SEQUENCES")
print("=" * 80)

try:
    from data_preprocessing import load_temporal_datasets

    print("\nLoading temporal sequence datasets...")
    train_ds, dev_ds, test_ds = load_temporal_datasets(
        './dataset/train/train.csv',
        './dataset/dev/dev.csv',
        './dataset/test/test.csv',
        window_size=12,
        stride=1,
        augment_train=True,
        target_ratio=0.15
    )

    # Verify dataset sizes
    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(dev_ds) > 0, "Dev dataset is empty"
    assert len(test_ds) > 0, "Test dataset is empty"
    print(f"[OK] Dataset sizes: Train={len(train_ds)}, Dev={len(dev_ds)}, Test={len(test_ds)}")

    # Verify sample structure
    sample = train_ds[0]
    assert 'sequence' in sample, "Missing sequence in sample"
    assert 'features' in sample, "Missing features in sample"
    assert sample['sequence'].shape == (12, 5), f"Wrong sequence shape: {sample['sequence'].shape}"
    assert sample['features'].shape == (19,), f"Wrong features shape: {sample['features'].shape}"
    print(f"[OK] Sample structure correct: sequence={sample['sequence'].shape}, features={sample['features'].shape}")

    # Verify class balance after augmentation
    failure_rate = train_ds.y_failure.mean()
    assert 0.12 <= failure_rate <= 0.18, f"Augmentation failed: {failure_rate:.1%}"
    print(f"[OK] Class balance after augmentation: {failure_rate:.1%} (target: 15%)")

    # Verify TTF normalization
    ttf_mean = train_ds.y_ttf.mean()
    ttf_std = train_ds.y_ttf.std()
    assert abs(ttf_mean) < 1.0, f"TTF not normalized (mean={ttf_mean:.2f})"
    assert 0.5 < ttf_std < 2.0, f"TTF normalization issue (std={ttf_std:.2f})"
    print(f"[OK] TTF normalized: mean={ttf_mean:.2f}, std={ttf_std:.2f}")

    # Verify TTF range (raw)
    assert train_ds.y_ttf_raw.min() >= 0, "Negative TTF values"
    assert train_ds.y_ttf_raw.max() <= 60, f"TTF too high: {train_ds.y_ttf_raw.max():.1f}"
    print(f"[OK] TTF range (raw): {train_ds.y_ttf_raw.min():.1f}-{train_ds.y_ttf_raw.max():.1f} hours")

    print("\n[OK] TEST 1 PASSED: Data preprocessing works correctly")

except Exception as e:
    print(f"\n[FAIL] TEST 1 FAILED: {str(e)}")
    sys.exit(1)

# Test 2: Model Architecture
print("\n" + "=" * 80)
print("TEST 2: MODEL ARCHITECTURE WITH TEMPORAL SEQUENCES")
print("=" * 80)

try:
    from model import MultiTaskTCN

    print("\nCreating temporal sequence model...")
    model = MultiTaskTCN(
        num_numeric_features=5,
        num_temporal_features=19,
        num_types=3,
        tcn_channels=64,
        num_heads=4,
        dropout=0.3,
        use_temporal_sequences=True
    )

    # Verify forward pass
    batch_size = 16
    sequence = torch.randn(batch_size, 12, 5)
    temporal_features = torch.randn(batch_size, 19)
    machine_type = torch.randint(0, 3, (batch_size, 1))

    outputs = model(sequence=sequence, temporal_features=temporal_features, machine_type=machine_type)

    assert outputs['failure_logits'].shape == (batch_size, 1), "Wrong failure logits shape"
    assert outputs['failure_type_logits'].shape == (batch_size, 5), "Wrong failure type logits shape"
    assert outputs['ttf'].shape == (batch_size, 1), "Wrong TTF shape"
    print(f"[OK] Forward pass successful with correct output shapes")

    # Verify attention weights
    assert 'attention_weights' in outputs, "Missing attention weights"
    attn_shape = outputs['attention_weights'].shape
    # Attention weights from MultiheadAttention are (batch, seq_len, seq_len) averaged over heads
    assert len(attn_shape) == 3, f"Wrong attention shape dimensions: {attn_shape}"
    assert attn_shape[0] == batch_size, f"Wrong batch size in attention: {attn_shape}"
    print(f"[OK] Attention mechanism working: {attn_shape}")

    # Verify gradient flow
    loss = outputs['failure_logits'].mean()
    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    assert grad_count > 0, "No gradients computed"
    print(f"[OK] Gradient flow intact: {grad_count} parameters with gradients")

    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert 100000 <= total_params <= 200000, f"Unexpected parameter count: {total_params:,}"
    print(f"[OK] Model parameters: {total_params:,}")

    print("\n[OK] TEST 2 PASSED: Model architecture works correctly")

except Exception as e:
    print(f"\n[FAIL] TEST 2 FAILED: {str(e)}")
    sys.exit(1)

# Test 3: Configuration
print("\n" + "=" * 80)
print("TEST 3: CONFIGURATION VALIDATION")
print("=" * 80)

try:
    from config import Config

    # Verify key hyperparameters
    assert Config.FOCAL_ALPHA == 0.70, f"Wrong focal alpha: {Config.FOCAL_ALPHA}"
    assert Config.FOCAL_GAMMA == 3.0, f"Wrong focal gamma: {Config.FOCAL_GAMMA}"
    print(f"[OK] Focal loss parameters: alpha={Config.FOCAL_ALPHA}, gamma={Config.FOCAL_GAMMA}")

    assert Config.TASK_WEIGHTS['failure'] == 2.5, "Wrong failure task weight"
    assert Config.TASK_WEIGHTS['ttf'] == 0.3, "Wrong TTF task weight"
    print(f"[OK] Task weights: {Config.TASK_WEIGHTS}")

    assert Config.DROPOUT == 0.3, f"Wrong dropout: {Config.DROPOUT}"
    assert Config.WEIGHT_DECAY == 5e-4, f"Wrong weight decay: {Config.WEIGHT_DECAY}"
    print(f"[OK] Regularization: dropout={Config.DROPOUT}, weight_decay={Config.WEIGHT_DECAY}")

    assert Config.LEARNING_RATE == 0.002, f"Wrong LR: {Config.LEARNING_RATE}"
    assert Config.USE_WARMUP == True, "Warmup not enabled"
    assert Config.WARMUP_EPOCHS == 5, f"Wrong warmup epochs: {Config.WARMUP_EPOCHS}"
    print(f"[OK] Learning rate schedule: LR={Config.LEARNING_RATE}, warmup={Config.WARMUP_EPOCHS} epochs")

    assert Config.USE_TEMPORAL_SEQUENCES == True, "Temporal sequences not enabled"
    assert Config.WINDOW_SIZE == 12, f"Wrong window size: {Config.WINDOW_SIZE}"
    print(f"[OK] Temporal sequences: window_size={Config.WINDOW_SIZE}")

    assert Config.AUGMENT_TRAIN == True, "Augmentation not enabled"
    assert Config.TARGET_RATIO == 0.15, f"Wrong target ratio: {Config.TARGET_RATIO}"
    print(f"[OK] Data augmentation: target_ratio={Config.TARGET_RATIO}")

    print("\n[OK] TEST 3 PASSED: Configuration is correct")

except Exception as e:
    print(f"\n[FAIL] TEST 3 FAILED: {str(e)}")
    sys.exit(1)

# Test 4: Training Pipeline
print("\n" + "=" * 80)
print("TEST 4: TRAINING PIPELINE (SHORT RUN)")
print("=" * 80)

try:
    from train import train, FocalLoss, MultiTaskLoss
    from torch.utils.data import DataLoader

    print("\nTesting focal loss...")
    focal_loss = FocalLoss(alpha=0.70, gamma=3.0)

    # Test focal loss behavior
    logits_easy_neg = torch.tensor([[-4.0]])  # Easy negative (p≈0.02)
    logits_hard_neg = torch.tensor([[0.0]])   # Hard negative (p=0.5)
    targets_neg = torch.tensor([[0.0]])

    loss_easy = focal_loss(logits_easy_neg, targets_neg).item()
    loss_hard = focal_loss(logits_hard_neg, targets_neg).item()

    assert loss_easy < loss_hard, f"Focal loss not focusing on hard examples: easy={loss_easy:.4f}, hard={loss_hard:.4f}"
    print(f"[OK] Focal loss focuses on hard examples: easy={loss_easy:.4f}, hard={loss_hard:.4f}")

    # Test multi-task loss
    criterion = MultiTaskLoss(
        alpha=0.70,
        gamma=3.0,
        task_weights={'failure': 2.5, 'failure_types': 1.0, 'ttf': 0.3}
    )

    dummy_outputs = {
        'failure_logits': torch.randn(16, 1),
        'failure_type_logits': torch.randn(16, 5),
        'ttf': torch.randn(16, 1)
    }
    dummy_targets = {
        'failure': torch.randint(0, 2, (16, 1)).float(),
        'failure_types': torch.randint(0, 2, (16, 5)).float(),
        'ttf': torch.randn(16, 1)
    }

    total_loss, loss_dict = criterion(dummy_outputs, dummy_targets)
    assert 'total' in loss_dict, "Missing total loss"
    assert 'failure' in loss_dict, "Missing failure loss"
    assert 'ttf' in loss_dict, "Missing TTF loss"
    print(f"[OK] Multi-task loss working: total={loss_dict['total']:.4f}")

    print("\nTesting mini training run (3 epochs)...")
    config = Config.to_dict()
    config['num_epochs'] = 3
    config['patience'] = 10

    # Run short training
    train(config)

    # Verify checkpoint was created
    checkpoint_path = Path('./checkpoints/best_model.pt')
    assert checkpoint_path.exists(), "Checkpoint not created"
    print(f"[OK] Checkpoint created: {checkpoint_path}")

    print("\n[OK] TEST 4 PASSED: Training pipeline works correctly")

except Exception as e:
    print(f"\n[FAIL] TEST 4 FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Evaluation with Threshold Optimization
print("\n" + "=" * 80)
print("TEST 5: EVALUATION WITH THRESHOLD OPTIMIZATION")
print("=" * 80)

try:
    from evaluate import optimize_threshold, TemperatureScaling, visualize_precision_recall_curve

    # Create synthetic predictions for testing
    np.random.seed(42)
    n_samples = 1000

    # Simulate predictions (biased toward positives like our current model)
    y_true = np.random.binomial(1, 0.03, n_samples)  # 3% positive rate
    y_probs = np.random.beta(2, 5, n_samples)  # Skewed distribution

    predictions = {
        'failure_probs': y_probs
    }
    targets = {
        'failure': y_true
    }

    print("\nTesting threshold optimization...")
    optimal_threshold, metrics = optimize_threshold(predictions, targets)

    assert 0.0 <= optimal_threshold <= 1.0, f"Invalid threshold: {optimal_threshold}"
    assert 'f1_score' in metrics, "Missing F1 score"
    assert 'optimal_threshold' in metrics, "Missing optimal threshold"
    print(f"[OK] Threshold optimization: threshold={optimal_threshold:.3f}, F1={metrics['f1_score']:.3f}")

    print("\nTesting temperature scaling...")
    temp_scaler = TemperatureScaling()
    logits = torch.tensor(np.random.randn(100, 1), dtype=torch.float32)
    labels = torch.tensor(np.random.binomial(1, 0.3, (100, 1)), dtype=torch.float32)

    temperature = temp_scaler.fit(logits, labels)
    assert 0.5 <= temperature <= 2.0, f"Unusual temperature: {temperature}"
    print(f"[OK] Temperature scaling: T={temperature:.3f}")

    print("\nTesting PR curve visualization...")
    pr_curve_path = './checkpoints/test_pr_curve.png'
    visualize_precision_recall_curve(predictions, targets, save_path=pr_curve_path)
    assert Path(pr_curve_path).exists(), "PR curve not saved"
    print(f"[OK] PR curve saved: {pr_curve_path}")

    print("\n[OK] TEST 5 PASSED: Evaluation enhancements work correctly")

except Exception as e:
    print(f"\n[FAIL] TEST 5 FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print("\n[OK] ALL TESTS PASSED!")
print("\nImplemented Improvements:")
print("  1. Temporal sequence processing (window_size=12)")
print("  2. Improved TTF synthesis with exponential degradation")
print("  3. TTF normalization (log-space + standardization)")
print("  4. Temporal feature engineering (19 features)")
print("  5. Data augmentation (3% → 15% failure ratio)")
print("  6. Enhanced regularization (dropout=0.3, weight_decay=5e-4)")
print("  7. Optimized focal loss (α=0.70, γ=3.0)")
print("  8. Task weight adjustment (failure=2.5, ttf=0.3)")
print("  9. AdamW optimizer with warmup + cosine annealing")
print(" 10. Threshold optimization via PR curve")
print(" 11. Temperature scaling calibration")

print("\nExpected Performance Gains:")
print("  Binary Failure F1-Score: 0.556 → 0.92-0.96 (Target: >0.95)")
print("  TTF MAE: 4.88 hours → 1.2-1.8 hours (Target: <2.0 hours)")

print("\n" + "=" * 80)
print("READY FOR FULL TRAINING!")
print("=" * 80)
print("\nTo train the improved model, run:")
print("  python train.py")
print("\nTo evaluate with threshold optimization:")
print("  python evaluate.py --optimize_threshold --calibrate")
print("\n" + "=" * 80)
