"""
Validation script for Phases 2-4 improvements.
Tests all advanced components before full training.
"""

import torch
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("PHASES 2-4 VALIDATION")
print("=" * 80)

# Test 1: SMOTE
print("\n[TEST 1] SMOTE Augmenter")
try:
    from data_preprocessing import SMOTEAugmenter

    smote = SMOTEAugmenter(k_neighbors=5, seed=42)

    # Create dummy data
    X_seq = np.random.randn(50, 12, 5)  # 50 samples
    y_fail = np.zeros(50)
    y_fail[:5] = 1  # 5 failures (10%)
    y_types = np.zeros((50, 5))
    y_ttf = np.random.rand(50)
    X_type = np.random.randint(0, 3, 50)

    X_aug, y_aug, _, _, _ = smote.fit_resample(
        X_seq, y_fail, y_types, y_ttf, X_type, target_ratio=0.30
    )

    assert len(X_aug) > len(X_seq), "SMOTE didn't generate samples"
    assert y_aug.mean() > y_fail.mean(), "Failure ratio didn't increase"

    print("[OK] SMOTE working")
    print(f"  Original: {len(X_seq)} samples, {y_fail.mean():.1%} failures")
    print(f"  Augmented: {len(X_aug)} samples, {y_aug.mean():.1%} failures")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Class-Balanced Batch Sampler
print("\n[TEST 2] Class-Balanced Batch Sampler")
try:
    from data_preprocessing import ClassBalancedBatchSampler

    labels = np.array([0]*80 + [1]*20)  # 20% positive
    sampler = ClassBalancedBatchSampler(labels, batch_size=16, seed=42)

    # Get first batch
    first_batch = next(iter(sampler))
    batch_labels = labels[first_batch]

    print("[OK] ClassBalancedBatchSampler working")
    print(f"  Batch size: {len(first_batch)}")
    print(f"  Batch balance: {batch_labels.mean():.1%} positive")
    print(f"  Expected: ~50% (class-balanced)")

    # Should be close to 50-50
    assert 0.4 <= batch_labels.mean() <= 0.6, "Batch not balanced"

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Two-Stage Classifier
print("\n[TEST 3] Two-Stage Classifier")
try:
    from model_advanced import TwoStageClassifier

    model = TwoStageClassifier(
        num_numeric_features=5,
        num_temporal_features=19,
        num_types=3,
        tcn_channels=64,
        num_heads=4,
        dropout=0.3,
        use_temporal_sequences=True
    )

    batch_size = 8
    sequence = torch.randn(batch_size, 12, 5)
    temporal_features = torch.randn(batch_size, 19)
    machine_type = torch.randint(0, 3, (batch_size, 1))

    outputs = model(sequence=sequence, temporal_features=temporal_features, machine_type=machine_type)

    assert 'stage1_logits' in outputs, "Missing stage1 logits"
    assert 'stage2_logits' in outputs, "Missing stage2 logits"
    assert 'failure_logits' in outputs, "Missing final logits"
    assert outputs['stage1_logits'].shape == (batch_size, 1), "Wrong stage1 shape"
    assert outputs['stage2_logits'].shape == (batch_size, 1), "Wrong stage2 shape"

    print("[OK] Two-Stage Classifier working")
    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  Stage 1 shape: {outputs['stage1_logits'].shape}")
    print(f"  Stage 2 shape: {outputs['stage2_logits'].shape}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: High-Capacity TCN
print("\n[TEST 4] High-Capacity TCN")
try:
    from model_advanced import HighCapacityTCN

    model_hc = HighCapacityTCN(
        num_numeric_features=5,
        num_temporal_features=19,
        num_types=3,
        tcn_channels=128,  # Increased
        num_heads=8,  # Increased
        dropout=0.3,
        use_temporal_sequences=True,
        binary_only=True
    )

    outputs_hc = model_hc(sequence=sequence, temporal_features=temporal_features, machine_type=machine_type)

    assert outputs_hc['failure_logits'].shape == (batch_size, 1), "Wrong output shape"

    total_params = sum(p.numel() for p in model_hc.parameters())

    print("[OK] High-Capacity TCN working")
    print(f"  Parameters: {total_params:,}")
    print(f"  Increase over baseline: ~{(total_params / 129177):.1f}x")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Alternative Losses
print("\n[TEST 5] Dice and Tversky Losses")
try:
    from train import DiceLoss, TverskyLoss

    dice_loss = DiceLoss(smooth=1.0)
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1.0)

    logits = torch.randn(16, 1)
    targets = torch.randint(0, 2, (16, 1)).float()

    dice_val = dice_loss(logits, targets)
    tversky_val = tversky_loss(logits, targets)

    assert 0 <= dice_val <= 2, "Dice loss out of range"
    assert 0 <= tversky_val <= 2, "Tversky loss out of range"

    print("[OK] Alternative losses working")
    print(f"  Dice loss: {dice_val.item():.4f}")
    print(f"  Tversky loss: {tversky_val.item():.4f}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Mixup Augmentation
print("\n[TEST 6] Mixup Augmentation")
try:
    from advanced_techniques import MixupAugmenter

    mixup = MixupAugmenter(alpha=0.2, prob=1.0)  # Always apply for testing

    batch = {
        'sequence': torch.randn(8, 12, 5),
        'features': torch.randn(8, 19),
        'failure': torch.randint(0, 2, (8, 1)).float(),
        'type': torch.randint(0, 3, (8, 1))
    }

    mixed_batch = mixup(batch)

    assert mixed_batch['sequence'].shape == batch['sequence'].shape, "Wrong shape"
    assert mixed_batch['features'].shape == batch['features'].shape, "Wrong shape"

    print("[OK] Mixup working")
    print(f"  Sequence shape: {mixed_batch['sequence'].shape}")
    print(f"  Mixed labels (soft): {mixed_batch['failure'].view(-1)[:4].tolist()}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: CutMix Augmentation
print("\n[TEST 7] CutMix Augmentation")
try:
    from advanced_techniques import CutMixAugmenter

    cutmix = CutMixAugmenter(alpha=1.0, prob=1.0)
    cutmix_batch = cutmix(batch)

    assert cutmix_batch['sequence'].shape == batch['sequence'].shape, "Wrong shape"

    print("[OK] CutMix working")
    print(f"  Sequence shape: {cutmix_batch['sequence'].shape}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Attention Regularization
print("\n[TEST 8] Attention Regularization")
try:
    from advanced_techniques import AttentionRegularization

    attn_reg = AttentionRegularization(regularization_strength=0.01, focus_recent=True)

    # Dummy attention weights
    attention_weights = torch.softmax(torch.randn(8, 12, 12), dim=-1)
    reg_loss = attn_reg(attention_weights, window_size=12)

    assert reg_loss.item() >= 0, "Regularization loss should be non-negative"

    print("[OK] Attention regularization working")
    print(f"  Regularization loss: {reg_loss.item():.6f}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Phase configurations
print("\n[TEST 9] Phase 2-4 Configurations")
try:
    from config import Phase2Config, Phase3Config, Phase4Config

    # Phase 2
    assert Phase2Config.USE_SMOTE == True, "SMOTE not enabled"
    assert Phase2Config.USE_CLASS_BALANCED_BATCHES == True, "Class balancing not enabled"
    assert Phase2Config.TARGET_RATIO == 0.35, "Wrong target ratio"

    print("[OK] Phase2Config:")
    print(f"  SMOTE: {Phase2Config.USE_SMOTE}")
    print(f"  Class-balanced batches: {Phase2Config.USE_CLASS_BALANCED_BATCHES}")
    print(f"  Target ratio: {Phase2Config.TARGET_RATIO:.0%}")

    # Phase 3
    assert Phase3Config.USE_TWO_STAGE == True, "Two-stage not enabled"
    assert Phase3Config.TCN_CHANNELS == 128, "Wrong TCN channels"
    assert Phase3Config.NUM_HEADS == 8, "Wrong num heads"

    print("\n[OK] Phase3Config:")
    print(f"  Two-stage: {Phase3Config.USE_TWO_STAGE}")
    print(f"  TCN channels: {Phase3Config.TCN_CHANNELS}")
    print(f"  Num heads: {Phase3Config.NUM_HEADS}")

    # Phase 4
    assert Phase4Config.USE_MIXUP == True, "Mixup not enabled"
    assert Phase4Config.USE_CUTMIX == True, "CutMix not enabled"
    assert Phase4Config.USE_PSEUDO_LABELING == True, "Pseudo-labeling not enabled"
    assert Phase4Config.USE_ATTENTION_REG == True, "Attention reg not enabled"

    print("\n[OK] Phase4Config:")
    print(f"  Mixup: {Phase4Config.USE_MIXUP}")
    print(f"  CutMix: {Phase4Config.USE_CUTMIX}")
    print(f"  Pseudo-labeling: {Phase4Config.USE_PSEUDO_LABELING}")
    print(f"  Attention reg: {Phase4Config.USE_ATTENTION_REG}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print("\n[OK] ALL TESTS PASSED!")

print("\nPhase 2 Features Validated:")
print("  1. SMOTE augmentation")
print("  2. Class-balanced batch sampler")
print("  3. Dice and Tversky losses")

print("\nPhase 3 Features Validated:")
print("  4. Two-stage classifier (detection + refinement)")
print("  5. High-capacity TCN (128 channels, 8 heads)")

print("\nPhase 4 Features Validated:")
print("  6. Mixup augmentation")
print("  7. CutMix augmentation")
print("  8. Attention regularization")
print("  9. Configurations for all phases")

print("\n" + "=" * 80)
print("READY FOR PHASES 2-4 TRAINING!")
print("=" * 80)

print("\nNext steps:")
print("  Phase 1: python train_ensemble.py (baseline)")
print("  Phase 2: python train_phase2.py (with SMOTE + class balancing)")
print("  Phase 3: python train_phase3.py (with two-stage + high capacity)")
print("  Phase 4: python train_phase4.py (with all advanced techniques)")

print("\n" + "=" * 80)
