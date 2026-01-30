"""
Quick tuning script for Phase 2 hyperparameters.
Tests different augmentation ratios and Tversky alpha/beta combinations.
"""

import sys
from pathlib import Path
from config import Phase2Config
from train_phase2 import train_phase2

# Configurations to test
configs = [
    # (target_ratio, alpha, beta, description)
    (0.35, 0.3, 0.7, "Baseline"),
    (0.40, 0.3, 0.7, "Higher augmentation"),
    (0.45, 0.3, 0.7, "Aggressive augmentation"),
    (0.40, 0.2, 0.8, "Higher recall focus"),
    (0.40, 0.1, 0.9, "Extreme recall focus"),
]

for i, (ratio, alpha, beta, desc) in enumerate(configs, 1):
    print("\n" + "=" * 80)
    print(f"CONFIGURATION {i}/{len(configs)}: {desc}")
    print(f"  TARGET_RATIO: {ratio}")
    print(f"  TVERSKY_ALPHA: {alpha}, TVERSKY_BETA: {beta}")
    print("=" * 80)

    # Modify config
    Phase2Config.TARGET_RATIO = ratio
    Phase2Config.TVERSKY_ALPHA = alpha
    Phase2Config.TVERSKY_BETA = beta

    # Train
    save_dir = Path(f'./checkpoints/phase2_tune/config_{i}_{desc.replace(" ", "_")}')
    try:
        train_phase2(seed=42, save_dir=save_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nConfiguration {i} failed with error: {e}")
        continue

print("\n" + "=" * 80)
print("TUNING COMPLETE")
print("=" * 80)
print("\nCompare results in: ./checkpoints/phase2_tune/")
