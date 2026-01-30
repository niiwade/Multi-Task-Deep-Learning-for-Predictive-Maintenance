"""
Configuration file for multi-task predictive maintenance model.
Centralizes all hyperparameters for easy experimentation.
"""

from pathlib import Path


class Config:
    """Configuration class for model training and evaluation."""

    # ===== Data Paths =====
    DATA_ROOT = Path('./dataset')
    TRAIN_PATH = DATA_ROOT / 'train' / 'train.csv'
    DEV_PATH = DATA_ROOT / 'dev' / 'dev.csv'
    TEST_PATH = DATA_ROOT / 'test' / 'test.csv'

    # ===== Output Paths =====
    CHECKPOINT_DIR = Path('./checkpoints')
    VISUALIZATION_DIR = Path('./visualizations')

    # ===== Model Architecture =====
    NUM_NUMERIC_FEATURES = 5  # Air temp, Process temp, Speed, Torque, Wear
    NUM_TEMPORAL_FEATURES = 19  # Temporal features (rates, acceleration, trends, etc.)
    NUM_TYPES = 3  # L, M, H
    TCN_CHANNELS = 64  # Number of channels in TCN layers
    NUM_HEADS = 4  # Number of attention heads
    DROPOUT = 0.3  # Dropout rate (increased from 0.2 for better regularization)

    # TCN dilations (defines receptive field)
    TCN_DILATIONS = [1, 2, 4, 8]
    TCN_KERNEL_SIZE = 3

    # Temporal sequence processing
    USE_TEMPORAL_SEQUENCES = True  # Use temporal sequences instead of single timesteps
    WINDOW_SIZE = 12  # Number of timesteps per sequence
    STRIDE = 1  # Sliding window stride (1 = overlapping)

    # ===== Training Hyperparameters =====
    BATCH_SIZE = 32
    NUM_EPOCHS = 80  # Reduced from 100 (converges faster with better LR schedule)
    LEARNING_RATE = 0.002  # Increased from 0.001 for warmup schedule
    WEIGHT_DECAY = 5e-4  # Increased from 1e-5 for stronger regularization

    # Learning rate schedule
    USE_WARMUP = True  # Use warmup + cosine annealing
    WARMUP_EPOCHS = 5  # Linear warmup for first 5 epochs
    COSINE_T0 = 15  # Cosine annealing restart period
    COSINE_T_MULT = 2  # Period multiplier after each restart
    COSINE_ETA_MIN = 1e-6  # Minimum learning rate

    # Early stopping
    PATIENCE = 15  # Stop if no improvement for N epochs

    # Gradient clipping
    GRAD_CLIP_NORM = 0.5  # Tighter clipping (from 1.0)

    # ===== Focal Loss Parameters =====
    # Alpha: weighting factor for positive class
    # 0.70 = 70% weight on positive (failure) class to compensate for 97:3 imbalance
    FOCAL_ALPHA = 0.70  # Increased from 0.25 to reduce false positives
    # Gamma: focusing parameter (higher = more focus on hard examples)
    FOCAL_GAMMA = 3.0  # Increased from 2.0 for stronger hard example focus

    # ===== Multi-Task Loss Weights =====
    TASK_WEIGHTS = {
        'failure': 2.5,        # Binary failure prediction (PRIORITY - increased from 1.0)
        'failure_types': 1.0,  # Multi-label failure type classification
        'ttf': 0.3            # Time-to-failure regression (reduced from 0.5, synthetic labels)
    }

    # ===== Data Preprocessing =====
    SYNTHESIZE_TTF = True  # Whether to generate synthetic TTF labels
    USE_IMPROVED_TTF = True  # Use improved exponential TTF synthesis (vs linear)
    USE_WEIGHTED_SAMPLING = True  # Use weighted sampler to handle imbalance

    # Data augmentation
    AUGMENT_TRAIN = True  # Apply data augmentation to training set
    TARGET_RATIO = 0.15  # Target failure ratio after augmentation (15% vs 3% baseline)

    # ===== Evaluation =====
    EVAL_BATCH_SIZE = 32
    NUM_VISUALIZATION_SAMPLES = 1000  # Number of samples for attention analysis

    # ===== Device =====
    USE_CUDA = True  # Use GPU if available

    # ===== Random Seeds =====
    SEED = 42

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary (for saving with checkpoints)."""
        return {
            'train_path': str(cls.TRAIN_PATH),
            'dev_path': str(cls.DEV_PATH),
            'test_path': str(cls.TEST_PATH),
            'checkpoint_dir': str(cls.CHECKPOINT_DIR),
            'num_numeric_features': cls.NUM_NUMERIC_FEATURES,
            'num_temporal_features': cls.NUM_TEMPORAL_FEATURES,
            'tcn_channels': cls.TCN_CHANNELS,
            'num_heads': cls.NUM_HEADS,
            'dropout': cls.DROPOUT,
            'use_temporal_sequences': cls.USE_TEMPORAL_SEQUENCES,
            'window_size': cls.WINDOW_SIZE,
            'stride': cls.STRIDE,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'use_warmup': cls.USE_WARMUP,
            'warmup_epochs': cls.WARMUP_EPOCHS,
            'patience': cls.PATIENCE,
            'focal_alpha': cls.FOCAL_ALPHA,
            'focal_gamma': cls.FOCAL_GAMMA,
            'task_weights': cls.TASK_WEIGHTS,
            'augment_train': cls.AUGMENT_TRAIN,
            'target_ratio': cls.TARGET_RATIO,
            'use_improved_ttf': cls.USE_IMPROVED_TTF
        }

    @classmethod
    def print_config(cls):
        """Pretty print configuration."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)

        print("\nData Paths:")
        print(f"  Train: {cls.TRAIN_PATH}")
        print(f"  Dev:   {cls.DEV_PATH}")
        print(f"  Test:  {cls.TEST_PATH}")

        print("\nModel Architecture:")
        print(f"  TCN Channels: {cls.TCN_CHANNELS}")
        print(f"  TCN Dilations: {cls.TCN_DILATIONS}")
        print(f"  Attention Heads: {cls.NUM_HEADS}")
        print(f"  Dropout: {cls.DROPOUT}")
        print(f"  Use Temporal Sequences: {cls.USE_TEMPORAL_SEQUENCES}")
        if cls.USE_TEMPORAL_SEQUENCES:
            print(f"  Window Size: {cls.WINDOW_SIZE}")
            print(f"  Stride: {cls.STRIDE}")

        print("\nTraining:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")
        if cls.USE_WARMUP:
            print(f"  Warmup Epochs: {cls.WARMUP_EPOCHS}")
        print(f"  Patience: {cls.PATIENCE}")
        print(f"  Grad Clip Norm: {cls.GRAD_CLIP_NORM}")

        print("\nFocal Loss:")
        print(f"  Alpha: {cls.FOCAL_ALPHA}")
        print(f"  Gamma: {cls.FOCAL_GAMMA}")

        print("\nTask Weights:")
        for task, weight in cls.TASK_WEIGHTS.items():
            print(f"  {task}: {weight}")

        print("\nOutput:")
        print(f"  Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"  Visualizations: {cls.VISUALIZATION_DIR}")

        print("=" * 60)


# ===== Alternative Configurations =====

class LightConfig(Config):
    """Lightweight configuration for quick experiments."""
    TCN_CHANNELS = 32
    NUM_HEADS = 2
    BATCH_SIZE = 64
    NUM_EPOCHS = 50


class HeavyConfig(Config):
    """Heavy configuration for maximum performance."""
    TCN_CHANNELS = 128
    NUM_HEADS = 8
    DROPOUT = 0.3
    BATCH_SIZE = 16
    TCN_DILATIONS = [1, 2, 4, 8, 16]  # Larger receptive field


class ImbalanceConfig(Config):
    """Configuration optimized for severe class imbalance."""
    FOCAL_ALPHA = 0.15  # More weight on positive class
    FOCAL_GAMMA = 3.0  # Stronger focus on hard examples
    TASK_WEIGHTS = {
        'failure': 2.0,  # Double weight on failure prediction
        'failure_types': 1.0,
        'ttf': 0.3
    }


class BinaryOnlyConfig(Config):
    """
    Single-task configuration for binary failure prediction only.
    Phase 1 of IMPROVEMENT_PLAN_V2.md - Removes multi-task interference.
    """
    # Remove TTF and failure type tasks
    TASK_WEIGHTS = {
        'failure': 1.0,  # Single task - no need for weighting
    }

    # More aggressive augmentation (15% -> 30%)
    TARGET_RATIO = 0.30  # 30% failure ratio

    # Asymmetric focal loss for safety-critical (FN worse than FP)
    FOCAL_ALPHA = 0.80  # Even more weight on failures
    FOCAL_GAMMA = 3.0
    USE_ASYMMETRIC_LOSS = True  # Enable asymmetric FN penalty
    ASYMMETRIC_BETA = 2.0  # FN costs 2x more than FP

    # Longer training for single task
    NUM_EPOCHS = 150  # From 80
    PATIENCE = 25  # From 15

    # Ensemble parameters
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]  # 5 models

    # Don't synthesize TTF (task removed)
    SYNTHESIZE_TTF = False
    USE_IMPROVED_TTF = False


class Phase2Config(BinaryOnlyConfig):
    """
    Phase 2 configuration: Advanced Class Balancing.
    Adds SMOTE and class-balanced batching on top of Phase 1.
    Expected F1: 0.75-0.85
    """
    # SMOTE parameters
    USE_SMOTE = True
    SMOTE_K_NEIGHBORS = 5

    # Class-balanced batching disabled (too restrictive with SMOTE)
    USE_CLASS_BALANCED_BATCHES = False

    # Conservative augmentation to find sweet spot
    TARGET_RATIO = 0.30  # 30% failures (original was 0.35, too aggressive)

    # Revert to original aggressive Tversky (was working in earlier runs)
    USE_TVERSKY_LOSS = True
    TVERSKY_ALPHA = 0.2  # FP penalty (original)
    TVERSKY_BETA = 0.8   # FN penalty (original)


class Phase3Config(Phase2Config):
    """
    Phase 3 configuration: Architecture Improvements.
    Two-stage classifier + high capacity model.
    Expected F1: 0.85-0.90
    """
    # Use two-stage classifier
    USE_TWO_STAGE = True
    STAGE1_THRESHOLD = 0.3  # Conservative stage 1

    # High-capacity model
    TCN_CHANNELS = 128  # From 64
    NUM_HEADS = 8  # From 4

    # Deeper TCN
    TCN_DILATIONS = [1, 2, 4, 8, 16]  # Added 16

    # Can also use high-capacity single-stage
    USE_HIGH_CAPACITY = False  # Set True to use HighCapacityTCN instead of TwoStage


class Phase4Config(Phase3Config):
    """
    Phase 4 configuration: Advanced Techniques.
    Mixup/CutMix + Pseudo-labeling + Attention regularization.
    Expected F1: 0.90-0.95 (target)
    """
    # Mixup augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.5

    # CutMix augmentation
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.3

    # Pseudo-labeling
    USE_PSEUDO_LABELING = True
    PSEUDO_CONFIDENCE_THRESHOLD = 0.95
    PSEUDO_UPDATE_FREQUENCY = 5  # Update every 5 epochs

    # Attention regularization
    USE_ATTENTION_REG = True
    ATTENTION_REG_STRENGTH = 0.01
    ATTENTION_FOCUS_RECENT = True

    # Temperature scaling for calibration
    USE_TEMPERATURE_SCALING = True

    # Extended training for advanced techniques
    NUM_EPOCHS = 200  # From 150
    PATIENCE = 30  # From 25


if __name__ == '__main__':
    # Test configuration
    Config.print_config()

    print("\n\nLightweight Configuration:")
    LightConfig.print_config()
