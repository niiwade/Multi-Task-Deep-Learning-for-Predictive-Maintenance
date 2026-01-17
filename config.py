"""
Configuration file for multi-task predictive maintenance model.
Centralizes all hyperparameters for easy experimentation.
"""

from pathlib import Path


class Config:
    """Configuration class for model training and evaluation."""

    # ===== Data Paths =====
    DATA_ROOT = Path('../dataset')
    TRAIN_PATH = DATA_ROOT / 'train' / 'train.csv'
    DEV_PATH = DATA_ROOT / 'dev' / 'dev.csv'
    TEST_PATH = DATA_ROOT / 'test' / 'test.csv'

    # ===== Output Paths =====
    CHECKPOINT_DIR = Path('./checkpoints')
    VISUALIZATION_DIR = Path('./visualizations')

    # ===== Model Architecture =====
    NUM_NUMERIC_FEATURES = 5  # Air temp, Process temp, Speed, Torque, Wear
    NUM_TYPES = 3  # L, M, H
    TCN_CHANNELS = 64  # Number of channels in TCN layers
    NUM_HEADS = 4  # Number of attention heads
    DROPOUT = 0.2  # Dropout rate

    # TCN dilations (defines receptive field)
    TCN_DILATIONS = [1, 2, 4, 8]
    TCN_KERNEL_SIZE = 3

    # ===== Training Hyperparameters =====
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5  # L2 regularization

    # Early stopping
    PATIENCE = 15  # Stop if no improvement for N epochs

    # Gradient clipping
    GRAD_CLIP_NORM = 1.0

    # ===== Focal Loss Parameters =====
    # Alpha: weighting factor for positive class (0.25 = more weight on negatives)
    FOCAL_ALPHA = 0.25
    # Gamma: focusing parameter (higher = more focus on hard examples)
    FOCAL_GAMMA = 2.0

    # ===== Multi-Task Loss Weights =====
    TASK_WEIGHTS = {
        'failure': 1.0,        # Binary failure prediction
        'failure_types': 1.0,  # Multi-label failure type classification
        'ttf': 0.5            # Time-to-failure regression (lower weight since synthesized)
    }

    # ===== Data Preprocessing =====
    SYNTHESIZE_TTF = True  # Whether to generate synthetic TTF labels
    USE_WEIGHTED_SAMPLING = True  # Use weighted sampler to handle imbalance

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
            'tcn_channels': cls.TCN_CHANNELS,
            'num_heads': cls.NUM_HEADS,
            'dropout': cls.DROPOUT,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'patience': cls.PATIENCE,
            'focal_alpha': cls.FOCAL_ALPHA,
            'focal_gamma': cls.FOCAL_GAMMA,
            'task_weights': cls.TASK_WEIGHTS
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

        print("\nTraining:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"  Patience: {cls.PATIENCE}")

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


if __name__ == '__main__':
    # Test configuration
    Config.print_config()

    print("\n\nLightweight Configuration:")
    LightConfig.print_config()
