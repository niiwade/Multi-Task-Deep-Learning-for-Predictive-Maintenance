"""
Complete pipeline runner for multi-task predictive maintenance.
Runs training, evaluation, and visualization in sequence.
"""

import argparse
import sys
from pathlib import Path
import torch

from config import Config, LightConfig, HeavyConfig, ImbalanceConfig
from train import train
from evaluate import evaluate_model, print_results
from visualize_attention import AttentionAnalyzer
from data_preprocessing import load_datasets
from model import MultiTaskTCN
from torch.utils.data import DataLoader


def run_full_pipeline(config_class=Config, skip_training=False, skip_eval=False, skip_viz=False):
    """
    Run the complete pipeline: training → evaluation → visualization.

    Args:
        config_class: Configuration class to use (Config, LightConfig, etc.)
        skip_training: Skip training if checkpoint exists
        skip_eval: Skip evaluation
        skip_viz: Skip visualization
    """
    print("\n" + "="*70)
    print("MULTI-TASK PREDICTIVE MAINTENANCE PIPELINE")
    print("="*70 + "\n")

    # Print configuration
    config_class.print_config()

    config = config_class.to_dict()
    checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pt'

    # ===== PHASE 1: TRAINING =====
    if not skip_training or not checkpoint_path.exists():
        print("\n" + "="*70)
        print("PHASE 1: TRAINING")
        print("="*70 + "\n")

        train(config)

        print("\n✓ Training completed!")
    else:
        print(f"\n⊳ Skipping training (checkpoint exists: {checkpoint_path})")

    # ===== PHASE 2: EVALUATION =====
    if not skip_eval:
        print("\n" + "="*70)
        print("PHASE 2: EVALUATION")
        print("="*70 + "\n")

        results = evaluate_model(
            checkpoint_path=str(checkpoint_path),
            test_path=config['test_path'],
            batch_size=config_class.EVAL_BATCH_SIZE
        )

        print_results(results)

        print("\n✓ Evaluation completed!")
    else:
        print("\n⊳ Skipping evaluation")

    # ===== PHASE 3: VISUALIZATION =====
    if not skip_viz:
        print("\n" + "="*70)
        print("PHASE 3: ATTENTION VISUALIZATION")
        print("="*70 + "\n")

        device = torch.device('cuda' if torch.cuda.is_available() and config_class.USE_CUDA else 'cpu')

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = MultiTaskTCN(
            num_numeric_features=config_class.NUM_NUMERIC_FEATURES,
            num_types=config_class.NUM_TYPES,
            tcn_channels=config['tcn_channels'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load test data
        train_dataset, _, _ = load_datasets(
            config['train_path'],
            config['dev_path'],
            config['test_path']
        )

        from data_preprocessing import MultiTaskDataset
        test_dataset = MultiTaskDataset(config['test_path'], scaler=train_dataset.scaler)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Create visualizations
        output_dir = config_class.VISUALIZATION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = AttentionAnalyzer(model, device)

        print("Generating feature importance visualization...")
        analyzer.visualize_feature_importance(
            test_loader,
            save_path=output_dir / 'feature_importance.png',
            num_samples=config_class.NUM_VISUALIZATION_SAMPLES
        )

        print("Generating attention analysis by failure type...")
        analyzer.visualize_attention_by_failure_type(
            test_loader,
            save_path=output_dir / 'attention_by_failure_type.png',
            num_samples=config_class.NUM_VISUALIZATION_SAMPLES
        )

        print("Analyzing sample predictions...")
        analyzer.analyze_sample_predictions(
            test_loader,
            num_samples=5,
            save_dir=output_dir / 'samples'
        )

        print(f"\n✓ Visualizations saved to {output_dir}")
    else:
        print("\n⊳ Skipping visualization")

    # ===== COMPLETION =====
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Model checkpoint: {checkpoint_path}")
    print(f"  Evaluation results: {Path(config['checkpoint_dir']) / 'evaluation_results.json'}")
    print(f"  Visualizations: {config_class.VISUALIZATION_DIR}")
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-task predictive maintenance pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python run_pipeline.py

  # Run with lightweight config for quick testing
  python run_pipeline.py --config light

  # Skip training if model exists
  python run_pipeline.py --skip-training

  # Only train, skip evaluation and visualization
  python run_pipeline.py --skip-eval --skip-viz

  # Run with heavy config for maximum performance
  python run_pipeline.py --config heavy

  # Run with imbalance-optimized config
  python run_pipeline.py --config imbalance
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='default',
        choices=['default', 'light', 'heavy', 'imbalance'],
        help='Configuration to use (default: default)'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training if checkpoint exists'
    )

    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation phase'
    )

    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization phase'
    )

    args = parser.parse_args()

    # Select configuration
    config_map = {
        'default': Config,
        'light': LightConfig,
        'heavy': HeavyConfig,
        'imbalance': ImbalanceConfig
    }

    config_class = config_map[args.config]

    # Run pipeline
    try:
        run_full_pipeline(
            config_class=config_class,
            skip_training=args.skip_training,
            skip_eval=args.skip_eval,
            skip_viz=args.skip_viz
        )
    except KeyboardInterrupt:
        print("\n\n⊗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n⊗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
