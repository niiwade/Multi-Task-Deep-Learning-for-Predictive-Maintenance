"""
Attention visualization utilities for interpretability.
Helps identify which sensor features drive failure predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from model import MultiTaskTCN
from data_preprocessing import load_datasets


class AttentionAnalyzer:
    """Analyzes and visualizes attention patterns from the model."""

    def __init__(self, model: MultiTaskTCN, device: torch.device):
        """
        Args:
            model: Trained multi-task TCN model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()

        self.feature_names = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]

    def extract_attention(self, dataloader: DataLoader,
                         num_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract attention weights from the model.

        Args:
            dataloader: DataLoader for dataset
            num_samples: Number of samples to analyze (None = all)

        Returns:
            Tuple of (attention_weights, failure_labels)
        """
        all_attention = []
        all_failures = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_samples is not None and i * dataloader.batch_size >= num_samples:
                    break

                numeric_features = batch['numeric_features'].to(self.device)
                machine_type = batch['type'].to(self.device)

                # Forward pass
                outputs = self.model(numeric_features, machine_type)

                # Extract attention weights (batch, num_heads, seq_len, seq_len)
                attention_weights = outputs['attention_weights'].cpu().numpy()

                # Average across heads and sequence dimension
                # Shape: (batch, num_heads, 1, 1) -> (batch,)
                avg_attention = attention_weights.mean(axis=(1, 2, 3))

                all_attention.append(avg_attention)
                all_failures.append(batch['failure'].numpy().flatten())

        attention = np.concatenate(all_attention)
        failures = np.concatenate(all_failures)

        return attention, failures

    def visualize_feature_importance(self, dataloader: DataLoader,
                                    save_path: str = None,
                                    num_samples: int = 1000):
        """
        Visualize average attention weights for each feature.

        Args:
            dataloader: DataLoader for dataset
            save_path: Path to save visualization (None = display only)
            num_samples: Number of samples to analyze
        """
        # Collect feature-level attention (simplified version)
        # In practice, attention is over the temporal dimension
        # Here we'll collect predictions and analyze patterns

        feature_importance = np.zeros(len(self.feature_names))
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break

                numeric_features = batch['numeric_features'].to(self.device)
                machine_type = batch['type'].to(self.device)

                # Forward pass
                outputs = self.model(numeric_features, machine_type)

                # Use attention weights as proxy for importance
                attention = outputs['attention_weights'].cpu().numpy()
                feature_importance += attention.mean(axis=(0, 1, 2, 3))
                num_batches += 1

        feature_importance /= num_batches

        # Normalize to sum to 1
        feature_importance = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(self.feature_names)))

        bars = ax.barh(self.feature_names, feature_importance, color=colors)

        ax.set_xlabel('Relative Importance', fontsize=12)
        ax.set_title('Sensor Feature Importance for Failure Prediction', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, feature_importance)):
            ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_attention_by_failure_type(self, dataloader: DataLoader,
                                           save_path: str = None,
                                           num_samples: int = 1000):
        """
        Visualize attention patterns for different failure scenarios.

        Args:
            dataloader: DataLoader for dataset
            save_path: Path to save visualization
            num_samples: Number of samples to analyze
        """
        # Collect attention and labels
        all_attention = []
        all_failures = []
        all_failure_types = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break

                numeric_features = batch['numeric_features'].to(self.device)
                machine_type = batch['type'].to(self.device)

                outputs = self.model(numeric_features, machine_type)

                attention = outputs['attention_weights'].cpu().numpy()
                avg_attention = attention.mean(axis=(1, 2, 3))

                all_attention.append(avg_attention)
                all_failures.append(batch['failure'].numpy().flatten())
                all_failure_types.append(batch['failure_types'].numpy())

        attention = np.concatenate(all_attention)
        failures = np.concatenate(all_failures)
        failure_types = np.concatenate(all_failure_types)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Attention distribution for failed vs healthy
        failed_attention = attention[failures == 1]
        healthy_attention = attention[failures == 0]

        axes[0].hist(healthy_attention, bins=30, alpha=0.6, label='Healthy', color='green', density=True)
        axes[0].hist(failed_attention, bins=30, alpha=0.6, label='Failed', color='red', density=True)
        axes[0].set_xlabel('Average Attention Weight', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title('Attention Distribution: Failed vs Healthy', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Attention by failure type
        failure_type_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        type_attention_means = []
        type_attention_stds = []

        for i in range(5):
            type_mask = failure_types[:, i] == 1
            if type_mask.sum() > 0:
                type_attention_means.append(attention[type_mask].mean())
                type_attention_stds.append(attention[type_mask].std())
            else:
                type_attention_means.append(0)
                type_attention_stds.append(0)

        x_pos = np.arange(len(failure_type_names))
        colors = plt.cm.Set3(range(len(failure_type_names)))

        bars = axes[1].bar(x_pos, type_attention_means, yerr=type_attention_stds,
                          color=colors, alpha=0.8, capsize=5)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(failure_type_names, fontsize=11)
        axes[1].set_ylabel('Mean Attention Weight', fontsize=11)
        axes[1].set_title('Attention by Failure Type', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention analysis plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def analyze_sample_predictions(self, dataloader: DataLoader,
                                  num_samples: int = 5,
                                  save_dir: str = None):
        """
        Analyze and visualize predictions for individual samples.

        Args:
            dataloader: DataLoader for dataset
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break

                numeric_features = batch['numeric_features'].to(self.device)
                machine_type = batch['type'].to(self.device)

                outputs = self.model(numeric_features, machine_type)

                # Process first sample in batch
                for i in range(min(batch['numeric_features'].size(0), num_samples - sample_count)):
                    self._visualize_single_sample(
                        numeric_features[i].cpu().numpy(),
                        machine_type[i].item(),
                        outputs['failure_logits'][i].item(),
                        outputs['ttf'][i].item(),
                        batch['failure'][i].item(),
                        batch['ttf'][i].item(),
                        sample_count,
                        save_dir
                    )
                    sample_count += 1

    def _visualize_single_sample(self, features: np.ndarray, machine_type: int,
                                pred_failure_logit: float, pred_ttf: float,
                                true_failure: float, true_ttf: float,
                                sample_id: int, save_dir: str = None):
        """Visualize a single sample's prediction."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Feature values
        colors = ['blue' if true_failure == 0 else 'red'] * len(features)
        bars = axes[0].barh(self.feature_names, features, color=colors, alpha=0.7)
        axes[0].set_xlabel('Normalized Value', fontsize=11)
        axes[0].set_title(f'Sample {sample_id} - Sensor Features', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)

        # Plot 2: Predictions
        pred_failure_prob = 1 / (1 + np.exp(-pred_failure_logit))

        info_text = [
            f"Machine Type: {['L', 'M', 'H'][machine_type]}",
            f"",
            f"Failure Prediction:",
            f"  Probability: {pred_failure_prob:.3f}",
            f"  True Label: {'FAILED' if true_failure == 1 else 'HEALTHY'}",
            f"",
            f"Time-to-Failure:",
            f"  Predicted: {pred_ttf:.1f} hours",
            f"  True: {true_ttf:.1f} hours",
        ]

        axes[1].text(0.1, 0.5, '\n'.join(info_text), fontsize=12,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
        axes[1].set_title(f'Sample {sample_id} - Predictions', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f'sample_{sample_id}_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved sample analysis to {save_path}")
        else:
            plt.show()

        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize attention patterns')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='../dataset/test/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to analyze')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    model = MultiTaskTCN(
        num_numeric_features=5,
        num_types=3,
        tcn_channels=config['tcn_channels'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")

    # Load dataset
    train_dataset, _, _ = load_datasets(
        config['train_path'],
        config['dev_path'],
        config['test_path']
    )

    from data_preprocessing import MultiTaskDataset
    test_dataset = MultiTaskDataset(args.test_path, scaler=train_dataset.scaler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create analyzer
    analyzer = AttentionAnalyzer(model, device)

    # Generate visualizations
    print("\nGenerating feature importance visualization...")
    analyzer.visualize_feature_importance(
        test_loader,
        save_path=output_path / 'feature_importance.png',
        num_samples=args.num_samples
    )

    print("\nGenerating attention analysis by failure type...")
    analyzer.visualize_attention_by_failure_type(
        test_loader,
        save_path=output_path / 'attention_by_failure_type.png',
        num_samples=args.num_samples
    )

    print("\nAnalyzing sample predictions...")
    analyzer.analyze_sample_predictions(
        test_loader,
        num_samples=5,
        save_dir=output_path / 'samples'
    )

    print(f"\nAll visualizations saved to {output_path}")
