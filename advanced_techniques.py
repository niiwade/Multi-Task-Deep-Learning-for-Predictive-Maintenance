"""
Advanced training techniques for Phase 4 of IMPROVEMENT_PLAN_V2.md.
Includes: Mixup/CutMix, Pseudo-Labeling, and Attention Regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class MixupAugmenter:
    """
    Mixup augmentation for temporal sequences.
    Phase 4 improvement from IMPROVEMENT_PLAN_V2.md.

    Creates virtual training examples by mixing pairs of examples and their labels.
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter (higher = more mixing)
            prob: Probability of applying mixup to a batch
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply mixup to a batch.

        Args:
            batch: Dictionary containing 'sequence', 'features', 'failure', etc.

        Returns:
            Mixed batch
        """
        if np.random.rand() > self.prob:
            return batch  # No mixup

        batch_size = batch['sequence'].size(0)

        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation for pairing
        index = torch.randperm(batch_size)

        # Mix sequences
        mixed_sequence = lam * batch['sequence'] + (1 - lam) * batch['sequence'][index]

        # Mix temporal features
        mixed_features = lam * batch['features'] + (1 - lam) * batch['features'][index]

        # Mix labels (soft labels)
        mixed_failure = lam * batch['failure'] + (1 - lam) * batch['failure'][index]

        # Machine type: use original (can't interpolate categorical)
        mixed_type = batch['type']

        mixed_batch = {
            'sequence': mixed_sequence,
            'features': mixed_features,
            'failure': mixed_failure,
            'type': mixed_type
        }

        # Copy other keys if present
        for key in ['failure_types', 'ttf']:
            if key in batch:
                mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]

        return mixed_batch


class CutMixAugmenter:
    """
    CutMix augmentation for temporal sequences.
    Phase 4 improvement from IMPROVEMENT_PLAN_V2.md.

    Replaces regions of one sequence with patches from another.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying cutmix
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply cutmix to a batch.

        Args:
            batch: Dictionary containing temporal sequences

        Returns:
            CutMix'd batch
        """
        if np.random.rand() > self.prob:
            return batch

        batch_size, window_size, num_features = batch['sequence'].shape

        # Sample mixing ratio
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size)

        # Cut a temporal window
        cut_size = int(window_size * (1 - lam))
        if cut_size == 0:
            return batch

        # Random position for cut
        cut_start = np.random.randint(0, window_size - cut_size + 1)
        cut_end = cut_start + cut_size

        # Create mixed sequence
        mixed_sequence = batch['sequence'].clone()
        mixed_sequence[:, cut_start:cut_end, :] = batch['sequence'][index, cut_start:cut_end, :]

        # Adjust label based on cut ratio
        # lam represents the proportion of original sample
        mixed_failure = lam * batch['failure'] + (1 - lam) * batch['failure'][index]

        mixed_batch = {
            'sequence': mixed_sequence,
            'features': batch['features'],  # Keep original temporal features
            'failure': mixed_failure,
            'type': batch['type']
        }

        # Copy other keys
        for key in ['failure_types', 'ttf']:
            if key in batch:
                mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]

        return mixed_batch


class PseudoLabeling:
    """
    Pseudo-labeling (self-training) strategy.
    Phase 4 improvement from IMPROVEMENT_PLAN_V2.md.

    Uses confident model predictions on unlabeled data as additional training samples.
    """

    def __init__(self, confidence_threshold: float = 0.95, update_frequency: int = 5):
        """
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels (0.95 = 95%)
            update_frequency: Update pseudo-labels every N epochs
        """
        self.confidence_threshold = confidence_threshold
        self.update_frequency = update_frequency
        self.pseudo_labels = None
        self.pseudo_indices = None

    def generate_pseudo_labels(self, model: nn.Module, unlabeled_loader,
                               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels from model predictions.

        Args:
            model: Trained model
            unlabeled_loader: DataLoader for unlabeled data
            device: Device to use

        Returns:
            Tuple of (pseudo_labels, confident_indices)
        """
        model.eval()

        all_probs = []
        all_indices = []

        with torch.no_grad():
            for idx, batch in enumerate(unlabeled_loader):
                # Forward pass
                machine_type = batch['type'].to(device)

                if 'sequence' in batch:
                    sequence = batch['sequence'].to(device)
                    features = batch['features'].to(device)
                    outputs = model(sequence=sequence, temporal_features=features,
                                  machine_type=machine_type)
                else:
                    numeric_features = batch['numeric_features'].to(device)
                    outputs = model(numeric_features=numeric_features, machine_type=machine_type)

                # Get probabilities
                probs = torch.sigmoid(outputs['failure_logits']).cpu()
                all_probs.append(probs)

                # Track original indices
                batch_indices = torch.arange(idx * unlabeled_loader.batch_size,
                                            min((idx + 1) * unlabeled_loader.batch_size,
                                                len(unlabeled_loader.dataset)))
                all_indices.append(batch_indices)

        all_probs = torch.cat(all_probs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        # Find confident predictions
        # Confident negative: prob < (1 - threshold)
        # Confident positive: prob > threshold
        confident_mask = (all_probs > self.confidence_threshold) | \
                        (all_probs < (1 - self.confidence_threshold))

        confident_indices = all_indices[confident_mask.squeeze()]
        pseudo_labels = (all_probs[confident_mask] > 0.5).float()

        print(f"Generated {len(pseudo_labels)} pseudo-labels "
              f"({confident_mask.sum().item() / len(all_probs) * 100:.1f}% of unlabeled data)")
        print(f"  Pseudo-positive rate: {pseudo_labels.mean().item():.1%}")

        return pseudo_labels, confident_indices


class AttentionRegularization(nn.Module):
    """
    Attention regularization loss.
    Phase 4 improvement from IMPROVEMENT_PLAN_V2.md.

    Guides attention to focus on important temporal regions (e.g., recent timesteps).
    """

    def __init__(self, regularization_strength: float = 0.01,
                 focus_recent: bool = True):
        """
        Args:
            regularization_strength: Weight for regularization loss
            focus_recent: If True, encourage attention on recent timesteps
        """
        super().__init__()
        self.strength = regularization_strength
        self.focus_recent = focus_recent

    def forward(self, attention_weights: torch.Tensor, window_size: int = 12) -> torch.Tensor:
        """
        Compute attention regularization loss.

        Args:
            attention_weights: Attention weights (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
            window_size: Temporal window size

        Returns:
            Regularization loss
        """
        # Average over heads if present
        if attention_weights.dim() == 4:
            # (batch, heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
            attention_weights = attention_weights.mean(dim=1)

        # Average over query dimension to get importance per timestep
        # (batch, seq_len, seq_len) -> (batch, seq_len)
        timestep_importance = attention_weights.mean(dim=1)

        if self.focus_recent:
            # Create target distribution: higher weight on recent timesteps
            # Recent timesteps are more predictive of imminent failure
            target_weights = torch.linspace(0.5, 1.5, window_size, device=attention_weights.device)
            target_weights = target_weights / target_weights.sum()  # Normalize
            target_weights = target_weights.unsqueeze(0)  # (1, seq_len)

            # KL divergence between attention and target
            # Minimize KL(attention || target) to guide attention toward recent timesteps
            log_attention = torch.log(timestep_importance + 1e-8)
            log_target = torch.log(target_weights + 1e-8)

            kl_loss = F.kl_div(log_attention, target_weights, reduction='batchmean')
            reg_loss = self.strength * kl_loss

        else:
            # Encourage uniform attention (exploratory)
            uniform_target = torch.ones(window_size, device=attention_weights.device) / window_size
            uniform_target = uniform_target.unsqueeze(0)

            log_attention = torch.log(timestep_importance + 1e-8)
            kl_loss = F.kl_div(log_attention, uniform_target, reduction='batchmean')
            reg_loss = self.strength * kl_loss

        return reg_loss


class ConfidenceCalibration:
    """
    Temperature scaling for probability calibration.
    Improves reliability of model confidence scores.
    """

    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """
        Fit temperature parameter on validation set.

        Args:
            logits: Model logits (before sigmoid)
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Optimal temperature
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated logits
        """
        return logits / self.temperature


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING ADVANCED TECHNIQUES")
    print("=" * 80)

    # Test Mixup
    print("\n[TEST 1] Mixup Augmentation")
    mixup = MixupAugmenter(alpha=0.2, prob=1.0)

    batch = {
        'sequence': torch.randn(8, 12, 5),
        'features': torch.randn(8, 19),
        'failure': torch.randint(0, 2, (8, 1)).float(),
        'type': torch.randint(0, 3, (8, 1))
    }

    mixed_batch = mixup(batch)
    print(f"[OK] Mixup: {batch['sequence'].shape} -> {mixed_batch['sequence'].shape}")
    print(f"  Original failure mean: {batch['failure'].mean():.3f}")
    print(f"  Mixed failure mean: {mixed_batch['failure'].mean():.3f}")

    # Test CutMix
    print("\n[TEST 2] CutMix Augmentation")
    cutmix = CutMixAugmenter(alpha=1.0, prob=1.0)
    cutmix_batch = cutmix(batch)
    print(f"[OK] CutMix: {batch['sequence'].shape} -> {cutmix_batch['sequence'].shape}")

    # Test Attention Regularization
    print("\n[TEST 3] Attention Regularization")
    attn_reg = AttentionRegularization(regularization_strength=0.01, focus_recent=True)
    attention_weights = torch.softmax(torch.randn(8, 12, 12), dim=-1)
    reg_loss = attn_reg(attention_weights, window_size=12)
    print(f"[OK] Attention regularization loss: {reg_loss.item():.6f}")

    # Test Temperature Scaling
    print("\n[TEST 4] Temperature Scaling")
    calib = ConfidenceCalibration()
    logits = torch.randn(100, 1)
    labels = torch.randint(0, 2, (100, 1)).float()
    temp = calib.fit(logits, labels)
    print(f"[OK] Optimal temperature: {temp:.3f}")

    print("\n" + "=" * 80)
    print("ADVANCED TECHNIQUES: READY")
    print("=" * 80)
