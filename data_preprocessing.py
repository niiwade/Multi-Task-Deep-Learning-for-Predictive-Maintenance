"""
Data preprocessing for multi-task predictive maintenance.
Handles feature normalization, time-to-failure (TTF) synthesis, temporal sequence creation,
and data augmentation for improved model performance.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, Optional, List
import warnings


class TTFSynthesizer:
    """Synthesizes realistic time-to-failure labels based on sensor patterns."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def synthesize_ttf(self, df: pd.DataFrame) -> np.ndarray:
        """
        Synthesize time-to-failure (in hours) based on sensor degradation patterns.

        Args:
            df: DataFrame with sensor features and failure labels

        Returns:
            Array of time-to-failure values (0 for failed machines, >0 for healthy)
        """
        ttf = np.zeros(len(df))

        # For failed machines (Machine failure == 1), TTF = 0
        failed_mask = df['Machine failure'] == 1
        ttf[failed_mask] = 0

        # For healthy machines, estimate TTF based on wear indicators
        healthy_mask = ~failed_mask

        # Normalize sensor values to [0, 1] for degradation scoring
        tool_wear_norm = df.loc[healthy_mask, 'Tool wear [min]'] / df['Tool wear [min]'].max()
        torque_norm = df.loc[healthy_mask, 'Torque [Nm]'] / df['Torque [Nm]'].max()
        speed_norm = 1 - (df.loc[healthy_mask, 'Rotational speed [rpm]'] / df['Rotational speed [rpm]'].max())
        temp_diff = (df.loc[healthy_mask, 'Process temperature [K]'] -
                     df.loc[healthy_mask, 'Air temperature [K]'])
        temp_diff_norm = temp_diff / temp_diff.max()

        # Degradation score (0 = new, 1 = critical)
        degradation = (0.4 * tool_wear_norm +
                      0.3 * torque_norm +
                      0.2 * speed_norm +
                      0.1 * temp_diff_norm)

        # Convert degradation to TTF (higher degradation = lower TTF)
        # Range: 1-100 hours for healthy machines
        base_ttf = 100 * (1 - degradation)

        # Add failure-type specific adjustments
        if 'TWF' in df.columns:
            # Tool wear failures happen gradually
            twf_risk = df.loc[healthy_mask, 'TWF']
            base_ttf *= (1 - 0.3 * twf_risk)

        if 'HDF' in df.columns:
            # Heat dissipation failures are more sudden
            hdf_risk = df.loc[healthy_mask, 'HDF']
            base_ttf *= (1 - 0.5 * hdf_risk)

        # Add realistic noise (±10%)
        noise = np.random.uniform(0.9, 1.1, size=len(base_ttf))
        ttf[healthy_mask] = np.maximum(1.0, base_ttf * noise)

        return ttf


class ImprovedTTFSynthesizer:
    """
    Physics-informed TTF synthesis using exponential degradation model.
    Addresses the limitations of linear TTF synthesis for better realism.
    """

    def __init__(self, base_life: float = 48.0, degradation_rate: float = 3.5, seed: int = 42):
        """
        Args:
            base_life: Base operational life in hours (max TTF for healthy machines)
            degradation_rate: Exponential decay rate (higher = faster failure)
            seed: Random seed for reproducibility
        """
        self.base_life = base_life
        self.degradation_rate = degradation_rate
        np.random.seed(seed)

    def synthesize_ttf(self, df: pd.DataFrame) -> np.ndarray:
        """
        Synthesize TTF using exponential degradation model.

        Model: TTF = base_life * exp(-degradation_rate * health_score)

        Args:
            df: DataFrame with sensor features and failure labels

        Returns:
            Array of time-to-failure values (0 for failed, 0.5-48 hours for healthy)
        """
        ttf = np.zeros(len(df))

        # Failed machines: TTF = 0
        failed_mask = df['Machine failure'] == 1
        ttf[failed_mask] = 0

        # Healthy machines: exponential degradation model
        healthy_mask = ~failed_mask
        n_healthy = healthy_mask.sum()

        if n_healthy == 0:
            return ttf

        # 1. Calculate health indicators (0=new, 1=critical)
        # Tool wear (normalize by realistic max, not dataset max)
        tool_wear_health = df.loc[healthy_mask, 'Tool wear [min]'].values / 200.0  # Max realistic wear
        tool_wear_health = np.clip(tool_wear_health, 0, 1)

        # Torque stress (higher torque = faster degradation)
        torque_values = df.loc[healthy_mask, 'Torque [Nm]'].values
        torque_stress = (torque_values - 40.0) / 20.0  # Centered at nominal 40 Nm
        torque_stress = np.clip(torque_stress, 0, 1)

        # Temperature stress (process temp - air temp indicates heat buildup)
        temp_diff = (df.loc[healthy_mask, 'Process temperature [K]'].values -
                    df.loc[healthy_mask, 'Air temperature [K]'].values)
        temp_stress = (temp_diff - 8.0) / 5.0  # Nominal diff = 8K
        temp_stress = np.clip(temp_stress, 0, 1)

        # Speed instability (deviation from optimal 1500 rpm)
        speed_values = df.loc[healthy_mask, 'Rotational speed [rpm]'].values
        speed_dev = np.abs(speed_values - 1500) / 300
        speed_stress = np.clip(speed_dev, 0, 1)

        # 2. Composite health score (weighted by failure mode influence)
        health_score = (0.45 * tool_wear_health +   # TWF dominant
                       0.25 * torque_stress +        # PWF, OSF
                       0.20 * temp_stress +          # HDF
                       0.10 * speed_stress)          # General wear

        # 3. Failure type specific adjustments
        if 'TWF' in df.columns:
            twf_risk = df.loc[healthy_mask, 'TWF'].values
            # TWF cases have longer lead time but definite failure
            health_score = np.where(twf_risk == 1,
                                   health_score * 1.3,  # 30% more degraded
                                   health_score)

        if 'HDF' in df.columns:
            hdf_risk = df.loc[healthy_mask, 'HDF'].values
            # HDF cases have shorter warning time (sudden failures)
            health_score = np.where(hdf_risk == 1,
                                   health_score * 1.5,  # 50% more critical
                                   health_score)

        # 4. Exponential TTF model (realistic failure acceleration)
        ttf_healthy = self.base_life * np.exp(-self.degradation_rate * health_score)

        # 5. Add realistic stochastic variation (±15%)
        noise = np.random.normal(1.0, 0.15, size=len(ttf_healthy))
        noise = np.clip(noise, 0.7, 1.3)  # Limit extreme outliers
        ttf_healthy = ttf_healthy * noise

        # Ensure minimum TTF for healthy machines (30 minutes)
        ttf_healthy = np.maximum(0.5, ttf_healthy)

        ttf[healthy_mask] = ttf_healthy

        return ttf


class TemporalAugmenter:
    """
    Data augmentation for temporal sequences to address class imbalance.
    Applies physics-informed perturbations to minority class samples.
    """

    def __init__(self, noise_level: float = 0.02, jitter_strength: float = 0.05, seed: int = 42):
        """
        Args:
            noise_level: Gaussian noise std (2% of sensor value range)
            jitter_strength: Temporal jittering range (±5% time warping)
            seed: Random seed
        """
        self.noise_level = noise_level
        self.jitter_strength = jitter_strength
        np.random.seed(seed)

    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal augmentation to a sequence.

        Args:
            sequence: (window_size, num_features) sensor readings

        Returns:
            Augmented sequence with same shape
        """
        augmented = sequence.copy()
        window_size, num_features = sequence.shape

        # 1. Gaussian noise (sensor measurement noise)
        noise = np.random.normal(0, self.noise_level, size=sequence.shape)
        augmented += noise

        # 2. Temporal jittering (slight time warping)
        jitter = np.random.uniform(1 - self.jitter_strength,
                                   1 + self.jitter_strength)

        # Resample to simulate time warping
        original_indices = np.arange(window_size)
        jittered_length = max(3, int(window_size * jitter))  # At least 3 points
        jittered_indices = np.linspace(0, window_size - 1, jittered_length)

        # Interpolate each feature
        augmented_jittered = np.zeros_like(sequence)
        for feat_idx in range(num_features):
            augmented_jittered[:, feat_idx] = np.interp(
                original_indices,
                jittered_indices,
                augmented[:min(len(jittered_indices), window_size), feat_idx]
            )

        # 3. Feature-specific perturbations (physics-informed)
        # Feature order: [Air temp, Process temp, Speed, Torque, Tool wear]

        # Tool wear: can only increase, add small positive drift
        augmented_jittered[:, 4] += np.random.uniform(0, 0.5)

        # Temperature: add cyclic variation (realistic thermal cycles)
        thermal_cycle = 0.3 * np.sin(np.linspace(0, 2*np.pi, window_size))
        augmented_jittered[:, 0] += thermal_cycle      # Air temp
        augmented_jittered[:, 1] += thermal_cycle * 1.2  # Process temp (stronger)

        return augmented_jittered

    def oversample_minority_class(self, X_sequences, y_failure, y_failure_types, y_ttf, X_type,
                                  target_ratio: float = 0.15):
        """
        Oversample failure class to reduce imbalance.

        Args:
            X_sequences: Sequence data (n_samples, window_size, num_features)
            y_failure: Binary failure labels
            y_failure_types: Multi-label failure types
            y_ttf: TTF values
            X_type: Machine type
            target_ratio: Desired failure ratio (0.15 = 15% failures)

        Returns:
            Augmented arrays with synthetic failure samples added
        """
        failure_indices = np.where(y_failure == 1)[0]
        healthy_indices = np.where(y_failure == 0)[0]

        current_ratio = len(failure_indices) / len(y_failure)

        if current_ratio >= target_ratio:
            print(f"Already balanced: {current_ratio:.1%} >= {target_ratio:.1%}")
            return X_sequences, y_failure, y_failure_types, y_ttf, X_type

        # Calculate synthetic samples needed
        n_synthetic = int(len(healthy_indices) * target_ratio / (1 - target_ratio)
                         - len(failure_indices))

        print(f"Generating {n_synthetic} synthetic failure samples...")
        print(f"  Original failure ratio: {current_ratio:.1%}")
        print(f"  Target failure ratio: {target_ratio:.1%}")

        # Generate synthetic samples
        synthetic_sequences = []
        synthetic_failure = []
        synthetic_failure_types = []
        synthetic_ttf = []
        synthetic_type = []

        for _ in range(n_synthetic):
            # Randomly select a failure sample
            idx = np.random.choice(failure_indices)

            # Augment it
            original_seq = X_sequences[idx]
            augmented_seq = self.augment_sequence(original_seq)

            synthetic_sequences.append(augmented_seq)
            synthetic_failure.append(y_failure[idx])
            synthetic_failure_types.append(y_failure_types[idx])
            synthetic_ttf.append(y_ttf[idx])
            synthetic_type.append(X_type[idx])

        # Concatenate original and synthetic
        X_sequences_aug = np.concatenate([X_sequences, np.array(synthetic_sequences)], axis=0)
        y_failure_aug = np.concatenate([y_failure, synthetic_failure])
        y_failure_types_aug = np.concatenate([y_failure_types, synthetic_failure_types], axis=0)
        y_ttf_aug = np.concatenate([y_ttf, synthetic_ttf])
        X_type_aug = np.concatenate([X_type, synthetic_type])

        new_ratio = y_failure_aug.sum() / len(y_failure_aug)
        print(f"  New failure ratio: {new_ratio:.1%}")

        return X_sequences_aug, y_failure_aug, y_failure_types_aug, y_ttf_aug, X_type_aug


class SMOTEAugmenter:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) for temporal sequences.
    Generates synthetic samples via interpolation in feature space.
    Phase 2 improvement from IMPROVEMENT_PLAN_V2.md.
    """

    def __init__(self, k_neighbors: int = 5, seed: int = 42):
        """
        Args:
            k_neighbors: Number of nearest neighbors for SMOTE
            seed: Random seed
        """
        self.k_neighbors = k_neighbors
        np.random.seed(seed)

    def fit_resample(self, X_sequences, y_failure, y_failure_types, y_ttf, X_type,
                     target_ratio: float = 0.30):
        """
        Apply SMOTE to balance classes via interpolation.

        Args:
            X_sequences: Sequence data (n_samples, window_size, num_features)
            y_failure: Binary failure labels
            y_failure_types: Multi-label failure types
            y_ttf: TTF values
            X_type: Machine type
            target_ratio: Desired failure ratio

        Returns:
            Augmented arrays with SMOTE-generated samples
        """
        failure_indices = np.where(y_failure == 1)[0]
        healthy_indices = np.where(y_failure == 0)[0]

        current_ratio = len(failure_indices) / len(y_failure)

        if current_ratio >= target_ratio:
            print(f"Already balanced: {current_ratio:.1%} >= {target_ratio:.1%}")
            return X_sequences, y_failure, y_failure_types, y_ttf, X_type

        # Calculate synthetic samples needed
        n_synthetic = int(len(healthy_indices) * target_ratio / (1 - target_ratio)
                         - len(failure_indices))

        print(f"Applying SMOTE: generating {n_synthetic} synthetic failure samples...")
        print(f"  Original failure ratio: {current_ratio:.1%}")
        print(f"  Target failure ratio: {target_ratio:.1%}")

        # Flatten sequences for nearest neighbor computation
        # Shape: (n_failures, window_size * num_features)
        X_failures_flat = X_sequences[failure_indices].reshape(len(failure_indices), -1)

        # Fit nearest neighbors on failure samples
        k = min(self.k_neighbors, len(failure_indices) - 1)
        if k < 1:
            # Not enough failures for SMOTE, fall back to simple duplication
            print(f"  Warning: Too few failures ({len(failure_indices)}) for SMOTE, using duplication")
            synthetic_indices = np.random.choice(failure_indices, size=n_synthetic, replace=True)
            X_sequences_aug = np.concatenate([X_sequences, X_sequences[synthetic_indices]], axis=0)
            y_failure_aug = np.concatenate([y_failure, y_failure[synthetic_indices]])
            y_failure_types_aug = np.concatenate([y_failure_types, y_failure_types[synthetic_indices]], axis=0)
            y_ttf_aug = np.concatenate([y_ttf, y_ttf[synthetic_indices]])
            X_type_aug = np.concatenate([X_type, X_type[synthetic_indices]])
            return X_sequences_aug, y_failure_aug, y_failure_types_aug, y_ttf_aug, X_type_aug

        nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because it includes the point itself
        nn.fit(X_failures_flat)

        # Generate synthetic samples
        synthetic_sequences = []
        synthetic_failure = []
        synthetic_failure_types = []
        synthetic_ttf = []
        synthetic_type = []

        for _ in range(n_synthetic):
            # Randomly select a failure sample
            idx_local = np.random.randint(0, len(failure_indices))
            idx_global = failure_indices[idx_local]

            # Find k nearest neighbors
            distances, indices_local = nn.kneighbors(X_failures_flat[idx_local:idx_local+1])

            # Select a random neighbor (excluding self at index 0)
            neighbor_local = np.random.choice(indices_local[0, 1:])  # Skip self
            neighbor_global = failure_indices[neighbor_local]

            # Interpolate between sample and neighbor
            # lambda in [0, 1]: lambda=0 gives original, lambda=1 gives neighbor
            lambda_interp = np.random.uniform(0, 1)

            # Interpolate sequences
            synthetic_seq = (lambda_interp * X_sequences[idx_global] +
                           (1 - lambda_interp) * X_sequences[neighbor_global])

            # Interpolate TTF (if not zero)
            if y_ttf[idx_global] > 0 and y_ttf[neighbor_global] > 0:
                synthetic_ttf_val = (lambda_interp * y_ttf[idx_global] +
                                   (1 - lambda_interp) * y_ttf[neighbor_global])
            else:
                synthetic_ttf_val = 0.0

            synthetic_sequences.append(synthetic_seq)
            synthetic_failure.append(1)  # Always failure
            # Use majority failure types from the two samples
            synthetic_failure_types.append(
                (y_failure_types[idx_global] + y_failure_types[neighbor_global] >= 1).astype(float)
            )
            synthetic_ttf.append(synthetic_ttf_val)
            # Use machine type from original sample
            synthetic_type.append(X_type[idx_global])

        # Concatenate original and synthetic
        X_sequences_aug = np.concatenate([X_sequences, np.array(synthetic_sequences)], axis=0)
        y_failure_aug = np.concatenate([y_failure, synthetic_failure])
        y_failure_types_aug = np.concatenate([y_failure_types, synthetic_failure_types], axis=0)
        y_ttf_aug = np.concatenate([y_ttf, synthetic_ttf])
        X_type_aug = np.concatenate([X_type, synthetic_type])

        new_ratio = y_failure_aug.sum() / len(y_failure_aug)
        print(f"  New failure ratio (SMOTE): {new_ratio:.1%}")

        return X_sequences_aug, y_failure_aug, y_failure_types_aug, y_ttf_aug, X_type_aug


class ClassBalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures 50-50 class balance in each batch.
    Phase 2 improvement from IMPROVEMENT_PLAN_V2.md.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, seed: int = 42):
        """
        Args:
            labels: Binary labels (0/1)
            batch_size: Batch size (should be even for perfect 50-50 split)
            seed: Random seed
        """
        self.labels = labels
        self.batch_size = batch_size
        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]
        self.n_positive = len(self.positive_indices)
        self.n_negative = len(self.negative_indices)
        self.n_batches = min(self.n_positive, self.n_negative) * 2 // batch_size

        np.random.seed(seed)

    def __iter__(self):
        # Shuffle indices
        positive_shuffled = np.random.permutation(self.positive_indices)
        negative_shuffled = np.random.permutation(self.negative_indices)

        # Repeat minority class if needed
        if self.n_positive < self.n_negative:
            # Repeat positive samples
            repeats = (self.n_negative // self.n_positive) + 1
            positive_shuffled = np.tile(positive_shuffled, repeats)[:self.n_negative]
        else:
            # Repeat negative samples
            repeats = (self.n_positive // self.n_negative) + 1
            negative_shuffled = np.tile(negative_shuffled, repeats)[:self.n_positive]

        # Ensure equal lengths
        min_length = min(len(positive_shuffled), len(negative_shuffled))
        positive_shuffled = positive_shuffled[:min_length]
        negative_shuffled = negative_shuffled[:min_length]

        # Create balanced batches
        n_per_class = self.batch_size // 2
        n_batches = min_length // n_per_class

        for i in range(n_batches):
            start_idx = i * n_per_class
            end_idx = (i + 1) * n_per_class

            batch_pos = positive_shuffled[start_idx:end_idx]
            batch_neg = negative_shuffled[start_idx:end_idx]

            # Combine and shuffle within batch
            batch = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch)

            yield batch.tolist()

    def __len__(self):
        return self.n_batches


class TemporalSequenceDataset(Dataset):
    """
    PyTorch Dataset that creates temporal sequences using sliding windows.
    Implements improved TTF synthesis, normalization, temporal features, and augmentation.
    """

    def __init__(self, csv_path: str,
                 scaler: Optional[StandardScaler] = None,
                 ttf_scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = False,
                 window_size: int = 12,
                 stride: int = 1,
                 augment: bool = False,
                 target_ratio: float = 0.15,
                 use_improved_ttf: bool = True):
        """
        Args:
            csv_path: Path to CSV file
            scaler: Pre-fitted StandardScaler for features (use for dev/test)
            ttf_scaler: Pre-fitted StandardScaler for TTF (use for dev/test)
            fit_scaler: Whether to fit scalers on this data (True for train only)
            window_size: Number of timesteps in each sequence (12 = ~2 min at 10s intervals)
            stride: Step size for sliding window (1 = overlapping sequences)
            augment: Whether to apply data augmentation (True for train only)
            target_ratio: Target failure ratio for augmentation (0.15 = 15%)
            use_improved_ttf: Use improved exponential TTF synthesis (vs linear)
        """
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        # Load data
        self.df = pd.read_csv(csv_path)

        # Feature columns (5 numeric features)
        self.numeric_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]

        # Type encoding: L=0, M=1, H=2
        if self.df['Type'].dtype == 'object':
            self.df['Type_encoded'] = self.df['Type'].map({'L': 0, 'M': 1, 'H': 2})
        else:
            self.df['Type_encoded'] = self.df['Type']

        # Extract labels
        self.failure_types_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

        # Synthesize TTF
        if use_improved_ttf:
            ttf_synthesizer = ImprovedTTFSynthesizer()
        else:
            ttf_synthesizer = TTFSynthesizer()

        self.df['TTF'] = ttf_synthesizer.synthesize_ttf(self.df)

        # Create pseudo-temporal sequences by sorting
        # Group by machine type and tool wear to simulate degradation progression
        self.df = self.df.sort_values(['Type_encoded', 'Tool wear [min]', 'Torque [Nm]']).reset_index(drop=True)

        # Create sliding windows
        self.X_sequences = []
        self.X_type = []
        self.y_failure = []
        self.y_failure_types = []
        self.y_ttf_raw = []

        for i in range(0, len(self.df) - window_size + 1, stride):
            window = self.df.iloc[i:i+window_size]

            # Use last timestep's labels
            last_row = window.iloc[-1]

            sequence = window[self.numeric_features].values  # (window_size, 5)
            self.X_sequences.append(sequence)
            self.X_type.append(last_row['Type_encoded'])
            self.y_failure.append(last_row['Machine failure'])
            self.y_failure_types.append(window[self.failure_types_cols].iloc[-1].values)
            self.y_ttf_raw.append(last_row['TTF'])

        self.X_sequences = np.array(self.X_sequences)  # (n_sequences, window_size, 5)
        self.X_type = np.array(self.X_type)
        self.y_failure = np.array(self.y_failure)
        self.y_failure_types = np.array(self.y_failure_types)
        self.y_ttf_raw = np.array(self.y_ttf_raw)

        # Normalize numeric features (per timestep)
        # Reshape to (n_sequences * window_size, 5) for scaling
        original_shape = self.X_sequences.shape
        X_flat = self.X_sequences.reshape(-1, len(self.numeric_features))

        if fit_scaler:
            self.scaler = StandardScaler()
            X_flat = self.scaler.fit_transform(X_flat)
        elif scaler is not None:
            self.scaler = scaler
            X_flat = self.scaler.transform(X_flat)
        else:
            raise ValueError("Must provide scaler or set fit_scaler=True")

        self.X_sequences = X_flat.reshape(original_shape)

        # Normalize TTF (log-space normalization)
        self.y_ttf_log = np.log1p(self.y_ttf_raw)  # log(1 + TTF)

        if fit_scaler:
            self.ttf_scaler = StandardScaler()
            self.y_ttf = self.ttf_scaler.fit_transform(
                self.y_ttf_log.reshape(-1, 1)
            ).flatten()
        elif ttf_scaler is not None:
            self.ttf_scaler = ttf_scaler
            self.y_ttf = self.ttf_scaler.transform(
                self.y_ttf_log.reshape(-1, 1)
            ).flatten()
        else:
            raise ValueError("Must provide ttf_scaler or set fit_scaler=True")

        # Apply data augmentation (training only)
        if augment:
            augmenter = TemporalAugmenter()
            (self.X_sequences, self.y_failure, self.y_failure_types,
             self.y_ttf, self.X_type) = augmenter.oversample_minority_class(
                self.X_sequences, self.y_failure, self.y_failure_types,
                self.y_ttf, self.X_type, target_ratio=target_ratio
            )

    def compute_temporal_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute temporal features from sequence window.

        Args:
            sequence: (window_size, num_features) sensor readings (normalized)

        Returns:
            Temporal features (19 features):
            - 5 rates of change (first derivatives)
            - 5 accelerations (second derivatives)
            - 5 variability measures (std)
            - 3 trend strengths (slopes)
            - 1 interaction feature (power proxy)
        """
        temporal_features = []
        window_size = sequence.shape[0]

        # 1. Rate of change (first derivative) - 5 features
        if window_size > 1:
            rates = np.diff(sequence, axis=0).mean(axis=0)  # (5,)
        else:
            rates = np.zeros(5)
        temporal_features.extend(rates)

        # 2. Acceleration (second derivative) - 5 features
        if window_size > 2:
            accel = np.diff(sequence, n=2, axis=0).mean(axis=0)  # (5,)
        else:
            accel = np.zeros(5)
        temporal_features.extend(accel)

        # 3. Variability (std) - instability indicator - 5 features
        std_features = sequence.std(axis=0)  # (5,)
        temporal_features.extend(std_features)

        # 4. Trend strength (linear regression slope) - 3 features
        # Only for critical features: tool wear, torque, temp diff
        if window_size > 1:
            x = np.arange(window_size)
            # Tool wear trend (feature index 4)
            wear_trend = np.polyfit(x, sequence[:, 4], 1)[0] if window_size > 1 else 0
            # Torque trend (feature index 3)
            torque_trend = np.polyfit(x, sequence[:, 3], 1)[0] if window_size > 1 else 0
            # Temp diff trend (features 0 and 1)
            temp_diff = sequence[:, 1] - sequence[:, 0]
            temp_trend = np.polyfit(x, temp_diff, 1)[0] if window_size > 1 else 0
            temporal_features.extend([wear_trend, torque_trend, temp_trend])
        else:
            temporal_features.extend([0, 0, 0])

        # 5. Interaction features - 1 feature
        # Torque * Speed (power indicator) - features 2 and 3
        power_proxy = (sequence[:, 3] * sequence[:, 2]).mean()
        temporal_features.append(power_proxy)

        return np.array(temporal_features)  # (14,)

    def denormalize_ttf(self, ttf_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized TTF back to hours.

        Args:
            ttf_normalized: Normalized TTF values

        Returns:
            TTF in hours
        """
        ttf_log = self.ttf_scaler.inverse_transform(
            ttf_normalized.reshape(-1, 1)
        ).flatten()
        ttf_hours = np.expm1(ttf_log)  # exp(x) - 1, inverse of log1p
        return np.maximum(0, ttf_hours)  # Ensure non-negative

    def __len__(self) -> int:
        return len(self.X_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sequence window
        sequence = self.X_sequences[idx]  # (window_size, 5)

        # Compute temporal features (rates, acceleration, trends, etc.)
        temporal_feats = self.compute_temporal_features(sequence)  # (19,)

        return {
            'sequence': torch.FloatTensor(sequence),  # (window_size, 5)
            'features': torch.FloatTensor(temporal_feats),  # (19,) - only temporal features
            'type': torch.LongTensor([self.X_type[idx]]),
            'failure': torch.FloatTensor([self.y_failure[idx]]),
            'failure_types': torch.FloatTensor(self.y_failure_types[idx]),
            'ttf': torch.FloatTensor([self.y_ttf[idx]])  # Normalized
        }


class MultiTaskDataset(Dataset):
    """PyTorch Dataset for multi-task predictive maintenance."""

    def __init__(self, csv_path: str, scaler: StandardScaler = None,
                 fit_scaler: bool = False, synthesize_ttf: bool = True):
        """
        Args:
            csv_path: Path to CSV file (train/dev/test split)
            scaler: Pre-fitted StandardScaler (use for dev/test)
            fit_scaler: Whether to fit scaler on this data (True for train only)
            synthesize_ttf: Whether to generate TTF labels
        """
        self.df = pd.read_csv(csv_path)

        # Feature columns
        self.numeric_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]

        # Type encoding: L=0, M=1, H=2
        # Handle both string ('L', 'M', 'H') and numeric (0, 1, 2) type columns
        if self.df['Type'].dtype == 'object':
            self.df['Type_encoded'] = self.df['Type'].map({'L': 0, 'M': 1, 'H': 2})
        else:
            # Type is already numeric
            self.df['Type_encoded'] = self.df['Type']

        # Extract features
        self.X_numeric = self.df[self.numeric_features].values
        self.X_type = self.df['Type_encoded'].values.astype(int)

        # Normalize numeric features
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X_numeric = self.scaler.fit_transform(self.X_numeric)
        elif scaler is not None:
            self.scaler = scaler
            self.X_numeric = self.scaler.transform(self.X_numeric)
        else:
            raise ValueError("Must provide scaler or set fit_scaler=True")

        # Task 1: Binary failure prediction
        self.y_failure = self.df['Machine failure'].values

        # Task 2: Failure type classification (multi-label)
        self.failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        self.y_failure_types = self.df[self.failure_types].values

        # Task 3: Time-to-failure regression
        if synthesize_ttf:
            ttf_synthesizer = TTFSynthesizer()
            self.y_ttf = ttf_synthesizer.synthesize_ttf(self.df)
        else:
            self.y_ttf = np.zeros(len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'numeric_features': torch.FloatTensor(self.X_numeric[idx]),
            'type': torch.LongTensor([self.X_type[idx]]),
            'failure': torch.FloatTensor([self.y_failure[idx]]),
            'failure_types': torch.FloatTensor(self.y_failure_types[idx]),
            'ttf': torch.FloatTensor([self.y_ttf[idx]])
        }


def load_datasets(train_path: str, dev_path: str, test_path: str) -> Tuple[MultiTaskDataset, MultiTaskDataset, MultiTaskDataset]:
    """
    Load train/dev/test datasets with proper scaling.

    Args:
        train_path: Path to training CSV
        dev_path: Path to validation CSV
        test_path: Path to test CSV

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    # Load training data and fit scaler
    train_dataset = MultiTaskDataset(train_path, fit_scaler=True)

    # Use fitted scaler for dev/test
    dev_dataset = MultiTaskDataset(dev_path, scaler=train_dataset.scaler)
    test_dataset = MultiTaskDataset(test_path, scaler=train_dataset.scaler)

    return train_dataset, dev_dataset, test_dataset


def load_temporal_datasets(train_path: str, dev_path: str, test_path: str,
                           window_size: int = 12, stride: int = 1,
                           augment_train: bool = True, target_ratio: float = 0.15) -> Tuple[TemporalSequenceDataset, TemporalSequenceDataset, TemporalSequenceDataset]:
    """
    Load train/dev/test datasets with temporal sequences and proper scaling.

    Args:
        train_path: Path to training CSV
        dev_path: Path to validation CSV
        test_path: Path to test CSV
        window_size: Number of timesteps per sequence
        stride: Sliding window stride
        augment_train: Whether to augment training data
        target_ratio: Target failure ratio for augmentation

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset) with temporal sequences
    """
    # Load training data and fit scalers
    print("Loading training data with temporal sequences...")
    train_dataset = TemporalSequenceDataset(
        train_path,
        fit_scaler=True,
        window_size=window_size,
        stride=stride,
        augment=augment_train,
        target_ratio=target_ratio
    )

    # Use fitted scalers for dev/test
    print("\nLoading validation data...")
    dev_dataset = TemporalSequenceDataset(
        dev_path,
        scaler=train_dataset.scaler,
        ttf_scaler=train_dataset.ttf_scaler,
        window_size=window_size,
        stride=stride,
        augment=False
    )

    print("\nLoading test data...")
    test_dataset = TemporalSequenceDataset(
        test_path,
        scaler=train_dataset.scaler,
        ttf_scaler=train_dataset.ttf_scaler,
        window_size=window_size,
        stride=stride,
        augment=False
    )

    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING TEMPORAL SEQUENCE DATASET (NEW)")
    print("=" * 80)

    # Test temporal sequence dataset
    train_ds, dev_ds, test_ds = load_temporal_datasets(
        './dataset/train/train.csv',
        './dataset/dev/dev.csv',
        './dataset/test/test.csv',
        window_size=12,
        stride=1,
        augment_train=True,
        target_ratio=0.15
    )

    print(f"\n{'='*80}")
    print(f"Dataset Sizes:")
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Dev: {len(dev_ds)} sequences")
    print(f"  Test: {len(test_ds)} sequences")

    # Sample data point
    sample = train_ds[0]
    print(f"\n{'='*80}")
    print("Sample data point structure:")
    for key, value in sample.items():
        if key == 'features':
            print(f"  {key}: {value.shape} (5 current + 19 temporal features)")
        else:
            print(f"  {key}: {value.shape}")

    # Check class distribution
    failures = sum(train_ds.y_failure)
    print(f"\n{'='*80}")
    print(f"Class Distribution (after augmentation):")
    print(f"  Failures: {int(failures)}/{len(train_ds)} ({100*failures/len(train_ds):.1f}%)")
    print(f"  Healthy: {int(len(train_ds) - failures)}/{len(train_ds)} ({100*(1-failures/len(train_ds)):.1f}%)")

    # TTF statistics
    print(f"\n{'='*80}")
    print(f"TTF Statistics (raw, before normalization):")
    print(f"  Min: {train_ds.y_ttf_raw.min():.2f} hours")
    print(f"  Max: {train_ds.y_ttf_raw.max():.2f} hours")
    print(f"  Mean: {train_ds.y_ttf_raw.mean():.2f} hours")
    print(f"  Std: {train_ds.y_ttf_raw.std():.2f} hours")

    print(f"\nTTF Statistics (normalized):")
    print(f"  Min: {train_ds.y_ttf.min():.2f}")
    print(f"  Max: {train_ds.y_ttf.max():.2f}")
    print(f"  Mean: {train_ds.y_ttf.mean():.2f}")
    print(f"  Std: {train_ds.y_ttf.std():.2f}")

    print(f"\n{'='*80}")
    print("Phase 1 Data Preprocessing: COMPLETE [OK]")
    print(f"{'='*80}")
