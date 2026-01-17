"""
Data preprocessing for multi-task predictive maintenance.
Handles feature normalization and time-to-failure (TTF) synthesis.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


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
        self.df['Type_encoded'] = self.df['Type'].map({'L': 0, 'M': 1, 'H': 2})

        # Extract features
        self.X_numeric = self.df[self.numeric_features].values
        self.X_type = self.df['Type_encoded'].values

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


if __name__ == '__main__':
    # Test the dataset
    train_ds, dev_ds, test_ds = load_datasets(
        '../dataset/train/train.csv',
        '../dataset/dev/dev.csv',
        '../dataset/test/test.csv'
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Dev size: {len(dev_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Sample data point
    sample = train_ds[0]
    print("\nSample data point:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape} - {value}")

    # Check class distribution
    failures = sum(train_ds.y_failure)
    print(f"\nFailure rate: {failures}/{len(train_ds)} ({100*failures/len(train_ds):.2f}%)")
