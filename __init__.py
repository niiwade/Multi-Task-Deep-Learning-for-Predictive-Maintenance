"""
Multi-Task Predictive Maintenance Package

A complete implementation of multi-task deep learning for industrial
predictive maintenance using Temporal Convolutional Networks (TCN)
with attention mechanisms.

Main Components:
- data_preprocessing: Dataset class with TTF synthesis
- model: TCN architecture with multi-head attention
- train: Training script with focal loss
- evaluate: Comprehensive evaluation metrics
- visualize_attention: Attention analysis and visualization
- config: Configuration management
"""

__version__ = '1.0.0'
__author__ = 'Revas Shield Team'

from .model import MultiTaskTCN, TemporalConvBlock, MultiHeadAttention
from .data_preprocessing import MultiTaskDataset, TTFSynthesizer, load_datasets
from .config import Config, LightConfig, HeavyConfig, ImbalanceConfig

__all__ = [
    'MultiTaskTCN',
    'TemporalConvBlock',
    'MultiHeadAttention',
    'MultiTaskDataset',
    'TTFSynthesizer',
    'load_datasets',
    'Config',
    'LightConfig',
    'HeavyConfig',
    'ImbalanceConfig',
]
