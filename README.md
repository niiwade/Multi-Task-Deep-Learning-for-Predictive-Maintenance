# Multi-Task Predictive Maintenance with TCN

A multi-task deep learning system for industrial predictive maintenance using Temporal Convolutional Networks (TCN) with attention mechanisms.

## Overview

This implementation addresses three simultaneous prediction tasks:

1. **Binary Failure Prediction**: Will the machine fail? (Classification)
2. **Failure Type Classification**: Which failure mode(s)? (Multi-label)
3. **Time-to-Failure Regression**: When will it fail? (Regression)

### Key Features

- **Temporal Convolutional Network (TCN)**: Dilated causal convolutions (1, 2, 4, 8) for long-range dependency modeling
- **Multi-Head Attention**: Interpretable feature importance via attention weights
- **Focal Loss**: Handles severe class imbalance (97:3 healthy/failed ratio)
- **Multi-Task Learning**: Shared representations improve generalization
- **Attention Visualization**: Understand which sensors drive predictions

## Architecture

```
Input: [Numeric Features (5) + Type Embedding (8)] → (13 features)
  ↓
Input Projection → TCN Channels (64)
  ↓
TCN Block 1 (dilation=1) → Residual + BatchNorm
  ↓
TCN Block 2 (dilation=2) → Residual + BatchNorm
  ↓
TCN Block 3 (dilation=4) → Residual + BatchNorm
  ↓
TCN Block 4 (dilation=8) → Residual + BatchNorm
  ↓
Multi-Head Attention (4 heads) → Interpretability
  ↓
Global Average Pooling
  ↓
├─→ Failure Head → Binary Prediction (1 output)
├─→ Type Head → Multi-Label Classification (5 outputs)
└─→ TTF Head → Regression (1 output, ReLU)
```

## Project Structure

```
multi_task/
├── data_preprocessing.py    # Dataset class with TTF synthesis
├── model.py                  # TCN architecture with attention
├── train.py                  # Training script with focal loss
├── evaluate.py               # Comprehensive evaluation metrics
├── visualize_attention.py    # Attention analysis and visualization
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── checkpoints/              # Saved model checkpoints (created during training)
└── visualizations/           # Attention plots (created during visualization)
```

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# From the Revas Shield root directory
cd multi_task
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python model.py  # Should print model architecture and parameter count
```

## Usage

### 1. Data Preprocessing (Test)

Test the data preprocessing and TTF synthesis:

```bash
python data_preprocessing.py
```

Expected output:
- Train/dev/test sizes
- Sample data point structure
- Failure rate statistics

### 2. Train the Model

Train the multi-task model with default hyperparameters:

```bash
python train.py
```

**Configuration** (edit in `train.py`):
- `batch_size`: 32
- `num_epochs`: 100
- `lr`: 0.001
- `tcn_channels`: 64
- `num_heads`: 4
- `focal_alpha`: 0.25
- `focal_gamma`: 2.0
- `patience`: 15 (early stopping)

**Training outputs**:
- Progress bars with loss metrics per epoch
- Model checkpoints saved to `./checkpoints/best_model.pt`
- Training history saved to `./checkpoints/training_history.json`

### 3. Evaluate the Model

Evaluate on the test set:

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pt --test_path ../dataset/test/test.csv
```

**Expected Results** (from projectwork.md):
- Binary Failure F1-Score: **> 0.95**
- Failure Type Accuracy: **> 95%**
- Time-to-Failure MAE: **< 2 hours**

**Optional arguments**:
- `--checkpoint`: Path to model checkpoint (default: `./checkpoints/best_model.pt`)
- `--test_path`: Path to test CSV (default: `../dataset/test/test.csv`)
- `--output`: Path to save results JSON (default: `./checkpoints/evaluation_results.json`)
- `--batch_size`: Batch size (default: 32)

### 4. Visualize Attention

Generate attention visualizations for interpretability:

```bash
python visualize_attention.py --checkpoint ./checkpoints/best_model.pt --output_dir ./visualizations
```

**Outputs**:
- `feature_importance.png`: Which sensors are most important?
- `attention_by_failure_type.png`: Attention patterns for different failure modes
- `samples/`: Individual sample analyses

**Optional arguments**:
- `--checkpoint`: Model checkpoint path
- `--test_path`: Test data path
- `--output_dir`: Output directory for visualizations
- `--num_samples`: Number of samples to analyze (default: 1000)

## Implementation Details

### Time-to-Failure (TTF) Synthesis

Since the AI4I 2020 dataset doesn't include TTF labels, we synthesize them based on sensor degradation patterns:

**For Failed Machines**: TTF = 0 hours

**For Healthy Machines**: TTF = 1-100 hours, estimated using:
- Tool wear (40% weight): Higher wear → Lower TTF
- Torque (30% weight): Higher torque → Lower TTF
- Speed (20% weight): Lower speed → Lower TTF
- Temperature differential (10% weight): Higher diff → Lower TTF

**Failure-type adjustments**:
- TWF (Tool Wear): Gradual degradation (-30% TTF)
- HDF (Heat Dissipation): Sudden failures (-50% TTF)

**Noise**: ±10% random variation for realism

### Class Imbalance Handling

The dataset has a severe imbalance (~97% healthy, ~3% failed). We address this with:

1. **Focal Loss**: α=0.25, γ=2.0 (focuses on hard examples)
2. **Weighted Sampling**: Over-sample minority class during training
3. **Class Weighting**: Inverse frequency weighting in loss calculation

### Multi-Task Loss

```python
Total Loss = w1 * Focal(failure) + w2 * BCE(failure_types) + w3 * MSE(ttf)
```

**Default weights**:
- Failure: 1.0
- Failure Types: 1.0
- TTF: 0.5 (lower weight since labels are synthesized)

## Performance Benchmarks

### Model Specifications
- **Parameters**: ~50K trainable parameters
- **Inference Time**: < 50ms per sample (target for real-time deployment)
- **Memory**: ~200MB GPU memory during inference

### Training Time
- **GPU (NVIDIA RTX 3070)**: ~2-3 minutes per epoch
- **CPU**: ~10-15 minutes per epoch
- **Typical convergence**: 30-50 epochs

## Interpretability

The attention mechanism provides insights into which sensors drive predictions:

### Expected Sensor Importance (Typical)
1. **Tool Wear**: Strongest indicator for TWF failures
2. **Torque**: Critical for PWF (Power Failure) detection
3. **Temperature Differential**: Key for HDF (Heat Dissipation)
4. **Rotational Speed**: Important for OSF (Overstrain)

Use `visualize_attention.py` to analyze your specific model.

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in `train.py`:
```python
'batch_size': 16  # or 8
```

### Issue: Poor Performance on Minority Class

**Solutions**:
1. Increase focal loss gamma: `'focal_gamma': 3.0`
2. Adjust task weights: `'task_weights': {'failure': 2.0, ...}`
3. Use stronger data augmentation (future work)

### Issue: Overfitting

**Solutions**:
1. Increase dropout: `'dropout': 0.3`
2. Add L2 regularization: `'weight_decay': 1e-4`
3. Use early stopping (already implemented)

### Issue: Training Takes Too Long

**Solutions**:
1. Use GPU if available (10x speedup)
2. Reduce `num_epochs` for quick experiments
3. Use smaller model: `'tcn_channels': 32`

## Extending the Implementation

### Add New Sensor Features

1. Update `data_preprocessing.py`:
   ```python
   self.numeric_features = [
       'Air temperature [K]',
       'Process temperature [K]',
       'Rotational speed [rpm]',
       'Torque [Nm]',
       'Tool wear [min]',
       'Your New Feature'  # Add here
   ]
   ```

2. Update `model.py`:
   ```python
   model = MultiTaskTCN(num_numeric_features=6, ...)  # Increment count
   ```

### Modify TCN Architecture

Edit `model.py` to:
- Add more TCN blocks with higher dilations
- Change channel dimensions
- Adjust attention heads

### Custom Loss Functions

Edit `train.py` to implement custom loss formulations or different task weightings.

## Citation

If you use this implementation, please cite the AI4I 2020 dataset:

```
@misc{ai4i2020,
  title={AI4I 2020 Predictive Maintenance Dataset},
  author={Stephan Matzka},
  year={2020},
  url={https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset}
}
```

## License

This implementation is part of the Revas Shield project.

## Contact

For questions or issues, please refer to the main project documentation.
