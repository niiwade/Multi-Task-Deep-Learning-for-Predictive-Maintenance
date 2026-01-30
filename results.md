1. Topic Overview
This project applies a multi-task deep learning model to industrial predictive maintenance,
simultaneously performing:
A. Binary failure prediction (will the machine fail?),
B. Failure type classification (which failure mode?), and
C. Time-to-failure regression (when will it fail?).
A Temporal Convolutional Network (TCN) with attention is used to provide fast, interpretable
predictions.
2. Current Knowledge
Predictive maintenance uses ML to reduce downtime and operational costs. Traditional models
mainly perform binary failure classification and overlook temporal patterns. Recent research shows
that TCNs outperform RNNs on sensor data due to parallelization, long-range dependency modeling,
and computational efficiency. Multi-task frameworks improve generalization by sharing
representations across related tasks. Attention mechanisms add interpretability by highlighting
influential sensor features.
. Relevance
Machine failures cause high costs and safety risks. A multi-task approach delivers richer insights by
predicting if, how, and when a machine will fail. Attention improves reliability by showing which
sensors drive decisions. This project addresses real-world challenges such as highly imbalanced
failure data (97:3 ratio), temporal pattern extraction, and interpretable industrial AI.
4. Data Description
The AI4I 2020 dataset (10,000 samples) includes air temperature, process temperature, rotational
speed, torque, and tool wear. It contains binary failure labels and five failure modes (TWF, HDF,
PWF, OSF, RNF). Data will be- split into 7,000 training, 1,500 validation, and 1,500 test samples
using stratified sampling.
5. Expected Results
We expect the model to achieve:
A. Failure detection: F1-score > 0.95
B. Failure type classification: Accuracy > 95%
C. Time-to-failure: MAE < 2 hours
The attention mechanism will reveal key sensor contributors, and the model will support real-time
inference (<50ms), making it suitable for deployment in industrial environments.
6. Methods, Tools & Plan
The project uses Python, PyTorch, pandas, NumPy, and scikit-learn. The model will be a TCN with
four dilated convolution layers (dilations 1, 2, 4, 8), multi-head attention, and three output heads.
Class imbalance is handled using weighted focal loss, class weighting, and weighted sampling