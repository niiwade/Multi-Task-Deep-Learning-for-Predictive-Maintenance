# Demo Dashboard (Streamlit)

This repo includes a small Streamlit app to demonstrate the trained model making predictions on samples from the test split.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Ensure you have data + a checkpoint

- Dataset CSVs are expected at:
  - `dataset/train/train.csv`
  - `dataset/dev/dev.csv`
  - `dataset/test/test.csv`
- A checkpoint is expected at one of:
  - `checkpoints/best_model.pt` (default)
  - `checkpoints/phase3/best_model.pt`
  - `checkpoints/phase2/best_model.pt`
  - `checkpoints/phase1/best_model.pt`

## 3) Run

```bash
streamlit run demo_dashboard.py
```

## What you’ll see

- Sensor window plot (normalized values over time)
- Failure probability + thresholded decision
- Failure-type probabilities (TWF/HDF/PWF/OSF/RNF)
- Time-to-failure (normalized + denormalized to hours)
- Attention weights heatmap (time → time)

