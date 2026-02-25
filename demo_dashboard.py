"""
Streamlit demo dashboard for the multi-task predictive maintenance model.

Run:
  pip install -r requirements.txt
  streamlit run demo_dashboard.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import streamlit as st

from config import Config
from data_preprocessing import load_temporal_datasets
from model import MultiTaskTCN


FAILURE_TYPE_NAMES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SENSOR_FEATURE_NAMES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


@dataclass(frozen=True)
class LoadedAssets:
    model: MultiTaskTCN
    device: torch.device
    checkpoint_path: Path
    use_temporal: bool
    window_size: int
    test_dataset: Any


def _default_checkpoint_candidates() -> list[Path]:
    candidates = [
        Path("checkpoints/best_model.pt"),
        Path("checkpoints/phase3/best_model.pt"),
        Path("checkpoints/phase2/best_model.pt"),
        Path("checkpoints/phase1/best_model.pt"),
    ]
    return [p for p in candidates if p.exists()]


def _infer_model_params_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    # type_embedding.weight: (num_types, embed_dim)
    num_types = int(state_dict["type_embedding.weight"].shape[0])
    # sequence_projection.weight: (tcn_channels, num_numeric_features, 1)
    tcn_channels = int(state_dict["sequence_projection.weight"].shape[0])
    num_numeric_features = int(state_dict["sequence_projection.weight"].shape[1])
    return num_numeric_features, num_types, tcn_channels


def _safe_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource(show_spinner=True)
def load_assets(checkpoint_path_str: str, prefer_cuda: bool, window_size: int) -> LoadedAssets:
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format (expected dict).")

    num_numeric_features, num_types, tcn_channels = _infer_model_params_from_state_dict(state_dict)
    device = _safe_device(prefer_cuda)

    model = MultiTaskTCN(
        num_numeric_features=num_numeric_features,
        num_temporal_features=Config.NUM_TEMPORAL_FEATURES,
        num_types=num_types,
        tcn_channels=tcn_channels,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        use_temporal_sequences=True,
        binary_only=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    train_ds, dev_ds, test_ds = load_temporal_datasets(
        str(Config.TRAIN_PATH),
        str(Config.DEV_PATH),
        str(Config.TEST_PATH),
        window_size=window_size,
        stride=Config.STRIDE,
        augment_train=False,
        target_ratio=Config.TARGET_RATIO,
    )

    return LoadedAssets(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        use_temporal=True,
        window_size=window_size,
        test_dataset=test_ds,
    )


def _predict_one(assets: LoadedAssets, idx: int) -> Dict[str, Any]:
    sample = assets.test_dataset[idx]
    sequence = sample["sequence"].unsqueeze(0).to(assets.device)
    temporal_features = sample["features"].unsqueeze(0).to(assets.device)
    machine_type = sample["type"].unsqueeze(0).to(assets.device)

    with torch.no_grad():
        outputs = assets.model(sequence=sequence, temporal_features=temporal_features, machine_type=machine_type)

    failure_prob = torch.sigmoid(outputs["failure_logits"]).squeeze().item()
    type_probs = torch.sigmoid(outputs["failure_type_logits"]).squeeze().detach().cpu().numpy().astype(float)
    ttf_norm = outputs["ttf"].squeeze().item()
    ttf_hours = float(assets.test_dataset.denormalize_ttf(np.array([ttf_norm]))[0])

    attn = outputs.get("attention_weights", None)
    if attn is not None:
        attn = attn.detach().cpu().numpy()

    return {
        "sample": sample,
        "failure_prob": float(failure_prob),
        "type_probs": type_probs,
        "ttf_norm": float(ttf_norm),
        "ttf_hours": ttf_hours,
        "attention_weights": attn,
    }


def _plot_sequence(sequence: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(sequence.shape[0])
    for j, name in enumerate(SENSOR_FEATURE_NAMES):
        ax.plot(x, sequence[:, j], label=name, linewidth=1.5)
    ax.set_title("Normalized sensor sequence (window)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized value")
    ax.grid(alpha=0.3)
    ax.legend(ncols=2, fontsize=8)
    st.pyplot(fig, clear_figure=True)


def _plot_attention(attn: np.ndarray):
    import matplotlib.pyplot as plt

    # PyTorch returns attn_weights as (batch, tgt_len, src_len) by default.
    if attn.ndim == 3:
        attn2 = attn[0]
    elif attn.ndim == 4:
        attn2 = attn[0].mean(axis=0)
    else:
        st.info("Attention weights shape not recognized.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(attn2, aspect="auto")
    ax.set_title("Attention weights (time → time)")
    ax.set_xlabel("Source timestep")
    ax.set_ylabel("Target timestep")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def main():
    st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
    st.title("Multi-Task Predictive Maintenance — Demo Dashboard")

    with st.sidebar:
        st.header("Model")
        candidates = _default_checkpoint_candidates()
        default_checkpoint = str(candidates[0]) if candidates else "checkpoints/best_model.pt"
        checkpoint_path_str = st.text_input("Checkpoint path", value=default_checkpoint)
        prefer_cuda = st.checkbox("Use CUDA if available", value=False)

        st.header("Data")
        window_size = st.slider("Window size", min_value=4, max_value=32, value=Config.WINDOW_SIZE, step=1)

    try:
        assets = load_assets(checkpoint_path_str, prefer_cuda, window_size)
    except Exception as e:
        st.error(f"Failed to load model/data: {e}")
        st.stop()

    n = len(assets.test_dataset)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        idx = st.slider("Test sample index", min_value=0, max_value=max(0, n - 1), value=0, step=1)
    with col_b:
        threshold = st.slider("Failure threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.01)

    pred = _predict_one(assets, idx)
    sample = pred["sample"]

    left, right = st.columns([1.2, 1.0])
    with left:
        _plot_sequence(sample["sequence"].numpy())
        if pred["attention_weights"] is not None:
            _plot_attention(pred["attention_weights"])

    with right:
        st.subheader("Predictions")
        failure_prob = pred["failure_prob"]
        st.metric("Failure probability", f"{failure_prob:.3f}")
        st.write(f"Decision @ threshold {threshold:.2f}: **{'FAIL' if failure_prob >= threshold else 'OK'}**")

        st.write("---")
        st.write("Failure type probabilities:")
        type_probs = pred["type_probs"]
        for name, p in zip(FAILURE_TYPE_NAMES, type_probs):
            st.write(f"- {name}: `{p:.3f}`")

        st.write("---")
        st.write("Time-to-failure (TTF):")
        st.write(f"- Normalized: `{pred['ttf_norm']:.3f}`")
        st.write(f"- Hours (denormalized): `{pred['ttf_hours']:.2f}`")

        st.write("---")
        st.subheader("Ground truth (test sample)")
        st.write(f"- Machine type: `{int(sample['type'].item())}` (0=L, 1=M, 2=H)")
        st.write(f"- Failure label: `{int(sample['failure'].item())}`")
        gt_types = sample["failure_types"].numpy().astype(int).tolist()
        st.write(f"- Failure types: `{dict(zip(FAILURE_TYPE_NAMES, gt_types))}`")

    st.caption(f"Loaded checkpoint: {assets.checkpoint_path} | Device: {assets.device.type}")


if __name__ == "__main__":
    main()

