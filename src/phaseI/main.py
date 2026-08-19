"""
main.py -- Phase I pipeline.

Pipeline:

    train_data
        ↓
    fit normalization
        ↓
    normalize
        ↓
    cut training windows
        ↓
    train Autoencoder
        ↓
    freeze encoder
        ↓
    extract latent sequences
"""

import numpy as np
import torch

from src.phaseI.core.data import fit_normalization, cut_windows
from src.phaseI.core.train import run_training, extract_latents_for_engine


def run_phase1(
    train_data: dict[int, np.ndarray],
    n_sensors: int,
    window_len: int = 30,
    latent_dim: int = 16,
    healthy_fraction: float = 0.25,
    train_stride: int = 1,
    bin_stride: int = 10,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
):
    """
    Chạy toàn bộ Phase I.

    Returns
    -------
    model:
        Autoencoder đã train và freeze.

    stats:
        NormalizationStats dùng lại cho Phase II/test.

    normalized_data:
        Dữ liệu train sau normalization.
    """

    # ========================================================
    # 1. Fit normalization trên development data
    # ========================================================

    healthy_data = []

    for series in train_data.values():

        n = len(series)
        cutoff = max(
            1,
            int(n * healthy_fraction),
        )

        healthy_data.append(
            series[:cutoff]
        )

    stats = fit_normalization(
        healthy_data
    )

    # ========================================================
    # 2. Normalize toàn bộ train trajectories
    # ========================================================

    normalized_data = {
        engine_id: stats.apply(series)
        for engine_id, series in train_data.items()
    }

    # ========================================================
    # 3. Cắt windows để train AE
    # ========================================================

    healthy_windows = []

    for series in normalized_data.values():

        n = len(series)

        cutoff = max(
            1,
            int(n * healthy_fraction),
        )

        windows = cut_windows(
            series[:cutoff],
            window_len=window_len,
            stride=train_stride,
        )

        if len(windows) > 0:
            healthy_windows.append(windows)

    if not healthy_windows:
        raise ValueError(
            "Không tạo được training window nào. "
            "Hãy kiểm tra window_len / healthy_fraction."
        )

    healthy_windows = np.concatenate(
        healthy_windows,
        axis=0,
    )

    # ========================================================
    # 4. Train AE
    # ========================================================

    model = run_training(
        healthy_windows=healthy_windows,
        n_sensors=n_sensors,
        window_len=window_len,
        latent_dim=latent_dim,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    # ========================================================
    # 5. Freeze Phase I
    # ========================================================

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False



    latent_data = {}

    for engine_id, series in normalized_data.items():

        latent_seq = extract_latents_for_engine(
            model=model,
            series=series,
            window_len=window_len,
            bin_stride=bin_stride,
            device=device,
        )

        latent_data[engine_id] = latent_seq

    return model, stats, normalized_data, latent_data
