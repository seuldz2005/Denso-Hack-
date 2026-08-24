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

    train_data đã được truncate tại elbow_cycle bởi prepare_cmapss_split()
    — toàn bộ data trong đây đều là healthy, không cần cắt thêm.

    Returns
    -------
    model:
        Autoencoder đã train và freeze encoder.

    stats:
        NormalizationStats dùng lại cho Phase II và inference.

    normalized_data:
        Toàn bộ train trajectories sau normalization.

    latent_data:
        {engine_id: latent_seq (n_bins, latent_dim)} — input trực tiếp cho Phase II.
    """

    # ========================================================
    # 1. Fit normalization trên development data
    #
    # train_data đã được truncate tại elbow_cycle bởi prepare_cmapss_split()
    # — toàn bộ data trong đây đều là healthy, fit normalization trên tất cả.
    # ========================================================

    stats = fit_normalization(
        list(train_data.values())
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
    #
    # Dùng toàn bộ normalized_data — đã là healthy (xem bước 1).
    # stride nhỏ (train_stride=1) để tối đa hóa số mẫu cho AE.
    # ========================================================

    training_windows = []

    for series in normalized_data.values():

        windows = cut_windows(
            series,
            window_len=window_len,
            stride=train_stride,
        )

        if len(windows) > 0:
            training_windows.append(windows)

    if not training_windows:
        raise ValueError(
            "Không tạo được training window nào. "
            "Hãy kiểm tra window_len và độ dài trajectory."
        )

    training_windows = np.concatenate(
        training_windows,
        axis=0,
    )

    # ========================================================
    # 4. Train AE
    # ========================================================

    model = run_training(
        training_windows=training_windows,
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
