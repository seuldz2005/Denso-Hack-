"""
main.py -- entry point của Phase I.

Dataset-specific code được gọi ở bên ngoài core pipeline.
File này chỉ kết nối:
    prepared trajectories
        → normalization + windows
        → AE training
        → latent extraction
"""

from src.phaseI.core.prepare_data import prepare_training_data
from src.phaseI.core.train import run_training

def run_phase1(train_data, n_sensors, window_len=30,
               latent_dim=16, n_epochs=30, lr=1e-3,
               batch_size=64, stride=1, seed=0, device="cpu"):

    # 1. Normalize + create training windows
    training_windows, stats, normalized_data = prepare_training_data(
        train_data,
        window_len=window_len,
        stride=stride,
    )

    # 2. Train AE
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

    return model, stats, normalized_data
