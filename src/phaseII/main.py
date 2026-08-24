"""
main.py -- Phase II pipeline.

Pipeline:

    latent_data (từ run_phase1())
        ↓
    compute_event_bins (elbow_cycle → degradation onset bin)
        ↓
    build_engine_records (cắt latent_seq tại [:event_bin+1])
        ↓
    stratified split train / val
        ↓
    train GRUHazard
"""

from src.phaseII.core.prepare_data import prepare_training_data
from src.phaseII.core.train import run_training


def run_phase2(
    latent_data: dict,
    train_metadata: dict,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    lam_monotonic: float = 0.1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    device: str = "cpu",
):
    """
    Chạy toàn bộ Phase II.

    Parameters
    ----------
    latent_data:
        Output của run_phase1() — {engine_id: latent_seq (n_bins, latent_dim)}.

    train_metadata:
        Output của prepare_cmapss_split() — {engine_id: {"is_extended": bool, ...}}.
        is_extended=True: engine có event_bin (positive example cho GRU).
        is_extended=False: engine censored (chưa đến degradation onset).

    Returns
    -------
    model:
        GRUHazard đã train.

    history:
        Dict chứa loss/C-index theo epoch.
    """

    train_records, val_records = prepare_training_data(
        latent_data=latent_data,
        train_metadata=train_metadata,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    model, history = run_training(
        train_records=train_records,
        val_records=val_records,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lam_monotonic=lam_monotonic,
        n_epochs=n_epochs,
        lr=lr,
        seed=random_seed,
        device=device,
    )

    return model, history
