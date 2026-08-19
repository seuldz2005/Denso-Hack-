from src.phaseII.core.prepare_data import prepare_training_data
from src.phaseII.core.train import run_training


train_records, val_records = prepare_training_data(
    latent_data=latent_data,
    event_bins=event_bins,
    train_ratio=0.8,
    random_seed=42,
)

model, history = run_training(
    train_records,
    val_records,
    latent_dim=16,
)
