from prepare_data import load_engines
from train import run_training, run_multi_seed

train_records, val_records = load_engines()
model, history = run_training(train_records, val_records, latent_dim=16)
