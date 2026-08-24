import sys
from pathlib import Path
from src.phaseI.main import run_phase1
from src.phaseII.main import run_phase2

from demo.core.data import (
    SplitConfig,
    prepare_cmapss_split,
)


config = SplitConfig(
    data_path="demo/data/train_FD001.txt",

    train_ratio=0.70,

    elbow_cycle=130,
    elbow_tolerance=5,

    extended_probability=0.30,
    extended_max_extra=50,

    min_cycles_before_failure=20,

    test_fraction_min=0.10,
    test_fraction_max=0.30,

    random_seed=42,
)
data = prepare_cmapss_split(config)

from src.preprocess_data.realtime import (
    simulate_realtime,
    RealtimeConfig,
)

realtime = simulate_realtime(
    engine_data=data["engine_data"],
    test_data=data["test_data"],
    observation_points=data["test_observation_points"],
    config=RealtimeConfig(
        update_probability=0.8,
        max_new_cycles=1,
        random_seed=42,
    ),
)


BIN_STRIDE = 10

model, stats, normalized_data, latent_data = run_phase1(
    train_data=data["train_data"],
    n_sensors=len(data["sensors"]),
    window_len=30,
    latent_dim=16,
    bin_stride=BIN_STRIDE,
)

gru_model, history = run_phase2(
    latent_data=latent_data,
    train_metadata=data["train_metadata"],
    latent_dim=16,
)
