"""
split_cmapss.py

Chuẩn bị C-MAPSS TRAIN cho experiment:

    .txt
      ↓
    DataFrame
      ↓
    engine trajectories
      ↓
    70% development / 30% test
      ↓
    development: truncate quanh elbow ~130
    test: initially right-censored

Chỉ sử dụng TRAIN của C-MAPSS.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SplitConfig:

    # Dataset
    data_path: str = "data/train_FD001.txt"

    # Engine split
    train_ratio: float = 0.70
    random_seed: int = 42

    # Piecewise-linear elbow
    elbow_cycle: int = 130
    elbow_tolerance: int = 10

    # Một phần development engines được giữ lâu hơn elbow
    extended_probability: float = 0.20
    extended_max_extra: int = 50

    # Không để development engine quá gần failure
    min_cycles_before_failure: int = 20

    # Initial censoring của test
    test_fraction_min: float = 0.10
    test_fraction_max: float = 0.30


# ============================================================
# C-MAPSS COLUMN DEFINITION
# ============================================================

SENSORS = [
    "T24",
    "T30",
    "T50",
    "P30",
    "Ps30",
    "phi",
    "BPR",
    "W31",
    "W32",
]


def load_cmapss_train(config: SplitConfig):
    """
    Load C-MAPSS TRAIN và trả về DataFrame + engine trajectories.
    """

    sensor_names = [
        "T2", "T24", "T30", "T50",
        "P2", "P15", "P30",
        "Nf", "Nc", "epr",
        "Ps30", "phi", "NRf", "NRc",
        "BPR", "farB", "htBleed",
        "Nfdmd", "PCNfRdmd",
        "W31", "W32",
    ]

    columns = [
        "unit_number",
        "cycles",
        "op1",
        "op2",
        "op3",
    ] + sensor_names

    # --------------------------------------------------------
    # Load raw C-MAPSS
    # --------------------------------------------------------

    df = pd.read_csv(
        config.data_path,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    # --------------------------------------------------------
    # Chỉ giữ metadata + sensors sử dụng
    # --------------------------------------------------------

    df = df[
        ["unit_number", "cycles"] + SENSORS
    ].copy()

    # --------------------------------------------------------
    # Chuyển thành:
    #
    # {
    #     engine_id: ndarray(n_cycles, n_sensors)
    # }
    # --------------------------------------------------------

    engine_data = {}

    for engine_id, group in df.groupby("unit_number"):

        group = group.sort_values("cycles")

        engine_data[int(engine_id)] = (
            group[SENSORS]
            .to_numpy(dtype=np.float32)
        )

    return df, engine_data


# ============================================================
# MAIN EXPERIMENT SPLIT
# ============================================================

def prepare_cmapss_split(config: SplitConfig = None):

    if config is None:
        config = SplitConfig()

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df, engine_data = load_cmapss_train(config)

    rng = np.random.default_rng(config.random_seed)

    # --------------------------------------------------------
    # 2. Split ENGINE 70 / 30
    # --------------------------------------------------------

    engine_ids = np.array(
        sorted(engine_data.keys())
    )

    rng.shuffle(engine_ids)

    n_train = int(
        len(engine_ids) * config.train_ratio
    )

    train_ids = sorted(
        engine_ids[:n_train].tolist()
    )

    test_ids = sorted(
        engine_ids[n_train:].tolist()
    )

    # --------------------------------------------------------
    # 3. Development / Train trajectories
    # --------------------------------------------------------

    train_data = {}

    for engine_id in train_ids:

        full = engine_data[engine_id]

        n_cycles = len(full)

        # Không được quan sát quá gần failure
        safe_max = max(
            1,
            n_cycles - config.min_cycles_before_failure
        )

        # Một số engine được giữ lâu hơn elbow
        if (
            rng.random()
            < config.extended_probability
        ):

            end = config.elbow_cycle + rng.integers(
                1,
                config.extended_max_extra + 1,
            )

        # Phần lớn engine chỉ quanh elbow
        else:

            end = rng.integers(
                max(
                    1,
                    config.elbow_cycle
                    - config.elbow_tolerance,
                ),
                config.elbow_cycle
                + config.elbow_tolerance
                + 1,
            )

        # Không vượt quá trajectory thực tế
        # và không tới sát failure
        end = min(end, safe_max)

        train_data[engine_id] = full[:end]

    # --------------------------------------------------------
    # 4. Test trajectories
    #
    # Ban đầu chỉ expose một phần nhỏ trajectory.
    # Mỗi engine có observation length khác nhau.
    # --------------------------------------------------------

    test_rng = np.random.default_rng(
        config.random_seed + 1
    )

    test_data = {}

    for engine_id in test_ids:

        full = engine_data[engine_id]

        n_cycles = len(full)

        fraction = test_rng.uniform(
            config.test_fraction_min,
            config.test_fraction_max,
        )

        end = max(
            1,
            int(n_cycles * fraction),
        )

        # Censor: không reveal failure
        end = min(
            end,
            n_cycles - 1,
        )

        test_data[engine_id] = full[:end]

    # --------------------------------------------------------
    # 5. Return tất cả những gì notebook cần
    # --------------------------------------------------------

    return {
        "df": df,

        # Toàn bộ trajectory gốc.
        # Chỉ dùng để kiểm tra / simulation.
        "engine_data": engine_data,

        # 70%
        "train_data": train_data,
        "train_ids": train_ids,

        # 30%
        "test_data": test_data,
        "test_ids": test_ids,

        # Sensors đang sử dụng
        "sensors": SENSORS,

        # Config
        "config": config,
    }
