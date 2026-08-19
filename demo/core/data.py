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
    test: initial observation state

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

    # Initial observation của test
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
    train_metadata = {}

    for engine_id in train_ids:

        full = engine_data[engine_id]
        n_cycles = len(full)

        safe_max = max(
            1,
            n_cycles - config.min_cycles_before_failure
        )

        is_extended = (
            rng.random() < config.extended_probability
        )

        if is_extended:
            end = config.elbow_cycle + rng.integers(
                1,
                config.extended_max_extra + 1,
            )
        else:
            end = rng.integers(
                max(1, config.elbow_cycle - config.elbow_tolerance),
                config.elbow_cycle + config.elbow_tolerance + 1,
            )

        end = min(end, safe_max)

        train_data[engine_id] = full[:end]

        train_metadata[engine_id] = {
            "is_extended": is_extended,
            "observed_cycles": end,
        }

    # --------------------------------------------------------
    # 4. Test INITIAL OBSERVATION
    #
    # Đây chỉ là trạng thái quan sát tại thời điểm
    # experiment bắt đầu.
    #
    # Chưa phải realtime simulation.
    # Realtime simulator sẽ reveal thêm cycle sau này.
    # --------------------------------------------------------

    test_rng = np.random.default_rng(
        config.random_seed + 1
    )

    test_data = {}
    test_observation_points = {}

    for engine_id in test_ids:

        full = engine_data[engine_id]

        n_cycles = len(full)

        # Mỗi engine bắt đầu experiment ở một
        # observation point khác nhau.
        fraction = test_rng.uniform(
            config.test_fraction_min,
            config.test_fraction_max,
        )

        end = max(
            1,
            int(n_cycles * fraction),
        )

        # Không expose failure ngay từ initial state.
        end = min(
            end,
            n_cycles - 1,
        )

        # Phần data hiện đang được quan sát.
        test_data[engine_id] = full[:end].copy()

        # Lưu lại current observation point.
        # Simulator sẽ dùng nó để biết cần reveal
        # từ cycle nào tiếp theo.
        test_observation_points[engine_id] = end

    # --------------------------------------------------------
    # 5. Return tất cả những gì notebook cần
    # --------------------------------------------------------

    return {
        "df": df,

        # FULL trajectories.
        # Giữ nguyên để simulator và evaluation sử dụng.
        # Không đưa trực tiếp vào model.
        "engine_data": engine_data,

        # 70% development
        "train_data": train_data,
        "train_ids": train_ids,
        "train_metadata": train_metadata,

        # 30% test
        # Chỉ chứa phần đã được observe tại thời điểm bắt đầu.
        "test_data": test_data,
        "test_ids": test_ids,

        # Cycle hiện tại mà mỗi test engine đã được observe.
        "test_observation_points": test_observation_points,

        # Sensors đang sử dụng
        "sensors": SENSORS,

        # Config
        "config": config,
    }
