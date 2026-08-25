"""
data.py

Chuẩn bị C-MAPSS TRAIN theo kiến trúc phân tầng 2 cấp (2-Tier Split Architecture):

TẬP DỮ LIỆU GỐC (100 Động cơ)
    + Split Dataset (70% TRAIN / 30% TEST)
        + TRAIN (70 Động cơ để học)
            Kịch bản bảo trì: 70% trong TRAIN là TBM / 30% trong TRAIN là CBM
            + TBM: Bảo trì định kỳ, cắt ngắn Cycle 130 - 135
            + CBM: Bảo trì dự đoán, cắt khi RUL còn 15 - 35
        + TEST (30 Động cơ để thử nghiệm)
            Mô phỏng máy đang chạy (Stream từ 15% - 65%)

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
    random_seed: int = 42

    # --- TẦNG 1: Tỉ lệ phân chia Train / Test ---
    train_ratio: float = 0.70

    # --- TẦNG 2: Kịch bản bảo trì trong tập Train ---
    tbm_ratio: float = 0.70  # 70% của Train là TBM (~49 máy), 30% là CBM (~21 máy)

    # 1. Kịch bản TBM (Time-Based Maintenance - Bảo trì định kỳ)
    tbm_cycle_min: int = 130
    tbm_cycle_max: int = 150

    # 2. Kịch bản CBM (Condition-Based Maintenance - Bảo trì dự đoán)
    cbm_rul_min: int = 15
    cbm_rul_max: int = 35

    # --- TẬP TEST: Mô phỏng máy đang chạy ---
    test_fraction_min: float = 0.15
    test_fraction_max: float = 0.65


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
    # 2. TẦNG 1: Split ENGINE 70% Train / 30% Test
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
    # 3. TẦNG 2: Kịch bản bảo trì cho tập Train
    #    - 70% TBM (Bảo trì định kỳ: cắt cycle 130-150)
    #    - 30% CBM (Bảo trì dự đoán: cắt khi RUL còn 15-35)
    # --------------------------------------------------------

    train_ids_shuffled = np.array(train_ids).copy()
    rng.shuffle(train_ids_shuffled)

    n_tbm = int(len(train_ids) * config.tbm_ratio)
    tbm_ids = sorted(train_ids_shuffled[:n_tbm].tolist())
    cbm_ids = sorted(train_ids_shuffled[n_tbm:].tolist())

    train_data = {}
    train_metadata = {}

    # 3.1. Xử lý nhóm TBM (Time-Based Maintenance)
    for engine_id in tbm_ids:
        full = engine_data[engine_id]
        n_cycles = len(full)

        target_cycle = rng.integers(
            config.tbm_cycle_min,
            config.tbm_cycle_max + 1,
        )

        end = min(target_cycle, n_cycles - 1)
        end = max(1, end)

        train_data[engine_id] = full[:end]
        train_metadata[engine_id] = {
            "maintenance_type": "TBM",
            "is_extended": False,  # Censored / Normal (chưa tới failure)
            "observed_cycles": end,
            "actual_rul_at_stop": n_cycles - end,
            "total_cycles": n_cycles,
        }

    # 3.2. Xử lý nhóm CBM (Condition-Based Maintenance)
    for engine_id in cbm_ids:
        full = engine_data[engine_id]
        n_cycles = len(full)

        target_rul = rng.integers(
            config.cbm_rul_min,
            config.cbm_rul_max + 1,
        )

        end = max(1, n_cycles - target_rul)
        end = min(end, n_cycles - 1)

        train_data[engine_id] = full[:end]
        train_metadata[engine_id] = {
            "maintenance_type": "CBM",
            "is_extended": True,  # Degraded / Event (quan sát đến khi phát hiện dấu hiệu suy thoái)
            "observed_cycles": end,
            "actual_rul_at_stop": n_cycles - end,
            "total_cycles": n_cycles,
        }

    # --------------------------------------------------------
    # 4. TẬP TEST: Initial Observation (Mô phỏng máy đang chạy: 15% - 65%)
    # Đây chỉ là trạng thái quan sát tại thời điểm
    # experiment bắt đầu.
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

        # 70% development (Train)
        "train_data": train_data,
        "train_ids": train_ids,
        "tbm_ids": tbm_ids,
        "cbm_ids": cbm_ids,
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
