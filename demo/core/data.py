"""
data.py

Prepare C-MAPSS TRAIN for a realistic degradation simulation.

Design
------
Original C-MAPSS:
    engine trajectory -> run-to-failure

Our simulation:
    1. Split engines into TRAIN / TEST.
    2. TRAIN:
        - mostly normal observed trajectories
        - a small number of trajectories contain rare abnormal patterns
        - trajectories are NEVER extended beyond their original length
    3. TEST:
        - only an initial fraction of each trajectory is observable
        - the remaining trajectory stays hidden for realtime simulation
    4. Keep original full trajectories untouched for evaluation.

Important semantic separation
-----------------------------
observation cutoff
    != degradation onset
    != failure event
    != rare abnormal event
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SplitConfig:

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data_path: str = "demo/data/train_FD002.txt"
    random_seed: int = 42

    # --------------------------------------------------------
    # TRAIN / TEST split
    # --------------------------------------------------------

    train_ratio: float = 0.70

    # --------------------------------------------------------
    # TRAIN observation
    #
    # This is NOT called "degradation onset".
    # It is only a reference observation boundary based on
    # the C-MAPSS simulation assumption.
    # --------------------------------------------------------

    observation_cycle_min: int = 130
    observation_cycle_max: int = 135

    # --------------------------------------------------------
    # Rare abnormal trajectory
    # --------------------------------------------------------

    rare_fault_probability: float = 0.20

    # Maximum duration of a synthetic abnormal segment
    rare_fault_max_duration: int = 10

    # Maximum number of cycles before observation cutoff
    # where the abnormal segment may begin.
    rare_fault_max_start_before_cutoff: int = 20

    # Magnitude of synthetic perturbation
    rare_fault_magnitude_min: float = 0.03
    rare_fault_magnitude_max: float = 0.10

    # Available synthetic rare-fault patterns
    rare_fault_types: tuple[str, ...] = (
        "plateau",
        "drop",
        "drift",
    )

    # --------------------------------------------------------
    # TEST
    #
    # Initial amount of data available to the model.
    # --------------------------------------------------------

    test_fraction_min: float = 0.10
    test_fraction_max: float = 0.30


# ============================================================
# OPERATIONAL SETTINGS & SENSOR DEFINITIONS
# ============================================================

OP_SETTINGS = [
    "op1",
    "op2",
    "op3",
]

ALL_SENSORS = [
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nfdmd",
    "PCNfRdmd",
    "W31",
    "W32",
]

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

DEFAULT_FAULT_METADATA = {
    "has_rare_fault": False,
    "fault_type": None,
    "fault_sensor": None,
    "fault_sensor_index": None,
    "fault_start": None,
    "fault_end": None,
    "fault_duration": 0,
    "fault_magnitude": 0.0,
}


# ============================================================
# LOAD C-MAPSS
# ============================================================

def load_cmapss_train(config: SplitConfig):
    """
    Load C-MAPSS TRAIN.

    Returns
    -------
    df:
        Original dataframe containing the selected sensors and operational settings.

    engine_data:
        Dictionary {engine_id: ndarray(n_cycles, n_sensors)}
        These trajectories are NEVER modified.

    op_data:
        Dictionary {engine_id: ndarray(n_cycles, 3)}
        Operational settings (op1, op2, op3) for each cycle.
    """

    columns = [
        "unit_number",
        "cycles",
    ] + OP_SETTINGS + ALL_SENSORS

    df = pd.read_csv(
        config.data_path,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    df = df[
        ["unit_number", "cycles"] + OP_SETTINGS + SENSORS
    ].copy()

    engine_data = {}
    op_data = {}

    for engine_id, group in df.groupby("unit_number"):

        group = group.sort_values("cycles")

        engine_data[int(engine_id)] = (
            group[SENSORS]
            .to_numpy(dtype=np.float32)
        )

        op_data[int(engine_id)] = (
            group[OP_SETTINGS]
            .to_numpy(dtype=np.float32)
        )

    return df, engine_data, op_data


# ============================================================
# OBSERVATION CUTOFF
# ============================================================

def sample_observation_cutoff(
    n_cycles: int,
    config: SplitConfig,
    rng: np.random.Generator,
) -> int:
    """
    Sample the observation cutoff for a training engine.

    The cutoff is a simulation boundary.

    It is NOT treated as:
        - degradation onset
        - failure time
        - ground-truth event

    If an engine is shorter than the requested reference region,
    its actual trajectory length is used safely.
    """

    target_cycle = rng.integers(
        config.observation_cycle_min,
        config.observation_cycle_max + 1,
    )

    cutoff = min(
        target_cycle,
        n_cycles,
    )

    return max(1, cutoff)


# ============================================================
# SENSOR NORMALIZATION FOR FAULT INJECTION (CHỈ DÙNG CHO NHIỄU)
# ============================================================

def _sensor_scale(
    trajectory: np.ndarray,
    sensor_index: int,
) -> float:
    """
    Ước lượng độ rung lắc tự nhiên (nhiễu) giữa 2 chu kỳ liền kề.
    Dùng để tạo nhiễu trắng cho Plateau.
    """
    values = trajectory[:, sensor_index]
    if len(values) < 3:
        return 1.0

    differences = np.diff(values)
    scale = np.median(np.abs(differences))
    if scale <= 1e-8:
        scale = np.std(differences)
    if scale <= 1e-8:
        scale = 1.0
    return float(scale)


# ============================================================
# RARE FAULT: PLATEAU (KẸT GIÁ TRỊ)
# ============================================================

def inject_plateau(
    trajectory: np.ndarray,
    start: int,
    end: int,
    sensor_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Tạo lỗi kẹt tín hiệu (Plateau) tại giá trị start + thêm nhiễu nhẹ.
    """
    result = trajectory.copy()
    values = result[:, sensor_index]
    scale = _sensor_scale(trajectory, sensor_index)

    plateau_value = values[start]
    noise = rng.normal(loc=0.0, scale=scale, size=end - start)
    result[start:end, sensor_index] = plateau_value + noise
    return result


# ============================================================
# RARE FAULT: DROP (SỤT BẬC THANG TẠM THỜI)
# ============================================================

def inject_drop(
    trajectory: np.ndarray,
    start: int,
    end: int,
    sensor_index: int,
    magnitude: float,
) -> np.ndarray:
    """
    Sụt giảm tức thì một lượng (3% - 10% giá trị cảm biến) trong [start:end].
    """
    result = trajectory.copy()
    values = trajectory[:, sensor_index]

    # Tính offset theo 3% - 10% giá trị thực của cảm biến
    base_val = np.abs(values[start]) if np.abs(values[start]) > 1e-4 else 1.0
    offset = magnitude * base_val

    duration = end - start
    if duration <= 0:
        return result

    result[start:end, sensor_index] -= offset
    return result


# ============================================================
# RARE FAULT: DRIFT (TRÔI DỐC TUYẾN TÍNH) - ĐÃ ĐỒNG BỘ
# ============================================================

def inject_drift(
    trajectory: np.ndarray,
    start: int,
    end: int,
    sensor_index: int,
    magnitude: float,
) -> np.ndarray:
    """
    Trôi dốc tuyến tính từ 0 đến -offset (3% - 10% giá trị cảm biến).
    """
    result = trajectory.copy()
    values = trajectory[:, sensor_index]

    # ĐỒNG BỘ: Tính offset theo giá trị thực của cảm biến
    base_val = np.abs(values[start]) if np.abs(values[start]) > 1e-4 else 1.0
    offset = magnitude * base_val

    duration = end - start
    if duration <= 0:
        return result

    drift = np.linspace(0.0, -offset, duration)
    result[start:end, sensor_index] += drift
    return result


# ============================================================
# RARE FAULT GENERATOR (ĐIỀU PHỐI TIÊM LỖI)
# ============================================================

def inject_rare_fault(
    trajectory: np.ndarray,
    cutoff: int,
    config: SplitConfig,
    rng: np.random.Generator,
):
    """
    Điều phối tiêm ngẫu nhiên 1 lỗi vào 1 cảm biến trước điểm cutoff.
    """
    result = trajectory.copy()
    n_sensors = result.shape[1]

    max_duration = min(config.rare_fault_max_duration, cutoff)

    # Trả về metadata mặc định an toàn nếu cutoff quá ngắn
    if max_duration < 2:
        return result, DEFAULT_FAULT_METADATA.copy()

    fault_type = rng.choice(config.rare_fault_types)
    duration = int(rng.integers(2, max_duration + 1))

    # FIX LOGIC VỊ TRÍ: Khớp chuẩn với rare_fault_max_start_before_cutoff
    earliest_start = max(0, cutoff - config.rare_fault_max_start_before_cutoff)
    latest_start = max(earliest_start, cutoff - duration)
    start = int(rng.integers(earliest_start, latest_start + 1))
    end = start + duration

    sensor_index = int(rng.integers(0, n_sensors))
    magnitude = float(rng.uniform(config.rare_fault_magnitude_min, config.rare_fault_magnitude_max))

    if fault_type == "plateau":
        result = inject_plateau(result, start, end, sensor_index, rng)
        # Với plateau, biên độ thực chất là độ lệch chuẩn nhiễu
        logged_magnitude = 0.0
    elif fault_type == "drop":
        result = inject_drop(result, start, end, sensor_index, magnitude)
        logged_magnitude = magnitude
    elif fault_type == "drift":
        result = inject_drift(result, start, end, sensor_index, magnitude)
        logged_magnitude = magnitude
    else:
        raise ValueError(f"Unknown rare fault type: {fault_type}")

    metadata = {
        "has_rare_fault": True,
        "fault_type": fault_type,
        "fault_sensor": SENSORS[sensor_index],
        "fault_sensor_index": sensor_index,
        "fault_start": start,
        "fault_end": end,
        "fault_duration": duration,
        "fault_magnitude": logged_magnitude,
    }

    return result, metadata

# ============================================================
# PREPARE TRAIN DATA
# ============================================================

def prepare_train_data(
    train_ids,
    engine_data,
    op_data,
    config: SplitConfig,
    rng: np.random.Generator,
):
    """
    Prepare observed training trajectories.

    Most engines:
        normal observation

    A small fraction:
        normal observation + rare abnormal pattern

    No trajectory is extended beyond its original length.
    """

    train_data = {}
    train_op_data = {}
    train_metadata = {}

    for engine_id in train_ids:

        full = engine_data[engine_id]
        full_op = op_data[engine_id]

        n_cycles = len(full)

        # ----------------------------------------------------
        # Observation cutoff
        # ----------------------------------------------------

        cutoff = sample_observation_cutoff(
            n_cycles,
            config,
            rng,
        )

        observed = full[:cutoff].copy()
        observed_op = full_op[:cutoff].copy()

        # ----------------------------------------------------
        # Rare fault decision
        # ----------------------------------------------------

        inject_fault = (
            rng.random()
            < config.rare_fault_probability
        )

        if inject_fault:

            observed, fault_metadata = inject_rare_fault(
                trajectory=observed,
                cutoff=len(observed),
                config=config,
                rng=rng,
            )

        else:

            fault_metadata = {
                "has_rare_fault": False,
                "fault_type": None,
                "fault_sensor": None,
                "fault_sensor_index": None,
                "fault_start": None,
                "fault_end": None,
                "fault_duration": 0,
                "fault_magnitude": 0.0,
            }

        # ----------------------------------------------------
        # Save observed trajectory & operational settings
        # ----------------------------------------------------

        train_data[engine_id] = observed
        train_op_data[engine_id] = observed_op

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        train_metadata[engine_id] = {
            "observed_cycles": len(observed),
            "actual_rul_at_stop": n_cycles - len(observed),
            "total_cycles": n_cycles,

            # Observation ends before original failure.
            "is_censored": len(observed) < n_cycles,

            **fault_metadata,
        }

    return train_data, train_op_data, train_metadata


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(
    test_ids,
    engine_data,
    op_data,
    config: SplitConfig,
):
    """
    Prepare initial observations for realtime testing.

    Only a small fraction of the trajectory is exposed.

    The complete original trajectory remains in engine_data.
    """

    test_rng = np.random.default_rng(
        config.random_seed + 1
    )

    test_data = {}
    test_op_data = {}
    test_observation_points = {}

    for engine_id in test_ids:

        full = engine_data[engine_id]
        full_op = op_data[engine_id]

        n_cycles = len(full)

        fraction = test_rng.uniform(
            config.test_fraction_min,
            config.test_fraction_max,
        )

        # ----------------------------------------------------
        # ceil instead of int()
        #
        # This prevents the actual observation fraction from
        # systematically falling below the requested minimum.
        # ----------------------------------------------------

        end = int(
            np.ceil(
                n_cycles * fraction
            )
        )

        end = max(1, end)

        # Do not expose the original failure at initialization.
        end = min(
            end,
            n_cycles - 1,
        )

        test_data[engine_id] = (
            full[:end].copy()
        )

        test_op_data[engine_id] = (
            full_op[:end].copy()
        )

        test_observation_points[engine_id] = end

    return (
        test_data,
        test_op_data,
        test_observation_points,
    )


# ============================================================
# MAIN SPLIT
# ============================================================

def prepare_cmapss_split(
    config: SplitConfig | None = None,
):
    """
    Main entry point.

    Returns
    -------
    dict
        {
            "df": original dataframe,
            "engine_data": original full trajectories (sensors),
            "op_data": original full operational settings,

            "train_data": observed training trajectories,
            "train_op_data": observed training operational settings,
            "train_ids": training engine IDs,
            "train_metadata": metadata,

            "test_data": initial test observations,
            "test_op_data": initial test operational settings,
            "test_ids": testing engine IDs,
            "test_observation_points": current observation points,

            "sensors": selected sensor names,
            "op_settings": operational setting names,
            "config": configuration,
        }
    """

    if config is None:
        config = SplitConfig()

    # ========================================================
    # 1. LOAD ORIGINAL DATA
    # ========================================================

    df, engine_data, op_data = load_cmapss_train(
        config
    )

    rng = np.random.default_rng(
        config.random_seed
    )

    # ========================================================
    # 2. ENGINE-LEVEL TRAIN / TEST SPLIT
    # ========================================================

    engine_ids = np.array(
        sorted(engine_data.keys())
    )

    rng.shuffle(engine_ids)

    n_train = int(
        len(engine_ids)
        * config.train_ratio
    )

    train_ids = sorted(
        engine_ids[:n_train].tolist()
    )

    test_ids = sorted(
        engine_ids[n_train:].tolist()
    )

    # ========================================================
    # 3. PREPARE TRAIN
    # ========================================================

    train_data, train_op_data, train_metadata = prepare_train_data(
        train_ids=train_ids,
        engine_data=engine_data,
        op_data=op_data,
        config=config,
        rng=rng,
    )

    # ========================================================
    # 4. PREPARE TEST
    # ========================================================

    (
        test_data,
        test_op_data,
        test_observation_points,
    ) = prepare_test_data(
        test_ids=test_ids,
        engine_data=engine_data,
        op_data=op_data,
        config=config,
    )

    # ========================================================
    # 5. RETURN
    # ========================================================

    return {
        # ----------------------------------------------------
        # Original data
        # ----------------------------------------------------

        "df": df,

        "engine_data": engine_data,
        "op_data": op_data,

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        "train_data": train_data,
        "train_op_data": train_op_data,

        "train_ids": train_ids,

        "train_metadata": train_metadata,

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        "test_data": test_data,
        "test_op_data": test_op_data,

        "test_ids": test_ids,

        "test_observation_points": (
            test_observation_points
        ),

        # ----------------------------------------------------
        # Sensors & Operational Settings
        # ----------------------------------------------------

        "sensors": SENSORS,
        "op_settings": OP_SETTINGS,

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        "config": config,
    }
