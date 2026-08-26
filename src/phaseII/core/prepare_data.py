"""
prepare_data.py -- Chuẩn bị dữ liệu cho Phase II.

Input:
    latent_data    : latent sequences từ encoder Phase I đã freeze.
    train_metadata : metadata từ prepare_cmapss_split(), chứa is_extended.

Nhiệm vụ:
    1. Xác định event_bin cho từng engine dựa trên is_extended:
       - Extended engine: event_bin = n_bins - 1 (bin cuối — engine được
         quan sát đến khi có dấu hiệu suy thoái, bin đó chính là event).
       - Normal engine: event_bin = None (censored — chưa vượt qua elbow).
    2. Build EngineRecord từ latent_seq + event_bin.
    3. Stratified split train / val giữ tỉ lệ extended/normal.

Phase II KHÔNG train lại encoder Phase I.
"""

import numpy as np

from src.phaseII.core.data import EngineRecord


# ============================================================
# BƯỚC 2 — Build EngineRecord
# ============================================================

def build_engine_records(
    latent_data: dict[int, np.ndarray],
    train_metadata: dict[int, dict],
) -> list[EngineRecord]:
    """
    Build một EngineRecord cho mỗi engine.

    Với mỗi engine:
    - n_bins: số latent timestep thực tế quan sát được.
    - event_observed:
        True  -> engine extended, event được quan sát.
        False -> engine normal, bị right-censored.
    - event_bin:
        n_bins - 1 nếu event được quan sát.
        None nếu bị right-censored.

    Không tạo hazard target [0, 0, ..., 1] ở đây.
    Target sẽ được tạo sau trong phase2/data.py.
    """

    records = []

    for engine_id, latent_seq in latent_data.items():

        # --------------------------------------------------
        # 1. Lấy thông tin extended / normal
        # --------------------------------------------------
        is_extended = train_metadata[engine_id]["is_extended"]

        # --------------------------------------------------
        # 2. Độ dài thực tế của latent sequence
        # --------------------------------------------------
        n_bins = latent_seq.shape[0]

        # --------------------------------------------------
        # 3. Xác định event
        # --------------------------------------------------
        if is_extended:
            event_observed = True
            event_bin = n_bins - 1
        else:
            event_observed = False
            event_bin = None

        # --------------------------------------------------
        # 4. Tạo EngineRecord
        # --------------------------------------------------
        record = EngineRecord(
            engine_id=str(engine_id),
            latent_seq=latent_seq.astype(np.float32),
            n_bins=n_bins,
            event_observed=event_observed,
            event_bin=event_bin,
        )

        records.append(record)

    return records


# ============================================================
# BƯỚC 3 — Stratified split
# ============================================================

def split_train_val(
    records: list[EngineRecord],
    train_metadata: dict[int, dict],
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    """
    Stratified split giữ tỉ lệ extended/normal trong cả train lẫn val.

    Extended engines (có event_bin) là positive examples duy nhất —
    phải đảm bảo cả hai split đều có, không để toàn bộ lọt vào một bên.
    """
    rng = np.random.default_rng(random_seed)

    normal = [
        r for r in records
        if not train_metadata[int(r.engine_id)]["is_extended"]
    ]

    extended = [
        r for r in records
        if train_metadata[int(r.engine_id)]["is_extended"]
    ]

    def split_group(group):
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n_train = int(len(group) * train_ratio)
        return (
            [group[i] for i in indices[:n_train]],
            [group[i] for i in indices[n_train:]],
        )

    train_normal,   val_normal   = split_group(normal)
    train_extended, val_extended = split_group(extended)

    train_records = train_normal + train_extended
    val_records   = val_normal   + val_extended

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    return train_records, val_records


# ============================================================
# ENTRY POINT
# ============================================================

def prepare_training_data(
    latent_data: dict[int, np.ndarray],
    train_metadata: dict[int, dict],
    train_ratio: float = 0.8,
    random_seed: int = 42,
):

    records = build_engine_records(
        latent_data=latent_data,
        train_metadata=train_metadata,
    )

    return split_train_val(
        records=records,
        train_metadata=train_metadata,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )
