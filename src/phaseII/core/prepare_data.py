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
    event_bin = bin cuối cùng của chính latent_seq engine đó, NẾU là
    extended -- không tính lại từ elbow_cycle cố định.

    Lý do: data.py đã ngẫu nhiên hóa độ dài quan sát cho từng engine rồi
    (qua extended_max_extra). Dùng lại đúng độ dài đó làm event_bin, thay
    vì áp một công thức chung, giữ được đúng sự đa dạng đã thiết kế và
    tránh length trở thành shortcut lộ liễu cho model.

    Về mặt ngữ nghĩa, điều này cũng khớp thực tế hơn: ta không thực sự
    biết chính xác cycle nào là "elbow" -- ta chỉ biết "quan sát dừng lại
    ở đây, và tại đây engine đã qua giai đoạn suy thoái" -- đúng như log
    bảo trì thật chỉ cho biết thời điểm ghi nhận, không cho biết thời
    điểm khởi phát chính xác.
    """
    records = []
    for engine_id, latent_seq in latent_data.items():
        is_extended = train_metadata[engine_id]["is_extended"]
        n_bins = latent_seq.shape[0]
        event_bin = (n_bins - 1) if is_extended else None

        records.append(
            EngineRecord(
                engine_id=str(engine_id),
                latent_seq=latent_seq.astype(np.float32),
                n_bins=n_bins,
                event_bin=event_bin,
            )
        )
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
