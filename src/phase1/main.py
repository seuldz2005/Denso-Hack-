"""
main.py -- entry point cho Phase I. CHỈ CẦN ĐIỀN vào hàm load_engines()
bên dưới bằng code chuẩn bị data bạn đã có sẵn -- mọi phần còn lại đã
sẵn sàng chạy, không cần sửa.
"""

import numpy as np
import torch

from core.windowing import EngineData
from core.train import run_training, extract_latents_for_engine
from core.metrics import z_drift_per_engine, fit_threshold, find_degradation_point, smooth_signal


CONFIG = {
    "window_len": 20,
    "train_stride": 1,
    "bin_stride": 10,
    "z_dim": 4,
    "n_epochs": 50,
    "lr": 1e-3,
    "engines_per_batch": 8,
    "val_fraction": 0.2,
    "patience": 5,
    "seed": 0,
}


def load_engines() -> list[EngineData]:
    """
    ============================ ĐIỀN VÀO ĐÂY ============================
    Dùng code chuẩn bị data bạn đã có sẵn (đọc file, tách X/W, chuẩn
    hóa...), rồi đóng gói kết quả thành list[EngineData], mỗi engine 1
    phần tử:

        EngineData(
            unit_number=...,   # int
            X=...,             # (n_cycles, n_sensors), đã chuẩn hóa
            W=...,             # (n_cycles, w_dim)
            cycle=...,         # (n_cycles,) số cycle thật
        )

    Trả về list chứa TẤT CẢ engine (train + validation gộp chung) --
    việc chia train/val nằm ở hàm split_train_val() bên dưới, không cần
    tự chia ở đây.
    ========================================================================
    """
    raise NotImplementedError("Điền code chuẩn bị data của bạn vào đây")


def split_train_val(engines: list[EngineData], val_fraction: float,
                     seed: int) -> tuple[list[EngineData], list[EngineData]]:
    """Chia theo ENGINE, không theo window -- 1 engine nằm trọn về 1
    bên, tránh leakage."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(engines))
    n_val = int(len(engines) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [engines[i] for i in train_idx], [engines[i] for i in val_idx]


def main():
    cfg = CONFIG

    all_engines = load_engines()
    train_engines, val_engines = split_train_val(all_engines, cfg["val_fraction"], cfg["seed"])
    print(f"{len(train_engines)} engine train, {len(val_engines)} engine validation")

    n_sensors = train_engines[0].X.shape[1]
    w_dim = train_engines[0].W.shape[1]

    model = run_training(
        engines=train_engines, val_engines=val_engines,
        n_sensors=n_sensors, w_dim=w_dim,
        window_len=cfg["window_len"], stride=cfg["train_stride"],
        z_dim=cfg["z_dim"], n_epochs=cfg["n_epochs"], lr=cfg["lr"],
        engines_per_batch=cfg["engines_per_batch"], seed=cfg["seed"],
        patience=cfg["patience"],
    )

    torch.save(model.state_dict(), "smart_ae.pt")
    print("Đã lưu model vào smart_ae.pt")

    # Sanity-check nhanh trên 1 engine validation
    example_engine = val_engines[0]
    latents = extract_latents_for_engine(model, example_engine,
                                          window_len=cfg["window_len"],
                                          bin_stride=cfg["bin_stride"])
    z_seq = latents.X   # LƯU Ý: field X ở đây thực chất là Z, không phải sensor
    drift = z_drift_per_engine(z_seq)
    drift_smooth = smooth_signal(drift)
    threshold = fit_threshold(drift_smooth[:5])
    dp_index = find_degradation_point(drift_smooth, threshold)

    if dp_index is not None:
        print(f"Engine {example_engine.unit_number}: phát hiện suy thoái "
              f"tại cycle {latents.cycle[dp_index]}")
    else:
        print(f"Engine {example_engine.unit_number}: chưa phát hiện suy thoái rõ ràng")


if __name__ == "__main__":
    main()
