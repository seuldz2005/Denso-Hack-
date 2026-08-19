"""
prepare.py -- chuẩn bị dữ liệu cho Phase I.

Nhận training trajectories đã được dataset-specific code chuẩn bị,
sau đó:
  1. fit một bộ NormalizationStats trên toàn bộ development trajectories;
  2. normalize tất cả trajectories bằng cùng bộ statistics;
  3. cắt sliding windows để đưa vào AE.

File này không biết C-MAPSS, DENSO hay bất kỳ dataset cụ thể nào.
"""

import numpy as np

from src.phaseI.core.data import fit_normalization, cut_windows


def prepare_training_data(train_data: dict[int, np.ndarray],
                           window_len: int, stride: int = 1):
    """
    train_data:
        {engine_id: (n_cycles, n_sensors)}

        Là development trajectories đã được split/truncate
        từ dataset-specific preparation.

    Trả về:
        training_windows: (n_windows, window_len, n_sensors)
        stats: NormalizationStats dùng lại cho inference/Phase II
        normalized_data: trajectories sau normalization
    """

    stats = fit_normalization(
        list(train_data.values())
    )

    normalized_data = {
        engine_id: stats.apply(series)
        for engine_id, series in train_data.items()
    }

    windows = []

    for series in normalized_data.values():
        w = cut_windows(
            series,
            window_len=window_len,
            stride=stride,
        )

        if len(w) > 0:
            windows.append(w)

    if not windows:
        n_sensors = next(iter(train_data.values())).shape[1]
        training_windows = np.empty(
            (0, window_len, n_sensors),
            dtype=np.float32,
        )
    else:
        training_windows = np.concatenate(
            windows,
            axis=0,
        )

    return training_windows, stats, normalized_data
