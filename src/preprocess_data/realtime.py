"""
realtime.py

Mô phỏng quá trình dữ liệu test được reveal dần theo thời gian.

Input:
    engine_data
        Full trajectory của mỗi engine.
        Chỉ dùng làm ground truth / nguồn dữ liệu ẩn.

    test_data
        Phần trajectory đã được observe ban đầu.

    observation_points
        Cycle hiện tại của mỗi engine.

Mỗi realtime step:
    - Một số engine nhận thêm cycle mới.
    - Một số engine không có measurement mới.
    - Không engine nào được nhìn thấy tương lai.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RealtimeConfig:
    # Xác suất một engine có measurement mới
    update_probability: float = 0.8

    # Mỗi lần update có thể nhận 1..max_new_cycles cycle
    max_new_cycles: int = 1

    # Seed để experiment reproducible
    random_seed: int = 42


def simulate_realtime(
    engine_data: dict[int, np.ndarray],
    test_data: dict[int, np.ndarray],
    observation_points: dict[int, int],
    config: RealtimeConfig = None,
):
    """
    Generator mô phỏng test data được reveal dần theo thời gian.

    Mỗi lần next() trả về snapshot hiện tại của toàn bộ test engines.

    Yields
    ------
    snapshot : dict[int, np.ndarray]
        Trajectory mà hệ thống hiện đang quan sát được.

    observation_points : dict[int, int]
        Cycle mới nhất đã được observe của mỗi engine.
    """

    if config is None:
        config = RealtimeConfig()

    rng = np.random.default_rng(config.random_seed)

    # Copy để không sửa trực tiếp test_data bên ngoài
    current_data = {
        engine_id: series.copy()
        for engine_id, series in test_data.items()
    }

    current_points = observation_points.copy()

    while True:

        updated = False

        for engine_id in current_data:

            full = engine_data[engine_id]
            current = current_points[engine_id]

            # Engine đã hết trajectory
            if current >= len(full):
                continue

            # Engine này chưa nhận measurement mới
            if rng.random() > config.update_probability:
                continue

            # Số cycle mới được reveal
            n_new = rng.integers(
                1,
                config.max_new_cycles + 1,
            )

            new_point = min(
                current + n_new,
                len(full),
            )

            current_data[engine_id] = (
                full[:new_point].copy()
            )

            current_points[engine_id] = new_point

            updated = True

        # Không còn engine nào có thể update
        if not updated:
            if all(
                current_points[eid] >= len(engine_data[eid])
                for eid in current_data
            ):
                break

        yield (
            current_data.copy(),
            current_points.copy(),
        )
