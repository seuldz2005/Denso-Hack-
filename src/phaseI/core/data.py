from dataclasses import dataclass
import numpy as np

@dataclass
class WindowBatch:
    X: np.ndarray            # (n_windows, window_len, n_sensors) -- feed vào network
    W: np.ndarray            # (n_windows, window_len, n_op_conditions) -- feed vào network (nếu bạn muốn W theo window, hoặc W tại timestep cuối window)
    unit_number: np.ndarray  # (n_windows,) -- KHÔNG feed vào network, chỉ để group/debug
    cycle: np.ndarray        # (n_windows,) -- KHÔNG feed vào network, cycle của timestep CUỐI trong mỗi window

def cut_windows(
    X: np.ndarray,       # (n_cycles, n_sensors) của 1 engine, đã chuẩn hóa
    W: np.ndarray,       # (n_cycles, n_op_conditions) cùng engine, đã tách riêng
    unit_number: int,    # id của engine này (hằng số cho cả engine)
    cycle: np.ndarray,   # (n_cycles,) số cycle thật, dùng để đặt nhãn window
    window_len: int,
    stride: int,
) -> WindowBatch:
    n = X.shape[0]
    if n < window_len:
        import warnings
        warnings.warn(f"Engine {unit_number}: {n} cycles < window_len={window_len}, skipped.")
        return WindowBatch(
            X=np.empty((0, window_len, X.shape[1]), dtype=X.dtype),
            W=np.empty((0, window_len, W.shape[1]), dtype=W.dtype),
            unit_number=np.empty((0,), dtype=int),
            cycle=np.empty((0,), dtype=int),
        )

    starts = range(0, n - window_len + 1, stride)
    X_windows = np.stack([X[s:s + window_len] for s in starts], axis=0)
    W_windows = np.stack([W[s:s + window_len] for s in starts], axis=0)
    # Quy ước: nhãn cycle của 1 window = cycle của timestep CUỐI CÙNG trong window,
    # vì đó là "thời điểm hiện tại" mà window đại diện cho khi inference real-time
    # (window nhìn lại quá khứ để mô tả trạng thái tại thời điểm cuối).
    end_cycles = np.array([cycle[s + window_len - 1] for s in starts])
    unit_numbers = np.full(len(starts), unit_number, dtype=int)

    return WindowBatch(X=X_windows, W=W_windows, unit_number=unit_numbers, cycle=end_cycles)
