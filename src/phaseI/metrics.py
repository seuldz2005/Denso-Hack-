"""
metrics.py -- đo lường và kiểm chứng cho Phase I.

Ba việc tách bạch:
  1. Tính reconstruction error thô (per-window, per-sensor-aware).
  2. Suy ra DP (Degradation Point) từ chuỗi error, với quy tắc "N window
     liên tiếp vượt ngưỡng mới tính" đã thống nhất, tránh báo động giả vì
     nhiễu một điểm.
  3. Kiểm chứng bằng Spearman correlation với RUL thật -- CHỈ dùng được
     trên C-MAPSS (có ground truth), không dùng được trên dữ liệu DENSO
     thật (xem lại: Silhouette Score mới là công cụ giám sát không cần
     nhãn cho giai đoạn triển khai thật).
"""

import numpy as np
import torch


@torch.no_grad()
def reconstruction_error_per_window(model, windows: torch.Tensor,
                                     condition: torch.Tensor | None = None,
                                     per_sensor_normalize: np.ndarray | None = None) -> np.ndarray:
    """
    windows: (n_windows, window_len, n_sensors)
    per_sensor_normalize: std của error từng sensor (tính từ vùng khỏe mạnh),
        dùng để chuẩn hóa error trước khi cộng gộp -- tránh một vài sensor
        phương sai cao lấn át toàn bộ error tổng, đúng rủi ro đã cảnh báo
        khi thiết kế pipeline AE.

    Trả về (n_windows,) -- một số error tổng hợp cho mỗi window.
    """
    model.eval()
    x_hat, _ = model(windows, condition)
    err = (x_hat - windows) ** 2                     # (n_windows, window_len, n_sensors)
    err = err.mean(dim=1)                              # gộp theo thời gian -> (n_windows, n_sensors)
    err = err.cpu().numpy()

    if per_sensor_normalize is not None:
        err = err / np.where(per_sensor_normalize == 0, 1.0, per_sensor_normalize)

    return err.mean(axis=1)                             # gộp theo sensor -> (n_windows,)


def fit_threshold(healthy_errors: np.ndarray, percentile: float = 95.0) -> float:
    """Ngưỡng lấy từ chính phân phối error trên vùng khỏe mạnh (không phải
    một số tùy ý) -- percentile 95 nghĩa là chấp nhận 5% false positive
    ngay trên chính dữ liệu train, một điểm khởi đầu hợp lý để tinh chỉnh
    sau khi có dữ liệu thật."""
    return float(np.percentile(healthy_errors, percentile))


def find_degradation_point(error_sequence: np.ndarray, threshold: float,
                            n_consecutive: int = 3) -> int | None:
    """
    error_sequence: (T,) đã làm mượt (nên EWMA/moving-average trước khi
    gọi hàm này -- xem smooth_errors bên dưới).

    Trả về index đầu tiên mà error vượt ngưỡng LIÊN TỤC ít nhất
    n_consecutive lần -- tránh gán DP chỉ vì một điểm nhiễu tức thời.
    Trả về None nếu chưa bao giờ đạt điều kiện này (engine vẫn khỏe mạnh
    trong toàn bộ đoạn quan sát được).
    """
    above = error_sequence > threshold
    run_length = 0
    for i, flag in enumerate(above):
        run_length = run_length + 1 if flag else 0
        if run_length >= n_consecutive:
            return i - n_consecutive + 1
    return None


def smooth_errors(error_sequence: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """EWMA đơn giản, không phụ thuộc thư viện ngoài."""
    smoothed = np.empty_like(error_sequence)
    smoothed[0] = error_sequence[0]
    for i in range(1, len(error_sequence)):
        smoothed[i] = alpha * error_sequence[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def spearman_validation(errors: np.ndarray, true_rul: np.ndarray) -> float:
    """
    CHỈ dùng trên C-MAPSS (hoặc bất kỳ dataset nào có RUL thật). Trả về hệ
    số tương quan Spearman -- kỳ vọng ÂM MẠNH (error cao khi RUL thấp).
    Đây là cổng kiểm tra đầu tiên, bắt buộc phải làm trước khi tin tưởng
    bất kỳ bước nào sau đó trong pipeline AE.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(errors, true_rul)
    return float(corr)
