"""
data.py -- Phase I data utilities.

Ba việc, tách rời rõ ràng, đúng thứ tự phải làm:
  1. Xác định vùng "khỏe mạnh" của mỗi engine (để biết train AE trên đoạn nào).
  2. Tính thống kê chuẩn hóa (mean/std) CHỈ từ vùng khỏe mạnh đó -- tránh
     đúng lỗi rò rỉ dữ liệu đã từng gặp trước đây (chuẩn hóa lẫn cả dữ liệu
     suy thoái vào thống kê "bình thường").
  3. Cắt sliding window từ chuỗi đã chuẩn hóa.

Không có gì trong file này biết đến model -- giữ tách biệt để AE, sau này
nếu đổi sang LSTM-AE/VAE, không phải sửa file này.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class NormalizationStats:
    """Lưu lại để dùng nhất quán cho mọi engine khác, và cho dữ liệu
    inference sau này -- KHÔNG tính lại thống kê mới cho từng engine, luôn
    dùng đúng bộ số này đã fit một lần từ vùng khỏe mạnh của tập tham chiếu.
    """
    mean: np.ndarray   # shape (n_sensors,)
    std: np.ndarray    # shape (n_sensors,)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.where(self.std == 0, 1.0, self.std)

def fit_normalization(healthy_series_list: list[np.ndarray]) -> NormalizationStats:
    """
    healthy_series_list: list các đoạn khỏe mạnh (từ nhiều engine tham
    chiếu, gộp lại) -- KHÔNG fit riêng cho từng engine, fit MỘT LẦN trên
    quần thể, dùng lại cho mọi engine sau đó (đúng nguyên tắc "fit once,
    transform many" đã thống nhất từ đầu).
    """
    stacked = np.concatenate(healthy_series_list, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return NormalizationStats(mean=mean, std=std)


def cut_windows(series: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """
    series: (n_cycles, n_sensors) đã chuẩn hóa.
    Trả về (n_windows, window_len, n_sensors).

    Dùng stride NHỎ (ví dụ 1) khi cắt window để TRAIN AE -- càng nhiều mẫu
    càng tốt cho việc học reconstruction.
    Dùng stride LỚN, canh đúng ranh giới bin (ví dụ mỗi 10 cycle) khi cắt
    window để TRÍCH XUẤT latent cho Phase II -- xem train.py, hàm
    extract_latents_for_engine, đây là nơi khác biệt bin/window được xử lý.
    """
    n = series.shape[0]
    if n < window_len:
        return np.empty((0, window_len, series.shape[1]), dtype=series.dtype)

    starts = range(0, n - window_len + 1, stride)
    windows = np.stack([series[s:s + window_len] for s in starts], axis=0)
    return windows
