"""
data.py -- Phase II data utilities.

Chịu trách nhiệm cho đúng hai việc, tách biệt rõ khỏi model/loss:
  1. Chuyển (event_cycle hoặc censor_cycle) của từng engine thành nhãn hazard
     theo bin thời gian rời rạc: y[t] = 1 tại đúng bin xảy ra sự kiện, 0 ở
     mọi bin trước đó. Engine bị censor chỉ có toàn 0, dừng đúng chỗ quan sát
     kết thúc -- KHÔNG suy đoán gì cho các bin sau đó.
  2. Gói các chuỗi latent (độ dài khác nhau theo từng engine, đúng vì các
     engine không đồng bộ) thành batch có padding, kèm mask để loss/metric
     không bao giờ tính nhầm trên phần padding giả.

Không có gì trong file này phụ thuộc vào kiến trúc model cụ thể (GRU/LSTM) --
đây là lý do nó tách file riêng, đúng chuẩn "modular, no redundant coupling"
đã thống nhất từ đầu.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class EngineRecord:
    """Một engine (hoặc một renewal-cycle segment, nếu đã cắt theo bảo dưỡng)."""
    engine_id: str
    latent_seq: np.ndarray   # shape (T, latent_dim) -- output encoder Phase I
    n_bins: int              # số bin thời gian engine này thực sự có dữ liệu
    event_bin: int | None    # bin xảy ra sự kiện (0-indexed); None nếu censored


def build_hazard_target(n_bins: int, event_bin: int | None) -> np.ndarray:
    """
    Trả về vector nhãn y có độ dài n_bins.

    - Nếu event_bin is None (censored): toàn bộ y = 0. Đây là điểm cốt lõi
      khiến pipeline dùng được dữ liệu censored -- không cần biết "sẽ hỏng
      lúc nào", chỉ cần biết "đến giờ chưa hỏng".
    - Nếu có event_bin: y = 0 ở mọi bin trước đó, y = 1 đúng tại event_bin.
      Các bin SAU event_bin không tồn tại trong sequence (engine dừng quan
      sát ngay khi sự kiện xảy ra) -- không được để lọt bin nào sau đó vào
      loss, nếu không sẽ tính loss trên dữ liệu không tồn tại.
    """
    y = np.zeros(n_bins, dtype=np.float32)
    if event_bin is not None:
        assert 0 <= event_bin < n_bins, "event_bin phải nằm trong [0, n_bins)"
        y[event_bin] = 1.0
    return y


def pad_batch(records: list[EngineRecord], latent_dim: int):
    """
    Gói một list EngineRecord (độ dài khác nhau) thành ba tensor cùng shape
    (batch, T_max, ...), kèm mask (batch, T_max) đánh dấu bin nào là thật.

    Trả về numpy arrays thuần (không phụ thuộc torch) -- converter sang
    torch.Tensor để ở ngoài file này, giữ data.py độc lập framework.
    """
    batch_size = len(records)
    t_max = max(r.n_bins for r in records)

    latents = np.zeros((batch_size, t_max, latent_dim), dtype=np.float32)
    targets = np.zeros((batch_size, t_max), dtype=np.float32)
    mask = np.zeros((batch_size, t_max), dtype=np.float32)

    for i, r in enumerate(records):
        t = r.n_bins
        latents[i, :t, :] = r.latent_seq[:t]
        targets[i, :t] = build_hazard_target(t, r.event_bin)
        mask[i, :t] = 1.0

    return latents, targets, mask
