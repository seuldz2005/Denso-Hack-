"""
loss.py -- reconstruction loss cho Phase I.

Tách file riêng dù chỉ có MSE, giữ đúng quy ước "mỗi trách nhiệm một file"
như phase2/ -- nếu sau này thêm KL-divergence (nâng cấp lên VAE) hoặc
domain-adversarial term, chỉ sửa file này, không đụng model.py/train.py.
"""

import torch


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE trung bình trên mọi phần tử (batch, window_len, n_sensors)."""
    return torch.mean((x_hat - x) ** 2)
