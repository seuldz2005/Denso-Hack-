"""
model.py -- Conv1D Autoencoder cho Phase I.

Kiến trúc nhẹ theo đúng lựa chọn đã chốt (không dùng LSTM-AE): 2 lớp
Conv1d nén chiều thời gian ở encoder, đối xứng ConvTranspose1d ở decoder.

`condition_dim` là CHỖ CẮM SẴN cho Conditional AE -- mặc định = 0 (bản
base, KHÔNG dùng). Chỉ tăng lên > 0 sau khi bài test FD002/FD004-style xác
nhận thực sự cần (xem lại phần thảo luận operating-condition confound).
Không cần sửa lại kiến trúc khi nâng cấp -- chỉ truyền condition vào lúc
gọi forward().
"""

import torch
import torch.nn as nn


class Conv1DAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, window_len: int, latent_dim: int = 16,
                 condition_dim: int = 0, hidden_channels: tuple[int, int] = (16, 8)):
        super().__init__()
        self.n_sensors = n_sensors
        self.window_len = window_len
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        c1, c2 = hidden_channels

        # --- Encoder: (batch, n_sensors, window_len) -> latent ---
        self.enc_conv = nn.Sequential(
            nn.Conv1d(n_sensors, c1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self._flat_len = self._infer_flat_len(window_len)
        self.enc_fc = nn.Linear(c2 * self._flat_len + condition_dim, latent_dim)

        # --- Decoder: latent -> (batch, n_sensors, window_len) ---
        self.dec_fc = nn.Linear(latent_dim + condition_dim, c2 * self._flat_len)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(c2, c1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(c1, n_sensors, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
        self._c2 = c2

    def _infer_flat_len(self, window_len: int) -> int:
        # hai lớp stride=2 -> giảm chiều thời gian còn khoảng window_len/4
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_sensors, window_len)
            out = self.enc_conv(dummy)
        return out.shape[-1]

    def encode(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, window_len, n_sensors) -> latent: (batch, latent_dim)"""
        h = self.enc_conv(x.transpose(1, 2))          # (batch, c2, flat_len)
        h = h.flatten(start_dim=1)                      # (batch, c2*flat_len)
        if condition is not None:
            h = torch.cat([h, condition], dim=-1)
        return self.enc_fc(h)

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """z: (batch, latent_dim) -> reconstruction: (batch, window_len, n_sensors)"""
        h = z if condition is None else torch.cat([z, condition], dim=-1)
        h = self.dec_fc(h)
        h = h.view(-1, self._c2, self._flat_len)
        out = self.dec_conv(h)                          # (batch, n_sensors, ~window_len)
        out = out[:, :, :self.window_len]                # cắt đúng độ dài nếu lệch do làm tròn
        return out.transpose(1, 2)                        # (batch, window_len, n_sensors)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None):
        """Trả về (reconstruction, latent) -- luôn trả cả hai, vì train.py
        cần reconstruction để tính loss, còn latent chính là thứ Phase II
        sẽ tiêu thụ sau này."""
        z = self.encode(x, condition)
        x_hat = self.decode(z, condition)
        return x_hat, z
