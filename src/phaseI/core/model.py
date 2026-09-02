"""
model.py -- "Smart AE": Conv1D Autoencoder theo kiến trúc ĐÃ ĐƠN GIẢN HÓA
của bài báo tạp chí (Bajarunas et al., RESS 2024) -- thay thế bản trước
(dựa theo bài hội nghị PHM 2023, có nhánh dự đoán Ŵ).

THAY ĐỔI SO VỚI BẢN TRƯỚC:
  - Bỏ hẳn nhánh enc_w_head và việc dự đoán Ŵ. Encoder giờ CHỈ output Z.
  - Decoder nhận W THẬT (quan sát được lúc train) ghép với Z, thay vì Ŵ
    dự đoán -- đúng theo Fig. 4 + Eq. (10)-(11) của bài tạp chí.
  - Bỏ hẳn loss dự đoán W (||W - Ŵ||) -- không còn cần thiết, vì decoder
    đã có sẵn W thật để giải thích phần do vận hành gây ra, Z không cần
    "gánh" việc đó nữa (lập luận của tác giả, không cần ép bằng loss phụ).
  - LƯU Ý: vẫn KHÔNG cần W lúc inference thực tế -- vì lúc dùng thật chỉ
    chạy encode(X) để lấy Z, không đụng tới decode() (decoder chỉ dùng
    lúc TRAIN để tính reconstruction loss). W thật vẫn cần lúc TRAIN
    (đưa vào decoder), giống hệt yêu cầu của bản cũ (cần W thật làm
    target cho loss dự đoán) -- không đổi gì về yêu cầu dữ liệu, chỉ
    đơn giản hóa cách dùng W.
"""

import torch
import torch.nn as nn


class SmartConv1DAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, window_len: int, w_dim: int,
                 z_dim: int = 8, hidden_channels: tuple[int, int] = (16, 8)):
        super().__init__()
        self.n_sensors = n_sensors
        self.window_len = window_len
        self.w_dim = w_dim
        self.z_dim = z_dim
        c1, c2 = hidden_channels

        # --- Encoder: (batch, n_sensors, window_len) -> Z ---
        self.enc_conv = nn.Sequential(
            nn.Conv1d(n_sensors, c1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self._flat_len = self._infer_flat_len(window_len)
        flat_dim = c2 * self._flat_len

        # CHỈ CÒN 1 nhánh -- không còn enc_w_head như bản trước.
        self.enc_z_head = nn.Linear(flat_dim, z_dim)

        # --- Decoder: (W thật, Z) -> (batch, n_sensors, window_len) ---
        self.dec_fc = nn.Linear(w_dim + z_dim, flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(c2, c1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(c1, n_sensors, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
        self._c2 = c2

    def _infer_flat_len(self, window_len: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_sensors, window_len)
            out = self.enc_conv(dummy)
        return out.shape[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_len, n_sensors) -> z: (batch, z_dim).
        CHỈ nhận X -- không có W, không có gì khác. Đây là hàm DUY NHẤT
        cần gọi lúc inference thực tế (deployment)."""
        h = self.enc_conv(x.transpose(1, 2))
        h = h.flatten(start_dim=1)
        return self.enc_z_head(h)

    def decode(self, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(w THẬT, z) -> reconstruction: (batch, window_len, n_sensors).
        CHỈ dùng lúc train -- không cần ở deployment."""
        h = torch.cat([w, z], dim=-1)
        h = self.dec_fc(h)
        h = h.view(-1, self._c2, self._flat_len)
        out = self.dec_conv(h)
        out = out[:, :, :self.window_len]
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        x: (batch, window_len, n_sensors) -- input encoder
        w: (batch, w_dim) -- W THẬT tại cycle cuối của window (nhãn,
           KHÔNG phải input encoder -- chỉ đưa thẳng vào decoder)

        Trả về (x_hat, z) -- 2 giá trị, KHÁC bản trước (3 giá trị:
        x_hat, w_hat, z). Mọi nơi gọi model (train.py, metrics.py) cần
        cập nhật lại cách gọi: giờ forward() cần thêm tham số w, và
        unpack chỉ còn 2 giá trị.
        """
        z = self.encode(x)
        x_hat = self.decode(w, z)
        return x_hat, z
