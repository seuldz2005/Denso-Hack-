"""
model.py -- GRU-hazard cho Phase II.

CHỈ nhận latent vector (đã qua encoder Phase I dùng chung) làm input --
không bao giờ nhận sensor thô. Nếu bạn thấy model.py cần import gì liên
quan đến sensor/window, đó là dấu hiệu interface với Phase I đã bị vi phạm.

Kiến trúc: GRU (many-to-many) -> Linear -> Sigmoid tại MỖI bước thời gian.
Chọn GRU thay LSTM theo đúng lập luận đã chốt: ít tham số hơn ~25%, phù hợp
dữ liệu ít sự kiện, sequence sau khi cắt theo renewal-cycle không đủ dài để
cần lợi thế cell-state riêng của LSTM.

Nếu sau này muốn thử LSTM để so sánh thực nghiệm (không phải vì "nghe có vẻ
mạnh hơn"), chỉ cần đổi nn.GRU thành nn.LSTM -- interface không đổi.
"""

import torch
import torch.nn as nn


class GRUHazard(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 32, num_layers: int = 1,
                 dropout: float = 0.0, extra_feature_dim: int = 0):
        """
        extra_feature_dim: chỗ cắm các feature phụ OPTIONAL (Label, DP,
        operating condition) đã bàn -- mặc định = 0 (baseline không dùng gì
        thêm). Chỉ tăng số này lên sau khi ablation chứng minh feature đó
        thực sự cải thiện C-index, không thêm mặc định.
        """
        super().__init__()
        input_dim = latent_dim + extra_feature_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, latent_seq: torch.Tensor, lengths: torch.Tensor,
                extra_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        latent_seq: (batch, T_max, latent_dim)
        lengths:    (batch,) độ dài THẬT của từng sequence (trước padding)
        extra_features: (batch, T_max, extra_feature_dim) hoặc None

        Trả về h: (batch, T_max) -- xác suất hazard tại mỗi bin, đã sigmoid.
        Giá trị tại các vị trí padding vẫn được tính (rẻ, không sao) nhưng
        PHẢI bị loại bỏ bằng mask ở bước tính loss/metric, không dùng trực
        tiếp giá trị tại đây.
        """
        x = latent_seq if extra_features is None else torch.cat(
            [latent_seq, extra_features], dim=-1
        )

        # pack_padded_sequence giúp GRU không lãng phí compute trên phần
        # padding, đồng thời tránh việc padding=0 làm nhiễu hidden state
        # của các sequence ngắn hơn t_max.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        h = torch.sigmoid(self.head(out)).squeeze(-1)  # (batch, T_max)
        return h
