"""
loss.py -- các thành phần loss cho Smart AE, cập nhật theo kiến trúc đơn
giản hóa (bài tạp chí Bajarunas et al., RESS 2024) -- bỏ hẳn
w_prediction_loss so với bản trước.
"""

import torch


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE trung bình trên mọi phần tử (batch, window_len, n_sensors).
    Không đổi so với bản trước."""
    return torch.mean((x_hat - x) ** 2)


def z_smoothness_loss(z_seq: torch.Tensor, same_engine_mask: torch.Tensor) -> torch.Tensor:
    """Không đổi so với bản trước -- xem giải thích chi tiết ở lần viết
    trước, công thức giữ nguyên."""
    diffs = z_seq[1:] - z_seq[:-1]
    sq_diffs = (diffs ** 2).sum(dim=-1)
    if same_engine_mask.sum() == 0:
        return torch.zeros((), device=z_seq.device)
    return sq_diffs[same_engine_mask].mean()


def z_monotonicity_loss(z_seq: torch.Tensor, same_engine_mask: torch.Tensor) -> torch.Tensor:
    """Không đổi so với bản trước -- và như đã xác nhận qua bài báo,
    công thức này gần như y hệt "Negative Gradient constraint" (Eq. 15)
    của bài tạp chí: L_NG = max{0, dZ/dt}."""
    diffs = z_seq[1:] - z_seq[:-1]
    penalty = torch.relu(-diffs).sum(dim=-1)
    if same_engine_mask.sum() == 0:
        return torch.zeros((), device=z_seq.device)
    return penalty[same_engine_mask].mean()


def smart_ae_loss(x_hat: torch.Tensor, x: torch.Tensor,
                   z_seq: torch.Tensor, same_engine_mask: torch.Tensor,
                   lambda_smooth: float = 1.0, lambda_mono: float = 1.0) -> dict[str, torch.Tensor]:
    """
    KHÔNG CÒN nhận w_hat/w_true -- không còn thành phần loss nào liên
    quan tới W (đã bỏ w_prediction_loss). Chỉ còn 3 thành phần: recon +
    smoothness + monotonicity.
    """
    l_recon = reconstruction_loss(x_hat, x)
    l_smooth = z_smoothness_loss(z_seq, same_engine_mask)
    l_mono = z_monotonicity_loss(z_seq, same_engine_mask)
    total = l_recon + lambda_smooth * l_smooth + lambda_mono * l_mono
    return {
        "total": total,
        "recon": l_recon,
        "smooth": l_smooth,
        "mono": l_mono,
    }
