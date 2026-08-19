"""
train.py -- huấn luyện AE + trích xuất latent cho Phase II.

Hai hàm chính, đúng ranh giới đã thống nhất với Phase II:
  - run_training(): CHỈ train AE, không biết gì về GRU-hazard.
  - extract_latents_for_engine(): chạy encoder ĐÃ ĐÓNG BĂNG, cắt window
    theo đúng ranh giới BIN (khác stride lúc train), trả về latent_seq
    sẵn sàng nhét thẳng vào phase2.data.EngineRecord.

File này không xác định healthy region, không split train/test và không
fit normalization. Các bước đó được xử lý trước khi gọi các hàm ở đây.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.phaseI.core.data import cut_windows
from src.phaseI.core.model import Conv1DAutoencoder
from src.phaseI.core.loss import reconstruction_loss


def run_training(training_windows: np.ndarray, n_sensors: int, window_len: int,
                 latent_dim: int = 16, n_epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 64, seed: int = 0, device: str = "cpu") -> Conv1DAutoencoder:
    """
    training_windows: (n_windows, window_len, n_sensors) -- các window đã
    được chuẩn bị từ DEVELOPMENT data sau bước split/truncation và
    normalization.

    Trả về model đã train xong -- người gọi tự quyết định .eval() +
    torch.no_grad() / đóng băng tham số ở bước sau, hàm này không tự làm
    thay để giữ trách nhiệm rõ ràng.
    """
    torch.manual_seed(seed)

    x = torch.from_numpy(training_windows).float()
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

    model = Conv1DAutoencoder(n_sensors=n_sensors, window_len=window_len,
                               latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = reconstruction_loss(x_hat, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch:3d} | recon_loss={epoch_loss / n_batches:.6f}")

    return model


@torch.no_grad()
def extract_latents_for_engine(model: Conv1DAutoencoder, series: np.ndarray,
                                window_len: int, bin_stride: int,
                                device: str = "cpu") -> np.ndarray:
    """
    series: (n_cycles, n_sensors) đã chuẩn hóa bằng đúng NormalizationStats
    được fit từ DEVELOPMENT/TRAINING data -- của MỘT engine.

    bin_stride: khoảng cách giữa các bin (ví dụ 10 cycle/bin) -- ĐÂY LÀ
    THAM SỐ QUYẾT ĐỊNH cách lấy các window cho Phase II, khác hẳn stride
    nhỏ dùng lúc train AE.

    Trả về latent_seq: (n_bins, latent_dim) -- đúng format
    EngineRecord.latent_seq mà phase2/data.py cần.
    """
    model.eval()
    windows = cut_windows(series, window_len=window_len, stride=bin_stride)

    if windows.shape[0] == 0:
        return np.empty((0, model.latent_dim), dtype=np.float32)

    x = torch.from_numpy(windows).float().to(device)
    _, z = model(x)

    return z.cpu().numpy()
