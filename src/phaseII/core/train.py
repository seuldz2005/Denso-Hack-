"""
train.py -- vòng lặp huấn luyện Phase II, khung để bạn cắm dữ liệu thật vào.

Giả định: bạn đã có sẵn, từ Phase I:
    - latent_seq mỗi engine: np.ndarray (T, latent_dim), từ encoder dùng
      chung, KHÔNG train lại encoder ở đây.
    - event_bin mỗi engine: int hoặc None (censored), theo bin thời gian
      đã định nghĩa (xem data.py).

Script này KHÔNG tự đọc C-MAPSS -- bạn cần viết một hàm load_engines()
riêng, trả về list[EngineRecord], rồi gọi run_training() bên dưới.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.phaseII.core.data import EngineRecord, pad_batch
from src.phaseII.core.model import GRUHazard
from src.phaseII.core.loss import combined_loss
from src.phaseII.core.metrics import concordance_index, median_survival_time, hazard_to_survival


class LatentSequenceDataset(Dataset):
    """Bọc list[EngineRecord] cho DataLoader; collate_fn thật sự nằm ở
    collate_fn() bên dưới vì cần padding tùy theo batch."""

    def __init__(self, records: list[EngineRecord]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate_fn(latent_dim: int):
    def collate_fn(batch: list[EngineRecord]):
        latents, targets, mask = pad_batch(batch, latent_dim)
        lengths = np.array([r.n_bins for r in batch])
        return (
            torch.from_numpy(latents),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
            torch.from_numpy(lengths),
            batch,  # giữ lại record gốc để tính C-index cuối epoch
        )
    return collate_fn


def run_training(train_records, val_records, latent_dim: int,
                  hidden_dim: int = 32, lam_monotonic: float = 0.1,
                  n_epochs: int = 50, lr: float = 1e-3, seed: int = 0,
                  device: str = "cpu"):
    """
    Trả về (model, history) -- history là dict chứa loss/C-index theo epoch,
    để bạn vẽ curve và log lại cho phần multi-seed evaluation sau này.

    ABLATION: để tắt Label/DP/operating-condition (baseline), đơn giản là
    không đưa extra_features vào EngineRecord.latent_seq trước khi gọi hàm
    này (tức nối feature phụ vào latent_seq từ bên ngoài, trước khi build
    EngineRecord -- giữ model.py/data.py không cần biết feature nào đang
    bật/tắt, tránh if/else rải khắp code).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    collate_fn = make_collate_fn(latent_dim)
    train_loader = DataLoader(LatentSequenceDataset(train_records), batch_size=16,
                               shuffle=True, collate_fn=collate_fn)

    model = GRUHazard(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_hazard_loss": [], "train_monotonic": [],
               "val_c_index": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {"total": 0.0, "hazard": 0.0, "monotonic": 0.0}
        n_batches = 0

        for latents, targets, mask, lengths, _ in train_loader:
            latents, targets, mask = latents.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            h = model(latents, lengths)
            losses = combined_loss(h, targets, mask, lam_monotonic=lam_monotonic)
            losses["total"].backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        history["train_loss"].append(epoch_losses["total"])
        history["train_hazard_loss"].append(epoch_losses["hazard"])
        history["train_monotonic"].append(epoch_losses["monotonic"])

        val_c = evaluate_c_index(model, val_records, latent_dim, device)
        history["val_c_index"].append(val_c)

        print(f"epoch {epoch:3d} | loss={epoch_losses['total']:.4f} "
              f"(hazard={epoch_losses['hazard']:.4f}, mono={epoch_losses['monotonic']:.4f}) "
              f"| val C-index={val_c:.4f}")

    return model, history


@torch.no_grad()
def evaluate_c_index(model: GRUHazard, records: list[EngineRecord],
                      latent_dim: int, device: str = "cpu") -> float:
    """
    Chạy model trên từng engine riêng lẻ (không batch, đơn giản hơn cho
    việc lấy đúng chuỗi hazard thật của từng cái), suy ra median survival
    time làm predicted_risk, rồi tính C-index so với event_bin/censor thật.
    """
    model.eval()

    event_times = []
    event_observed = []
    predicted_risk = []  # càng cao = càng nguy hiểm => dùng -median_survival

    for r in records:
        latent = torch.from_numpy(r.latent_seq[None, :r.n_bins, :]).to(device)
        length = torch.tensor([r.n_bins])
        h = model(latent, length).cpu().numpy()[0]

        med = median_survival_time(h)
        predicted_risk.append(-med)  # đảo dấu theo quy ước lifelines

        if r.event_bin is not None:
            event_times.append(r.event_bin)
            event_observed.append(1)
        else:
            event_times.append(r.n_bins - 1)
            event_observed.append(0)

    return concordance_index(
        np.array(event_times), np.array(predicted_risk), np.array(event_observed)
    )


def run_multi_seed(train_records, val_records, latent_dim: int, seeds=(0, 1, 2, 3, 4),
                    **kwargs):
    """
    Lặp lại run_training với nhiều seed, trả về mean/std của C-index cuối
    cùng -- việc còn tồn đọng từ đầu cuộc trò chuyện, giờ chỉ cần gọi hàm
    này thay vì tự viết lại vòng lặp mỗi lần.
    """
    final_c_indices = []
    for seed in seeds:
        _, history = run_training(train_records, val_records, latent_dim,
                                   seed=seed, **kwargs)
        final_c_indices.append(history["val_c_index"][-1])

    final_c_indices = np.array(final_c_indices)
    print(f"\nC-index qua {len(seeds)} seed: "
          f"{final_c_indices.mean():.4f} ± {final_c_indices.std():.4f}")
    return final_c_indices
