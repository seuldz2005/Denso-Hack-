from src.phaseI.core.train import extract_latents_for_engine
def encode_snapshot(
    model,
    snapshot,
    stats,
    window_len=30,
    bin_stride=10,
    device="cpu",
):
    normalized = stats.apply(snapshot)

    return extract_latents_for_engine(
        model=model,
        series=normalized,
        window_len=window_len,
        bin_stride=bin_stride,
        device=device,
    )
