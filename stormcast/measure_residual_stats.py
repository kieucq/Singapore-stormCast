"""Measure channel-wise statistics of StormCast regression residuals."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from physicsnemo.core import Module
from physicsnemo.distributed import DistributedManager
from torch.utils.data import DataLoader

from datasets import dataset_classes
from utils.nn import regression_model_forward, unpack_batch


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="singv_diffusion_10years_100k",
)
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Use the training manifest because sigma_data should describe
    # the data on which diffusion is trained.
    dataset_cls = dataset_classes[cfg.dataset.name]
    dataset = dataset_cls(cfg.dataset, train=True)

    channels = list(dataset.state_channels())
    num_channels = len(channels)

    batch_size = int(cfg.get("stats_batch_size", 1))
    num_workers = int(cfg.get("stats_num_workers", 2))
    max_samples = int(cfg.get("max_samples", -1))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    regression_model = (
        Module.from_checkpoint(cfg.model.regression_weights)
        .eval()
        .requires_grad_(False)
        .to(device)
    )

    invariant_array = dataset.get_invariants()
    invariant_base = None

    if invariant_array is not None:
        invariant_base = torch.as_tensor(
            invariant_array,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

    # Running totals for every channel.
    # Float64 reduces numerical error when accumulating many grid cells.
    count = torch.zeros(num_channels, dtype=torch.float64)
    total = torch.zeros(num_channels, dtype=torch.float64)
    total_squared = torch.zeros(num_channels, dtype=torch.float64)

    samples_processed = 0

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            (
                background,
                state,
                input_mask,
                target_mask,
                lead_time_label,
            ) = unpack_batch(batch, device)

            input_state, target_state = state
            current_batch_size = input_state.shape[0]

            invariant_tensor = None
            if invariant_base is not None:
                invariant_tensor = invariant_base.expand(
                    current_batch_size, -1, -1, -1
                )

            regression_output = regression_model_forward(
                regression_model,
                input_state,
                background,
                invariant_tensor,
                lead_time_label=lead_time_label,
                condition_list=cfg.model.regression_conditions,
            )

            # Match the masking used during diffusion training.
            if input_mask is not None:
                regression_output = regression_output.masked_fill(
                    input_mask == 0,
                    0.0,
                )

            # This is the clean target supplied to EDM diffusion.
            residual = target_state - regression_output

            if target_mask is None:
                valid = torch.ones_like(residual, dtype=torch.bool)
            else:
                valid = target_mask != 0

            invalid_finite = valid & ~torch.isfinite(residual)
            if invalid_finite.any():
                raise RuntimeError(
                    "Non-finite regression residual found in a valid cell."
                )

            residual64 = residual.to(torch.float64)
            valid64 = valid.to(torch.float64)

            # Sum over batch, y and x, retaining the channel dimension.
            reduce_dims = (0, 2, 3)

            count += valid64.sum(dim=reduce_dims).cpu()
            total += (residual64 * valid64).sum(dim=reduce_dims).cpu()
            total_squared += (
                residual64.square() * valid64
            ).sum(dim=reduce_dims).cpu()

            samples_processed += current_batch_size

            if batch_number % 50 == 0:
                print(
                    f"Processed {samples_processed}/{len(dataset)} samples",
                    flush=True,
                )

            if max_samples > 0 and samples_processed >= max_samples:
                break

    if torch.any(count == 0):
        missing = [
            channels[i]
            for i in range(num_channels)
            if count[i].item() == 0
        ]
        raise RuntimeError(f"No valid values found for channels: {missing}")

    # Population variance across all valid training values.
    mean = total / count
    variance = total_squared / count - mean.square()
    variance = torch.clamp(variance, min=0.0)

    std = torch.sqrt(variance)
    rms = torch.sqrt(total_squared / count)

    results = pd.DataFrame(
        {
            "channel": channels,
            "residual_mean": mean.numpy(),
            "residual_std": std.numpy(),
            "residual_rms": rms.numpy(),
            "valid_count": count.numpy().astype(np.int64),
        }
    )

    output_dir = Path(
        str(
            cfg.get(
                "stats_output",
                "~/scratch/retraining/residual_stats",
            )
        )
    ).expanduser()

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "residual_channel_statistics.csv"
    npz_path = output_dir / "residual_channel_statistics.npz"

    results.to_csv(csv_path, index=False)

    np.savez(
        npz_path,
        channels=np.asarray(channels),
        residual_mean=mean.numpy(),
        residual_std=std.numpy(),
        residual_rms=rms.numpy(),
        valid_count=count.numpy().astype(np.int64),
        samples_processed=np.asarray(samples_processed),
    )

    print("\nChannel statistics:")
    print(results.to_string(index=False))

    print("\nResidual standard-deviation summary:")
    print(f"Minimum: {std.min().item():.6f}")
    print(f"Median:  {std.median().item():.6f}")
    print(f"Maximum: {std.max().item():.6f}")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved NPZ: {npz_path}")


if __name__ == "__main__":
    main()