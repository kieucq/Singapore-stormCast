#!/usr/bin/env python3
"""
Compute per-channel normalization statistics for prepared SINGV state files.

This script reads a CSV manifest describing six-hour input-target state pairs,
collects every unique prepared state referenced by the manifest, and computes
normalization statistics independently for each state channel.

Input
-----
manifest : positional argument
    Path to a CSV file containing the columns:

        input_time,input_file,target_time,target_file

    Example row:

        1995-01-01T01:00:00,prepared/prepared_19950101_0100.nc,
        1995-01-01T07:00:00,prepared/prepared_19950101_0700.nc

    Only the ``input_file`` and ``target_file`` columns are used. Relative file
    paths are resolved against ``--data-root``.

--data-root : optional
    Root directory used to resolve relative paths in the manifest.
    Defaults to ``~/scratch/retraining``.

--output-dir : optional
    Directory in which the generated NPZ and CSV files are written.
    Defaults to ``~/scratch/retraining/normalisation_stats``.

Prepared state format
---------------------
Each referenced NetCDF file must contain:

    state(time, channel, y, x)

with exactly one time step. The script therefore reads each file as an array of
shape:

    (channel, y, x)

The file must also contain a ``channel`` coordinate giving the channel names in
the same order for every state file.

Processing
----------
Input and target files are deduplicated before processing, so a state shared by
two neighbouring forecast pairs contributes only once.

For each channel, the script computes:

- number and fraction of valid values
- mean
- population standard deviation, using ``ddof=0``
- raw minimum and maximum
- normalized minimum and maximum

Surface channels are required to be complete. Pressure-level variables must
share an identical valid-cell mask at each pressure level. Expected NaNs in
masked pressure-level cells are ignored, while infinite values are treated as
errors. Statistics are accumulated one file at a time using float64 running
accumulators, so the full dataset is never loaded into memory.

Output
------
Two files are generated automatically from the manifest name and written to
``--output-dir``:

    <manifest-prefix>_normalisation.npz
    <manifest-prefix>_normalisation.csv

For example:

    week_1995_01_pairs.csv

produces:

    week_1995_01_normalisation.npz
    week_1995_01_normalisation.csv

The NPZ file contains machine-readable arrays for channel names, count, valid
fraction, mean, standard deviation, raw and normalized extrema, image shape,
number of states, manifest path, and ``ddof``.

The CSV file contains the same per-channel statistics in human-readable form.

No input NetCDF file is modified.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr


SURFACE_CHANNELS = ("tas", "uas", "vas", "psl", "pr")
PRESSURE_PREFIXES = ("ta", "ua", "va", "hus", "zg")

EXPECTED_MANIFEST_COLUMNS = ("input_file", "target_file")
EXPECTED_STATE_DIMS = ("time", "channel", "y", "x")

DEFAULT_DATA_ROOT = Path("~/scratch/retraining")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "normalisation_stats"


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata inferred from the first prepared state file."""

    channel_names: tuple[str, ...]
    image_shape: tuple[int, int]


class RunningChannelStats:
    """Numerically stable streaming statistics for independent channels.

    Statistics are accumulated using the parallel/Welford merge equations.
    Each call to ``update`` accepts one state array with shape
    ``(channel, y, x)``. NaNs are ignored, while infinities are rejected.
    """

    def __init__(self, channel_names: Sequence[str]) -> None:
        if not channel_names:
            raise ValueError("At least one channel is required.")

        self.channel_names = tuple(str(name) for name in channel_names)
        num_channels = len(self.channel_names)

        self.count = np.zeros(num_channels, dtype=np.int64)
        self.mean = np.zeros(num_channels, dtype=np.float64)
        self.m2 = np.zeros(num_channels, dtype=np.float64)
        self.minimum = np.full(num_channels, np.inf, dtype=np.float64)
        self.maximum = np.full(num_channels, -np.inf, dtype=np.float64)

    @property
    def num_channels(self) -> int:
        """Return the number of tracked channels."""

        return len(self.channel_names)

    def update(self, state: np.ndarray) -> None:
        """Merge statistics from one state array."""

        if state.ndim != 3:
            raise ValueError(
                f"Expected state with 3 dimensions (channel, y, x), got "
                f"shape {state.shape}."
            )

        if state.shape[0] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {state.shape[0]}."
            )

        for index in range(self.num_channels):
            channel = state[index]
            valid = ~np.isnan(channel)
            batch_count = int(valid.sum())

            if batch_count == 0:
                continue

            values = channel[valid].astype(np.float64, copy=False)
            batch_mean = float(np.mean(values, dtype=np.float64))
            deviations = values - batch_mean
            batch_m2 = float(np.dot(deviations, deviations))
            batch_minimum = float(np.min(values))
            batch_maximum = float(np.max(values))

            old_count = int(self.count[index])
            old_mean = float(self.mean[index])
            combined_count = old_count + batch_count

            if old_count == 0:
                combined_mean = batch_mean
                combined_m2 = batch_m2
            else:
                delta = batch_mean - old_mean
                combined_mean = old_mean + delta * batch_count / combined_count
                combined_m2 = (
                    self.m2[index]
                    + batch_m2
                    + delta * delta * old_count * batch_count / combined_count
                )

            self.count[index] = combined_count
            self.mean[index] = combined_mean
            self.m2[index] = combined_m2
            self.minimum[index] = min(self.minimum[index], batch_minimum)
            self.maximum[index] = max(self.maximum[index], batch_maximum)

    def finalize(self) -> dict[str, np.ndarray]:
        """Return validated population statistics for every channel."""

        if np.any(self.count == 0):
            missing = [
                self.channel_names[index]
                for index in np.flatnonzero(self.count == 0)
            ]
            raise ValueError(
                "No valid values were found for the following channels: "
                + ", ".join(missing)
            )

        variance = self.m2 / self.count.astype(np.float64)

        tolerance = np.finfo(np.float64).eps * np.maximum(
            1.0, np.abs(self.mean) ** 2
        )
        materially_negative = variance < -tolerance
        if np.any(materially_negative):
            bad = [
                self.channel_names[index]
                for index in np.flatnonzero(materially_negative)
            ]
            raise ValueError(
                "Negative variance encountered for channels: " + ", ".join(bad)
            )

        variance = np.maximum(variance, 0.0)
        standard_deviation = np.sqrt(variance)

        if not np.all(np.isfinite(self.mean)):
            raise ValueError("At least one channel mean is not finite.")

        if not np.all(np.isfinite(standard_deviation)):
            raise ValueError("At least one channel standard deviation is not finite.")

        if np.any(standard_deviation <= 0):
            bad = [
                self.channel_names[index]
                for index in np.flatnonzero(standard_deviation <= 0)
            ]
            raise ValueError(
                "Standard deviation must be positive for every channel. "
                "Non-positive channels: "
                + ", ".join(bad)
            )

        if not np.all(np.isfinite(self.minimum)):
            raise ValueError("At least one channel minimum is not finite.")

        if not np.all(np.isfinite(self.maximum)):
            raise ValueError("At least one channel maximum is not finite.")

        return {
            "count": self.count.copy(),
            "mean": self.mean.copy(),
            "std": standard_deviation,
            "minimum": self.minimum.copy(),
            "maximum": self.maximum.copy(),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute per-channel normalization statistics from prepared SINGV "
            "NetCDF state files referenced by a six-hour pair manifest."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="CSV manifest containing input_file and target_file columns.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Root directory used to resolve relative paths in the manifest. "
            "Default: ~/scratch/retraining"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory in which automatically named NPZ and CSV outputs are "
            "written. Default: ~/scratch/retraining/normalisation_stats"
        ),
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    """Expand ``~`` and return an absolute, normalized path."""

    return path.expanduser().resolve()


def derive_output_paths(
    manifest_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate output filenames from the manifest filename.

    For example, ``week_1995_01_pairs.csv`` becomes:

    - ``week_1995_01_normalisation.npz``
    - ``week_1995_01_normalisation.csv``
    """

    base_name = manifest_path.stem.removesuffix("_pairs")

    return (
        output_dir / f"{base_name}_normalisation.npz",
        output_dir / f"{base_name}_normalisation.csv",
    )


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Read and validate manifest rows."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")

        missing_columns = [
            column
            for column in EXPECTED_MANIFEST_COLUMNS
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Manifest {manifest_path} is missing required column(s): "
                + ", ".join(missing_columns)
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest contains no data rows: {manifest_path}")

    for row_number, row in enumerate(rows, start=2):
        for column in EXPECTED_MANIFEST_COLUMNS:
            value = (row.get(column) or "").strip()
            if not value:
                raise ValueError(
                    f"Manifest row {row_number} has an empty {column!r} value."
                )

    return rows


def collect_unique_state_paths(
    rows: Iterable[dict[str, str]],
    data_root: Path,
) -> list[Path]:
    """Collect unique input and target paths while preserving first occurrence."""

    unique_paths: list[Path] = []
    seen: set[Path] = set()

    for row in rows:
        for column in EXPECTED_MANIFEST_COLUMNS:
            path = Path(row[column].strip()).expanduser()
            if not path.is_absolute():
                path = data_root / path
            path = path.resolve()

            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

    return unique_paths


def decode_channel_names(values: np.ndarray) -> tuple[str, ...]:
    """Convert an xarray channel coordinate into a tuple of strings."""

    names: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return tuple(names)


def validate_missing_values(
    state: np.ndarray,
    channel_names: Sequence[str],
    path: Path,
) -> None:
    """Validate surface completeness and pressure-mask consistency."""

    channel_to_index = {
        name: index for index, name in enumerate(channel_names)
    }

    for channel_name in SURFACE_CHANNELS:
        if channel_name not in channel_to_index:
            raise ValueError(
                f"{path}: required surface channel is missing: {channel_name}"
            )

        missing_count = int(
            np.isnan(state[channel_to_index[channel_name]]).sum()
        )
        if missing_count:
            raise ValueError(
                f"{path}: surface channel {channel_name} contains "
                f"{missing_count} NaNs."
            )

    pressure_levels = [
        name.removeprefix("ta_")
        for name in channel_names
        if name.startswith("ta_")
    ]

    if not pressure_levels:
        raise ValueError(
            f"{path}: no pressure-level temperature channels found."
        )

    for level in pressure_levels:
        level_channels = [
            f"{prefix}_{level}" for prefix in PRESSURE_PREFIXES
        ]

        missing_channels = [
            name for name in level_channels
            if name not in channel_to_index
        ]
        if missing_channels:
            raise ValueError(
                f"{path}: pressure level {level} is missing channel(s): "
                + ", ".join(missing_channels)
            )

        reference_mask = np.isfinite(
            state[channel_to_index[level_channels[0]]]
        )

        for channel_name in level_channels[1:]:
            current_mask = np.isfinite(
                state[channel_to_index[channel_name]]
            )

            if not np.array_equal(reference_mask, current_mask):
                mismatch_count = int(
                    np.count_nonzero(reference_mask != current_mask)
                )
                raise ValueError(
                    f"{path}: pressure masks differ between "
                    f"{level_channels[0]} and {channel_name} at "
                    f"{mismatch_count} cells."
                )


def load_and_validate_state(
    path: Path,
    expected_metadata: DatasetMetadata | None,
) -> tuple[np.ndarray, DatasetMetadata]:
    """Load one prepared state and validate its schema and metadata."""

    if not path.is_file():
        raise FileNotFoundError(f"Prepared state file does not exist: {path}")

    with xr.open_dataset(path) as dataset:
        if "state" not in dataset:
            raise ValueError(f"{path} does not contain a 'state' variable.")

        if "channel" not in dataset:
            raise ValueError(f"{path} does not contain a 'channel' coordinate.")

        state_variable = dataset["state"]
        actual_dims = tuple(state_variable.dims)
        if actual_dims != EXPECTED_STATE_DIMS:
            raise ValueError(
                f"{path}: expected state dimensions {EXPECTED_STATE_DIMS}, "
                f"got {actual_dims}."
            )

        if state_variable.sizes["time"] != 1:
            raise ValueError(
                f"{path}: expected exactly one time step, got "
                f"{state_variable.sizes['time']}."
            )

        channel_names = decode_channel_names(dataset["channel"].values)
        expected_channel_count = state_variable.sizes["channel"]
        if len(channel_names) != expected_channel_count:
            raise ValueError(
                f"{path}: channel coordinate contains {len(channel_names)} "
                f"names but state has {expected_channel_count} channels."
            )

        image_shape = (
            int(state_variable.sizes["y"]),
            int(state_variable.sizes["x"]),
        )
        metadata = DatasetMetadata(
            channel_names=channel_names,
            image_shape=image_shape,
        )

        if expected_metadata is not None:
            if metadata.channel_names != expected_metadata.channel_names:
                raise ValueError(
                    f"{path}: channel names or channel order do not match "
                    "the first prepared state file."
                )

            if metadata.image_shape != expected_metadata.image_shape:
                raise ValueError(
                    f"{path}: spatial shape {metadata.image_shape} does not "
                    f"match expected shape {expected_metadata.image_shape}."
                )

        state = np.asarray(state_variable.isel(time=0).values)

    expected_shape = (
        len(metadata.channel_names),
        metadata.image_shape[0],
        metadata.image_shape[1],
    )
    if state.shape != expected_shape:
        raise ValueError(
            f"{path}: expected loaded state shape {expected_shape}, "
            f"got {state.shape}."
        )

    if not np.issubdtype(state.dtype, np.number):
        raise TypeError(f"{path}: state dtype must be numeric, got {state.dtype}.")

    if np.isinf(state).any():
        locations = np.argwhere(np.isinf(state))
        first = tuple(int(value) for value in locations[0])
        channel_name = metadata.channel_names[first[0]]
        raise ValueError(
            f"{path}: state contains an infinite value at "
            f"(channel={first[0]} [{channel_name}], y={first[1]}, x={first[2]})."
        )
    
    validate_missing_values(
        state,
        metadata.channel_names,
        path,
    )

    return state, metadata


def compute_derived_statistics(
    metadata: DatasetMetadata,
    statistics: dict[str, np.ndarray],
    *,
    num_states: int,
) -> dict[str, np.ndarray]:
    """Compute valid fractions and normalized extrema."""

    if num_states <= 0:
        raise ValueError("num_states must be positive.")

    total_cells_per_channel = (
        num_states
        * metadata.image_shape[0]
        * metadata.image_shape[1]
    )

    return {
        "valid_fraction": (
            statistics["count"].astype(np.float64)
            / float(total_cells_per_channel)
        ),
        "normalized_minimum": (
            statistics["minimum"] - statistics["mean"]
        ) / statistics["std"],
        "normalized_maximum": (
            statistics["maximum"] - statistics["mean"]
        ) / statistics["std"],
    }


def write_npz(
    output_path: Path,
    metadata: DatasetMetadata,
    statistics: dict[str, np.ndarray],
    *,
    num_states: int,
    num_manifest_rows: int,
    manifest_path: Path,
) -> None:
    """Write machine-readable normalization statistics."""

    derived = compute_derived_statistics(
        metadata,
        statistics,
        num_states=num_states,
    )

    np.savez(
        output_path,
        channels=np.asarray(metadata.channel_names, dtype=np.str_),
        count=statistics["count"],
        valid_fraction=derived["valid_fraction"],
        mean=statistics["mean"],
        std=statistics["std"],
        minimum=statistics["minimum"],
        maximum=statistics["maximum"],
        normalized_minimum=derived["normalized_minimum"],
        normalized_maximum=derived["normalized_maximum"],
        num_states=np.asarray(num_states, dtype=np.int64),
        num_manifest_rows=np.asarray(num_manifest_rows, dtype=np.int64),
        manifest=np.asarray(str(manifest_path), dtype=np.str_),
        image_shape=np.asarray(metadata.image_shape, dtype=np.int64),
        ddof=np.asarray(0, dtype=np.int64),
    )


def write_summary_csv(
    output_path: Path,
    metadata: DatasetMetadata,
    statistics: dict[str, np.ndarray],
    *,
    num_states: int,
) -> None:
    """Write a human-readable per-channel summary."""

    derived = compute_derived_statistics(
        metadata,
        statistics,
        num_states=num_states,
    )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "channel",
                "count",
                "valid_fraction",
                "mean",
                "std",
                "min",
                "max",
                "normalized_min",
                "normalized_max",
            ]
        )

        for index, channel_name in enumerate(metadata.channel_names):
            writer.writerow(
                [
                    index,
                    channel_name,
                    int(statistics["count"][index]),
                    f"{derived['valid_fraction'][index]:.17g}",
                    f"{statistics['mean'][index]:.17g}",
                    f"{statistics['std'][index]:.17g}",
                    f"{statistics['minimum'][index]:.17g}",
                    f"{statistics['maximum'][index]:.17g}",
                    f"{derived['normalized_minimum'][index]:.17g}",
                    f"{derived['normalized_maximum'][index]:.17g}",
                ]
            )


def print_summary(
    *,
    manifest_path: Path,
    num_manifest_rows: int,
    num_states: int,
    metadata: DatasetMetadata,
    statistics: dict[str, np.ndarray],
    output_path: Path,
    summary_csv_path: Path,
) -> None:
    """Print an execution summary and formatted per-channel table."""

    derived = compute_derived_statistics(
        metadata,
        statistics,
        num_states=num_states,
    )

    print()
    print("SINGV NORMALIZATION STATISTICS")
    print("==============================")
    print(f"Manifest:             {manifest_path}")
    print(f"Manifest rows:        {num_manifest_rows}")
    print(f"Unique states:        {num_states}")
    print(f"Channels:             {len(metadata.channel_names)}")
    print(
        f"Image shape:          "
        f"{metadata.image_shape[0]} x {metadata.image_shape[1]}"
    )
    print(f"Output NPZ:           {output_path}")
    print(f"Output summary CSV:   {summary_csv_path}")
    print()
    print(
        f"{'Idx':>3}  {'Channel':<14} {'Count':>12} {'Valid':>9} "
        f"{'Mean':>14} {'Std':>14} {'Norm min':>12} {'Norm max':>12}"
    )
    print("-" * 106)

    for index, channel_name in enumerate(metadata.channel_names):
        print(
            f"{index:3d}  "
            f"{channel_name:<14} "
            f"{int(statistics['count'][index]):12d} "
            f"{derived['valid_fraction'][index]:9.6f} "
            f"{statistics['mean'][index]:14.7g} "
            f"{statistics['std'][index]:14.7g} "
            f"{derived['normalized_minimum'][index]:12.6g} "
            f"{derived['normalized_maximum'][index]:12.6g}"
        )


def main() -> None:
    """Run the normalization-statistics workflow."""

    args = parse_args()

    manifest_path = normalize_path(args.manifest)
    data_root = normalize_path(args.data_root)
    output_dir = normalize_path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path, summary_csv_path = derive_output_paths(
        manifest_path,
        output_dir,
    )

    rows = read_manifest_rows(manifest_path)
    state_paths = collect_unique_state_paths(rows, data_root)

    if not state_paths:
        raise ValueError("No prepared state paths were found in the manifest.")

    print("Collecting SINGV normalization statistics")
    print("------------------------------------------")
    print(f"Manifest rows: {len(rows)}")
    print(f"Unique states: {len(state_paths)}")

    first_state, metadata = load_and_validate_state(state_paths[0], None)
    running_stats = RunningChannelStats(metadata.channel_names)
    running_stats.update(first_state)

    print(f"Channels:      {len(metadata.channel_names)}")
    print(
        f"Image shape:   "
        f"{metadata.image_shape[0]} x {metadata.image_shape[1]}"
    )

    progress_width = len(str(len(state_paths)))

    print(
        f"[{1:>{progress_width}}/{len(state_paths)}] "
        f"{state_paths[0]}"
    )

    for index, state_path in enumerate(state_paths[1:], start=2):
        state, _ = load_and_validate_state(state_path, metadata)
        running_stats.update(state)

        print(
            f"[{index:>{progress_width}}/{len(state_paths)}] "
            f"{state_path}"
        )

    statistics = running_stats.finalize()

    write_npz(
        output_path,
        metadata,
        statistics,
        num_states=len(state_paths),
        num_manifest_rows=len(rows),
        manifest_path=manifest_path,
    )
    write_summary_csv(
        summary_csv_path,
        metadata,
        statistics,
        num_states=len(state_paths),
    )

    print_summary(
        manifest_path=manifest_path,
        num_manifest_rows=len(rows),
        num_states=len(state_paths),
        metadata=metadata,
        statistics=statistics,
        output_path=output_path,
        summary_csv_path=summary_csv_path,
    )


if __name__ == "__main__":
    main()