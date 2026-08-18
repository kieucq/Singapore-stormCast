#!/usr/bin/env python3
"""
Build one six-hour SINGV training pair.

For a requested input time t, this script:

1. assemble the native-grid SINGV states at t and t + 6 hours;
2. ensure that the raw monthly ERA5 files containing t exist;
3. prepare SINGV_t, ERA5_t, and SINGV_t+6h;
4. validate the complete training sample;
5. optionally record it in a six-column manifest.

Example
-------
python build_pair.py --datetime 2014-12-01T01:00

Overwrite only the prepared files
---------------------------------
python build_pair.py \
    --datetime 2014-12-01T01:00 \
    --overwrite-prepared

Default outputs
---------------
~/scratch/retraining/assembled/assembled_YYYYMMDD_HHMM.nc
~/scratch/retraining/prepared/prepared_YYYYMMDD_HHMM.nc
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

import assemble_state as assembler
import prepare_state as preparer

from background import download_era5 as era5_downloader
from background import prepare_era5 as era5_preparer


RETRAINING_DIR = Path("~/scratch/retraining").expanduser()
PAIR_INTERVAL = timedelta(hours=6)


@dataclass(frozen=True)
class TrainingPair:
    """Paths and valid times for one complete six-hour training sample."""

    input_time: datetime
    input_file: Path
    background_time: datetime
    background_file: Path
    target_time: datetime
    target_file: Path


def parse_datetime(value: str) -> datetime:
    """Parse and validate one supported UTC SINGV valid time."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ISO datetime {value!r}: {exc}"
        ) from exc

    if dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        raise argparse.ArgumentTypeError(
            "Datetime must be exactly on the hour."
        )

    if dt.hour not in assembler.VALID_PRESSURE_HOURS:
        valid_hours = sorted(assembler.VALID_PRESSURE_HOURS)
        raise argparse.ArgumentTypeError(
            f"Hour must be one of {valid_hours} UTC."
        )

    return dt


def ensure_assembled(
    dt: datetime,
    *,
    overwrite: bool,
) -> Path:
    """Create or reuse one assembled state."""
    output_path = assembler.default_output_path(dt).expanduser()

    if output_path.exists() and not overwrite:
        print(f"Reusing assembled state: {output_path}")
        return output_path

    return assembler.assemble_state(
        dt,
        output_path,
        overwrite=overwrite,
    )


def ensure_prepared(
    assembled_path: Path,
    *,
    overwrite: bool,
    quiet: bool,
) -> Path:
    """Create or reuse one prepared state."""
    assembled_path = assembled_path.expanduser()
    output_path = preparer.make_output_path(
        assembled_path,
        preparer.DEFAULT_OUTPUT_DIR,
    )

    if output_path.exists() and not overwrite:
        print(f"Reusing prepared state:  {output_path}")
        return output_path

    print(f"Preparing state:         {assembled_path}")

    return preparer.prepare_state(
        assembled_path,
        output_path,
        overwrite=overwrite,
        verbose=not quiet,
    )


def ensure_background(
    input_time: datetime,
    *,
    overwrite: bool,
    quiet: bool,
) -> Path:
    """Create or reuse the prepared ERA5 background at the input time."""
    output_path = era5_preparer.make_output_path(
        input_time,
        era5_preparer.DEFAULT_OUTPUT_DIR,
    )

    if output_path.exists() and not overwrite:
        print(f"Reusing ERA5 background: {output_path}")
        return output_path

    print(f"Preparing ERA5 background: {input_time.isoformat()}")

    return era5_preparer.prepare_era5(
        input_time,
        output_path=output_path,
        overwrite=overwrite,
        verbose=not quiet,
    )


def _read_scalar_time(dataset: xr.Dataset, path: Path) -> np.datetime64:
    """Read one scalar datetime64[ns] time coordinate."""
    if "time" not in dataset:
        raise ValueError(f"Prepared file has no time coordinate: {path}")

    values = np.asarray(dataset["time"].values)

    if values.size != 1:
        raise ValueError(
            f"Expected exactly one time in {path}, found {values.size}."
        )

    return np.datetime64(values.reshape(-1)[0], "ns")


def validate_training_sample(pair: TrainingPair) -> None:
    """Validate the SINGV input, ERA5 background, and SINGV target."""
    with (
        xr.open_dataset(pair.input_file, decode_times=True) as input_ds,
        xr.open_dataset(pair.background_file, decode_times=True) as background_ds,
        xr.open_dataset(pair.target_file, decode_times=True) as target_ds,
    ):
        input_time = _read_scalar_time(input_ds, pair.input_file)
        background_time = _read_scalar_time(background_ds, pair.background_file)
        target_time = _read_scalar_time(target_ds, pair.target_file)

        expected_input_time = np.datetime64(pair.input_time, "ns")
        expected_background_time = np.datetime64(pair.background_time, "ns")
        expected_target_time = np.datetime64(pair.target_time, "ns")

        if input_time != expected_input_time:
            raise ValueError(
                f"Input file time is {input_time}, expected "
                f"{expected_input_time}: {pair.input_file}"
            )

        if background_time != expected_background_time:
            raise ValueError(
                f"Background file time is {background_time}, expected "
                f"{expected_background_time}: {pair.background_file}"
            )

        if background_time != input_time:
            raise ValueError(
                "ERA5 background time does not match the SINGV input time: "
                f"{background_time} vs {input_time}"
            )

        if target_time != expected_target_time:
            raise ValueError(
                f"Target file time is {target_time}, expected "
                f"{expected_target_time}: {pair.target_file}"
            )

        interval = target_time - input_time
        if interval != np.timedelta64(6, "h"):
            raise ValueError(
                f"Prepared states are separated by {interval}, not 6 hours."
            )

        if input_ds["state"].dims != target_ds["state"].dims:
            raise ValueError(
                "Input and target state dimensions differ: "
                f"{input_ds['state'].dims} vs {target_ds['state'].dims}"
            )

        if input_ds["state"].shape != target_ds["state"].shape:
            raise ValueError(
                "Input and target state shapes differ: "
                f"{input_ds['state'].shape} vs {target_ds['state'].shape}"
            )

        for coordinate in ("channel", "plev", "y", "x", "latitude", "longitude", "latitude_bounds", "longitude_bounds"):
            if coordinate not in input_ds or coordinate not in target_ds:
                raise ValueError(
                    f"Coordinate {coordinate!r} is missing from one of the "
                    "prepared files."
                )

        if "background" not in background_ds:
            raise ValueError(
                f"ERA5 file has no background variable: "
                f"{pair.background_file}"
            )

        background = background_ds["background"]

        expected_background_dims = (
            "time",
            "channel",
            "y",
            "x",
        )
        if background.dims != expected_background_dims:
            raise ValueError(
                f"ERA5 background dimensions are {background.dims}; "
                f"expected {expected_background_dims}."
            )

        expected_background_shape = (1, 26, 624, 624)
        if background.shape != expected_background_shape:
            raise ValueError(
                f"ERA5 background shape is {background.shape}; "
                f"expected {expected_background_shape}."
            )

        expected_channels = np.asarray(
            era5_preparer.CHANNEL_NAMES,
            dtype=str,
        )
        actual_channels = np.asarray(
            background_ds["channel"].values,
            dtype=str,
        )

        if not np.array_equal(actual_channels, expected_channels):
            raise ValueError(
                "ERA5 background channel order is incorrect.\n"
                f"Expected: {expected_channels.tolist()}\n"
                f"Found:    {actual_channels.tolist()}"
            )

        if not np.all(np.isfinite(background.values)):
            raise ValueError(
                f"ERA5 background contains NaN or infinite values: "
                f"{pair.background_file}"
            )

        if background_ds.attrs.get("normalization") != "none":
            raise ValueError(
                "ERA5 background must be unnormalized; found "
                f"normalization={background_ds.attrs.get('normalization')!r}."
            )

        for coordinate in (
            "y",
            "x",
            "latitude",
            "longitude",
            "latitude_bounds",
            "longitude_bounds",
        ):
            if coordinate not in input_ds or coordinate not in background_ds:
                raise ValueError(
                    f"Coordinate {coordinate!r} is missing from the "
                    "SINGV input or ERA5 background."
                )

            if not np.array_equal(
                input_ds[coordinate].values,
                background_ds[coordinate].values,
            ):
                raise ValueError(
                    f"SINGV input and ERA5 background {coordinate} "
                    "coordinates differ."
                )

            if not np.array_equal(
                input_ds[coordinate].values,
                target_ds[coordinate].values,
            ):
                raise ValueError(
                    f"Input and target {coordinate} coordinates differ."
                )

    print("Validated training sample:")
    print(
        f"  input:      {pair.input_time.isoformat()}  "
        f"{pair.input_file}"
    )
    print(
        f"  background: {pair.background_time.isoformat()}  "
        f"{pair.background_file}"
    )
    print(
        f"  target:     {pair.target_time.isoformat()}  "
        f"{pair.target_file}"
    )


def _manifest_path_value(path: Path) -> str:
    """Store paths relative to the retraining root when possible."""
    resolved_path = path.resolve()
    resolved_root = RETRAINING_DIR.resolve()

    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def record_pair(
    pair: TrainingPair,
    manifest_path: Path,
) -> None:
    """Append one pair to the manifest unless it is already present."""
    manifest_path = manifest_path.expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "input_time",
        "input_file",
        "background_time",
        "background_file",
        "target_time",
        "target_file",
    ]

    row = {
        "input_time": pair.input_time.isoformat(),
        "input_file": _manifest_path_value(pair.input_file),
        "background_time": pair.background_time.isoformat(),
        "background_file": _manifest_path_value(pair.background_file),
        "target_time": pair.target_time.isoformat(),
        "target_file": _manifest_path_value(pair.target_file),
    }

    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != fieldnames:
                raise ValueError(
                    f"Manifest has columns {reader.fieldnames}; "
                    f"expected {fieldnames}: {manifest_path}"
                )

            for existing_row in reader:
                if all(
                    existing_row.get(key) == value
                    for key, value in row.items()
                ):
                    print(f"Pair already recorded:    {manifest_path}")
                    return

    write_header = (
        not manifest_path.exists()
        or manifest_path.stat().st_size == 0
    )

    with manifest_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    print(f"Recorded pair:           {manifest_path}")


def build_pair(
    input_time: datetime,
    *,
    overwrite_assembled: bool = False,
    overwrite_prepared: bool = False,
    manifest_path: Path | None = None,
    quiet: bool = False,
) -> TrainingPair:
    """Build and validate one complete t -> t+6h training sample."""
    target_time = input_time + PAIR_INTERVAL

    print("SINGV SIX-HOUR PAIR BUILD")
    print("=========================")
    print(f"Input time:             {input_time.isoformat()}")
    print(f"Target time:            {target_time.isoformat()}")

    assembled_input = ensure_assembled(
        input_time,
        overwrite=overwrite_assembled,
    )
    era5_downloader.ensure_era5_month(input_time)
    assembled_target = ensure_assembled(
        target_time,
        overwrite=overwrite_assembled,
    )

    # Regenerating an assembled state invalidates any prepared file derived
    # from the previous assembled version.
    effective_overwrite_prepared = (
        overwrite_prepared or overwrite_assembled
    )

    prepared_input = ensure_prepared(
        assembled_input,
        overwrite=effective_overwrite_prepared,
        quiet=quiet,
    )
    prepared_background = ensure_background(
        input_time,
        overwrite=effective_overwrite_prepared,
        quiet=quiet,
    )
    prepared_target = ensure_prepared(
        assembled_target,
        overwrite=effective_overwrite_prepared,
        quiet=quiet,
    )

    pair = TrainingPair(
        input_time=input_time,
        input_file=prepared_input,
        background_time=input_time,
        background_file=prepared_background,
        target_time=target_time,
        target_file=prepared_target,
    )

    validate_training_sample(pair)

    if manifest_path is not None:
        record_pair(pair, manifest_path)

    return pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble and prepare two consecutive SINGV states, validate "
            "the six-hour pair, and record it in a manifest."
        )
    )
    parser.add_argument(
        "--datetime",
        required=True,
        type=parse_datetime,
        help=(
            "Input valid time in ISO format, e.g. 2014-12-01T01:00. "
            "Hour must be 01, 07, 13, or 19 UTC."
        ),
    )
    parser.add_argument(
        "--overwrite-assembled",
        action="store_true",
        help=(
            "Regenerate both assembled files. This also regenerates their "
            "prepared files."
        ),
    )
    parser.add_argument(
        "--overwrite-prepared",
        action="store_true",
        help=(
            "Regenerate the prepared SINGV input, ERA5 background, "
            "and SINGV target while reusing upstream data."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path at which to record the completed six-column training sample.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed preparation diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        build_pair(
            args.datetime,
            overwrite_assembled=args.overwrite_assembled,
            overwrite_prepared=args.overwrite_prepared,
            manifest_path=args.manifest,
            quiet=args.quiet,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()