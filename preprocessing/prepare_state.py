#!/usr/bin/env python3
"""
Prepare one assembled SINGV state for StormCast retraining.

This preparation stage:

1. validates one 960 x 960 assembled SINGV state;
2. crops 12 pixels from every edge;
3. regrids 936 x 936 -> 624 x 624;
4. keeps the 14 native SINGV pressure levels;
5. stacks the fixed 75-channel state;
6. builds the pressure-level valid-fraction mask;
7. saves an unnormalized NetCDF file.

Pressure cells below the valid-fraction threshold are stored as NaN.
Normalization and post-normalization zero filling are intentionally deferred
to a later stage.

Usage
-----
python prepare_state.py PATH_TO_ASSEMBLED_FILE

Example
-------
python prepare_state.py assembled_20141201_0100.nc

Default output
--------------
The prepared output directory configured in paths.py, with filename:

prepared_YYYYMMDD_HHMM.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

import prep_utils as prep
import paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop, regrid, mask, and stack one assembled SINGV state "
            "for StormCast regression retraining."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to assembled_YYYYMMDD_HHMM.nc",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.PREPARED_DIR,
        help=f"Output directory (default: {paths.PREPARED_DIR})",
    )
    parser.add_argument(
        "--valid-fraction-threshold",
        type=float,
        default=prep.VALID_FRACTION_THRESHOLD,
        help=(
            "A target pressure-level cell is valid when its conservatively "
            "remapped valid fraction is at least this value "
            f"(default: {prep.VALID_FRACTION_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed diagnostics and the output summary.",
    )
    return parser.parse_args()


def make_output_path(input_path: Path, output_dir: Path) -> Path:
    stem = input_path.stem

    if stem.startswith("assembled_"):
        timestamp = stem.removeprefix("assembled_")
        filename = f"prepared_{timestamp}.nc"
    else:
        filename = f"prepared_{stem}.nc"

    return output_dir.expanduser() / filename


def save_dataset(
    dataset: xr.Dataset,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    encoding = {
        "state": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, 1, 156, 156),
        },
        "pressure_valid": {
            "dtype": "uint8",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": None,
            "chunksizes": (1, 1, 156, 156),
        },
        "pressure_valid_fraction": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, 1, 156, 156),
        },
        "time": {
            "dtype": "int64",
            "units": "hours since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "_FillValue": None,
        },
        "plev": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "y": {
            "dtype": "int32",
            "_FillValue": None,
        },
        "x": {
            "dtype": "int32",
            "_FillValue": None,
        },
        "latitude": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "longitude": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "latitude_bounds": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "longitude_bounds": {
            "dtype": "float64",
            "_FillValue": None,
        },
    }

    dataset.to_netcdf(
        output_path,
        format="NETCDF4",
        encoding=encoding,
    )


def print_summary(dataset: xr.Dataset, output_path: Path) -> None:
    state = dataset["state"]
    mask = dataset["pressure_valid"].astype(bool)

    surface = state.isel(channel=slice(0, len(prep.SURFACE_VARIABLES))).values
    pressure = state.isel(channel=slice(len(prep.SURFACE_VARIABLES), None)).values

    size_mb = output_path.stat().st_size / (1024**2)

    print("\nOutput summary")
    print("--------------")
    print("Path:                 ", output_path)
    print(f"File size:             {size_mb:.1f} MB")
    print("State dimensions:      ", state.dims)
    print("State shape:           ", state.shape)
    print("State dtype:           ", state.dtype)
    print("Pressure mask shape:   ", mask.shape)
    print("Surface finite:        ", bool(np.isfinite(surface).all()))
    print("Pressure NaN fraction: ", f"{np.isnan(pressure).mean():.6f}")
    print("Mask valid fraction:   ", f"{mask.values.mean():.6f}")
    print("First channels:        ", dataset["channel"].values[:8].tolist())
    print("Last channels:         ", dataset["channel"].values[-5:].tolist())


def prepare_state(
    input_path: Path,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    valid_fraction_threshold: float = prep.VALID_FRACTION_THRESHOLD,
    verbose: bool = True,
) -> Path:
    """
    Prepare one assembled SINGV state and save it as a NetCDF file.

    Parameters
    ----------
    input_path
        Path to assembled_YYYYMMDD_HHMM.nc.
    output_path
        Explicit prepared output path. When omitted, the default prepared
        directory and filename are used.
    overwrite
        Replace an existing output file.
    valid_fraction_threshold
        Minimum remapped valid fraction required for a pressure-level cell.
    verbose
        Print per-variable diagnostics and the output summary.

    Returns
    -------
    Path
        Path to the prepared NetCDF file.
    """
    input_path = input_path.expanduser()

    if output_path is None:
        output_path = make_output_path(
            input_path,
            paths.PREPARED_DIR,
        )
    else:
        output_path = output_path.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Check before performing the expensive preprocessing.
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use overwrite=True to replace it."
        )

    with xr.open_dataset(
        input_path,
        mask_and_scale=True,
        decode_times=True,
    ) as source:
        processed = prep.preprocess_one_state(
            source,
            valid_fraction_threshold=valid_fraction_threshold,
            verbose=verbose,
        )

    processed.attrs["source_file"] = str(input_path.resolve())

    save_dataset(
        processed,
        output_path,
        overwrite=overwrite,
    )

    if verbose:
        print_summary(processed, output_path)

    return output_path


def main() -> None:
    args = parse_args()
    start = perf_counter()

    input_path = args.input_file.expanduser()
    output_path = make_output_path(
        input_path,
        args.output_dir,
    )

    print("SINGV STATE PREPARATION")
    print("=======================")
    print("Input:                ", input_path)
    print("Output:               ", output_path)
    print("Crop:                  12 pixels per side")
    print("Grid:                  960x960 -> 936x936 -> 624x624")
    print("Valid threshold:       ", args.valid_fraction_threshold)
    print("Normalization:          deferred")

    prepare_state(
        input_path,
        output_path,
        overwrite=args.overwrite,
        valid_fraction_threshold=args.valid_fraction_threshold,
        verbose=not args.quiet,
    )

    elapsed_minutes = (perf_counter() - start) / 60.0
    print(f"\nCompleted successfully in {elapsed_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()