#!/usr/bin/env python3
"""
Convert one collated SingV3 NetCDF file into an ncview-friendly
representation of the StormCast input fields.

Usage:
    python prepare_singv_input.py PATH_TO_SINGV_FILE

Example:
    python prepare_singv_input.py \
        ~/scratch/pretrained/singv_collated/singv_collated_20141201_0100.nc

Output:
    ~/scratch/pretrained/singv_inputs/singv_input_20141201_0100.nc

This script performs conversion, validation, and NetCDF storage only.
It does not run StormCast inference.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

from earth2studio.data import GFS_FX
from earth2studio.models.px import StormCast

import utils as ut
from paths import INPUT_DIR


# ── Command-line arguments ────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one collated SingV3 NetCDF file into a "
            "ncview-friendly representation of the StormCast input fields."
        )
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the collated SingV3 NetCDF file.",
    )

    return parser.parse_args()


def make_output_path(input_path):
    """
    Derive the prepared-input filename from the collated SingV filename.

    Example:
        singv_collated_20141201_0100.nc
        → singv_input_20141201_0100.nc
    """
    stem = input_path.stem

    if stem.startswith("singv_collated_"):
        timestamp = stem.removeprefix("singv_collated_")
        output_name = f"singv_input_{timestamp}.nc"
    else:
        output_name = f"singv_input_{stem}.nc"

    return INPUT_DIR / output_name


# ── StormCast coordinate setup ────────────────────────────────────────────────

def get_stormcast_coordinates():
    """
    Load StormCast only to obtain its required variables and grid coordinates.

    No inference is performed here.
    """
    print("\nLoading StormCast input coordinates...")

    package = StormCast.load_default_package()

    model = StormCast.load_model(
        package,
        conditioning_data_source=GFS_FX(),
    )

    coords = model.input_coords()

    variables = (
        np.asarray(coords["variable"])
        .astype(str)
        .tolist()
    )

    hrrr_y = np.asarray(coords["hrrr_y"])
    hrrr_x = np.asarray(coords["hrrr_x"])

    ny = len(hrrr_y)
    nx = len(hrrr_x)

    print(f"Variables:   {len(variables)}")
    print(f"Target grid: {ny} × {nx}")

    if len(variables) != 99:
        raise ValueError(
            f"Expected 99 StormCast variables, got {len(variables)}"
        )

    if (ny, nx) != (512, 640):
        raise ValueError(
            f"Expected StormCast grid (512, 640), got {(ny, nx)}"
        )

    return variables, hrrr_y, hrrr_x


# ── Diagnostics ────────────────────────────────────────────────────────────────

def verify_local_data_wrapper(
    data,
    variables,
    hrrr_y,
    hrrr_x,
):
    """Verify that MyLocalData returns the expected array."""
    source = ut.MyLocalData(data)

    input_time = data["time"].values[0]

    sample = source(
        time=[input_time],
        variable=variables,
    )

    expected_shape = (
        1,
        len(variables),
        len(hrrr_y),
        len(hrrr_x),
    )

    if sample.shape != expected_shape:
        raise ValueError(
            f"MyLocalData returned {sample.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(sample.values).all():
        raise ValueError(
            "MyLocalData returned non-finite values."
        )

    print("\nEarth2Studio wrapper verified")
    print("Dimensions:", sample.dims)
    print("Shape:     ", sample.shape)
    print("Dtype:     ", sample.dtype)


# ── Save output ───────────────────────────────────────────────────────────────

def save_input(
    data,
    output_path,
):
    """
    Save the converted fields in an ncview-friendly layout.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = ut.sc_to_ncview(data)

    if output_path.exists():
        print(f"Warning: overwriting existing file: {output_path}")

    encoding = {}

    if "time" in dataset.coords:
        encoding["time"] = {
            "dtype": "int32",
            "units": "hours since 1970-01-01 00:00:00",
            "_FillValue": None,
        }

    for coord in ["hybrid_level", "p_hybrid_level"]:
        if coord in dataset.coords:
            encoding[coord] = {
                "dtype": "int32",
                "_FillValue": None,
            }

    dataset.to_netcdf(
        output_path,
        format="NETCDF4_CLASSIC",
        encoding=encoding,
    )

    size_mb = output_path.stat().st_size / (1024 ** 2)

    print("\nSaved ncview-compatible file:")
    print(output_path)
    print(f"File size: {size_mb:.1f} MB")
    print("Variables:", list(dataset.data_vars))


# ── Main execution ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    start = perf_counter()

    input_path = args.input_file.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"SingV3 input file not found: {input_path}"
        )

    output_path = make_output_path(input_path)

    print("SINGV3 → STORMCAST FIELD BUILDER")
    print("================================")
    print("SingV3 source:       ", input_path)
    print("Output file:         ", output_path)

    variables, hrrr_y, hrrr_x = (
        get_stormcast_coordinates()
    )

    with xr.open_dataset(
        input_path,
        mask_and_scale=True,
    ) as ds_singv:

        # ── Preprocess Subterranean Zero-Masks (Thermodynamic Grid Only) ──────
        if "ta" in ds_singv:
            print("\nPreprocessing: Sanitizing subterranean zero-masks for T-grid...")
            # Isolate genuine atmospheric cells (where temperature is above absolute zero)
            valid_atmosphere = ds_singv["ta"] > 0.0
            
            # Wipe out the 0.0 cliffs and vertically backward-fill down pressure levels
            for var in ["ta", "zg", "hus"]:
                if var in ds_singv:
                    ds_singv[var] = ds_singv[var].where(valid_atmosphere).bfill(dim="plev")
                    
            # Specific humidity physical floor safety guard
            if "hus" in ds_singv:
                ds_singv["hus"] = ds_singv["hus"].clip(min=0.0)
        # ──────────────────────────────────────────────────────────────────────

        data, singv_time = ut.build_stormcast_input(
            ds_singv=ds_singv,
            variables=variables,
            hrrr_y=hrrr_y,
            hrrr_x=hrrr_x,
            verbose=True,
        )

    data.attrs["source_file"] = str(input_path)

    print("\nConversion summary")
    print("------------------")
    print("Actual SingV3 time:", singv_time)
    print("Stored valid time: ", data["time"].values[0])
    print("Dimensions:        ", data.dims)
    print("Shape:             ", data.shape)
    print("Dtype:             ", data.dtype)

    verify_local_data_wrapper(
        data=data,
        variables=variables,
        hrrr_y=hrrr_y,
        hrrr_x=hrrr_x,
    )

    save_input(
        data=data,
        output_path=output_path,
    )

    elapsed_minutes = (perf_counter() - start) / 60

    print("\nBuild completed successfully.")
    print(f"Elapsed time: {elapsed_minutes:.1f} minutes")


if __name__ == "__main__":
    main()