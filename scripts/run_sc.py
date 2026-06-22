#!/usr/bin/env python3
"""
Run StormCast using an ncview-compatible SingV input file.

The original SingV time is temporarily replaced with a GFS-supported date.
The resulting forecast is only intended to test whether StormCast accepts
the converted input; it is not physically meaningful.

Usage:
    python run_sc.py INPUT_FILE

Example:
    python run_sc.py \
        ~/scratch/singv_sc/singv_sc_20131004_0700.nc
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

import earth2studio.run as run
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast

import utils_singv3 as ut


OUTPUT_DIR = Path("~/scratch/output").expanduser()
N_STEPS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run StormCast using a converted SingV input file."
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to singv_sc_*.nc",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input_file.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{input_path.stem}_output.zarr"

    print("Input: ", input_path)
    print("Output:", output_path)

    # Load StormCast with its normal GFS conditioning.
    package = StormCast.load_default_package()

    model = StormCast.load_model(
        package,
        conditioning_data_source=ARCO(),
    )

    variables = (
        np.asarray(model.input_coords()["variable"])
        .astype(str)
        .tolist()
    )

    # Convert the ncview-friendly file back to StormCast's 99 channels.
    with xr.open_dataset(
        input_path,
        mask_and_scale=True,
    ) as ds:

        data = ut.ncview_to_sc(
            ds=ds,
            variables=variables,
        ).load()

    original_time = data["time"].values[0]

    run_time = (
        np.datetime64(original_time, "us")
        .astype(datetime)
    )

    print("SingV input time:       ", original_time)
    print("ERA5 conditioning time: ", run_time)

    my_data = ut.MyLocalData(data)

    io = ZarrBackend(
        str(output_path),
        backend_kwargs={"overwrite": True},
    )

    print("\nRunning StormCast...")
    print("Warning: the GFS conditioning does not match the SingV input.")

    run.deterministic(
        time=[run_time],
        nsteps=N_STEPS,
        prognostic=model,
        data=my_data,
        io=io,
    )

    print("\nInference completed.")
    print("Saved Zarr:", output_path)

    print("\nConverting output to ncview-compatible NetCDF...")

    ds_out = xr.open_zarr(output_path)

    flat_output = ds_out.to_array(dim="variable")
    ds_ncview = ut.sc_to_ncview(flat_output)

    netcdf_path = output_path.with_suffix(".nc")

    encoding = {
        "time": {
            "dtype": "int32",
            "units": "hours since 1970-01-01 00:00:00",
        },
        "lead_time": {
            "dtype": "int32",
            "units": "hours",
        },
        "hybrid_level": {
            "dtype": "int32",
        },
        "p_hybrid_level": {
            "dtype": "int32",
        },
    }

    ds_ncview.to_netcdf(
        netcdf_path,
        encoding=encoding,
    )

    ds_out.close()

    print("Saved NetCDF:", netcdf_path)


if __name__ == "__main__":
    main()