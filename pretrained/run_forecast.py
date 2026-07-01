#!/usr/bin/env python3

"""
Run StormCast inference using an ncview-compatible SingV input file.

The forecast is intended only to test whether the pretrained StormCast model
accepts the converted SingV input. It is not physically meaningful.

Usage:
    python run_forecast.py INPUT_FILE

Example:
    python run_forecast.py \
        ~/scratch/pretrained/singv_inputs/singv_input_20131004_0700.nc

Outputs:
    ~/scratch/pretrained/singv_forecasts/singv_forecast_20131004_0700.zarr
    ~/scratch/pretrained/singv_forecasts/singv_forecast_20131004_0700.nc
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

import utils as ut


PRETRAINED_DIR = Path("~/scratch/pretrained").expanduser()
OUTPUT_DIR = PRETRAINED_DIR / "singv_forecasts"

N_STEPS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run StormCast using a converted SingV input file."
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a singv_input_*.nc file.",
    )

    return parser.parse_args()

def print_geographic_grid(name, lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Convert 0–360° longitude to −180–180° for readability.
    lon_180 = ((lon + 180.0) % 360.0) - 180.0

    centre_y = lat.shape[0] // 2
    centre_x = lat.shape[1] // 2

    print(f"\n{name}")
    print("-" * len(name))
    print("Shape:          ", lat.shape)
    print(
        "Latitude range:",
        float(np.nanmin(lat)),
        "to",
        float(np.nanmax(lat)),
    )
    print(
        "Longitude range:",
        float(np.nanmin(lon_180)),
        "to",
        float(np.nanmax(lon_180)),
    )
    print(
        "Centre point:   ",
        float(lat[centre_y, centre_x]),
        float(lon_180[centre_y, centre_x]),
    )

def set_conditioning_grid_from_input(model, ds):
    """
    Set StormCast's ERA5 conditioning target to the geographic
    region represented by the converted SingV input file.
    """
    required_attrs = [
        "target_lat_min",
        "target_lat_max",
        "target_lon_min",
        "target_lon_max",
    ]

    missing = [
        attr
        for attr in required_attrs
        if attr not in ds.attrs
    ]

    if missing:
        raise ValueError(
            f"Input file is missing grid metadata: {missing}"
        )

    ny = model.lat.shape[0]
    nx = model.lat.shape[1]

    target_lats = np.linspace(
        ds.attrs["target_lat_min"],
        ds.attrs["target_lat_max"],
        ny,
        dtype=np.float32,
    )

    target_lons = np.linspace(
        ds.attrs["target_lon_min"],
        ds.attrs["target_lon_max"],
        nx,
        dtype=np.float32,
    )

    target_lons_2d, target_lats_2d = np.meshgrid(
        target_lons,
        target_lats,
    )

    model.lat = target_lats_2d
    model.lon = target_lons_2d


def main():
    args = parse_args()

    input_path = args.input_file.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_stem = input_path.stem

    if input_stem.startswith("singv_input_"):
        timestamp = input_stem.removeprefix("singv_input_")
        output_name = f"singv_forecast_{timestamp}.zarr"
    else:
        output_name = f"singv_forecast_{input_stem}.zarr"

    output_path = OUTPUT_DIR / output_name

    print("Input: ", input_path)
    print("Output:", output_path)

    # Load StormCast with ERA5 reanalysis conditioning from ARCO.
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

        set_conditioning_grid_from_input(
            model=model,
            ds=ds,
        )

        print_geographic_grid(
            "StormCast ARCO conditioning target",
            model.lat,
            model.lon,
        )

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
            "_FillValue": None,
        },
        "lead_time": {
            "dtype": "int32",
            "units": "hours",
            "_FillValue": None,
        },
        "hybrid_level": {
            "dtype": "int32",
            "_FillValue": None,
        },
        "p_hybrid_level": {
            "dtype": "int32",
            "_FillValue": None,
        },
    }

    ds_ncview.to_netcdf(
        netcdf_path,
        format="NETCDF4_CLASSIC",
        encoding=encoding,
    )

    ds_out.close()

    print("Saved NetCDF:", netcdf_path)


if __name__ == "__main__":
    main()