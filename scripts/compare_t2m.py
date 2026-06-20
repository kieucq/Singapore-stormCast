#!/usr/bin/env python3

"""
Compare raw SingV surface temperature with converted StormCast temperature.

This script reads:

- A raw SingV NetCDF file containing the surface-temperature variable ``tas``
- A converted StormCast-compatible NetCDF file containing ``t2m``

It reproduces the same horizontal regridding used during the SingV-to-StormCast
conversion, then compares the expected regridded temperature field against the
saved ``t2m`` field.

The script reports:

- Raw and converted array shapes
- Temperature ranges
- Maximum absolute difference
- Mean absolute difference
- Root mean square error (RMSE)

Usage
-----
    python compare_t2m.py RAW_FILE CONVERTED_FILE

Example
-------
    python compare_t2m.py \
        ~/scratch/singv_raw/singv_raw_20131004_0700.nc \
        ~/scratch/singv_sc/singv_sc_20131004_0700.nc

A correctly converted file should normally produce differences close to zero,
apart from small floating-point rounding errors.
"""


import argparse

import numpy as np
import xarray as xr

import utils_singv3 as ut


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare raw SingV tas with converted StormCast t2m."
    )

    parser.add_argument(
        "raw_file",
        help="Path to singv_raw_*.nc",
    )

    parser.add_argument(
        "converted_file",
        help="Path to singv_sc_*.nc",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with xr.open_dataset(
        args.raw_file,
        mask_and_scale=True,
    ) as ds_raw, xr.open_dataset(
        args.converted_file,
        mask_and_scale=True,
    ) as ds_sc:

        # Remove any singleton dimensions, leaving (lat, lon)
        raw_tas = (
            ds_raw["tas"]
            .squeeze(drop=True)
            .values
            .astype(np.float32)
        )

        saved_t2m = (
            ds_sc["t2m"]
            .squeeze(drop=True)
            .values
            .astype(np.float32)
        )

        src_lats = ds_raw["lat"].values
        src_lons = ds_raw["lon"].values

        ny, nx = saved_t2m.shape

        # Reproduce exactly the target grid used by make_grid_spec()
        target_lats = np.linspace(
            src_lats.min(),
            src_lats.max(),
            ny,
        )

        target_lons = np.linspace(
            src_lons.min(),
            src_lons.max(),
            nx,
        )

        expected_t2m = ut.regrid(
            field=raw_tas,
            src_lats=src_lats,
            src_lons=src_lons,
            target_lats=target_lats,
            target_lons=target_lons,
        )

        expected_t2m = ut.fill_nan_nearest(expected_t2m)

        difference = saved_t2m - expected_t2m

        print("Raw tas shape:       ", raw_tas.shape)
        print("Converted t2m shape: ", saved_t2m.shape)

        print("\nRaw tas range:")
        print(float(np.nanmin(raw_tas)), "to", float(np.nanmax(raw_tas)))

        print("\nExpected regridded range:")
        print(
            float(np.nanmin(expected_t2m)),
            "to",
            float(np.nanmax(expected_t2m)),
        )

        print("\nSaved t2m range:")
        print(
            float(np.nanmin(saved_t2m)),
            "to",
            float(np.nanmax(saved_t2m)),
        )

        print("\nComparison:")
        print(
            "Maximum absolute difference:",
            float(np.max(np.abs(difference))),
        )
        print(
            "Mean absolute difference:",
            float(np.mean(np.abs(difference))),
        )
        print(
            "RMSE:",
            float(np.sqrt(np.mean(difference**2))),
        )


if __name__ == "__main__":
    main()
