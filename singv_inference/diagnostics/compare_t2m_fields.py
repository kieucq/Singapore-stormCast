#!/usr/bin/env python3

"""
Compare collated SingV surface temperature with the prepared StormCast input.

This script reads:

- A collated SingV NetCDF file containing the surface-temperature variable
  ``tas``
- A prepared StormCast input NetCDF file containing ``t2m``

It reproduces the horizontal regridding used during the SingV-to-StormCast
conversion, then compares the expected regridded temperature field against the
saved ``t2m`` field.

The script reports:

- Collated and prepared-input array shapes
- Temperature ranges
- Maximum absolute difference
- Mean absolute difference
- Root mean square error (RMSE)

Usage
-----
    python compare_t2m_fields.py COLLATED_FILE INPUT_FILE

Example
-------
    python compare_t2m_fields.py \
        ~/scratch/pretrained/singv_collated/singv_collated_20131004_0700.nc \
        ~/scratch/pretrained/singv_inputs/singv_input_20131004_0700.nc

A correctly prepared input should normally produce differences close to zero,
apart from small floating-point rounding errors.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as ut


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare collated SingV tas with prepared StormCast t2m."
    )

    parser.add_argument(
        "collated_file",
        type=Path,
        help="Path to a singv_collated_*.nc file.",
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a singv_input_*.nc file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    collated_path = args.collated_file.expanduser()
    input_path = args.input_file.expanduser()

    with xr.open_dataset(
        collated_path,
        mask_and_scale=True,
    ) as ds_collated, xr.open_dataset(
        input_path,
        mask_and_scale=True,
    ) as ds_input:

        # Remove any singleton dimensions, leaving (lat, lon)
        collated_tas = (
            ds_collated["tas"]
            .squeeze(drop=True)
            .values
            .astype(np.float32)
        )

        saved_t2m = (
            ds_input["t2m"]
            .squeeze(drop=True)
            .values
            .astype(np.float32)
        )

        src_lats = ds_collated["lat"].values
        src_lons = ds_collated["lon"].values

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
            field=collated_tas,
            src_lats=src_lats,
            src_lons=src_lons,
            target_lats=target_lats,
            target_lons=target_lons,
        )

        expected_t2m = ut.fill_nan_nearest(expected_t2m)

        difference = saved_t2m - expected_t2m

        print("Collated tas shape: ", collated_tas.shape)
        print("Prepared t2m shape: ", saved_t2m.shape)

        print("\nCollated tas range:")
        print(float(np.nanmin(collated_tas)), "to", float(np.nanmax(collated_tas)))

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
