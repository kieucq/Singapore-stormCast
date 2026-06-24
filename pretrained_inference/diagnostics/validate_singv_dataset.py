#!/usr/bin/env python3
"""
Validate one collated SingV3 NetCDF file.

This script checks that the required SingV3 variables and coordinates exist,
prints missing-value diagnostics, and summarizes the ta/hus/zg zero masks.

Usage:
    python validate_singv_dataset.py COLLATED_FILE

Example:
    python validate_singv_dataset.py \
        ~/scratch/pretrained/singv_collated/singv_collated_20141201_0100.nc
"""

import argparse
import sys
from pathlib import Path

import xarray as xr

# Allow direct execution from the diagnostics directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils as ut

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collated_file",
        type=Path,
        help="Path to a singv_collated_*.nc file.",
    )
    args = parser.parse_args()

    path = args.collated_file.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with xr.open_dataset(
        path,
        mask_and_scale=True,
    ) as ds:
        ut.validate_singv_dataset(ds, verbose=True)


if __name__ == "__main__":
    main()