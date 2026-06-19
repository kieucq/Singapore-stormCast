"""
Validate one collected SingV3 NetCDF file.

This script checks that the required SingV3 variables and coordinates exist,
prints NaN diagnostics, and prints the ta/hus/zg zero-mask summary.

Usage:
        python validate_singv3_dataset.py <file name>.nc
"""

import argparse
import xarray as xr

import utils_singv3 as ut

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to SingV3 NetCDF file")
    args = parser.parse_args()

    ds = xr.open_dataset(args.path, mask_and_scale=True)

    ut.validate_singv_dataset(ds, verbose=True)


if __name__ == "__main__":
    main()
