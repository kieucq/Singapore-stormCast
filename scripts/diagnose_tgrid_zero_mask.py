"""
Diagnose the shared zero mask in SingV3 T-grid pressure-level variables.

This script checks whether `ta`, `hus`, and `zg` have identical zero-valued
regions across pressure levels. These zero regions are suspected to represent
missing or below-surface pressure-level data.

Usage:
    python diagnose_tgrid_zero_mask.py <file name>.nc
"""

import argparse
import xarray as xr

from utils_singv3 import (
    summarize_tgrid_zero_mask,
    print_tgrid_zero_mask_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to SingV3 NetCDF file")
    args = parser.parse_args()

    ds = xr.open_dataset(args.path, mask_and_scale=True)

    stats = summarize_tgrid_zero_mask(ds)
    print_tgrid_zero_mask_summary(stats)


if __name__ == "__main__":
    main()
