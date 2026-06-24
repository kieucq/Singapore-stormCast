#!/usr/bin/env python3
"""
Collect the SingV3 variables required by the pretrained StormCast input
pipeline for one requested datetime, and combine them into one NetCDF file.

Usage:
    python collect_singv.py --datetime 2014-12-01T01:00

Optional explicit output path:
    python collect_singv.py \
        --datetime 2014-12-01T01:00 \
        --output PATH_TO_OUTPUT_FILE

Default output:
    ~/scratch/pretrained/singv_collated/singv_collated_20141201_0100.nc

Pressure-level variables are available every six hours at 01/07/13/19 UTC,
according to the archive filename convention. Surface variables are available
hourly. The requested hour must therefore be one of 01, 07, 13, or 19 UTC.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr

# ── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = Path(
    "/home/project/13004327/data_service/model_data/V3_Historical/"
    "V3-WMC-2/CCRS/ERA5/historical/reanalysis/SINGV-RCM/vn5"
)

PRETRAINED_DIR = Path("~/scratch/pretrained").expanduser()
OUTPUT_DIR = PRETRAINED_DIR / "singv_collated"

# Variables to collect, with their frequency directory and StormCast role
SURFACE_VARS = {
    "tas":  "t2m",    # near-surface temperature, 1.5m in SINGV
    "uas":  "u10m",   # 10m u-wind
    "vas":  "v10m",   # 10m v-wind
    "psl":  "mslp",   # mean sea level pressure
    "pr": "refc_proxy",  # stratiform rainfall flux -> approximate reflectivity proxy
}

PRESSURE_VARS = {
    "ua": "u",   # zonal wind
    "va": "v",   # meridional wind
    "ta": "t",   # temperature
    "hus": "q",  # specific humidity
    "zg": "Z",   # geopotential height (already in metres)
}

VALID_PRESSURE_HOURS = {1, 7, 13, 19}


# ── File discovery ────────────────────────────────────────────────────────

def find_surface_file(varname: str, dt: datetime) -> Path:
    """
    1hr files are one per day:
        <var>_..._1hr_YYYYMMDD0000-YYYYMMDD2300.nc
    """
    yyyymm = dt.strftime("%Y%m")
    yyyymmdd = dt.strftime("%Y%m%d")
    pattern = f"{varname}_*_1hr_{yyyymmdd}0000-{yyyymmdd}2300.nc"

    directory = BASE_DIR / "1hr" / varname / yyyymm
    matches = sorted(directory.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No 1hr file found for {varname} on {yyyymmdd} in {directory}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple 1hr files matched for {varname} on {yyyymmdd}: {matches}")

    return matches[0]


def find_pressure_file(varname: str, dt: datetime) -> Path:
    """
    6hr files are one per day, named with the date of the FIRST (01:00) timestep:
        <var>_..._6hr_YYYYMMDD0100-YYYYMMDD1900.nc
    """
    yyyymm = dt.strftime("%Y%m")
    yyyymmdd = dt.strftime("%Y%m%d")
    pattern = f"{varname}_*_6hr_{yyyymmdd}0100-{yyyymmdd}1900.nc"

    directory = BASE_DIR / "6hr" / varname / yyyymm
    matches = sorted(directory.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No 6hr file found for {varname} on {yyyymmdd} in {directory}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple 6hr files matched for {varname} on {yyyymmdd}: {matches}")

    return matches[0]


# ── Extraction ────────────────────────────────────────────────────────────

def extract_surface_field(varname: str, dt: datetime) -> xr.DataArray:
    """Extract a single hourly timestep from a 1hr surface file."""
    path = find_surface_file(varname, dt)
    ds = xr.open_dataset(path, decode_times=False)

    # time coordinate is "hours since <file's reference date> 00:00:00"
    # hour-of-day == dt.hour, since each file covers exactly one day (00..23)
    field = ds[varname].isel(time=dt.hour, drop=True)

    field = field.load()  # read into memory before closing
    ds.close()

    return field.assign_coords(valid_time=np.datetime64(dt, "ns"))


def extract_pressure_field(varname: str, dt: datetime) -> xr.DataArray:
    """Extract a single 6hr timestep (01/07/13/19) from a 6hr pressure-level file."""
    path = find_pressure_file(varname, dt)
    ds = xr.open_dataset(path, decode_times=False)

    # 6hr files contain 4 timesteps: index 0=01h, 1=07h, 2=13h, 3=19h
    time_index = VALID_PRESSURE_HOURS_SORTED.index(dt.hour)
    field = ds[varname].isel(time=time_index, drop=True)

    field = field.load()
    ds.close()

    return field.assign_coords(valid_time=np.datetime64(dt, "ns"))


VALID_PRESSURE_HOURS_SORTED = sorted(VALID_PRESSURE_HOURS)


# ── Main ──────────────────────────────────────────────────────────────────

def collect(dt: datetime, output_path: Path):
    if dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        raise ValueError("Requested datetime must be exactly on the hour.")

    if dt.hour not in VALID_PRESSURE_HOURS:
        raise ValueError(
            f"Requested hour {dt.hour:02d}:00 is not a valid pressure-level "
            f"timestep. Must be one of {sorted(VALID_PRESSURE_HOURS)} (UTC)."
        )

    print(f"Collecting SINGV-RCM data for {dt.isoformat()}...")

    data_vars = {}

    # Surface (1hr) variables
    print("\nSurface variables (1hr):")
    for varname, role in SURFACE_VARS.items():
        print(f"  {varname:6s} ({role})...", end=" ", flush=True)
        field = extract_surface_field(varname, dt)
        data_vars[varname] = field
        print(f"shape={field.shape}, min={float(field.min()):.4g}, max={float(field.max()):.4g}")

    # Pressure-level (6hr) variables
    print("\nPressure-level variables (6hr):")
    for varname, role in PRESSURE_VARS.items():
        print(f"  {varname:6s} ({role})...", end=" ", flush=True)
        field = extract_pressure_field(varname, dt)
        data_vars[varname] = field
        print(f"shape={field.shape}, min={float(field.min()):.4g}, max={float(field.max()):.4g}")

    # Combine into a single dataset
    ds_out = xr.Dataset(data_vars)
    ds_out.attrs["source"] = "SINGV-RCM ERA5-driven reanalysis (CCRS), vn5"
    ds_out.attrs["requested_valid_time"] = dt.isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_path)
    print(f"\nWrote: {output_path}")

def default_output_path(dt: datetime) -> Path:
    filename = f"singv_collated_{dt.strftime('%Y%m%d_%H%M')}.nc"
    return OUTPUT_DIR / filename

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datetime", required=True,
        help="Requested valid time, ISO format e.g. 2014-12-01T01:00. "
             "Hour must be one of 01, 07, 13, 19 (UTC).",
    )
    parser.add_argument(
        "--output", required=False, type=Path, default=None,
        help=(
            "Optional output NetCDF path. If omitted, the file is written to "
            "~/scratch/pretrained/singv_collated/ using the name "
            "singv_collated_YYYYMMDD_HHMM.nc."
        ),
    )
    args = parser.parse_args()

    try:
        dt = datetime.fromisoformat(args.datetime)
    except ValueError as e:
        print(f"Error parsing --datetime: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        args.output.expanduser()
        if args.output is not None
        else default_output_path(dt)
    )

    try:
        collect(dt, output_path)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()