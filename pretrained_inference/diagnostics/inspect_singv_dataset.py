#!/usr/bin/env python3
"""
Print simple diagnostic tables for one collated SingV3 NetCDF file.

Usage:
    python inspect_singv_dataset.py COLLATED_FILE

Example:
    python inspect_singv_dataset.py \
        ~/scratch/pretrained/singv_collated/singv_collated_19950115_1900.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


SURFACE_VARS = ["tas", "uas", "vas", "psl", "pr"]
PRESSURE_VARS = ["ta", "hus", "ua", "va", "zg"]


def fmt(x: Any, ndigits: int = 6) -> str:
    """Readable formatting for floats / missing values."""
    if x is None:
        return "NA"

    try:
        if np.isnan(x):
            return "NA"
    except TypeError:
        pass

    if isinstance(x, (float, np.floating)):
        return f"{x:.{ndigits}g}"

    return str(x)


def get_fill_value(raw_da: xr.DataArray) -> float | None:
    """
    Return the declared NetCDF missing value.

    This is read from the raw dataset opened with mask_and_scale=False,
    so _FillValue / missing_value still exist in attrs.
    """
    for key in ["_FillValue", "missing_value"]:
        if key in raw_da.attrs:
            value = np.asarray(raw_da.attrs[key])
            if value.size == 1:
                return float(value.item())

    return None


def select_first_time_if_present(da: xr.DataArray) -> xr.DataArray:
    """If a variable still has a time dimension, inspect only the first time."""
    if "time" in da.dims:
        return da.isel(time=0, drop=True)
    return da


def raw_fill_fraction(raw_arr: np.ndarray, fill_value: float | None) -> float | None:
    """Fraction of raw stored values equal to _FillValue / missing_value."""
    if fill_value is None:
        return None

    raw_arr = np.asarray(raw_arr)

    if raw_arr.size == 0:
        return None

    return float(np.isclose(raw_arr, fill_value).mean())


def stats(arr: np.ndarray) -> dict[str, float | None]:
    """Basic stats, ignoring NaN/Inf."""
    arr = np.asarray(arr)

    if arr.size == 0:
        return {
            "missing_fraction": None,
            "zero_fraction": None,
            "min": None,
            "p01": None,
            "mean": None,
            "p99": None,
            "max": None,
            "min_nonzero": None,
            "max_nonzero": None,
        }

    finite = np.isfinite(arr)

    out: dict[str, float | None] = {
        "missing_fraction": float(1.0 - finite.mean()),
        "zero_fraction": None,
        "min": None,
        "p01": None,
        "mean": None,
        "p99": None,
        "max": None,
        "min_nonzero": None,
        "max_nonzero": None,
    }

    if not finite.any():
        return out

    vals = arr[finite]
    zero = np.isclose(vals, 0.0)

    out["zero_fraction"] = float(zero.mean())
    out["min"] = float(np.nanmin(vals))
    out["p01"] = float(np.nanpercentile(vals, 1))
    out["mean"] = float(np.nanmean(vals))
    out["p99"] = float(np.nanpercentile(vals, 99))
    out["max"] = float(np.nanmax(vals))

    nonzero = vals[~zero]
    if nonzero.size:
        out["min_nonzero"] = float(np.nanmin(nonzero))
        out["max_nonzero"] = float(np.nanmax(nonzero))

    return out


def print_surface_table(ds: xr.Dataset, ds_raw: xr.Dataset) -> None:
    print("\nSurface variables")

    header = (
        f"{'var':<8} {'units':<14} {'raw_fill%':>10} {'missing%':>10} {'zero%':>10} "
        f"{'min':>12} {'p01':>12} {'mean':>12} {'p99':>12} {'max':>12}"
    )

    print(header)
    print("-" * len(header))

    for var in SURFACE_VARS:
        if var not in ds:
            print(f"{var:<8} NOT FOUND")
            continue

        da = select_first_time_if_present(ds[var])
        da_raw = select_first_time_if_present(ds_raw[var])

        fill_value = get_fill_value(ds_raw[var])
        fill_frac = raw_fill_fraction(da_raw.values, fill_value)
        s = stats(da.values)

        print(
            f"{var:<8} "
            f"{ds[var].attrs.get('units', 'NA'):<14} "
            f"{fmt(100 * fill_frac if fill_frac is not None else None):>10} "
            f"{fmt(100 * s['missing_fraction']):>10} "
            f"{fmt(100 * s['zero_fraction']):>10} "
            f"{fmt(s['min']):>12} "
            f"{fmt(s['p01']):>12} "
            f"{fmt(s['mean']):>12} "
            f"{fmt(s['p99']):>12} "
            f"{fmt(s['max']):>12}"
        )


def print_pressure_tables(ds: xr.Dataset, ds_raw: xr.Dataset) -> None:
    print("\nPressure-level variables")

    if "plev" not in ds:
        raise ValueError("No plev coordinate found in dataset.")

    plevs = np.asarray(ds["plev"].values, dtype=float)

    for var in PRESSURE_VARS:
        if var not in ds:
            print(f"\n{var}: NOT FOUND")
            continue

        print(
            f"\n{var}  "
            f"units={ds[var].attrs.get('units', 'NA')}  "
            f"long_name={ds[var].attrs.get('long_name', 'NA')}"
        )

        header = (
            f"{'plev Pa':>10} {'hPa':>8} {'raw_fill%':>10} {'missing%':>10} {'zero%':>10} "
            f"{'min':>12} {'p01':>12} {'mean':>12} {'p99':>12} {'max':>12} "
            f"{'min_nonzero':>12} {'max_nonzero':>12}"
        )

        print(header)
        print("-" * len(header))

        da_all = select_first_time_if_present(ds[var])
        da_raw_all = select_first_time_if_present(ds_raw[var])

        fill_value = get_fill_value(ds_raw[var])

        for plev in plevs:
            da = da_all.sel(plev=plev)
            da_raw = da_raw_all.sel(plev=plev)

            fill_frac = raw_fill_fraction(da_raw.values, fill_value)
            s = stats(da.values)

            print(
                f"{plev:10.0f} {plev / 100.0:8.1f} "
                f"{fmt(100 * fill_frac if fill_frac is not None else None):>10} "
                f"{fmt(100 * s['missing_fraction']):>10} "
                f"{fmt(100 * s['zero_fraction']):>10} "
                f"{fmt(s['min']):>12} "
                f"{fmt(s['p01']):>12} "
                f"{fmt(s['mean']):>12} "
                f"{fmt(s['p99']):>12} "
                f"{fmt(s['max']):>12} "
                f"{fmt(s['min_nonzero']):>12} "
                f"{fmt(s['max_nonzero']):>12}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collated_file",
        type=Path,
        help="Path to a singv_collated_*.nc file.",
    )
    args = parser.parse_args()

    path = args.collated_file.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ds = xr.open_dataset(path, mask_and_scale=True)
    ds_raw = xr.open_dataset(path, mask_and_scale=False)

    if "valid_time" in ds:
        print(f"valid_time: {ds['valid_time'].values}")

    print_surface_table(ds, ds_raw)
    print_pressure_tables(ds, ds_raw)


if __name__ == "__main__":
    main()