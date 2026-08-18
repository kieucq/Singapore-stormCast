#!/usr/bin/env python3
"""
Prepare one ERA5 background state from monthly downloaded data for StormCast preprocessing.

This preparation stage:

1. validates the matching raw monthly ERA5 pressure-level and single-level files;
2. selects one exact UTC timestamp;
3. converts ERA5 geopotential ``z`` to geopotential height in metres;
4. interpolates all ERA5 fields onto the exact prepared SINGV target grid;
5. stacks the fixed 26-channel background tensor;
6. saves an unnormalised NetCDF file.

The prepared SINGV file at the same valid time is treated as the authoritative
source of the target 624 x 624 latitude/longitude grid and corresponding
cell-edge bounds.

Usage
-----
python prepare_era5.py --datetime 1995-01-01T01:00

Example
-------
python prepare_era5.py \
    --datetime 1995-01-01T01:00

Default output
--------------
Prepared backgrounds are written beneath BACKGROUND_PREPARED_DIR configured
in paths.py, with names such as:

    background_19950101_0100.nc
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

import paths


VALID_UTC_HOURS = (1, 7, 13, 19)
STANDARD_GRAVITY = 9.80665

PRESSURE_VARIABLES = ("u", "v", "z", "t", "q")
PRESSURE_LEVELS_HPA = (1000, 850, 500, 250)
SINGLE_VARIABLES = ("u10", "v10", "t2m", "tcwv", "msl", "sp")

CHANNEL_NAMES = [
    *(f"u_{level}" for level in PRESSURE_LEVELS_HPA),
    *(f"v_{level}" for level in PRESSURE_LEVELS_HPA),
    *(f"z_{level}" for level in PRESSURE_LEVELS_HPA),
    *(f"t_{level}" for level in PRESSURE_LEVELS_HPA),
    *(f"q_{level}" for level in PRESSURE_LEVELS_HPA),
    "u10",
    "v10",
    "t2m",
    "tcwv",
    "mslp",
    "sp",
]

CHANNEL_UNITS = [
    *(["m s-1"] * len(PRESSURE_LEVELS_HPA)),
    *(["m s-1"] * len(PRESSURE_LEVELS_HPA)),
    *(["m"] * len(PRESSURE_LEVELS_HPA)),
    *(["K"] * len(PRESSURE_LEVELS_HPA)),
    *(["kg kg-1"] * len(PRESSURE_LEVELS_HPA)),
    "m s-1",
    "m s-1",
    "K",
    "kg m-2",
    "Pa",
    "Pa",
]

if len(CHANNEL_NAMES) != 26:
    raise RuntimeError(f"Expected 26 background channels, got {len(CHANNEL_NAMES)}")


def parse_valid_datetime(value: str) -> datetime:
    """Parse and validate one ERA5/SINGV valid time."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime {value!r}; expected YYYY-MM-DDTHH:MM."
        ) from exc

    if parsed.minute != 0 or parsed.second != 0 or parsed.microsecond != 0:
        raise argparse.ArgumentTypeError(
            "Datetime must be on an exact UTC hour boundary."
        )

    if parsed.hour not in VALID_UTC_HOURS:
        raise argparse.ArgumentTypeError(
            f"Hour must be one of {VALID_UTC_HOURS}; got {parsed.hour:02d}:00."
        )

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate one raw ERA5 monthly background state onto the exact "
            "prepared SINGV grid and save a 26-channel unnormalised NetCDF file."
        )
    )
    parser.add_argument(
        "--datetime",
        dest="valid_time",
        required=True,
        type=parse_valid_datetime,
        help="UTC valid time in YYYY-MM-DDTHH:MM format; hour must be 01, 07, 13, or 19.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=paths.BACKGROUND_RAW_DIR,
        help=(
            "Root directory containing raw monthly ERA5 files "
            f"(default: {paths.BACKGROUND_RAW_DIR})"
        ),
    )   
    parser.add_argument(
        "--singv-prepared-dir",
        type=Path,
        default=paths.PREPARED_DIR,
        help=(
            "Directory containing prepared SINGV states used as the authoritative "
            f"target grid (default: {paths.PREPARED_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.BACKGROUND_PREPARED_DIR,
        help=f"Output directory (default: {paths.BACKGROUND_PREPARED_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output file path. Overrides --output-dir when supplied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed diagnostics and the output summary.",
    )
    return parser.parse_args()


def make_output_path(valid_time: datetime, output_dir: Path) -> Path:
    timestamp = valid_time.strftime("%Y%m%d_%H%M")
    return output_dir.expanduser() / f"background_{timestamp}.nc"


def raw_pressure_path(valid_time: datetime, raw_root: Path) -> Path:
    year = valid_time.strftime("%Y")
    year_month = valid_time.strftime("%Y%m")
    return raw_root.expanduser() / year / f"era5_pressure_{year_month}.nc"


def raw_single_path(valid_time: datetime, raw_root: Path) -> Path:
    year = valid_time.strftime("%Y")
    year_month = valid_time.strftime("%Y%m")
    return raw_root.expanduser() / year / f"era5_single_{year_month}.nc"


def singv_prepared_path(valid_time: datetime, prepared_dir: Path) -> Path:
    timestamp = valid_time.strftime("%Y%m%d_%H%M")
    return prepared_dir.expanduser() / f"prepared_{timestamp}.nc"


def validate_raw_pressure_dataset(ds: xr.Dataset) -> None:
    required_coords = ("valid_time", "pressure_level", "latitude", "longitude")
    missing_coords = [name for name in required_coords if name not in ds.coords]
    if missing_coords:
        raise ValueError(f"Raw pressure dataset is missing coordinates: {missing_coords}")

    missing_variables = [name for name in PRESSURE_VARIABLES if name not in ds.data_vars]
    if missing_variables:
        raise ValueError(f"Raw pressure dataset is missing variables: {missing_variables}")

    expected_dims = ("valid_time", "pressure_level", "latitude", "longitude")
    for name in PRESSURE_VARIABLES:
        if ds[name].dims != expected_dims:
            raise ValueError(
                f"{name} has dimensions {ds[name].dims}; expected {expected_dims}."
            )

    actual_levels = np.asarray(ds["pressure_level"].values, dtype=np.float64)
    expected_levels = np.asarray(PRESSURE_LEVELS_HPA, dtype=np.float64)
    if actual_levels.shape != expected_levels.shape or not np.array_equal(
        actual_levels, expected_levels
    ):
        raise ValueError(
            "Unexpected ERA5 pressure levels or ordering.\n"
            f"Expected: {expected_levels.tolist()}\n"
            f"Found:    {actual_levels.tolist()}"
        )

    longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
    latitude = np.asarray(ds["latitude"].values, dtype=np.float64)

    if not np.all(np.diff(longitude) > 0):
        raise ValueError("ERA5 longitude must be strictly increasing.")

    if not (np.all(np.diff(latitude) > 0) or np.all(np.diff(latitude) < 0)):
        raise ValueError("ERA5 latitude must be strictly monotonic.")


def validate_raw_single_dataset(ds: xr.Dataset) -> None:
    required_coords = ("valid_time", "latitude", "longitude")
    missing_coords = [name for name in required_coords if name not in ds.coords]
    if missing_coords:
        raise ValueError(f"Raw single-level dataset is missing coordinates: {missing_coords}")

    missing_variables = [name for name in SINGLE_VARIABLES if name not in ds.data_vars]
    if missing_variables:
        raise ValueError(f"Raw single-level dataset is missing variables: {missing_variables}")

    expected_dims = ("valid_time", "latitude", "longitude")
    for name in SINGLE_VARIABLES:
        if ds[name].dims != expected_dims:
            raise ValueError(
                f"{name} has dimensions {ds[name].dims}; expected {expected_dims}."
            )

    longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
    latitude = np.asarray(ds["latitude"].values, dtype=np.float64)

    if not np.all(np.diff(longitude) > 0):
        raise ValueError("ERA5 longitude must be strictly increasing.")

    if not (np.all(np.diff(latitude) > 0) or np.all(np.diff(latitude) < 0)):
        raise ValueError("ERA5 latitude must be strictly monotonic.")


def validate_matching_raw_grids(
    pressure_ds: xr.Dataset,
    single_ds: xr.Dataset,
) -> None:
    pressure_lat = np.asarray(pressure_ds["latitude"].values, dtype=np.float64)
    pressure_lon = np.asarray(pressure_ds["longitude"].values, dtype=np.float64)
    single_lat = np.asarray(single_ds["latitude"].values, dtype=np.float64)
    single_lon = np.asarray(single_ds["longitude"].values, dtype=np.float64)

    if not np.array_equal(pressure_lat, single_lat):
        raise ValueError("Pressure and single-level ERA5 latitude coordinates do not match.")

    if not np.array_equal(pressure_lon, single_lon):
        raise ValueError("Pressure and single-level ERA5 longitude coordinates do not match.")


def validate_singv_target_dataset(ds: xr.Dataset, valid_time: np.datetime64) -> None:
    required_coords = (
        "time",
        "y",
        "x",
        "latitude",
        "longitude",
        "latitude_bounds",
        "longitude_bounds",
    )
    missing_coords = [name for name in required_coords if name not in ds.coords]
    if missing_coords:
        raise ValueError(
            f"Prepared SINGV dataset is missing required coordinates: {missing_coords}"
        )

    if ds.sizes.get("time") != 1:
        raise ValueError(
            f"Prepared SINGV dataset must contain exactly one time step, found {ds.sizes.get('time')}."
        )

    expected_shape = (624, 624)
    actual_shape = (ds.sizes.get("y"), ds.sizes.get("x"))
    if actual_shape != expected_shape:
        raise ValueError(
            f"Prepared SINGV grid must be {expected_shape}, got {actual_shape}."
        )

    if ds["latitude"].dims != ("y",):
        raise ValueError(f"SINGV latitude has dims {ds['latitude'].dims}; expected ('y',).")

    if ds["longitude"].dims != ("x",):
        raise ValueError(f"SINGV longitude has dims {ds['longitude'].dims}; expected ('x',).")

    singv_time = np.datetime64(np.asarray(ds["time"].values).squeeze(), "ns")
    if singv_time != valid_time:
        raise ValueError(
            "Prepared SINGV valid time does not match requested datetime.\n"
            f"Requested: {valid_time}\n"
            f"Found:     {singv_time}"
        )

    latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
    longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
    if not np.all(np.diff(latitude) > 0):
        raise ValueError("Prepared SINGV latitude must be strictly increasing.")
    if not np.all(np.diff(longitude) > 0):
        raise ValueError("Prepared SINGV longitude must be strictly increasing.")


def ensure_target_within_source(
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
) -> None:
    source_lat_min = float(np.min(source_latitude))
    source_lat_max = float(np.max(source_latitude))
    source_lon_min = float(np.min(source_longitude))
    source_lon_max = float(np.max(source_longitude))

    target_lat_min = float(np.min(target_latitude))
    target_lat_max = float(np.max(target_latitude))
    target_lon_min = float(np.min(target_longitude))
    target_lon_max = float(np.max(target_longitude))

    if target_lat_min < source_lat_min or target_lat_max > source_lat_max:
        raise ValueError(
            "Target latitude range is not fully contained within the ERA5 source grid.\n"
            f"Target: {target_lat_min} to {target_lat_max}\n"
            f"Source: {source_lat_min} to {source_lat_max}"
        )

    if target_lon_min < source_lon_min or target_lon_max > source_lon_max:
        raise ValueError(
            "Target longitude range is not fully contained within the ERA5 source grid.\n"
            f"Target: {target_lon_min} to {target_lon_max}\n"
            f"Source: {source_lon_min} to {source_lon_max}"
        )


def prepare_pressure_selection(
    pressure_ds: xr.Dataset,
    valid_time: np.datetime64,
) -> xr.Dataset:
    selection = pressure_ds.sel(valid_time=valid_time, pressure_level=list(PRESSURE_LEVELS_HPA))
    selection = selection.sortby("latitude")
    selection = selection.sortby("longitude")

    # Convert geopotential to geopotential height.
    selection = selection.copy()
    selection["z"] = selection["z"] / STANDARD_GRAVITY
    selection["z"].attrs["units"] = "m"
    selection["z"].attrs["long_name"] = "geopotential height"
    selection["z"].attrs["converted_from"] = "ERA5 geopotential"
    selection["z"].attrs["standard_gravity"] = STANDARD_GRAVITY

    return selection


def prepare_single_selection(
    single_ds: xr.Dataset,
    valid_time: np.datetime64,
) -> xr.Dataset:
    selection = single_ds.sel(valid_time=valid_time)
    selection = selection.sortby("latitude")
    selection = selection.sortby("longitude")
    return selection


def interpolate_to_target_grid(
    source: xr.Dataset,
    *,
    target_latitude: xr.DataArray,
    target_longitude: xr.DataArray,
) -> xr.Dataset:
    interpolated = source.interp(
        latitude=target_latitude,
        longitude=target_longitude,
        method="linear",
    )

    return interpolated


def build_background_array(
    pressure_interp: xr.Dataset,
    single_interp: xr.Dataset,
) -> np.ndarray:
    arrays: list[np.ndarray] = []

    for variable in ("u", "v", "z", "t", "q"):
        for level in PRESSURE_LEVELS_HPA:
            array = np.asarray(
                pressure_interp[variable].sel(pressure_level=level).values,
                dtype=np.float32,
            )
            arrays.append(array)

    arrays.extend(
        np.asarray(single_interp[name].values, dtype=np.float32)
        for name in SINGLE_VARIABLES
    )

    if len(arrays) != 26:
        raise RuntimeError(f"Expected 26 channels, assembled {len(arrays)}")

    stacked = np.stack(arrays, axis=0)

    if stacked.shape != (26, 624, 624):
        raise ValueError(
            f"Expected stacked background shape (26, 624, 624), got {stacked.shape}."
        )

    if not np.all(np.isfinite(stacked)):
        raise ValueError("Interpolated background contains NaN or infinite values.")

    return stacked


def build_output_dataset(
    background: np.ndarray,
    *,
    singv_ds: xr.Dataset,
    pressure_path: Path,
    single_path: Path,
) -> xr.Dataset:
    valid_time = np.asarray(singv_ds["time"].values)

    dataset = xr.Dataset(
        data_vars={
            "background": (
                ("time", "channel", "y", "x"),
                background[np.newaxis, ...],
                {
                    "long_name": "ERA5 background channels interpolated onto prepared SINGV grid",
                },
            ),
            "channel_units": (
                ("channel",),
                np.asarray(CHANNEL_UNITS, dtype="U16"),
            ),
        },
        coords={
            "time": (
                ("time",),
                valid_time,
                dict(singv_ds["time"].attrs),
            ),
            "channel": (
                ("channel",),
                np.asarray(CHANNEL_NAMES, dtype="U8"),
            ),
            "y": (
                ("y",),
                np.asarray(singv_ds["y"].values, dtype=np.int32),
                dict(singv_ds["y"].attrs),
            ),
            "x": (
                ("x",),
                np.asarray(singv_ds["x"].values, dtype=np.int32),
                dict(singv_ds["x"].attrs),
            ),
            "latitude": (
                ("y",),
                np.asarray(singv_ds["latitude"].values, dtype=np.float64),
                dict(singv_ds["latitude"].attrs),
            ),
            "longitude": (
                ("x",),
                np.asarray(singv_ds["longitude"].values, dtype=np.float64),
                dict(singv_ds["longitude"].attrs),
            ),
            "latitude_bounds": (
                singv_ds["latitude_bounds"].dims,
                np.asarray(singv_ds["latitude_bounds"].values, dtype=np.float64),
                dict(singv_ds["latitude_bounds"].attrs),
            ),
            "longitude_bounds": (
                singv_ds["longitude_bounds"].dims,
                np.asarray(singv_ds["longitude_bounds"].values, dtype=np.float64),
                dict(singv_ds["longitude_bounds"].attrs),
            ),
        },
        attrs={
            "source": "ERA5 hourly reanalysis",
            "processing_stage": "prepared_background",
            "normalisation": "none",
            "grid_source": "prepared SINGV state",
            "interpolation": "xarray.interp linear",
            "pressure_levels_hpa": ", ".join(str(level) for level in PRESSURE_LEVELS_HPA),
            "geopotential_conversion": f"z divided by {STANDARD_GRAVITY} to obtain geopotential height in metres",
            "channel_order": ", ".join(CHANNEL_NAMES),
            "raw_pressure_file": str(pressure_path),
            "raw_single_file": str(single_path),
        },
    )

    return dataset


def save_dataset(
    dataset: xr.Dataset,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    encoding = {
        "background": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, 1, 156, 156),
        },
        "time": {
            "dtype": "int64",
            "units": "hours since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "_FillValue": None,
        },
        "channel": {
            "dtype": "S1",
        },
        "y": {
            "dtype": "int32",
            "_FillValue": None,
        },
        "x": {
            "dtype": "int32",
            "_FillValue": None,
        },
        "latitude": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "longitude": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "latitude_bounds": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "longitude_bounds": {
            "dtype": "float64",
            "_FillValue": None,
        },
    }

    dataset.to_netcdf(
        output_path,
        format="NETCDF4",
        encoding=encoding,
    )


def print_summary(dataset: xr.Dataset, output_path: Path) -> None:
    background = dataset["background"]
    size_mb = output_path.stat().st_size / (1024**2)

    print("\nOutput summary")
    print("--------------")
    print("Path:                  ", output_path)
    print(f"File size:              {size_mb:.1f} MB")
    print("Background dimensions:  ", background.dims)
    print("Background shape:       ", background.shape)
    print("Background dtype:       ", background.dtype)
    print("Finite values:          ", bool(np.isfinite(background.values).all()))
    print("First channels:         ", dataset["channel"].values[:8].tolist())
    print("Last channels:          ", dataset["channel"].values[-6:].tolist())


def prepare_era5(
    valid_time: datetime,
    output_path: Path | None = None,
    *,
    raw_root: Path = paths.BACKGROUND_RAW_DIR,
    singv_prepared_dir: Path = paths.PREPARED_DIR,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Prepare one ERA5 background state and save it as a NetCDF file.

    Parameters
    ----------
    valid_time
        UTC valid time. Hour must be 01, 07, 13, or 19.
    output_path
        Explicit prepared output path. When omitted, BACKGROUND_PREPARED_DIR
        configured in paths.py is used.
    raw_root
        Root directory containing raw monthly ERA5 files.
    singv_prepared_dir
        Directory containing prepared SINGV states used as the authoritative
        target grid.
    overwrite
        Replace an existing output file.
    verbose
        Print diagnostics and the output summary.

    Returns
    -------
    Path
        Path to the prepared ERA5 background NetCDF file.
    """
    if valid_time.hour not in VALID_UTC_HOURS or any(
        (valid_time.minute, valid_time.second, valid_time.microsecond)
    ):
        raise ValueError(
            f"Valid time must be one of the supported UTC hours {VALID_UTC_HOURS} on an exact hour boundary."
        )

    if output_path is None:
        output_path = make_output_path(valid_time, paths.BACKGROUND_PREPARED_DIR)
    else:
        output_path = output_path.expanduser()

    pressure_path = raw_pressure_path(valid_time, raw_root)
    single_path = raw_single_path(valid_time, raw_root)
    singv_path = singv_prepared_path(valid_time, singv_prepared_dir)

    missing_paths = [path for path in (pressure_path, single_path, singv_path) if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Required input files not found:\n" + "\n".join(f"- {path}" for path in missing_paths)
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use overwrite=True to replace it."
        )

    requested_np_time = np.datetime64(valid_time, "ns")

    with (
        xr.open_dataset(pressure_path, mask_and_scale=True) as pressure_ds,
        xr.open_dataset(single_path, mask_and_scale=True) as single_ds,
        xr.open_dataset(singv_path, mask_and_scale=True) as singv_ds,
    ):
        validate_raw_pressure_dataset(pressure_ds)
        validate_raw_single_dataset(single_ds)
        validate_matching_raw_grids(pressure_ds, single_ds)
        validate_singv_target_dataset(singv_ds, requested_np_time)

        target_latitude = singv_ds["latitude"]
        target_longitude = singv_ds["longitude"]

        source_latitude = np.asarray(pressure_ds["latitude"].values, dtype=np.float64)
        source_longitude = np.asarray(pressure_ds["longitude"].values, dtype=np.float64)
        ensure_target_within_source(
            np.asarray(target_latitude.values, dtype=np.float64),
            np.asarray(target_longitude.values, dtype=np.float64),
            source_latitude,
            source_longitude,
        )

        pressure_selected = prepare_pressure_selection(pressure_ds, requested_np_time)
        single_selected = prepare_single_selection(single_ds, requested_np_time)

        pressure_interp = interpolate_to_target_grid(
            pressure_selected,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
        )
        single_interp = interpolate_to_target_grid(
            single_selected,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
        )

        background = build_background_array(pressure_interp, single_interp)
        output_dataset = build_output_dataset(
            background,
            singv_ds=singv_ds,
            pressure_path=pressure_path,
            single_path=single_path,
        )

    save_dataset(output_dataset, output_path, overwrite=overwrite)

    if verbose:
        print_summary(output_dataset, output_path)

    return output_path


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser()
    singv_prepared_dir = args.singv_prepared_dir.expanduser()

    output_path = args.output
    if output_path is None:
        output_path = make_output_path(args.valid_time, args.output_dir)

    if not args.quiet:
        pressure_path = raw_pressure_path(args.valid_time, raw_root)
        single_path = raw_single_path(args.valid_time, raw_root)
        target_singv = singv_prepared_path(args.valid_time, singv_prepared_dir)
        print("ERA5 BACKGROUND PREPARATION")
        print("===========================")
        print("Valid time:       ", args.valid_time.isoformat(timespec="minutes"))
        print("Pressure file:    ", pressure_path)
        print("Single-level file:", single_path)
        print("Target SINGV:     ", target_singv)
        print("Output:           ", output_path.expanduser())
        print("Overwrite:        ", args.overwrite)

    start = perf_counter()
    written = prepare_era5(
        args.valid_time,
        output_path=output_path,
        raw_root=raw_root,
        singv_prepared_dir=singv_prepared_dir,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )
    elapsed = perf_counter() - start

    if not args.quiet:
        print(f"\nFinished in {elapsed:.1f} s")
        print("Wrote:", written)


if __name__ == "__main__":
    main()
