#!/usr/bin/env python3
"""
Download monthly ERA5 background data for SINGV StormCast retraining.

The downloader follows the ERA5 conditioning specification used in the
original StormCast paper:

Pressure-level data (20 channels)
---------------------------------
Five variables at 1000, 850, 500, and 250 hPa:

    u_component_of_wind
    v_component_of_wind
    geopotential
    temperature
    specific_humidity

Single-level data (6 channels)
------------------------------

    10m_u_component_of_wind
    10m_v_component_of_wind
    2m_temperature
    total_column_water_vapour
    mean_sea_level_pressure
    surface_pressure

Only 01:00, 07:00, 13:00, and 19:00 UTC are requested. These are the ERA5
background times paired with SINGV input states at the same valid times.

Files are downloaded one calendar month at a time. If a requested date range
starts or ends partway through a month, the complete month is downloaded so
that each monthly file is reusable by later dataset ranges.

Default retrieval area
----------------------
The default area is a 0.25-degree-aligned box that fully encloses the prepared
SINGV domain and provides a small interpolation margin:

    north = 10.25
    west  = 92.75
    south = -7.50
    east  = 110.75

The raw ERA5 data are not yet on the 624 x 624 SINGV grid. A later preparation
stage must interpolate these files onto the exact latitude and longitude
coordinates stored in each prepared SINGV state.

Usage
-----
Download all required raw ERA5 data for January 1995:

    python download_era5.py \
        --start-date 1995-01-01 \
        --end-date 1995-01-31

Preview the requests without contacting CDS:

    python download_era5.py \
        --start-date 1995-01-01 \
        --end-date 1995-02-07 \
        --dry-run

Download only one product type:

    python download_era5.py \
        --start-date 1995-01-01 \
        --end-date 1995-01-31 \
        --products pressure

Default outputs
---------------

    ~/scratch/retraining/background/raw/YYYY/
        era5_pressure_YYYYMM.nc
        era5_single_YYYYMM.nc

Requirements
------------
- cdsapi >= 0.7.7
- a valid ~/.cdsapirc
- accepted CDS licences for both ERA5 datasets
- xarray and a NetCDF backend for post-download validation
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import xarray as xr


# ── Fixed StormCast ERA5 specification ───────────────────────────────────────

PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"

PRESSURE_VARIABLES = (
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
)

SINGLE_LEVEL_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_column_water_vapour",
    "mean_sea_level_pressure",
    "surface_pressure",
)

PRESSURE_LEVELS_HPA = ("1000", "850", "500", "250")
VALID_TIMES_UTC = ("01:00", "07:00", "13:00", "19:00")

# CDS area ordering is north, west, south, east.
DEFAULT_AREA = (10.25, 92.75, -7.50, 110.75)

DEFAULT_OUTPUT_DIR = Path("~/scratch/retraining/background/raw")
PRODUCTS = ("pressure", "single")
EARLIEST_ERA5_DATE = date(1940, 1, 1)


@dataclass(frozen=True, order=True)
class YearMonth:
    """One calendar month."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 1:
            raise ValueError(f"Invalid year: {self.year}")
        if not 1 <= self.month <= 12:
            raise ValueError(f"Invalid month: {self.month}")

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def compact(self) -> str:
        return f"{self.year:04d}{self.month:02d}"


@dataclass(frozen=True)
class ProductSpec:
    """CDS dataset and output details for one ERA5 product type."""

    name: str
    dataset: str
    variables: tuple[str, ...]
    filename_prefix: str
    pressure_levels: tuple[str, ...] | None = None


PRODUCT_SPECS: dict[str, ProductSpec] = {
    "pressure": ProductSpec(
        name="pressure",
        dataset=PRESSURE_DATASET,
        variables=PRESSURE_VARIABLES,
        filename_prefix="era5_pressure",
        pressure_levels=PRESSURE_LEVELS_HPA,
    ),
    "single": ProductSpec(
        name="single",
        dataset=SINGLE_LEVEL_DATASET,
        variables=SINGLE_LEVEL_VARIABLES,
        filename_prefix="era5_single",
    ),
}


# ── Command-line parsing ─────────────────────────────────────────────────────


def parse_iso_date(value: str) -> date:
    """Parse a YYYY-MM-DD command-line date."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc

    if parsed < EARLIEST_ERA5_DATE:
        raise argparse.ArgumentTypeError(
            f"ERA5 retrieval starts at {EARLIEST_ERA5_DATE.isoformat()}; "
            f"received {parsed.isoformat()}."
        )

    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download monthly ERA5 pressure-level and single-level background "
            "data for SINGV StormCast retraining."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        required=True,
        help=(
            "Inclusive start date, YYYY-MM-DD. The complete containing month "
            "is downloaded."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        required=True,
        help=(
            "Inclusive end date, YYYY-MM-DD. The complete containing month "
            "is downloaded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory for monthly raw files. Files are written below a "
            f"YYYY subdirectory. Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCTS,
        default=list(PRODUCTS),
        help="Product types to download. Default: pressure single",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=DEFAULT_AREA,
        help=(
            "CDS retrieval area in north west south east order. "
            f"Default: {' '.join(str(value) for value in DEFAULT_AREA)}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing monthly files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned CDS requests without submitting them.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Do not open downloaded NetCDF files for structural validation. "
            "Not recommended except for debugging CDS format changes."
        ),
    )
    return parser.parse_args()


# ── Request construction ─────────────────────────────────────────────────────


def iter_months(start_date: date, end_date: date) -> Iterable[YearMonth]:
    """Yield every calendar month intersecting an inclusive date range."""

    if start_date > end_date:
        raise ValueError(
            f"Start date {start_date} is later than end date {end_date}."
        )

    year = start_date.year
    month = start_date.month
    final = YearMonth(end_date.year, end_date.month)

    while True:
        current = YearMonth(year, month)
        yield current

        if current == final:
            break

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def month_days(month: YearMonth) -> list[str]:
    """Return every valid day number for one calendar month."""

    count = calendar.monthrange(month.year, month.month)[1]
    return [f"{day:02d}" for day in range(1, count + 1)]


def validate_area(area: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate and normalize a CDS north-west-south-east area."""

    if len(area) != 4:
        raise ValueError("Area must contain north, west, south, and east.")

    north, west, south, east = (float(value) for value in area)

    if not (-90.0 <= south < north <= 90.0):
        raise ValueError(
            "Area latitude bounds must satisfy -90 <= south < north <= 90; "
            f"received north={north}, south={south}."
        )

    if not (-180.0 <= west < east <= 360.0):
        raise ValueError(
            "Area longitude bounds must satisfy -180 <= west < east <= 360; "
            f"received west={west}, east={east}."
        )

    return north, west, south, east


def build_request(
    month: YearMonth,
    product: ProductSpec,
    area: Sequence[float],
) -> dict[str, Any]:
    """Construct one current CDS API request for a complete calendar month."""

    north, west, south, east = validate_area(area)

    request: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "variable": list(product.variables),
        "year": [f"{month.year:04d}"],
        "month": [f"{month.month:02d}"],
        "day": month_days(month),
        "time": list(VALID_TIMES_UTC),
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],
    }

    if product.pressure_levels is not None:
        request["pressure_level"] = list(product.pressure_levels)

    return request


def output_path(
    output_root: Path,
    month: YearMonth,
    product: ProductSpec,
) -> Path:
    """Return the canonical output path for one monthly product."""

    return (
        output_root.expanduser()
        / f"{month.year:04d}"
        / f"{product.filename_prefix}_{month.compact}.nc"
    )


# ── Download validation ──────────────────────────────────────────────────────


def _find_name(dataset: xr.Dataset, candidates: Sequence[str]) -> str | None:
    """Return the first candidate present as a coordinate or dimension."""

    for name in candidates:
        if name in dataset or name in dataset.dims:
            return name
    return None


def _decode_time_count(dataset: xr.Dataset, path: Path) -> int:
    """Return the number of valid times in a downloaded ERA5 file."""

    time_name = _find_name(dataset, ("valid_time", "time"))
    if time_name is None:
        raise ValueError(
            f"{path}: no 'valid_time' or 'time' coordinate was found."
        )

    values = np.asarray(dataset[time_name].values)
    if values.ndim != 1:
        raise ValueError(
            f"{path}: expected a one-dimensional time coordinate, got "
            f"shape {values.shape}."
        )

    return int(values.size)


def validate_downloaded_file(
    path: Path,
    month: YearMonth,
    product: ProductSpec,
    area: Sequence[float],
) -> None:
    """Validate the basic structure and coverage of one raw ERA5 NetCDF file."""

    if not path.is_file():
        raise FileNotFoundError(f"Downloaded file does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {path}")

    if zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is a ZIP archive, not an unarchived NetCDF file. "
            "The CDS output format may have changed."
        )

    expected_times = len(month_days(month)) * len(VALID_TIMES_UTC)
    north, west, south, east = validate_area(area)

    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            if not dataset.data_vars:
                raise ValueError(f"{path}: NetCDF file contains no data variables.")

            actual_times = _decode_time_count(dataset, path)
            if actual_times != expected_times:
                raise ValueError(
                    f"{path}: expected {expected_times} valid times "
                    f"({len(month_days(month))} days x "
                    f"{len(VALID_TIMES_UTC)} times/day), found {actual_times}."
                )

            latitude_name = _find_name(dataset, ("latitude", "lat"))
            longitude_name = _find_name(dataset, ("longitude", "lon"))

            if latitude_name is None or longitude_name is None:
                raise ValueError(
                    f"{path}: latitude/longitude coordinates were not found."
                )

            latitude = np.asarray(dataset[latitude_name].values, dtype=np.float64)
            longitude = np.asarray(dataset[longitude_name].values, dtype=np.float64)

            if latitude.ndim != 1 or longitude.ndim != 1:
                raise ValueError(
                    f"{path}: expected one-dimensional latitude and longitude; "
                    f"got {latitude.shape} and {longitude.shape}."
                )

            if latitude.size < 2 or longitude.size < 2:
                raise ValueError(
                    f"{path}: latitude and longitude must each contain at least "
                    "two points."
                )

            tolerance = 1e-8
            if float(latitude.min()) > south + tolerance:
                raise ValueError(
                    f"{path}: southernmost latitude {latitude.min()} does not "
                    f"cover requested south boundary {south}."
                )
            if float(latitude.max()) < north - tolerance:
                raise ValueError(
                    f"{path}: northernmost latitude {latitude.max()} does not "
                    f"cover requested north boundary {north}."
                )
            if float(longitude.min()) > west + tolerance:
                raise ValueError(
                    f"{path}: westernmost longitude {longitude.min()} does not "
                    f"cover requested west boundary {west}."
                )
            if float(longitude.max()) < east - tolerance:
                raise ValueError(
                    f"{path}: easternmost longitude {longitude.max()} does not "
                    f"cover requested east boundary {east}."
                )

            if product.pressure_levels is not None:
                level_name = _find_name(
                    dataset,
                    ("pressure_level", "level", "plev", "isobaricInhPa"),
                )
                if level_name is None:
                    raise ValueError(
                        f"{path}: pressure-level coordinate was not found."
                    )

                level_values = np.asarray(
                    dataset[level_name].values,
                    dtype=np.float64,
                ).reshape(-1)

                # CDS normally supplies pressure levels in hPa, but accept Pa
                # as well so validation remains robust to metadata conventions.
                if level_values.size and float(np.nanmax(level_values)) > 2000.0:
                    level_values = level_values / 100.0

                levels = {
                    int(round(float(value)))
                    for value in level_values
                }
                expected_levels = {int(value) for value in product.pressure_levels}

                if levels != expected_levels:
                    raise ValueError(
                        f"{path}: pressure levels {sorted(levels)} do not match "
                        f"expected {sorted(expected_levels)}."
                    )

    except OSError as exc:
        raise ValueError(f"Could not open downloaded NetCDF file {path}: {exc}") from exc


# ── Retrieval ────────────────────────────────────────────────────────────────


def create_cds_client() -> Any:
    """Create a configured CDS API client with a clear dependency error."""

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "The 'cdsapi' package is not installed. Activate the intended "
            "environment and run: python -m pip install 'cdsapi>=0.7.7'"
        ) from exc

    try:
        return cdsapi.Client()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the CDS API client. Check ~/.cdsapirc and "
            "confirm that the required ERA5 dataset licences have been accepted."
        ) from exc


def download_one(
    client: Any,
    month: YearMonth,
    product: ProductSpec,
    destination: Path,
    area: Sequence[float],
    *,
    overwrite: bool,
    validate: bool,
) -> str:
    """Download or reuse one monthly ERA5 product.

    Returns one of ``"downloaded"`` or ``"reused"``.
    """

    destination = destination.expanduser()

    if destination.exists() and not overwrite:
        if validate:
            validate_downloaded_file(destination, month, product, area)
        print(f"Reusing:    {destination}")
        return "reused"

    destination.parent.mkdir(parents=True, exist_ok=True)

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()

    request = build_request(month, product, area)

    print(f"Requesting: {product.name:8s} {month.label}")
    print(f"Target:     {destination}")

    try:
        client.retrieve(product.dataset, request, str(partial))

        if validate:
            validate_downloaded_file(partial, month, product, area)
        elif not partial.is_file() or partial.stat().st_size == 0:
            raise ValueError(f"CDS produced no usable output at {partial}")

        os.replace(partial, destination)

    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    size_mb = destination.stat().st_size / (1024**2)
    print(f"Downloaded: {destination} ({size_mb:.1f} MiB)")
    return "downloaded"


def ensure_era5_month(
    valid_time: datetime,
    *,
    output_root: Path = DEFAULT_OUTPUT_DIR,
    area: Sequence[float] = DEFAULT_AREA,
) -> tuple[Path, Path]:
    """
    Ensure that both raw ERA5 files exist for the month containing valid_time.

    Existing files are reused. Missing files are downloaded from CDS.
    The returned paths are ordered as pressure-level, single-level.
    """
    month = YearMonth(valid_time.year, valid_time.month)
    output_root = output_root.expanduser()
    area = validate_area(area)

    pressure_product = PRODUCT_SPECS["pressure"]
    single_product = PRODUCT_SPECS["single"]

    pressure_path = output_path(
        output_root,
        month,
        pressure_product,
    )
    single_path = output_path(
        output_root,
        month,
        single_product,
    )

    if pressure_path.is_file() and single_path.is_file():
        print(f"Reusing raw ERA5 month: {month.label}")
        return pressure_path, single_path

    client = create_cds_client()

    download_one(
        client,
        month,
        pressure_product,
        pressure_path,
        area,
        overwrite=False,
        validate=True,
    )
    download_one(
        client,
        month,
        single_product,
        single_path,
        area,
        overwrite=False,
        validate=True,
    )

    return pressure_path, single_path


def print_dry_run(
    months: Sequence[YearMonth],
    products: Sequence[ProductSpec],
    output_root: Path,
    area: Sequence[float],
) -> None:
    """Print all planned requests without creating a CDS client."""

    for month in months:
        for product in products:
            destination = output_path(output_root, month, product)
            request = build_request(month, product, area)

            print("\n" + "=" * 79)
            print(f"Product: {product.name}")
            print(f"Month:   {month.label}")
            print(f"Dataset: {product.dataset}")
            print(f"Target:  {destination}")
            print("Request:")
            print(json.dumps(request, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.start_date > args.end_date:
        raise ValueError(
            f"Start date {args.start_date} is later than end date {args.end_date}."
        )

    area = validate_area(args.area)
    months = list(iter_months(args.start_date, args.end_date))
    products = [PRODUCT_SPECS[name] for name in args.products]
    output_root = args.output_dir.expanduser()

    print("ERA5 BACKGROUND DOWNLOAD")
    print("========================")
    print(f"Requested dates: {args.start_date} through {args.end_date}")
    print(
        "Calendar months: "
        f"{months[0].label} through {months[-1].label} "
        f"({len(months)} month(s))"
    )
    print(f"UTC times:       {', '.join(VALID_TIMES_UTC)}")
    print(
        "Area N/W/S/E:    "
        f"{area[0]}, {area[1]}, {area[2]}, {area[3]}"
    )
    print(f"Products:        {', '.join(product.name for product in products)}")
    print(f"Output root:     {output_root}")
    print(f"Dry run:         {args.dry_run}")

    if args.dry_run:
        print_dry_run(months, products, output_root, area)
        return

    client = create_cds_client()

    downloaded = 0
    reused = 0

    for month in months:
        for product in products:
            destination = output_path(output_root, month, product)
            result = download_one(
                client,
                month,
                product,
                destination,
                area,
                overwrite=args.overwrite,
                validate=not args.skip_validation,
            )

            if result == "downloaded":
                downloaded += 1
            else:
                reused += 1

    print("\nDownload summary")
    print("----------------")
    print(f"Downloaded: {downloaded}")
    print(f"Reused:     {reused}")
    print(f"Total:      {downloaded + reused}")


if __name__ == "__main__":
    main()
