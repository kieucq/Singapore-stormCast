#!/usr/bin/env python3
"""
Audit the SINGV-RCM archive used for StormCast retraining.

The default audit checks every expected daily file from 1995-01-01 through
2014-12-31 without loading the full weather fields. It verifies:

- every expected month directory exists;
- every expected date has exactly one matching NetCDF file;
- each matched file is non-empty and opens successfully;
- the expected variable, dimensions, sizes, and coordinates are present;
- decoded timestamps match the date and cadence encoded in the filename;
- latitude and longitude are finite, strictly increasing, and unchanged;
- pressure levels match the fixed 14-level retraining specification;
- variable dtype is numeric;
- dtype, units, packing, missing-value, and time metadata remain consistent.

Use --scan-data to additionally load every weather field and check its
numerical contents. The full scan is much slower.

Reports are rewritten after each variable, so completed results survive if a
long job is interrupted.

Default outputs
---------------
~/scratch/retraining/audit/audit_issues.csv
~/scratch/retraining/audit/audit_summary.csv

Examples
--------
python audit.py

python audit.py \
    --start-date 2014-01-01 \
    --end-date 2014-01-31

python audit.py --scan-data
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


# ── Archive specification ────────────────────────────────────────────────────

BASE_DIR = Path(
    "/home/project/13004327/data_service/model_data/V3_Historical/"
    "V3-WMC-2/CCRS/ERA5/historical/reanalysis/SINGV-RCM/vn5"
)

DEFAULT_OUTPUT_DIR = Path("~/scratch/retraining/audit").expanduser()
DEFAULT_START_DATE = date(1995, 1, 1)
DEFAULT_END_DATE = date(2014, 12, 31)

EXPECTED_GRID_SHAPE = (960, 960)
EXPECTED_PRESSURE_LEVELS_PA = np.asarray(
    [
        100000,
        92500,
        85000,
        80000,
        75000,
        70000,
        60000,
        50000,
        40000,
        30000,
        20000,
        10000,
        5000,
        1000,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class VariableSpec:
    """Expected structure for one archived SINGV variable."""

    frequency: str
    variable: str
    dimensions: tuple[str, ...]
    hours: tuple[int, ...]


SPECS = [
    *[
        VariableSpec(
            frequency="1hr",
            variable=variable,
            dimensions=("time", "lat", "lon"),
            hours=tuple(range(24)),
        )
        for variable in ("pr", "tas", "uas", "vas", "psl")
    ],
    *[
        VariableSpec(
            frequency="6hr",
            variable=variable,
            dimensions=("time", "plev", "lat", "lon"),
            hours=(1, 7, 13, 19),
        )
        for variable in ("ta", "ua", "va", "zg", "hus")
    ],
]


@dataclass
class Summary:
    """Per-variable counts written to audit_summary.csv."""

    frequency: str
    variable: str
    expected_files: int
    matched_files: int = 0
    missing_dates: int = 0
    duplicate_dates: int = 0
    unreadable_files: int = 0
    structural_issues: int = 0
    metadata_issues: int = 0
    data_issues: int = 0
    scanned_files: int = 0
    global_min: float | None = None
    global_max: float | None = None
    maximum_missing_fraction: float | None = None


# ── Date and filename helpers ────────────────────────────────────────────────

def parse_date(value: str) -> date:
    """Parse a command-line date in YYYY-MM-DD format."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def iter_dates(start: date, end: date) -> Iterable[date]:
    """Yield every date in an inclusive interval."""
    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


def iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    """Yield each year-month pair intersecting an inclusive date interval."""
    year, month = start.year, start.month

    while (year, month) <= (end.year, end.month):
        yield year, month

        month += 1
        if month == 13:
            month = 1
            year += 1


def expected_times(day: date, hours: tuple[int, ...]) -> list[str]:
    """Return the exact minute-resolution timestamps expected for one file."""
    return [f"{day.isoformat()}T{hour:02d}:00" for hour in hours]


def normalize_times(values: np.ndarray) -> list[str]:
    """Convert decoded timestamps to comparable minute-resolution strings."""
    normalized: list[str] = []

    for value in np.asarray(values).reshape(-1):
        if isinstance(value, np.datetime64):
            text = np.datetime_as_string(
                value.astype("datetime64[m]"),
                unit="m",
            )
        else:
            # Supports cftime values if xarray uses them for decoding.
            text = str(value).replace(" ", "T")[:16]

        normalized.append(text)

    return normalized


def filename_date(path: Path, spec: VariableSpec) -> date | None:
    """Extract the daily date from one expected SINGV archive filename."""
    if spec.frequency == "1hr":
        expression = (
            rf"^{re.escape(spec.variable)}_.*_1hr_"
            r"(?P<start>\d{8})0000-(?P<end>\d{8})2300\.nc$"
        )
    else:
        expression = (
            rf"^{re.escape(spec.variable)}_.*_6hr_"
            r"(?P<start>\d{8})0100-(?P<end>\d{8})1900\.nc$"
        )

    match = re.match(expression, path.name)

    if match is None or match.group("start") != match.group("end"):
        return None

    try:
        return datetime.strptime(match.group("start"), "%Y%m%d").date()
    except ValueError:
        return None


# ── Issue accounting ─────────────────────────────────────────────────────────

def add_issue(
    issues: list[dict[str, str]],
    summary: Summary,
    *,
    severity: str,
    category: str,
    day: date | None,
    path: Path | None,
    message: str,
) -> None:
    """Record one audit issue and update its summary category."""
    issues.append(
        {
            "severity": severity,
            "category": category,
            "frequency": summary.frequency,
            "variable": summary.variable,
            "date": "" if day is None else day.isoformat(),
            "path": "" if path is None else str(path),
            "message": message,
        }
    )

    if category == "missing_date":
        summary.missing_dates += 1
    elif category == "duplicate_date":
        summary.duplicate_dates += 1
    elif category == "unreadable_file":
        summary.unreadable_files += 1
    elif category.startswith("metadata_"):
        summary.metadata_issues += 1
    elif category.startswith("data_"):
        summary.data_issues += 1
    else:
        summary.structural_issues += 1


# ── File discovery ───────────────────────────────────────────────────────────

def discover_files(
    spec: VariableSpec,
    start: date,
    end: date,
    issues: list[dict[str, str]],
    summary: Summary,
) -> dict[date, list[Path]]:
    """
    Discover matching files and group them by the date encoded in each name.

    Files placed in the wrong YYYYMM directory are reported and are not
    accepted as satisfying the expected date, because the production
    collector would not find them there.
    """
    files_by_date: dict[date, list[Path]] = {}

    for year, month in iter_months(start, end):
        month_name = f"{year:04d}{month:02d}"
        directory = BASE_DIR / spec.frequency / spec.variable / month_name

        if not directory.is_dir():
            add_issue(
                issues,
                summary,
                severity="error",
                category="missing_directory",
                day=date(year, month, 1),
                path=directory,
                message="Expected month directory does not exist.",
            )
            continue

        for path in sorted(directory.glob("*.nc")):
            parsed = filename_date(path, spec)

            if parsed is None:
                add_issue(
                    issues,
                    summary,
                    severity="warning",
                    category="unexpected_filename",
                    day=None,
                    path=path,
                    message="Filename does not match the expected daily pattern.",
                )
                continue

            if (parsed.year, parsed.month) != (year, month):
                add_issue(
                    issues,
                    summary,
                    severity="error",
                    category="misplaced_file",
                    day=parsed,
                    path=path,
                    message=(
                        f"Filename date belongs to {parsed:%Y%m}, but the file "
                        f"is stored in directory {month_name}."
                    ),
                )
                continue

            if start <= parsed <= end:
                files_by_date.setdefault(parsed, []).append(path)

    return files_by_date


# ── Metadata and coordinate checks ───────────────────────────────────────────

def metadata_value(variable: xr.DataArray, name: str) -> Any:
    """Read metadata whether xarray exposes it in attrs or encoding."""
    if name in variable.attrs:
        return variable.attrs[name]

    return variable.encoding.get(name)


def metadata_snapshot(
    variable: xr.DataArray,
    time_coordinate: xr.DataArray,
) -> dict[str, Any]:
    """Capture metadata whose changes could alter interpretation of the data."""
    return {
        "dtype": str(variable.dtype),
        "units": metadata_value(variable, "units"),
        "_FillValue": metadata_value(variable, "_FillValue"),
        "missing_value": metadata_value(variable, "missing_value"),
        "scale_factor": metadata_value(variable, "scale_factor"),
        "add_offset": metadata_value(variable, "add_offset"),
        "time_dtype": str(time_coordinate.dtype),
        # The raw time-units reference date legitimately changes each day.
        # Decoded timestamps are validated separately, so only the calendar
        # needs to remain consistent here.
        "time_calendar": metadata_value(time_coordinate, "calendar"),
    }


def metadata_equal(first: Any, second: Any) -> bool:
    """Compare scalar or array metadata, treating NaNs as equal."""
    try:
        return bool(np.array_equal(first, second, equal_nan=True))
    except (TypeError, ValueError):
        return first == second


def check_coordinate(
    coordinate: xr.DataArray,
    *,
    name: str,
    expected_dimension: str,
    expected_order: str = "increasing",
    day: date,
    path: Path,
    issues: list[dict[str, str]],
    summary: Summary,
) -> np.ndarray | None:
    """Validate one one-dimensional numerical coordinate."""
    if coordinate.dims != (expected_dimension,):
        add_issue(
            issues,
            summary,
            severity="error",
            category=f"{name}_dimensions",
            day=day,
            path=path,
            message=(
                f"Coordinate {name!r} has dimensions {coordinate.dims}; "
                f"expected {(expected_dimension,)}."
            ),
        )
        return None

    values = np.asarray(coordinate.values)

    if not np.issubdtype(values.dtype, np.number):
        add_issue(
            issues,
            summary,
            severity="error",
            category=f"{name}_dtype",
            day=day,
            path=path,
            message=(
                f"Coordinate {name!r} has dtype {values.dtype}; "
                "expected a numerical dtype."
            ),
        )
        return None

    values = values.astype(np.float64, copy=False)

    if not np.isfinite(values).all():
        add_issue(
            issues,
            summary,
            severity="error",
            category=f"{name}_nonfinite",
            day=day,
            path=path,
            message=f"Coordinate {name!r} contains non-finite values.",
        )

    if expected_order not in {"increasing", "decreasing"}:
        raise ValueError(
            "expected_order must be 'increasing' or 'decreasing'."
        )

    if values.size >= 2:
        differences = np.diff(values)
        correctly_ordered = (
            np.all(differences > 0)
            if expected_order == "increasing"
            else np.all(differences < 0)
        )

        if not correctly_ordered:
            add_issue(
                issues,
                summary,
                severity="error",
                category=f"{name}_order",
                day=day,
                path=path,
                message=(
                    f"Coordinate {name!r} is not strictly "
                    f"{expected_order}."
                ),
            )

    return values


# ── Header audit ─────────────────────────────────────────────────────────────

def audit_header(
    path: Path,
    day: date,
    spec: VariableSpec,
    issues: list[dict[str, str]],
    summary: Summary,
    references: dict[str, Any],
) -> bool:
    """
    Audit one file without loading its weather field.

    Returns True when the file opens and contains the expected variable, so an
    optional data scan can proceed. Structural problems are still recorded.
    """
    try:
        if path.stat().st_size == 0:
            add_issue(
                issues,
                summary,
                severity="error",
                category="empty_file",
                day=day,
                path=path,
                message="File size is zero bytes.",
            )
            return False

        with xr.open_dataset(
            path,
            decode_times=True,
            mask_and_scale=False,
            cache=False,
        ) as dataset:
            if spec.variable not in dataset:
                add_issue(
                    issues,
                    summary,
                    severity="error",
                    category="missing_variable",
                    day=day,
                    path=path,
                    message=f"Expected variable {spec.variable!r} is absent.",
                )
                return False

            variable = dataset[spec.variable]

            if not np.issubdtype(variable.dtype, np.number):
                add_issue(
                    issues,
                    summary,
                    severity="error",
                    category="variable_dtype",
                    day=day,
                    path=path,
                    message=(
                        f"Variable dtype is {variable.dtype}; "
                        "expected a numerical dtype."
                    ),
                )

            if variable.dims != spec.dimensions:
                add_issue(
                    issues,
                    summary,
                    severity="error",
                    category="variable_dimensions",
                    day=day,
                    path=path,
                    message=(
                        f"Dimensions are {variable.dims}; "
                        f"expected {spec.dimensions}."
                    ),
                )

            expected_sizes = {
                "time": len(spec.hours),
                "lat": EXPECTED_GRID_SHAPE[0],
                "lon": EXPECTED_GRID_SHAPE[1],
            }

            if spec.frequency == "6hr":
                expected_sizes["plev"] = len(EXPECTED_PRESSURE_LEVELS_PA)

            for dimension, expected_size in expected_sizes.items():
                actual_size = dataset.sizes.get(dimension)

                if actual_size != expected_size:
                    add_issue(
                        issues,
                        summary,
                        severity="error",
                        category=f"{dimension}_size",
                        day=day,
                        path=path,
                        message=(
                            f"Dimension {dimension!r} has length "
                            f"{actual_size}; expected {expected_size}."
                        ),
                    )

            required_coordinates = ["time", "lat", "lon"]

            if spec.frequency == "6hr":
                required_coordinates.append("plev")

            missing_coordinates = [
                name
                for name in required_coordinates
                if name not in dataset.coords
            ]

            if missing_coordinates:
                add_issue(
                    issues,
                    summary,
                    severity="error",
                    category="missing_coordinates",
                    day=day,
                    path=path,
                    message=f"Missing coordinates: {missing_coordinates}.",
                )
                return True

            actual_times = normalize_times(dataset["time"].values)
            wanted_times = expected_times(day, spec.hours)

            if actual_times != wanted_times:
                add_issue(
                    issues,
                    summary,
                    severity="error",
                    category="time_values",
                    day=day,
                    path=path,
                    message=(
                        f"Decoded times are {actual_times}; "
                        f"expected {wanted_times}."
                    ),
                )

            latitude = check_coordinate(
                dataset["lat"],
                name="latitude",
                expected_dimension="lat",
                day=day,
                path=path,
                issues=issues,
                summary=summary,
            )
            longitude = check_coordinate(
                dataset["lon"],
                name="longitude",
                expected_dimension="lon",
                day=day,
                path=path,
                issues=issues,
                summary=summary,
            )

            if latitude is not None:
                if "latitude" not in references:
                    references["latitude"] = latitude.copy()
                elif not np.array_equal(latitude, references["latitude"]):
                    add_issue(
                        issues,
                        summary,
                        severity="error",
                        category="latitude_values",
                        day=day,
                        path=path,
                        message="Latitude coordinate differs from the reference.",
                    )

            if longitude is not None:
                if "longitude" not in references:
                    references["longitude"] = longitude.copy()
                elif not np.array_equal(longitude, references["longitude"]):
                    add_issue(
                        issues,
                        summary,
                        severity="error",
                        category="longitude_values",
                        day=day,
                        path=path,
                        message="Longitude coordinate differs from the reference.",
                    )

            if spec.frequency == "6hr":
                levels = check_coordinate(
                    dataset["plev"],
                    name="pressure",
                    expected_dimension="plev",
                    expected_order="decreasing",
                    day=day,
                    path=path,
                    issues=issues,
                    summary=summary,
                )

                if (
                    levels is not None
                    and not np.array_equal(
                        levels,
                        EXPECTED_PRESSURE_LEVELS_PA,
                    )
                ):
                    add_issue(
                        issues,
                        summary,
                        severity="error",
                        category="pressure_levels",
                        day=day,
                        path=path,
                        message=(
                            "Pressure levels or ordering differ from the "
                            "expected 14-level specification."
                        ),
                    )

            reference_key = f"metadata:{spec.frequency}:{spec.variable}"
            current_metadata = metadata_snapshot(
                variable,
                dataset["time"],
            )

            if reference_key not in references:
                references[reference_key] = current_metadata
            else:
                reference_metadata = references[reference_key]

                for name, reference_value in reference_metadata.items():
                    current_value = current_metadata[name]

                    if not metadata_equal(current_value, reference_value):
                        add_issue(
                            issues,
                            summary,
                            severity="error",
                            category=f"metadata_{name}",
                            day=day,
                            path=path,
                            message=(
                                f"{name} changed: "
                                f"reference={reference_value!r}, "
                                f"current={current_value!r}."
                            ),
                        )

        return True

    except Exception as exc:
        add_issue(
            issues,
            summary,
            severity="error",
            category="unreadable_file",
            day=day,
            path=path,
            message=f"{type(exc).__name__}: {exc}",
        )
        return False


# ── Optional full data scan ──────────────────────────────────────────────────

def audit_data(
    path: Path,
    day: date,
    spec: VariableSpec,
    issues: list[dict[str, str]],
    summary: Summary,
) -> None:
    """Load and inspect one complete weather field."""
    try:
        with xr.open_dataset(
            path,
            decode_times=False,
            mask_and_scale=True,
            cache=False,
        ) as dataset:
            values = np.asarray(dataset[spec.variable].values)

        finite = np.isfinite(values)
        finite_count = int(np.count_nonzero(finite))
        missing_fraction = 1.0 - finite_count / values.size

        if np.isinf(values).any():
            add_issue(
                issues,
                summary,
                severity="error",
                category="data_infinite",
                day=day,
                path=path,
                message="Decoded data contain positive or negative infinity.",
            )

        if finite_count == 0:
            add_issue(
                issues,
                summary,
                severity="error",
                category="data_all_missing",
                day=day,
                path=path,
                message="Every decoded value is missing or non-finite.",
            )
            minimum = None
            maximum = None
        else:
            # where= avoids allocating a second array containing every valid
            # value, which matters for large 6-hour pressure files.
            minimum = float(
                np.min(values, where=finite, initial=np.inf)
            )
            maximum = float(
                np.max(values, where=finite, initial=-np.inf)
            )

            if minimum == maximum:
                add_issue(
                    issues,
                    summary,
                    severity="warning",
                    category="data_constant",
                    day=day,
                    path=path,
                    message=f"Every finite value equals {minimum!r}.",
                )

        summary.scanned_files += 1
        summary.maximum_missing_fraction = (
            missing_fraction
            if summary.maximum_missing_fraction is None
            else max(
                summary.maximum_missing_fraction,
                missing_fraction,
            )
        )

        if minimum is not None:
            summary.global_min = (
                minimum
                if summary.global_min is None
                else min(summary.global_min, minimum)
            )

        if maximum is not None:
            summary.global_max = (
                maximum
                if summary.global_max is None
                else max(summary.global_max, maximum)
            )

    except Exception as exc:
        add_issue(
            issues,
            summary,
            severity="error",
            category="data_read_failure",
            day=day,
            path=path,
            message=f"{type(exc).__name__}: {exc}",
        )


# ── Per-variable orchestration ───────────────────────────────────────────────

def print_progress(
    index: int,
    total: int,
    summary: Summary,
    progress_every: int,
) -> None:
    """Print periodic progress, including the final date."""
    if not progress_every:
        return

    if index % progress_every == 0 or index == total:
        print(
            f"  checked {index}/{total} dates; "
            f"matched={summary.matched_files}, "
            f"unreadable={summary.unreadable_files}"
        )


def audit_variable(
    spec: VariableSpec,
    dates: list[date],
    *,
    scan_data: bool,
    progress_every: int,
    issues: list[dict[str, str]],
    references: dict[str, Any],
) -> Summary:
    """Audit every expected daily file for one variable."""
    summary = Summary(
        frequency=spec.frequency,
        variable=spec.variable,
        expected_files=len(dates),
    )

    print(f"\nAuditing {spec.frequency}/{spec.variable}")
    print("-" * (11 + len(spec.frequency) + len(spec.variable)))

    files_by_date = discover_files(
        spec,
        dates[0],
        dates[-1],
        issues,
        summary,
    )

    for index, day in enumerate(dates, start=1):
        matches = files_by_date.get(day, [])

        if not matches:
            add_issue(
                issues,
                summary,
                severity="error",
                category="missing_date",
                day=day,
                path=None,
                message="No matching daily NetCDF file was found.",
            )
            print_progress(
                index,
                len(dates),
                summary,
                progress_every,
            )
            continue

        if len(matches) > 1:
            add_issue(
                issues,
                summary,
                severity="error",
                category="duplicate_date",
                day=day,
                path=None,
                message=(
                    "Multiple matching files were found: "
                    + "; ".join(str(path) for path in matches)
                ),
            )
            print_progress(
                index,
                len(dates),
                summary,
                progress_every,
            )
            continue

        path = matches[0]
        summary.matched_files += 1

        header_readable = audit_header(
            path,
            day,
            spec,
            issues,
            summary,
            references,
        )

        if scan_data and header_readable:
            audit_data(
                path,
                day,
                spec,
                issues,
                summary,
            )

        print_progress(
            index,
            len(dates),
            summary,
            progress_every,
        )

    other_issues = (
        summary.structural_issues
        + summary.metadata_issues
        + summary.data_issues
    )

    print(
        f"  expected={summary.expected_files}, "
        f"matched={summary.matched_files}, "
        f"missing={summary.missing_dates}, "
        f"duplicates={summary.duplicate_dates}, "
        f"unreadable={summary.unreadable_files}, "
        f"other issues={other_issues}"
    )

    return summary


# ── Report writing ───────────────────────────────────────────────────────────

def write_csv_atomically(
    path: Path,
    columns: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Write a CSV through a temporary file, then replace the final report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def write_issues(
    path: Path,
    issues: list[dict[str, str]],
) -> None:
    """Write the issue table."""
    columns = [
        "severity",
        "category",
        "frequency",
        "variable",
        "date",
        "path",
        "message",
    ]

    write_csv_atomically(path, columns, issues)


def write_summary(
    path: Path,
    summaries: list[Summary],
) -> None:
    """Write one summary row per completed variable."""
    columns = [
        "frequency",
        "variable",
        "expected_files",
        "matched_files",
        "missing_dates",
        "duplicate_dates",
        "unreadable_files",
        "structural_issues",
        "metadata_issues",
        "data_issues",
        "scanned_files",
        "global_min",
        "global_max",
        "maximum_missing_fraction",
    ]

    rows = [
        {
            column: getattr(summary, column)
            for column in columns
        }
        for summary in summaries
    ]

    write_csv_atomically(path, columns, rows)


# ── Command-line interface ───────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=DEFAULT_START_DATE,
        help=f"First date to audit (default: {DEFAULT_START_DATE}).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=DEFAULT_END_DATE,
        help=f"Last date to audit (default: {DEFAULT_END_DATE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--scan-data",
        action="store_true",
        help=(
            "Load every weather field and inspect numerical values. "
            "This is much slower than the default header audit."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help=(
            "Print progress after this many dates per variable; "
            "use 0 to disable (default: 250)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_date > args.end_date:
        raise ValueError(
            "--start-date must not be later than --end-date."
        )

    if args.progress_every < 0:
        raise ValueError(
            "--progress-every must be non-negative."
        )

    if not BASE_DIR.is_dir():
        raise FileNotFoundError(
            f"SINGV base directory not found: {BASE_DIR}"
        )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    issues_path = output_dir / "audit_issues.csv"
    summary_path = output_dir / "audit_summary.csv"

    dates = list(iter_dates(args.start_date, args.end_date))
    issues: list[dict[str, str]] = []
    references: dict[str, Any] = {}
    summaries: list[Summary] = []

    print("SINGV-RCM ARCHIVE AUDIT")
    print("=======================")
    print(f"Base directory: {BASE_DIR}")
    print(
        f"Date range:     {args.start_date} "
        f"through {args.end_date}"
    )
    print(f"Expected days:  {len(dates)}")
    print(f"Full data scan: {args.scan_data}")
    print(f"Reports:        {output_dir}")

    for spec in SPECS:
        summary = audit_variable(
            spec,
            dates,
            scan_data=args.scan_data,
            progress_every=args.progress_every,
            issues=issues,
            references=references,
        )
        summaries.append(summary)

        # Preserve all completed work if a later variable is interrupted.
        write_issues(issues_path, issues)
        write_summary(summary_path, summaries)

    errors = sum(
        issue["severity"] == "error"
        for issue in issues
    )
    warnings = sum(
        issue["severity"] == "warning"
        for issue in issues
    )

    print("\nAudit complete")
    print("--------------")
    print(f"Errors:   {errors}")
    print(f"Warnings: {warnings}")
    print(f"Issues:   {issues_path}")
    print(f"Summary:  {summary_path}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()