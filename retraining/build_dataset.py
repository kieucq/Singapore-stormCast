#!/usr/bin/env python3
"""
Build SINGV six-hour input-target pairs over an explicit date range.

The user supplies:

- a dataset split: ``training``, ``validation``, or ``testing``;
- an inclusive start date;
- an inclusive end date.

The script automatically names the dataset:

    <split>_<start-YYYYMMDD>_<end-YYYYMMDD>

For each candidate time ``t``, it:

1. considers the pair t -> t + 6 hours;
2. skips pairs whose input or target falls on a known missing-data date;
3. calls build_pair.build_pair() for each remaining pair;
4. records successful pairs in a CSV manifest;
5. reuses completed work when the same command is run again.

Example
-------
Build a training range:

    python build_dataset.py \
        --split training \
        --start-date 1995-01-01 \
        --end-date 1995-01-31

This produces:

    ~/scratch/retraining/manifests/
        training_19950101_19950131.csv

Add ``--dry-run`` to preview the work without building states or modifying
manifest files.

Notes
-----
- Dates are interpreted as UTC.
- Valid state times are 01, 07, 13, and 19 UTC.
- Date ranges are inclusive.
- A pair is included only when both its input and target lie inside the
  requested range.
- Pairs therefore do not cross split boundaries.
- Known missing source files are skipped rather than interpolated.
- Unexpected missing files stop the run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import build_pair as pair_builder


RETRAINING_DIR = Path("~/scratch/retraining").expanduser()
DEFAULT_MANIFEST_DIR = RETRAINING_DIR / "manifests"
PAIR_INTERVAL = timedelta(hours=6)

FIRST_VALID_TIME = time(hour=1)
LAST_VALID_TIME = time(hour=19)

DATASET_SPLITS = ("training", "validation", "testing")

# Each listed daily file contains all 24 hourly values for that variable.
# Therefore all four six-hour states on the affected date are unavailable.
KNOWN_MISSING_DAILY_FILES: dict[date, tuple[str, ...]] = {
    date(1998, 2, 21): ("uas",),
    date(2002, 3, 16): ("vas",),
    date(2010, 9, 21): ("psl",),
}

PAIR_COLUMNS = [
    "input_time",
    "input_file",
    "target_time",
    "target_file",
]


@dataclass(frozen=True)
class DatasetRange:
    """Generated name and inclusive UTC date bounds for one dataset run."""

    name: str
    start_date: date
    end_date: date


@dataclass
class BuildCounts:
    """Counters displayed when a dataset run finishes."""

    candidates: int = 0
    built: int = 0
    already_complete: int = 0
    skipped_known: int = 0


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD command-line date."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def dataset_name(split: str, start_date: date, end_date: date) -> str:
    """Generate a consistent dataset name from its split and date range."""

    return f"{split}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"


def iter_candidate_input_times(
    start_date: date,
    end_date: date,
) -> Iterable[datetime]:
    """
    Yield every candidate input time whose target remains inside the range.

    For example, on an interior day this includes 19 UTC -> 01 UTC on the
    following day. On the final date, 19 UTC is excluded because its target
    would fall outside the requested range.
    """
    if start_date > end_date:
        raise ValueError("Start date must not be later than end date.")

    current = datetime.combine(start_date, FIRST_VALID_TIME)
    final_state_time = datetime.combine(end_date, LAST_VALID_TIME)

    while current + PAIR_INTERVAL <= final_state_time:
        yield current
        current += PAIR_INTERVAL


def unavailable_state_reason(
    valid_time: datetime,
) -> str | None:
    """Return the known reason a state is unavailable, if any."""
    variables = KNOWN_MISSING_DAILY_FILES.get(valid_time.date())

    if variables is None:
        return None

    return (
        f"{valid_time.date().isoformat()} is missing daily source "
        f"file(s) for {', '.join(variables)}"
    )


def unavailable_pair_reason(
    input_time: datetime,
) -> str | None:
    """Return why a pair touches a known unavailable state."""
    target_time = input_time + PAIR_INTERVAL
    reasons: list[str] = []

    input_reason = unavailable_state_reason(input_time)
    if input_reason is not None:
        reasons.append(f"input state: {input_reason}")

    target_reason = unavailable_state_reason(target_time)
    if target_reason is not None:
        reasons.append(f"target state: {target_reason}")

    if not reasons:
        return None

    return "; ".join(reasons)


def manifest_path_value(path: Path) -> str:
    """Store paths relative to the retraining root whenever possible."""
    resolved_path = path.expanduser().resolve()
    resolved_root = RETRAINING_DIR.resolve()

    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def resolve_manifest_file(value: str) -> Path:
    """Resolve a path stored in a generated pair manifest."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return RETRAINING_DIR / path


def load_pair_manifest(
    path: Path,
) -> dict[str, dict[str, str]]:
    """Load completed pairs, rejecting malformed or duplicate rows."""
    if not path.exists():
        return {}

    rows_by_input_time: dict[str, dict[str, str]] = {}

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != PAIR_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}; "
                f"expected {PAIR_COLUMNS}."
            )

        for line_number, row in enumerate(reader, start=2):
            input_time = row["input_time"]

            if not input_time:
                raise ValueError(
                    f"Empty input_time in {path} at line {line_number}."
                )

            if input_time in rows_by_input_time:
                raise ValueError(
                    f"Duplicate input_time {input_time!r} in {path}."
                )

            rows_by_input_time[input_time] = row

    return rows_by_input_time


def append_csv_row(
    path: Path,
    columns: list[str],
    row: dict[str, str],
) -> None:
    """Append one CSV row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def pair_row(
    pair: pair_builder.TrainingPair,
) -> dict[str, str]:
    """Convert a successfully built pair to one manifest row."""
    return {
        "input_time": pair.input_time.isoformat(),
        "input_file": manifest_path_value(pair.input_file),
        "target_time": pair.target_time.isoformat(),
        "target_file": manifest_path_value(pair.target_file),
    }


def manifest_row_is_complete(
    row: dict[str, str],
    expected_input_time: datetime,
) -> bool:
    """
    Check that an existing row describes this pair and both files still exist.

    The prepared files themselves were validated when the row was created.
    Use --revalidate-existing to run build_pair again for completed rows.
    """
    expected_target_time = expected_input_time + PAIR_INTERVAL

    if row["input_time"] != expected_input_time.isoformat():
        return False

    if row["target_time"] != expected_target_time.isoformat():
        raise ValueError(
            "Existing manifest target time does not match the requested "
            f"six-hour pair for {expected_input_time.isoformat()}."
        )

    input_file = resolve_manifest_file(row["input_file"])
    target_file = resolve_manifest_file(row["target_file"])

    return input_file.is_file() and target_file.is_file()


def build_dataset(
    dataset_range: DatasetRange,
    *,
    manifest_dir: Path,
    overwrite_assembled: bool,
    overwrite_prepared: bool,
    revalidate_existing: bool,
    verbose: bool,
    dry_run: bool,
    progress_every: int,
) -> BuildCounts:
    """Build every usable pair for one resolved range."""
    manifest_dir = manifest_dir.expanduser()

    if not dry_run:
        manifest_dir.mkdir(parents=True, exist_ok=True)

    pair_manifest = manifest_dir / f"{dataset_range.name}.csv"

    existing_pairs = load_pair_manifest(pair_manifest)

    candidate_times = list(
        iter_candidate_input_times(
            dataset_range.start_date,
            dataset_range.end_date,
        )
    )

    counts = BuildCounts(candidates=len(candidate_times))

    print("SINGV DATASET BUILD")
    print("===================")
    print(f"Name:              {dataset_range.name}")
    print(
        f"Date range:        {dataset_range.start_date} "
        f"through {dataset_range.end_date}"
    )
    print(f"Candidate pairs:   {len(candidate_times)}")
    print(f"Pair manifest:     {pair_manifest}")
    print(f"Dry run:           {dry_run}")

    for index, input_time in enumerate(candidate_times, start=1):
        target_time = input_time + PAIR_INTERVAL
        input_key = input_time.isoformat()
        known_reason = unavailable_pair_reason(input_time)

        if known_reason is not None:
            counts.skipped_known += 1

            print(
                f"Skipping {input_time.isoformat()} -> "
                f"{target_time.isoformat()}: {known_reason}"
            )
        
            continue

        existing_row = existing_pairs.get(input_key)

        if (
            existing_row is not None
            and manifest_row_is_complete(
                existing_row,
                input_time,
            )
            and not revalidate_existing
            and not overwrite_assembled
            and not overwrite_prepared
        ):
            counts.already_complete += 1

            if (
                progress_every
                and (
                    index % progress_every == 0
                    or index == len(candidate_times)
                )
            ):
                print(
                    f"Progress {index}/{len(candidate_times)}: "
                    f"built={counts.built}, "
                    f"existing={counts.already_complete}, "
                    f"skipped={counts.skipped_known}"
                )

            continue

        if dry_run:
            print(
                f"Would build {input_time.isoformat()} -> "
                f"{target_time.isoformat()}"
            )
            continue

        try:
            pair = pair_builder.build_pair(
                input_time,
                overwrite_assembled=overwrite_assembled,
                overwrite_prepared=overwrite_prepared,
                manifest_path=None,
                quiet=not verbose,
            )

        except FileNotFoundError as error:
            raise RuntimeError(
                f"Unexpected missing file while building "
                f"{input_time.isoformat()} -> {target_time.isoformat()}: {error}"
            ) from error

        row = pair_row(pair)

        if existing_row is None:
            append_csv_row(pair_manifest, PAIR_COLUMNS, row)
            existing_pairs[input_key] = row
        elif existing_row != row:
            raise ValueError(
                f"Rebuilt pair {input_key} does not match its existing manifest row. "
                f"Existing: {existing_row}; new: {row}"
            )

        counts.built += 1

        if (
            progress_every
            and (
                index % progress_every == 0
                or index == len(candidate_times)
            )
        ):
            print(
                f"Progress {index}/{len(candidate_times)}: "
                f"built={counts.built}, "
                f"existing={counts.already_complete}, "
                f"skipped={counts.skipped_known}"
            )

    print("\nDataset build complete")
    print("----------------------")
    print(f"Candidate pairs:          {counts.candidates}")
    print(f"Built or revalidated:     {counts.built}")
    print(f"Already complete:         {counts.already_complete}")
    print(f"Skipped known gaps:       {counts.skipped_known}")
    print(f"Pair manifest:            {pair_manifest}")

    return counts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        required=True,
        choices=DATASET_SPLITS,
        help="Dataset split being created.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=parse_date,
        help="First UTC date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=parse_date,
        help="Last UTC date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=(
            "Directory for dataset manifests "
            f"(default: {DEFAULT_MANIFEST_DIR})."
        ),
    )
    parser.add_argument(
        "--overwrite-assembled",
        action="store_true",
        help=(
            "Regenerate assembled states. Their prepared states are also "
            "regenerated."
        ),
    )
    parser.add_argument(
        "--overwrite-prepared",
        action="store_true",
        help="Regenerate prepared states while reusing assembled states.",
    )
    parser.add_argument(
        "--revalidate-existing",
        action="store_true",
        help=(
            "Call build_pair again for rows already present in the pair "
            "manifest."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed preparation diagnostics for every new state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List actions and counts without creating states or changing "
            "manifests."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help=(
            "Print progress every N candidate pairs; use 0 to disable "
            "(default: 25)."
        ),
    )

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("--start-date must not be later than --end-date.")

    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")

    return args


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    requested_range = DatasetRange(
        name=dataset_name(args.split, args.start_date, args.end_date),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    try:
        build_dataset(
            requested_range,
            manifest_dir=args.manifest_dir,
            overwrite_assembled=args.overwrite_assembled,
            overwrite_prepared=args.overwrite_prepared,
            revalidate_existing=args.revalidate_existing,
            verbose=args.verbose,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
        )
    except (
        FileExistsError,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()