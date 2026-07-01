#!/usr/bin/env python3
"""
Build SINGV six-hour training-pair manifests over a date range.

This script sits above build_pair.py. For each candidate time t, it:

1. considers the pair t -> t + 6 hours;
2. skips pairs whose input or target falls on a known missing-data date;
3. calls build_pair.build_pair() for every usable pair;
4. records successful pairs in a CSV manifest;
5. records skipped pairs in a separate CSV manifest;
6. reuses completed work when the same command is run again.

Command structure
-----------------
The general command is:

    python build_dataset.py MODE [MODE ARGUMENTS] [OPTIONS]

MODE must be either:

    split
        Use one of the predefined date ranges listed below.

    range
        Use a custom date range supplied on the command line.

Using split mode
----------------
Syntax:

    python build_dataset.py split SPLIT_NAME

Available split names:

    smoke_test
        1998-02-20 through 1998-02-22.
        Small end-to-end test containing valid pairs and the known missing
        uas file on 1998-02-21.

    train
        1995-01-01 through 2012-12-31.

    validation
        2013-01-01 through 2013-12-31.

    test
        2014-01-01 through 2014-12-31.

Examples:

    python build_dataset.py split smoke_test
    python build_dataset.py split train
    python build_dataset.py split validation
    python build_dataset.py split test

Using range mode
----------------
Use range mode for a temporary or custom date interval.

Syntax:

    python build_dataset.py range \
        --name NAME \
        --start-date YYYY-MM-DD \
        --end-date YYYY-MM-DD

Example:

    python build_dataset.py range \
        --name january_2000 \
        --start-date 2000-01-01 \
        --end-date 2000-01-31

The --name value is used in the output manifest filenames.

Dry runs
--------
Add --dry-run to preview what would be built or skipped without creating
prepared files or changing manifests.

Examples:

    python build_dataset.py split smoke_test --dry-run

    python build_dataset.py range \
        --name january_2000 \
        --start-date 2000-01-01 \
        --end-date 2000-01-31 \
        --dry-run

Default outputs
---------------
For a dataset named NAME:

    ~/scratch/retraining/manifests/NAME_pairs.csv
    ~/scratch/retraining/manifests/NAME_skipped.csv

The pairs manifest records usable input and target prepared files.

The skipped manifest records excluded pairs and the reason each pair was
excluded.

Notes
-----
- Dates are interpreted as UTC.
- Valid state times are 01, 07, 13, and 19 UTC.
- Date ranges are inclusive.
- A pair is included only when both its input and target lie inside the
  requested range.
- Pairs therefore do not cross train, validation, or test boundaries.
- Known missing source files are skipped rather than interpolated.
- An unexpected missing file stops the run after being recorded.
- Re-running the same command skips pairs that are already complete.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import build_pair as pair_builder


RETRAINING_DIR = Path("~/scratch/retraining").expanduser()
DEFAULT_MANIFEST_DIR = RETRAINING_DIR / "manifests"
PAIR_INTERVAL = timedelta(hours=6)

FIRST_VALID_TIME = time(hour=1)
LAST_VALID_TIME = time(hour=19)

NAMED_RANGES: dict[str, tuple[date, date]] = {
    "smoke_test": (
        date(1998, 2, 20),
        date(1998, 2, 22),
    ),
    "train": (
        date(1995, 1, 1),
        date(2012, 12, 31),
    ),
    "validation": (
        date(2013, 1, 1),
        date(2013, 12, 31),
    ),
    "test": (
        date(2014, 1, 1),
        date(2014, 12, 31),
    ),
}

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

SKIPPED_COLUMNS = [
    "input_time",
    "target_time",
    "category",
    "reason",
]


@dataclass(frozen=True)
class DatasetRange:
    """Resolved name and inclusive UTC date bounds for one dataset run."""

    name: str
    start_date: date
    end_date: date


@dataclass
class BuildCounts:
    """Counters displayed when a dataset run finishes."""

    candidates: int = 0
    planned: int = 0
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


def validate_name(value: str) -> str:
    """Require a simple filesystem-safe custom dataset name."""
    if not value:
        raise argparse.ArgumentTypeError("Dataset name must not be empty.")

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if any(character not in allowed for character in value):
        raise argparse.ArgumentTypeError(
            "Dataset name may contain only letters, digits, '-' and '_'."
        )

    return value


def resolve_range(args: argparse.Namespace) -> DatasetRange:
    """Resolve either a named split or an explicitly supplied date range."""
    if args.mode == "split":
        start_date, end_date = NAMED_RANGES[args.split_name]
        return DatasetRange(
            name=args.split_name,
            start_date=start_date,
            end_date=end_date,
        )

    return DatasetRange(
        name=args.name,
        start_date=args.start_date,
        end_date=args.end_date,
    )


def iter_candidate_input_times(
    start_date: date,
    end_date: date,
) -> Iterator[datetime]:
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


def load_skipped_keys(path: Path) -> set[tuple[str, str, str]]:
    """Load keys already recorded in the skipped-pair manifest."""
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str]] = set()

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != SKIPPED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}; "
                f"expected {SKIPPED_COLUMNS}."
            )

        for row in reader:
            keys.add(
                (
                    row["input_time"],
                    row["target_time"],
                    row["category"],
                )
            )

    return keys


def append_csv_row(
    path: Path,
    columns: list[str],
    row: dict[str, str],
) -> None:
    """Append and flush one CSV row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if write_header:
            writer.writeheader()

        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


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


def record_skip(
    skipped_path: Path,
    skipped_keys: set[tuple[str, str, str]],
    *,
    input_time: datetime,
    category: str,
    reason: str,
) -> None:
    """Record one skipped pair unless the same skip is already present."""
    target_time = input_time + PAIR_INTERVAL
    key = (
        input_time.isoformat(),
        target_time.isoformat(),
        category,
    )

    if key in skipped_keys:
        return

    append_csv_row(
        skipped_path,
        SKIPPED_COLUMNS,
        {
            "input_time": input_time.isoformat(),
            "target_time": target_time.isoformat(),
            "category": category,
            "reason": reason,
        },
    )
    skipped_keys.add(key)

def should_report_progress(
    index: int,
    total: int,
    progress_every: int,
) -> bool:
    """Return whether progress should be printed for this candidate."""
    return (
        progress_every > 0
        and (
            index % progress_every == 0
            or index == total
        )
    )


def print_progress(
    index: int,
    total: int,
    counts: BuildCounts,
) -> None:
    """Print one concise dataset-build progress line."""
    print(
        f"Progress {index}/{total}: "
        f"planned={counts.planned}, "
        f"built={counts.built}, "
        f"existing={counts.already_complete}, "
        f"skipped={counts.skipped_known}"
    )


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

    pair_manifest = manifest_dir / f"{dataset_range.name}_pairs.csv"
    skipped_manifest = (
        manifest_dir / f"{dataset_range.name}_skipped.csv"
    )

    existing_pairs = load_pair_manifest(pair_manifest)
    skipped_keys = load_skipped_keys(skipped_manifest)

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
    print(f"Skipped manifest:  {skipped_manifest}")
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

            if not dry_run:
                record_skip(
                    skipped_manifest,
                    skipped_keys,
                    input_time=input_time,
                    category="known_missing_daily_file",
                    reason=known_reason,
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

            if should_report_progress(
                index,
                len(candidate_times),
                progress_every,
            ):
                print_progress(
                    index,
                    len(candidate_times),
                    counts,
                )

            continue

        if dry_run:
            counts.planned += 1
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
        except FileNotFoundError as exc:
            reason = str(exc)

            record_skip(
                skipped_manifest,
                skipped_keys,
                input_time=input_time,
                category="runtime_missing_file",
                reason=reason,
            )

            raise RuntimeError(
                f"Unexpected missing file while building "
                f"{input_time.isoformat()} -> "
                f"{target_time.isoformat()}: {reason}"
            ) from exc

        row = pair_row(pair)

        if existing_row is None:
            append_csv_row(
                pair_manifest,
                PAIR_COLUMNS,
                row,
            )
            existing_pairs[input_key] = row
        elif existing_row != row:
            raise ValueError(
                f"Rebuilt pair {input_key} does not match its "
                f"existing manifest row. Existing: {existing_row}; "
                f"new: {row}"
            )

        counts.built += 1

        if should_report_progress(
            index,
            len(candidate_times),
            progress_every,
        ):
            print_progress(
                index,
                len(candidate_times),
                counts,
            )

    print("\nDataset build complete")
    print("----------------------")
    print(f"Candidate pairs:          {counts.candidates}")

    if dry_run:
        print(f"Would build:              {counts.planned}")
    else:
        print(f"Built or revalidated:     {counts.built}")

    print(f"Already complete:         {counts.already_complete}")
    print(f"Skipped known gaps:       {counts.skipped_known}")
    print(f"Pair manifest:            {pair_manifest}")
    print(f"Skipped manifest:         {skipped_manifest}")

    return counts


def add_common_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add options shared by named-split and custom-range modes."""
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=(
            "Directory for pair and skipped manifests "
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
        help=(
            "Regenerate prepared states while reusing assembled states."
        ),
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Build one predefined smoke_test/train/validation/test range.",
    )
    split_parser.add_argument(
        "split_name",
        choices=tuple(NAMED_RANGES),
        help="Predefined range to build.",
    )
    add_common_arguments(split_parser)

    range_parser = subparsers.add_parser(
        "range",
        help="Build an explicitly supplied inclusive date range.",
    )
    range_parser.add_argument(
        "--name",
        required=True,
        type=validate_name,
        help="Filesystem-safe name used for output manifests.",
    )
    range_parser.add_argument(
        "--start-date",
        required=True,
        type=parse_date,
        help="First UTC date in YYYY-MM-DD format.",
    )
    range_parser.add_argument(
        "--end-date",
        required=True,
        type=parse_date,
        help="Last UTC date in YYYY-MM-DD format.",
    )
    add_common_arguments(range_parser)

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")

    dataset_range = resolve_range(args)

    if dataset_range.start_date > dataset_range.end_date:
        raise ValueError(
            "--start-date must not be later than --end-date."
        )

    try:
        build_dataset(
            dataset_range,
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
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()