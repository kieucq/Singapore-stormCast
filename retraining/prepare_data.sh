#!/usr/bin/env bash
#
# Build SINGV training and validation datasets, then compute normalisation
# statistics from the training dataset only.
#
# Usage:
#
#   bash retraining/prepare_data.sh \
#     TRAIN_START TRAIN_END \
#     VALIDATION_START VALIDATION_END
#
# Example:
#
#   bash retraining/prepare_data.sh \
#     1995-01-01 1996-12-31 \
#     1997-01-01 1997-04-30
#

set -euo pipefail


usage() {
    cat <<'EOF'
Build SINGV training and validation datasets.

Usage:
  prepare_data.sh TRAIN_START TRAIN_END VALIDATION_START VALIDATION_END

Dates must use YYYY-MM-DD format.

Example:
  bash retraining/prepare_data.sh \
    1995-01-01 1996-12-31 \
    1997-01-01 1997-04-30

The script:
  1. builds or resumes the requested training dataset;
  2. builds or resumes the requested validation dataset;
  3. computes normalisation statistics from the training dataset only.
EOF
}


if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 4 ]]; then
    echo "Error: expected four date arguments." >&2
    echo >&2
    usage >&2
    exit 2
fi


TRAIN_START="$1"
TRAIN_END="$2"
VALIDATION_START="$3"
VALIDATION_END="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/scratch/retraining}"


# Validate all dates before beginning expensive processing.
python - \
    "$TRAIN_START" \
    "$TRAIN_END" \
    "$VALIDATION_START" \
    "$VALIDATION_END" <<'PY'
import sys
from datetime import date

try:
    train_start, train_end, validation_start, validation_end = (
        date.fromisoformat(value) for value in sys.argv[1:]
    )
except ValueError as error:
    raise SystemExit(f"Error: invalid date: {error}")

if train_start > train_end:
    raise SystemExit(
        "Error: training start date is later than training end date."
    )

if validation_start > validation_end:
    raise SystemExit(
        "Error: validation start date is later than validation end date."
    )

if train_end >= validation_start:
    raise SystemExit(
        "Error: training and validation ranges overlap, or validation does "
        "not begin after training."
    )
PY


TRAIN_START_COMPACT="${TRAIN_START//-/}"
TRAIN_END_COMPACT="${TRAIN_END//-/}"

VALIDATION_START_COMPACT="${VALIDATION_START//-/}"
VALIDATION_END_COMPACT="${VALIDATION_END//-/}"

TRAIN_NAME="training_${TRAIN_START_COMPACT}_${TRAIN_END_COMPACT}"
VALIDATION_NAME="validation_${VALIDATION_START_COMPACT}_${VALIDATION_END_COMPACT}"

TRAIN_MANIFEST="$DATA_ROOT/manifests/${TRAIN_NAME}.csv"
VALIDATION_MANIFEST="$DATA_ROOT/manifests/${VALIDATION_NAME}.csv"

NORMALISATION_NPZ="$DATA_ROOT/normalisation_stats/${TRAIN_NAME}_normalisation_stats.npz"
NORMALISATION_CSV="$DATA_ROOT/normalisation_stats/${TRAIN_NAME}_normalisation_stats.csv"


echo "SINGV DATA PREPARATION"
echo "======================"
echo "Training dates:     $TRAIN_START through $TRAIN_END"
echo "Validation dates:   $VALIDATION_START through $VALIDATION_END"
echo "Data root:          $DATA_ROOT"
echo "Training manifest:  $TRAIN_MANIFEST"
echo "Validation manifest: $VALIDATION_MANIFEST"
echo


echo "STAGE 1: BUILDING TRAINING DATASET"
echo "=================================="

python "$SCRIPT_DIR/build_dataset.py" \
    --split training \
    --start-date "$TRAIN_START" \
    --end-date "$TRAIN_END"


echo
echo "STAGE 2: BUILDING VALIDATION DATASET"
echo "===================================="

python "$SCRIPT_DIR/build_dataset.py" \
    --split validation \
    --start-date "$VALIDATION_START" \
    --end-date "$VALIDATION_END"


if [[ ! -f "$TRAIN_MANIFEST" ]]; then
    echo "Error: training manifest was not created:" >&2
    echo "  $TRAIN_MANIFEST" >&2
    exit 1
fi

if [[ ! -f "$VALIDATION_MANIFEST" ]]; then
    echo "Error: validation manifest was not created:" >&2
    echo "  $VALIDATION_MANIFEST" >&2
    exit 1
fi


echo
echo "STAGE 3: COMPUTING TRAINING NORMALISATION STATISTICS"
echo "===================================================="

python "$SCRIPT_DIR/compute_normalisation_stats.py" \
    "$TRAIN_MANIFEST"


if [[ ! -f "$NORMALISATION_NPZ" ]]; then
    echo "Error: normalisation NPZ was not created:" >&2
    echo "  $NORMALISATION_NPZ" >&2
    exit 1
fi

if [[ ! -f "$NORMALISATION_CSV" ]]; then
    echo "Error: normalisation summary CSV was not created:" >&2
    echo "  $NORMALISATION_CSV" >&2
    exit 1
fi


echo
echo "DATA PREPARATION COMPLETE"
echo "========================="
echo "Training manifest:"
echo "  $TRAIN_MANIFEST"
echo
echo "Validation manifest:"
echo "  $VALIDATION_MANIFEST"
echo
echo "Normalisation statistics:"
echo "  $NORMALISATION_NPZ"
echo "  $NORMALISATION_CSV"