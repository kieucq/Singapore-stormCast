#!/usr/bin/env bash
#
# Prepare SINGV training and validation data, compute normalisation
# statistics, and then train the StormCast regression model in the same PBS
# allocation.
#
# Usage:
#
#   bash retraining/prepare_and_train.sh \
#     TRAIN_START TRAIN_END \
#     VALIDATION_START VALIDATION_END \
#     CONFIG_NAME
#
# Example:
#
#   bash retraining/prepare_and_train.sh \
#     1995-01-01 1996-12-31 \
#     1997-01-01 1997-04-30 \
#     singv_regression_2years
#

set -euo pipefail


usage() {
    cat <<'EOF'
Prepare SINGV data and then train StormCast.

Usage:
  prepare_and_train.sh \
    TRAIN_START TRAIN_END \
    VALIDATION_START VALIDATION_END \
    CONFIG_NAME

Example:
  bash retraining/prepare_and_train.sh \
    1995-01-01 1996-12-31 \
    1997-01-01 1997-04-30 \
    singv_regression_2years
EOF
}


if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 5 ]]; then
    echo "Error: expected four dates and one training config name." >&2
    echo >&2
    usage >&2
    exit 2
fi


TRAIN_START="$1"
TRAIN_END="$2"
VALIDATION_START="$3"
VALIDATION_END="$4"
CONFIG_NAME="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/scratch/retraining}"

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

TRAINING_CONFIG="$SCRIPT_DIR/config/${CONFIG_NAME}.yaml"


if [[ ! -f "$SCRIPT_DIR/prepare_data.sh" ]]; then
    echo "Error: preparation script does not exist:" >&2
    echo "  $SCRIPT_DIR/prepare_data.sh" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/train_regression.sh" ]]; then
    echo "Error: training script does not exist:" >&2
    echo "  $SCRIPT_DIR/train_regression.sh" >&2
    exit 1
fi

if [[ ! -f "$TRAINING_CONFIG" ]]; then
    echo "Error: training configuration does not exist:" >&2
    echo "  $TRAINING_CONFIG" >&2
    exit 1
fi


echo "SINGV PREPARATION AND STORMCAST TRAINING"
echo "========================================"
echo "Training dates:     $TRAIN_START through $TRAIN_END"
echo "Validation dates:   $VALIDATION_START through $VALIDATION_END"
echo "Training config:    $CONFIG_NAME"
echo "Data root:          $DATA_ROOT"
echo


echo "PHASE 1: DATA PREPARATION"
echo "========================="

bash "$SCRIPT_DIR/prepare_data.sh" \
    "$TRAIN_START" \
    "$TRAIN_END" \
    "$VALIDATION_START" \
    "$VALIDATION_END"


echo
echo "Checking preparation outputs..."

for REQUIRED_FILE in \
    "$TRAIN_MANIFEST" \
    "$VALIDATION_MANIFEST" \
    "$NORMALISATION_NPZ" \
    "$NORMALISATION_CSV"
do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "Error: required preparation output does not exist:" >&2
        echo "  $REQUIRED_FILE" >&2
        exit 1
    fi

    echo "Found: $REQUIRED_FILE"
done


echo
echo "PHASE 2: STORMCAST REGRESSION TRAINING"
echo "======================================"

bash "$SCRIPT_DIR/train_regression.sh" \
    "$CONFIG_NAME"


echo
echo "PREPARATION AND TRAINING COMPLETE"
echo "================================="
echo "Finished: $(date)"