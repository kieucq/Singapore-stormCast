#!/usr/bin/env bash
#
# Build the first larger SINGV regression dataset.
#
# The script:
#   1. builds three months of training pairs;
#   2. builds the following month of validation pairs;
#   3. computes normalisation statistics from the training data only.
#
# Usage:
#   conda activate stormcast
#   bash retraining/prepare_regression_data.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$HOME/scratch/retraining"

TRAIN_START="1995-01-01"
TRAIN_END="1995-03-31"

VALIDATION_START="1995-04-01"
VALIDATION_END="1995-04-30"

TRAIN_START_COMPACT="${TRAIN_START//-/}"
TRAIN_END_COMPACT="${TRAIN_END//-/}"

TRAIN_NAME="training_${TRAIN_START_COMPACT}_${TRAIN_END_COMPACT}"
TRAIN_MANIFEST="$DATA_ROOT/manifests/${TRAIN_NAME}.csv"

echo "Building training dataset..."
python "$SCRIPT_DIR/build_dataset.py" \
  --split training \
  --start-date "$TRAIN_START" \
  --end-date "$TRAIN_END"

echo
echo "Building validation dataset..."
python "$SCRIPT_DIR/build_dataset.py" \
  --split validation \
  --start-date "$VALIDATION_START" \
  --end-date "$VALIDATION_END"

echo
echo "Computing training normalisation statistics..."
python "$SCRIPT_DIR/compute_normalisation_stats.py" \
  "$TRAIN_MANIFEST"

echo
echo "Regression data preparation complete."
echo "Training manifest:"
echo "  $TRAIN_MANIFEST"
echo
echo "Validation manifest:"
echo "  $DATA_ROOT/manifests/validation_19950401_19950430.csv"
echo
echo "Normalisation statistics:"
echo "  $DATA_ROOT/normalisation_stats/${TRAIN_NAME}_normalisation_stats.npz"