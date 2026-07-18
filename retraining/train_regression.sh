#!/usr/bin/env bash
#
# Run a StormCast regression training configuration.
#
# Usage:
#   bash retraining/train_regression.sh CONFIG_NAME
#
# Example:
#   bash retraining/train_regression.sh singv_regression_pilot
#

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 CONFIG_NAME" >&2
    exit 2
fi

CONFIG_NAME="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STORMCAST_DIR="$REPO_ROOT/stormcast"

TRAINING_CONFIG="$STORMCAST_DIR/config/${CONFIG_NAME}.yaml"

TRAINING_CONFIG="$SCRIPT_DIR/config/${CONFIG_NAME}.yaml"

if [[ ! -d "$DATASET_CONFIG_DIR" ]]; then
    echo "Dataset configuration directory not found:" >&2
    echo "  $DATASET_CONFIG_DIR" >&2
    exit 1
fi

if [[ ! -f "$DATASET_ADAPTER" ]]; then
    echo "Dataset adapter not found:" >&2
    echo "  $DATASET_ADAPTER" >&2
    exit 1
fi

if [[ ! -f "$TRAINING_CONFIG" ]]; then
    echo "Training configuration not found:" >&2
    echo "  $TRAINING_CONFIG" >&2
    exit 1
fi

mkdir -p \
    "$STORMCAST_DIR/datasets" \
    "$STORMCAST_DIR/config/dataset"

for DATASET_CONFIG in "$DATASET_CONFIG_DIR"/*.yaml; do
    ln -sfn \
        "$DATASET_CONFIG" \
        "$STORMCAST_DIR/config/dataset/$(basename "$DATASET_CONFIG")"
done

ln -sfn \
    "$TRAINING_CONFIG" \
    "$STORMCAST_DIR/config/${CONFIG_NAME}.yaml"

cd "$STORMCAST_DIR"

echo "Resolved configuration"
echo "======================"
python train.py \
    --config-name "$CONFIG_NAME" \
    --cfg job

echo
echo "Starting training..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name "$CONFIG_NAME"