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
STORMCAST_DIR="$HOME/scratch/physicsnemo-v2.0.0/examples/weather/stormcast"

DATASET_CONFIG="$SCRIPT_DIR/config/dataset/singv_regression.yaml"
TRAINING_CONFIG="$SCRIPT_DIR/config/${CONFIG_NAME}.yaml"

if [[ ! -f "$DATASET_CONFIG" ]]; then
    echo "Dataset configuration not found:" >&2
    echo "  $DATASET_CONFIG" >&2
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

ln -sfn \
    "$SCRIPT_DIR/regression_dataset_adapter.py" \
    "$STORMCAST_DIR/datasets/regression_dataset_adapter.py"

ln -sfn \
    "$DATASET_CONFIG" \
    "$STORMCAST_DIR/config/dataset/singv_regression.yaml"

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