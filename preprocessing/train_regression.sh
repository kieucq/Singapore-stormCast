#!/usr/bin/env bash
#
# Run a StormCast regression training configuration.
#
# Usage:
#   bash retraining/train_regression.sh CONFIG_NAME
#
# Example:
#   bash retraining/train_regression.sh singv_regression_10years_100k
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

if [[ ! -d "$STORMCAST_DIR" ]]; then
    echo "StormCast directory not found:" >&2
    echo "  $STORMCAST_DIR" >&2
    exit 1
fi

if [[ ! -f "$TRAINING_CONFIG" ]]; then
    echo "Training configuration not found:" >&2
    echo "  $TRAINING_CONFIG" >&2
    exit 1
fi

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