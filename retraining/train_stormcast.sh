#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 CONFIG_NAME" >&2
    exit 2
fi

CONFIG_NAME="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STORMCAST_DIR="$PROJECT_ROOT/stormcast"
CONFIG_PATH="$STORMCAST_DIR/config/${CONFIG_NAME}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Training configuration not found:" >&2
    echo "  $CONFIG_PATH" >&2
    exit 1
fi

cd "$STORMCAST_DIR"

echo "Resolved configuration"
echo "======================"

python train.py \
    --config-name "$CONFIG_NAME" \
    --cfg job

echo
echo "Starting StormCast training"
echo "==========================="

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name "$CONFIG_NAME"
