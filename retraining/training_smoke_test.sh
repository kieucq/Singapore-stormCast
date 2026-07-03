#!/usr/bin/env bash
#
# Run an end-to-end StormCast regression smoke test using the custom SINGV
# dataset adapter.
#
# The script:
#   1. Links the SINGV adapter and Hydra configurations into the official
#      StormCast example directory.
#   2. Constructs the dataset and validates one real input-target sample.
#   3. Prints the final composed Hydra configuration.
#   4. Runs two training steps and one validation step.
#
# This test verifies that data loading, normalization, model construction,
# training, validation, and checkpoint saving work together. It does not
# produce a scientifically useful trained model.
#
# Usage:
#   conda activate stormcast
#   bash retraining/training_smoke_test.sh
#
# Inputs:
#   - retraining/singv.py
#   - retraining/config/dataset/singv.yaml
#   - retraining/config/singv_regression_smoke.yaml
#   - prepared SINGV data, manifests, and normalization statistics under
#     ~/scratch/retraining
#
# Outputs:
#   A timestamped smoke-test run under:
#   /scratch/users/nus/e1155933/retraining/runs/singv-regression-smoke/
#


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STORMCAST_ROOT="$HOME/scratch/physicsnemo-v2.0.0/examples/weather/stormcast"

ADAPTER="$PROJECT_ROOT/retraining/singv.py"
DATASET_CONFIG="$PROJECT_ROOT/retraining/config/dataset/singv.yaml"
SMOKE_CONFIG="$PROJECT_ROOT/retraining/config/singv_regression_smoke.yaml"

RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"

# ----------------------------------------------------------------------
# 1. Expose the adapter and configurations to StormCast
# ----------------------------------------------------------------------

ln -sfn "$ADAPTER" "$STORMCAST_ROOT/datasets/singv.py"
ln -sfn "$DATASET_CONFIG" "$STORMCAST_ROOT/config/dataset/singv.yaml"
ln -sfn "$SMOKE_CONFIG" "$STORMCAST_ROOT/config/singv_regression_smoke.yaml"

cd "$STORMCAST_ROOT"

# ----------------------------------------------------------------------
# 2. Verify registration and load one real sample
# ----------------------------------------------------------------------

python - <<'PY'
from omegaconf import OmegaConf

from datasets import dataset_classes


config = OmegaConf.load("config/dataset/singv.yaml")
dataset_name = config.name

assert dataset_name in dataset_classes

dataset_class = dataset_classes[dataset_name]
dataset = dataset_class(config, train=True)
sample = dataset[0]

input_state, target_state = sample["state"]
expected_state_shape = (
    len(dataset.state_channels()),
    *dataset.image_shape(),
)

assert input_state.shape == expected_state_shape
assert target_state.shape == expected_state_shape
assert sample["mask"].shape == expected_state_shape
assert sample["background"].shape == (0, *dataset.image_shape())

assert dataset.latitude().shape == dataset.image_shape()
assert dataset.longitude().shape == dataset.image_shape()

print("SINGV adapter and sample test passed.")
print("Pairs:", len(dataset))
print("State shape:", expected_state_shape)
PY

# ----------------------------------------------------------------------
# 3. Display the final composed Hydra configuration
# ----------------------------------------------------------------------

python train.py \
  --config-name singv_regression_smoke \
  --cfg job \
  "training.run_id=$RUN_ID"

# ----------------------------------------------------------------------
# 4. Run the two-step training smoke test
# ----------------------------------------------------------------------

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  train.py \
  --config-name singv_regression_smoke \
  "training.run_id=$RUN_ID"