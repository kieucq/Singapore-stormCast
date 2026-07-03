# SINGV → StormCast retraining

This folder contains the data-preparation and training-integration pipeline for
adapting StormCast to six-hour SINGV-RCM states.

The pipeline converts archived SINGV files into 75-channel `624 × 624` state
files, groups them into six-hour input-target pairs, computes training-set
normalisation statistics, and exposes the resulting dataset to the official
StormCast trainer.

## Pipeline overview

```text
SINGV-RCM archive
        │
        ├── audit_singv_data.py
        │       Check archive coverage, structure, metadata, and optionally data
        │
        ▼
assemble_state.py
        │       Extract one native-grid state at 01/07/13/19 UTC
        ▼
assembled/assembled_YYYYMMDD_HHMM.nc
        │
        ▼
prepare_state.py + prep_utils.py
        │       Crop, regrid, mask, and stack 75 channels
        ▼
prepared/prepared_YYYYMMDD_HHMM.nc
        │
        ▼
build_pair.py
        │       Create and validate one t → t+6 h pair
        ▼
build_dataset.py
        │       Build all pairs in a training/validation/testing date range
        ▼
manifests/<split>_<start>_<end>_pairs.csv
        │
        ├── compute_normalisation_stats.py
        │       Compute per-channel statistics from the training manifest only
        │
        ▼
normalisation_stats/<training-range>_normalisation.npz
        │
        ▼
singv.py
        │       StormCast dataset adapter
        ▼
training_smoke_test.sh
                End-to-end two-step StormCast regression test
```

## File reference

### `audit_singv_data.py`

Audits the original SINGV-RCM archive before dataset generation.

It checks expected files over a date range, verifies that they can be opened,
and checks variables, dimensions, times, coordinates, pressure levels, units,
dtype, and related metadata. The optional `--scan-data` mode also reads the
weather fields and inspects their numerical values, but is much slower.

Typical use:

```bash
python audit_singv_data.py   --start-date 1995-01-01   --end-date 1995-01-31
```

Reports are written under:

```text
~/scratch/retraining/audit/
```

### `assemble_state.py`

Builds one native-grid SINGV state for a requested valid time.

It finds the five surface files and five pressure-level files required for that
time, extracts the correct timestep, combines the variables, and writes one
NetCDF file on the original `960 × 960` grid.

Supported times are `01`, `07`, `13`, and `19` UTC.

```bash
python assemble_state.py --datetime 2014-12-01T01:00
```

Default output:

```text
~/scratch/retraining/assembled/assembled_20141201_0100.nc
```

### `prep_utils.py`

Contains constants and helper functions used by `prepare_state.py`.

This includes:

- the five surface variables;
- the five pressure-level variables;
- the 14 pressure levels;
- the permanent 75-channel ordering;
- source, cropped, and target grid definitions;
- crop and regridding helpers;
- pressure-validity masking logic.

This is a helper module and is not normally run directly.

### `prepare_state.py`

Converts one assembled native-grid state into the format used for retraining.

It:

1. validates the assembled `960 × 960` state;
2. crops 12 pixels from every edge;
3. regrids `936 × 936` to `624 × 624`;
4. keeps the 14 native SINGV pressure levels;
5. stacks five surface channels and 70 pressure-level channels;
6. stores invalid pressure-level cells as `NaN`;
7. writes an unnormalised NetCDF file.

```bash
python prepare_state.py   ~/scratch/retraining/assembled/assembled_20141201_0100.nc
```

Default output:

```text
~/scratch/retraining/prepared/prepared_20141201_0100.nc
```

### `build_pair.py`

Builds one six-hour input-target pair.

For an input time `t`, it assembles and prepares the states at `t` and
`t + 6 hours`, reuses existing files where possible, validates that both
prepared states are structurally compatible, and can record the pair in a CSV
manifest.

```bash
python build_pair.py --datetime 2014-12-01T01:00
```

This is useful for debugging one pair. For a full date range, use
`build_dataset.py` instead.

### `build_dataset.py`

Builds every usable six-hour pair in an explicit date range.

The user chooses one of three split labels:

```text
training
validation
testing
```

The manifest name is generated automatically from the split and dates:

```text
<split>_<start-YYYYMMDD>_<end-YYYYMMDD>_pairs.csv
```

Preview a build:

```bash
python build_dataset.py   --split training   --start-date 1995-01-01   --end-date 1995-01-31   --dry-run
```

Run the build:

```bash
python build_dataset.py   --split training   --start-date 1995-01-01   --end-date 1995-01-31
```

The script:

- keeps pairs fully inside the requested date range;
- skips known missing-data dates;
- records unexpected missing files and stops;
- reuses completed pairs when rerun;
- writes separate successful-pair and skipped-pair manifests.

Default outputs:

```text
~/scratch/retraining/manifests/
├── training_19950101_19950131_pairs.csv
└── training_19950101_19950131_skipped.csv
```

### `compute_normalisation_stats.py`

Computes per-channel statistics from prepared states referenced by a pair
manifest.

It deduplicates shared states, ignores expected `NaN` pressure cells, rejects
infinite values, and computes count, mean, population standard deviation,
minimum, maximum, and validity information for each channel.

Run this on the **training manifest only**:

```bash
python compute_normalisation_stats.py   ~/scratch/retraining/manifests/training_19950101_19950131_pairs.csv
```

It automatically creates:

```text
~/scratch/retraining/normalisation_stats/
├── training_19950101_19950131_normalisation.npz
└── training_19950101_19950131_normalisation.csv
```

The NPZ file is consumed by `singv.py`. The CSV is for human inspection.

Do not compute separate validation or testing statistics. Training,
validation, and testing states must all use statistics calculated from the
training split.

### `singv.py`

Defines `SINGVDataset`, the custom StormCast dataset adapter.

It:

- reads training or validation pairs from a CSV manifest;
- loads prepared NetCDF states;
- loads training-set normalisation statistics;
- normalises each channel;
- replaces invalid normalised cells with zero;
- returns a target mask so invalid cells do not contribute to the loss;
- returns data in the structure expected by the StormCast trainer.

The dataset currently has:

```text
75 state channels
0 background channels
624 × 624 spatial shape
6-hour input-target interval
```

This module is imported by StormCast and is not normally run directly.

### `training_smoke_test.sh`

Runs a small end-to-end StormCast regression test.

It:

1. links the custom dataset adapter and YAML files into the official StormCast
   example directory;
2. constructs the dataset and loads one real sample;
3. checks the returned shapes;
4. prints the final Hydra configuration;
5. runs two training steps and one validation step;
6. saves a timestamped checkpoint.

From the repository root:

```bash
conda activate stormcast
bash retraining/training_smoke_test.sh
```

A successful smoke test proves that the dataset, model, loss, backpropagation,
validation, and checkpoint-saving pipeline work together. It does not produce
a scientifically useful model.

### `config/`

Contains Hydra YAML configuration files.

```text
config/
├── <experiment>.yaml
└── dataset/
    └── <dataset>.yaml
```

Files directly inside `config/` are top-level experiment recipes. They select
and combine the dataset, model, training, sampler, and Hydra configurations,
then apply experiment-specific overrides.

Files inside `config/dataset/` configure only the dataset, such as:

- the dataset class;
- data root;
- training manifest;
- validation manifest;
- normalisation file.

The top-level experiment YAML refers to a dataset YAML through a Hydra default
such as:

```yaml
defaults:
  - dataset/singv_smoke
```

### `__init__.py`

Marks this directory as a Python package. It is not normally run directly.

### `__pycache__/`

Automatically generated Python bytecode cache. Do not edit it. It does not
contain source code and should normally be excluded from Git.

## Generated scratch structure

The pipeline writes generated data outside the Git repository:

```text
~/scratch/retraining/
├── audit/
├── assembled/
├── prepared/
├── manifests/
├── normalisation_stats/
└── runs/
```

Keep source code, YAML configuration, and this README in Git. Treat the
contents of `~/scratch/retraining/` as generated data.

## Typical workflow

### 1. Activate the environment

```bash
conda activate stormcast
```

Run large archive audits and dataset builds on a compute node rather than a
login node.

### 2. Audit the requested dates

```bash
python retraining/audit_singv_data.py   --start-date 1995-01-01   --end-date 1995-02-07
```

### 3. Preview the dataset ranges

```bash
python retraining/build_dataset.py   --split training   --start-date 1995-01-01   --end-date 1995-01-31   --dry-run

python retraining/build_dataset.py   --split validation   --start-date 1995-02-01   --end-date 1995-02-07   --dry-run
```

### 4. Build training and validation data

```bash
python retraining/build_dataset.py   --split training   --start-date 1995-01-01   --end-date 1995-01-31

python retraining/build_dataset.py   --split validation   --start-date 1995-02-01   --end-date 1995-02-07
```

### 5. Compute training statistics

```bash
python retraining/compute_normalisation_stats.py   ~/scratch/retraining/manifests/training_19950101_19950131_pairs.csv
```

### 6. Update the dataset YAML

Point the dataset configuration to:

```yaml
train_manifest: manifests/training_19950101_19950131_pairs.csv
validation_manifest: manifests/validation_19950201_19950207_pairs.csv
normalisation_path: normalisation_stats/training_19950101_19950131_normalisation.npz
```

### 7. Run the smoke test

```bash
bash retraining/training_smoke_test.sh
```

## Important reminders

- Valid SINGV pressure-level times are `01`, `07`, `13`, and `19` UTC.
- Each model sample predicts the state six hours after its input state.
- Date ranges are inclusive.
- Pairs do not cross the boundary of their requested split.
- Normalisation statistics must come from training data only.
- Invalid pressure cells remain `NaN` in prepared files, are filled with zero
  after normalisation, and are excluded from the loss using the target mask.
- `build_pair.py` is for one pair; `build_dataset.py` is for a full range.
- `singv.py` and `prep_utils.py` are imported modules, not normal command-line
  entry points.
- The smoke configuration should remain small and separate from the eventual
  full training configuration.