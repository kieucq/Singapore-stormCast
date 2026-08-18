# SINGV → StormCast preprocessing

This folder contains the preprocessing pipeline used to adapt StormCast to
six-hour SINGV-RCM states with ERA5 synoptic background conditioning.

The pipeline:

- extracts SINGV states from the archived SINGV-RCM dataset;
- crops and regrids them to a `624 × 624` grid;
- prepares matching ERA5 background fields;
- groups the data into six-hour input-background-target samples;
- computes training-set normalisation statistics;
- provides the resulting files and manifests to the StormCast training code.

Each prepared SINGV state contains 75 channels:

- 5 surface variables:
  - `tas` — near-surface air temperature;
  - `uas` — eastward near-surface wind;
  - `vas` — northward near-surface wind;
  - `psl` — mean sea-level pressure;
  - `pr` — precipitation rate;
- 5 pressure-level variables:
  - `ta` — air temperature;
  - `ua` — eastward wind;
  - `va` — northward wind;
  - `hus` — specific humidity;
  - `zg` — geopotential height;
- each pressure-level variable is represented at 14 levels:
  `1000`, `925`, `850`, `800`, `750`, `700`, `600`, `500`, `400`, `300`,
  `200`, `100`, `50`, and `10` hPa.

These variables were selected to mirror the meteorological content of the
original StormCast state configuration as closely as possible using the
variables and pressure levels available in the SINGV-RCM archive.

Each prepared ERA5 background contains 26 channels:

- `u`, `v`, `z`, `t`, and `q` at 1000, 850, 500, and 250 hPa,
  where `z` is converted from geopotential to geopotential height;
- `u10`, `v10`, `t2m`, `tcwv`, `mslp`, and `sp`.

Each training sample therefore represents:

```text
SINGV(t) + ERA5(t) → SINGV(t + 6 h)
```

## Pipeline overview

For normal use, preprocessing requires only two main commands:

```text
build_dataset.py
        │
        │  Automatically:
        │
        ├── assembles the required SINGV states
        ├── prepares the SINGV states
        ├── downloads the required ERA5 data
        ├── prepares matching ERA5 backgrounds
        ├── validates six-hour training samples
        └── writes the dataset manifest
        │
        ▼
manifests/<split>_<start>_<end>.csv
        │
        ▼
compute_normalisation_stats.py
        │
        ├── SINGV state normalisation statistics
        └── ERA5 background normalisation statistics
```

The individual preprocessing stages called internally by `build_dataset.py`
are:

```text
                            build_dataset.py
                                   │
                                   ▼
                              build_pair.py
                                   │
                 ┌─────────────────┴──────────────────┐
                 │                                    │
                 ▼                                    ▼
        SINGV preprocessing                    ERA5 preprocessing
                 │                                    │
        assemble_state.py                  background/download_era5.py
                 │                                    │
                 ▼                                    ▼
        assembled/assembled_*.nc              background/raw/*.nc
                 │                                    │
        prepare_state.py                    background/prepare_era5.py
        + prep_utils.py                                │
                 │                                    ▼
                 ▼                         background/prepared/background_*.nc
        prepared/prepared_*.nc                        │
                 │                                    │
                 └─────────────────┬──────────────────┘
                                   │
                                   ▼
                     SINGV(t) + ERA5(t) → SINGV(t+6h)
                                   │
                                   ▼
                  manifests/<split>_<start>_<end>.csv
```

`audit_singv_data.py` is an optional preliminary check of the source archive.
It is useful when validating a new date range or archive installation, but it
is not required for each dataset build.

---

## File reference

### `paths.py`

Defines the filesystem locations shared by the preprocessing pipeline.

The main paths are:

```text
DATA_ROOT
ASSEMBLED_DIR
PREPARED_DIR
MANIFEST_DIR
NORMALISATION_DIR
AUDIT_DIR
BACKGROUND_RAW_DIR
BACKGROUND_PREPARED_DIR
SINGV_ARCHIVE_ROOT
```

Edit `DATA_ROOT` and `SINGV_ARCHIVE_ROOT` when moving the pipeline to another
system. The remaining paths are derived automatically.

---

### `audit_singv_data.py`

Audits the original SINGV-RCM archive before dataset generation.

The default audit checks every expected daily file from 1995 through 2014 and
verifies:

- archive coverage;
- filename structure;
- variables and dimensions;
- timestamps;
- latitude and longitude coordinates;
- pressure levels;
- dtype and units;
- packing and missing-value metadata.

The optional `--scan-data` mode additionally reads the full weather fields and
checks their numerical contents, but is substantially slower.

Example:

```bash
python preprocessing/audit_singv_data.py \
    --start-date 1995-01-01 \
    --end-date 1995-01-31
```

Reports are written to `AUDIT_DIR` configured in `paths.py`:

```text
audit_issues.csv
audit_summary.csv
```

---

### `assemble_state.py`

Builds one native-grid SINGV state for a requested valid time.

It locates the five required surface files and five required pressure-level
files, extracts the appropriate timestep, and combines them into one NetCDF
file on the original `960 × 960` SINGV grid.

The surface variables are:

```text
tas
uas
vas
psl
pr
```

The pressure-level variables are:

```text
ta
ua
va
hus
zg
```

Supported valid times are:

```text
01 UTC
07 UTC
13 UTC
19 UTC
```

Example:

```bash
python preprocessing/assemble_state.py \
    --datetime 2014-12-01T01:00
```

The default output is written to `ASSEMBLED_DIR`:

```text
assembled_20141201_0100.nc
```

---

### `prep_utils.py`

Contains the numerical constants and preprocessing functions used by
`prepare_state.py`.

This includes:

- the five surface variables;
- the five pressure-level variables;
- the 14 SINGV pressure levels;
- the permanent 75-channel ordering;
- source, cropped, and target grid definitions;
- bilinear regridding;
- conservative regridding;
- mask-aware pressure-field regridding;
- pressure-validity masking.

This is a helper module and is not normally run directly.

---

### `prepare_state.py`

Converts one assembled SINGV state into the format used by StormCast training.

It:

1. validates the assembled `960 × 960` state;
2. crops 12 pixels from every edge;
3. reduces the grid from `936 × 936` to `624 × 624`;
4. preserves the 14 native SINGV pressure levels;
5. stacks 5 surface channels and 70 pressure-level channels;
6. constructs pressure-level validity masks;
7. stores invalid pressure-level cells as `NaN`;
8. writes an unnormalised NetCDF state.

Example:

```bash
python preprocessing/prepare_state.py \
    ~/scratch/stormcast-data/assembled/assembled_20141201_0100.nc
```

The default output is written to `PREPARED_DIR`:

```text
prepared_20141201_0100.nc
```

The output state has shape:

```text
(time=1, channel=75, y=624, x=624)
```

---

## ERA5 background preparation

StormCast training uses a lower-resolution synoptic background in addition to
the high-resolution SINGV state.

For each input time `t`, the pipeline prepares an ERA5 background at the same
valid time and interpolates it onto exactly the same `624 × 624` grid as the
prepared SINGV state.

### `background/download_era5.py`

Downloads the monthly ERA5 files required for background conditioning when
they are not already available locally.

Raw ERA5 data are stored beneath:

```text
BACKGROUND_RAW_DIR
```

The required background channels are constructed from:

```text
Pressure levels:
u, v, z, t, q
at 1000, 850, 500, and 250 hPa

Single levels:
u10
v10
t2m
tcwv
mslp
sp
```

This gives 26 background channels in total.

Normal dataset construction calls the downloader automatically through
`build_pair.py` when required.

---

### `background/prepare_era5.py`

Converts the raw ERA5 data at one valid time into the background field used by
StormCast.

It:

- extracts the requested ERA5 timestep;
- selects the required pressure and single-level fields;
- converts geopotential to geopotential height where required;
- interpolates ERA5 onto the exact prepared SINGV grid;
- stacks the fixed 26-channel background;
- validates that the result is finite;
- writes an unnormalised NetCDF file.

Prepared backgrounds are written beneath:

```text
BACKGROUND_PREPARED_DIR
```

with names such as:

```text
background_20141201_0100.nc
```

The output has shape:

```text
(time=1, channel=26, y=624, x=624)
```

---

### `build_pair.py`

Builds and validates one complete six-hour training sample.

For an input time `t`, it ensures that the following exist:

```text
prepared SINGV state at t
prepared ERA5 background at t
prepared SINGV state at t + 6 h
```

It therefore constructs:

```text
SINGV(t) + ERA5(t) → SINGV(t + 6 h)
```

The script reuses existing intermediate files where possible and validates:

- valid times;
- the six-hour input-target separation;
- state dimensions and shapes;
- ERA5 channel ordering;
- ERA5 finite values;
- matching SINGV and ERA5 grids.

Example:

```bash
python preprocessing/build_pair.py \
    --datetime 2014-12-01T01:00
```

Useful options include:

```text
--overwrite-assembled
--overwrite-prepared
--manifest
--quiet
```

`build_pair.py` is primarily useful for testing or debugging individual
samples. Use `build_dataset.py` for complete date ranges.

---

### `build_dataset.py`

Builds every usable six-hour sample within an explicit date range.

The user supplies one of three split labels:

```text
training
validation
testing
```

The manifest name is generated automatically:

```text
<split>_<start-YYYYMMDD>_<end-YYYYMMDD>.csv
```

For example:

```text
training_19950101_19950131.csv
```

Preview a dataset build:

```bash
python preprocessing/build_dataset.py \
    --split training \
    --start-date 1995-01-01 \
    --end-date 1995-01-31 \
    --dry-run
```

Run the build:

```bash
python preprocessing/build_dataset.py \
    --split training \
    --start-date 1995-01-01 \
    --end-date 1995-01-31
```

The script:

- considers every six-hour pair within the requested range;
- prevents pairs from crossing the requested split boundary;
- skips known missing SINGV dates;
- stops on unexpected missing files;
- reuses completed samples when rerun;
- optionally revalidates existing samples;
- writes one six-column CSV manifest.

The manifest columns are:

```text
input_time
input_file
background_time
background_file
target_time
target_file
```

Generated file paths are stored relative to `DATA_ROOT` whenever possible,
making manifests portable when the data root is moved.

The default manifest directory is:

```text
MANIFEST_DIR
```

---

### `compute_normalisation_stats.py`

Computes separate per-channel normalisation statistics for:

1. prepared SINGV states;
2. prepared ERA5 backgrounds.

The script reads a six-column dataset manifest.

Input and target SINGV states are deduplicated before processing, so a state
shared by neighbouring forecast pairs contributes only once.

For every channel, the script calculates:

- count;
- valid fraction;
- mean;
- population standard deviation (`ddof=0`);
- raw minimum and maximum;
- normalised minimum and maximum.

Expected `NaN` values in masked SINGV pressure cells are ignored. Infinite
values are rejected.

ERA5 backgrounds are required to contain only finite values.

Normalisation statistics should be computed from the **training split only**.

Example:

```bash
python preprocessing/compute_normalisation_stats.py \
    ~/scratch/stormcast-data/manifests/training_19950101_19950131.csv
```

Four files are generated in `NORMALISATION_DIR`:

```text
training_19950101_19950131_normalisation_stats.npz
training_19950101_19950131_normalisation_stats.csv

training_19950101_19950131_background_normalisation_stats.npz
training_19950101_19950131_background_normalisation_stats.csv
```

The NPZ files are machine-readable and are used during training.

The CSV files contain the same per-channel information for inspection.

Validation and testing data must use the statistics computed from the training
split. Do not calculate independent validation or testing statistics.

---

## StormCast dataset integration

The custom dataset integration lives under:

```text
stormcast/datasets/
```

It reads the generated manifests and prepared NetCDF files and provides samples
to the StormCast trainer.

For each sample, the training dataset supplies:

```text
state:       75 × 624 × 624
background:  26 × 624 × 624
target:      75 × 624 × 624
```

Training-set normalisation statistics are applied separately to the SINGV state
and ERA5 background.

Invalid SINGV pressure cells are stored as `NaN` during preprocessing. During
dataset loading they are handled using the pressure mask so that invalid
below-ground regions do not contribute to the training loss.

---

## Generated data structure

With the default `DATA_ROOT` in `paths.py`, generated data have the following
structure:

```text
~/scratch/stormcast-data/
├── audit/
│   ├── audit_issues.csv
│   └── audit_summary.csv
│
├── assembled/
│   └── assembled_YYYYMMDD_HHMM.nc
│
├── prepared/
│   └── prepared_YYYYMMDD_HHMM.nc
│
├── background/
│   ├── raw/
│   └── prepared/
│       └── background_YYYYMMDD_HHMM.nc
│
├── manifests/
│   └── <split>_<start>_<end>.csv
│
└── normalisation_stats/
    ├── <training-range>_normalisation_stats.npz
    ├── <training-range>_normalisation_stats.csv
    ├── <training-range>_background_normalisation_stats.npz
    └── <training-range>_background_normalisation_stats.csv
```

Generated data should remain outside the Git repository.

---

## Typical workflow

The following commands assume they are run from the repository root.

### 1. Configure paths

Edit:

```text
preprocessing/paths.py
```

Set the local:

```python
DATA_ROOT
SINGV_ARCHIVE_ROOT
```

---

### 2. Activate the environment

```bash
conda activate stormcast
```

Large archive audits and dataset builds should be run on a compute node rather
than a login node.

---

### 3. Optional: audit the requested SINGV period

```bash
python preprocessing/audit_singv_data.py \
    --start-date 1995-01-01 \
    --end-date 1995-02-07
```

---

### 4. Preview dataset generation

Training:

```bash
python preprocessing/build_dataset.py \
    --split training \
    --start-date 1995-01-01 \
    --end-date 1995-01-31 \
    --dry-run
```

Validation:

```bash
python preprocessing/build_dataset.py \
    --split validation \
    --start-date 1995-02-01 \
    --end-date 1995-02-07 \
    --dry-run
```

`build_dataset.py` is the main preprocessing entry point. It automatically
assembles and prepares the required SINGV states, downloads and prepares ERA5
backgrounds, validates each six-hour sample, and writes the manifest.

---

### 5. Build training and validation datasets

```bash
python preprocessing/build_dataset.py \
    --split training \
    --start-date 1995-01-01 \
    --end-date 1995-01-31
```

```bash
python preprocessing/build_dataset.py \
    --split validation \
    --start-date 1995-02-01 \
    --end-date 1995-02-07
```

This generates both the SINGV states and matching ERA5 backgrounds required by
each sample.

---

### 6. Compute training normalisation statistics

```bash
python preprocessing/compute_normalisation_stats.py \
    ~/scratch/stormcast-data/manifests/training_19950101_19950131.csv
```

This generates separate statistics for the SINGV state and ERA5 background.

---

### 7. Configure StormCast training

Point the StormCast dataset configuration to:

```text
training manifest
validation manifest
SINGV normalisation NPZ
ERA5 background normalisation NPZ
```

All validation and testing samples must use the statistics calculated from the
training split.

---

### 8. Run training

Training is launched through the StormCast training code and the corresponding
Hydra experiment configuration under:

```text
stormcast/config/
```

The preprocessing pipeline itself does not contain model checkpoints or
generated training runs.

---

## Important reminders

- Valid SINGV pressure-level times are `01`, `07`, `13`, and `19` UTC.
- Every model sample predicts the SINGV state six hours after its input.
- ERA5 background conditioning is taken at the same valid time as the SINGV
  input.
- SINGV states contain 75 channels.
- ERA5 backgrounds contain 26 channels.
- Prepared SINGV and ERA5 fields use the same `624 × 624` grid.
- Date ranges supplied to `build_dataset.py` are inclusive.
- Pairs do not cross the requested dataset boundary.
- Known missing SINGV source dates are skipped rather than interpolated.
- Normalisation statistics must be calculated from training data only.
- Invalid pressure-level SINGV cells remain `NaN` in prepared files and are
  excluded appropriately during training.
- `build_pair.py` is intended for individual samples; `build_dataset.py` is
  intended for complete ranges.
- `prep_utils.py` and `paths.py` are helper modules rather than normal
  command-line entry points.
- Generated NetCDF files, manifests, audit reports, statistics, logs, and
  checkpoints should not be committed to Git.
- `__pycache__/` is generated automatically by Python and should be excluded
  from Git.