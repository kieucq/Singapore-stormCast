# Introduction to pipeline

The relevant repository structure is:

```text
Singapore-stormCast/
├── preprocessing/
├── stormcast/
│   ├── config/
│   ├── datasets/
│   ├── utils/
│   ├── train.py
│   └── inference.py
└── PIPELINE_GUIDE.md
```

The `stormcast` directory contains the StormCast training and inference code, with modifications for SINGV data. The `preprocessing` directory contains the SINGV and ERA5 preprocessing pipeline.

Generated data, model checkpoints and inference outputs are stored under a common data root. By default:

```text
~/scratch/stormcast-data/
```

After running the full pipeline, this directory will have approximately the following structure:

```text
stormcast-data/
├── assembled/
├── background/
│   ├── raw/
│   └── prepared/
├── manifests/
├── normalisation_stats/
├── prepared/
├── runs/
└── inference/
```

> For the rest of this guide, you are assumed to be in the `Singapore-stormCast` directory unless stated otherwise.

# Configure file paths

Before running the pipeline, configure the data paths in:

```text
preprocessing/paths.py
```

Set `DATA_ROOT` to the location where generated data and model outputs should be stored. By default:

```python
DATA_ROOT = Path("~/scratch/stormcast-data").expanduser()
```

Also verify that `SINGV_ARCHIVE_ROOT` points to the SINGV-V3 archive. The immediate contents of the correct directory should look like:

```text
SINGV_ARCHIVE_ROOT/
├── 10min/
├── 12min/
├── 1hr/
├── 3hr/
├── 6hr/
├── Amon/
└── day/
```

On NSCC, the archive is currently located at:

```text
/home/project/13004327/data_service/model_data/V3_Historical/V3-WMC-2/CCRS/ERA5/historical/reanalysis/SINGV-RCM/vn5
```

Finally, ensure that `data_root` in:

```text
stormcast/config/dataset/singv_1month.yaml
```

**points to the exact same location** as `DATA_ROOT` in `preprocessing/paths.py`.

# Prepare the training dataset

Dataset preparation is handled by:

```text
preprocessing/build_dataset.py
```

The script requires:

- a dataset split: `training`, `validation` or `testing`;
- an inclusive start date;
- an inclusive end date.

To prepare the one-month training dataset used in this example:

```bash
python preprocessing/build_dataset.py \
    --split training \
    --start-date 1995-01-01 \
    --end-date 1995-01-31
```

This creates the training manifest:

```text
~/scratch/stormcast-data/manifests/training_19950101_19950131.csv
```

Each training sample has the form:

```text
SINGV(t) + ERA5(t) → SINGV(t + 6 h)
```

`build_dataset.py` prepares both the high-resolution SINGV state and the ERA5 background required by the model.

Prepared SINGV data are stored under:

```text
~/scratch/stormcast-data/assembled/
~/scratch/stormcast-data/prepared/
```

ERA5 background data are stored under:

```text
~/scratch/stormcast-data/background/raw/
~/scratch/stormcast-data/background/prepared/
```

Existing files are reused. Missing SINGV states are assembled from `SINGV_ARCHIVE_ROOT` and preprocessed automatically. Missing ERA5 background fields are downloaded, prepared and interpolated onto the SINGV grid automatically.

# Prepare the validation dataset

Prepare the one-week validation dataset with:

```bash
python preprocessing/build_dataset.py \
    --split validation \
    --start-date 1995-02-01 \
    --end-date 1995-02-07
```

This creates:

```text
~/scratch/stormcast-data/manifests/validation_19950201_19950207.csv
```

# Compute normalisation statistics

Normalisation statistics are computed from the training dataset using:

```text
preprocessing/compute_normalisation_stats.py
```

Run:

```bash
python preprocessing/compute_normalisation_stats.py \
    ~/scratch/stormcast-data/manifests/training_19950101_19950131.csv
```

Separate statistics are computed for the SINGV state and ERA5 background.

The files used by the training code are:

```text
~/scratch/stormcast-data/normalisation_stats/training_19950101_19950131_normalisation_stats.npz
```

and:

```text
~/scratch/stormcast-data/normalisation_stats/training_19950101_19950131_background_normalisation_stats.npz
```

A CSV version of the SINGV statistics is also produced for inspection:

```text
~/scratch/stormcast-data/normalisation_stats/training_19950101_19950131_normalisation_stats.csv
```

# Run model training

StormCast training and inference settings are provided through Hydra configuration files.

The main configuration files for this example are:

```text
stormcast/config/
├── singv_regression_1month.yaml   # regression entry point
├── singv_diffusion_1month.yaml    # diffusion entry point
├── singv_inference_1month.yaml    # inference entry point
│
├── dataset/                       # dataset definitions
├── model/                         # model settings and conditioning
├── training/                      # training settings
├── inference/                     # inference output settings
├── sampler/                       # diffusion sampler settings
└── hydra/                         # Hydra runtime settings
```

The three `singv_*_1month.yaml` files in the main `config` directory are the entry points used by this guide. They combine the detailed settings stored in the subdirectories.

Common settings that a user may want to change, such as checkpoint paths, inference start time and forecast length, are kept in the entry-point files. Lower-level dataset, model, training and sampler settings are kept in the corresponding subdirectories.

The original StormCast example configs — `regression.yaml`, `regression_lite.yaml`, `diffusion.yaml` and `diffusion_lite.yaml` — are also kept in the main `config` directory for reference.

## Regression training

Training and inference must be run from inside the `stormcast` directory.

Enter the directory:

```bash
cd stormcast
```

Then, on a GPU compute node, run:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name singv_regression_1month
```

The example regression run trains for 1,000 steps.

Its checkpoints, validation plots and logs are stored under:

```text
~/scratch/stormcast-data/runs/singv-regression-1month/training-1000steps/
```

## Diffusion training

Regression training must be completed before diffusion training because the diffusion model uses the trained regression model as part of its conditioning.

While remaining in the `stormcast` directory, run:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name singv_diffusion_1month
```

The example diffusion run also trains for 1,000 steps.

Its checkpoints, validation plots and logs are stored under:

```text
~/scratch/stormcast-data/runs/singv-diffusion-1month/training-1000steps/
```

Both regression and diffusion validation use the following diagnostic variables:

```text
tas
pr
ta_1000
ua_1000
ta_850
zg_500
```

# Run model inference

Inference must also be run from inside the `stormcast` directory on a GPU compute node.

Run:

```bash
python inference.py \
    --config-name singv_inference_1month
```

The example configuration starts from:

```text
1995-02-01T01:00
```

which is the first input time in the validation dataset.

Each autoregressive forecast step corresponds to 6 hours. The example uses:

```yaml
n_steps: 1
```

and therefore produces a forecast for:

```text
1995-02-01T07:00
```

The forecast start time and number of forecast steps can be changed in:

```text
stormcast/config/singv_inference_1month.yaml
```

StormCast predicts the full SINGV state. For this example, the following six variables are selected for inference plots and numerical output:

```text
tas
pr
ta_1000
ua_1000
ta_850
zg_500
```

For each selected variable, the modified inference code produces:

- a regression-only forecast;
- a regression + diffusion forecast, where the diffusion model provides a residual correction to the regression forecast.

The main outputs for this example are PNG comparison plots showing the model forecasts against the corresponding SINGV targets.

The inference outputs are stored under:

```text
~/scratch/stormcast-data/inference/singv-inference-1month/regression-diffusion-comparison/
```

# Quick run summary

After configuring the file paths as described previously, the full example pipeline is:

```bash
# From Singapore-stormCast/

python preprocessing/build_dataset.py \
    --split training \
    --start-date 1995-01-01 \
    --end-date 1995-01-31

python preprocessing/build_dataset.py \
    --split validation \
    --start-date 1995-02-01 \
    --end-date 1995-02-07

python preprocessing/compute_normalisation_stats.py \
    ~/scratch/stormcast-data/manifests/training_19950101_19950131.csv
```

Then enter the StormCast directory:

```bash
cd stormcast
```

On a GPU compute node, run regression training:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name singv_regression_1month
```

After regression training finishes, run diffusion training:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name singv_diffusion_1month
```

Then run inference:

```bash
python inference.py \
    --config-name singv_inference_1month
```

The final inference plots are saved under:

```text
~/scratch/stormcast-data/inference/singv-inference-1month/regression-diffusion-comparison/
```