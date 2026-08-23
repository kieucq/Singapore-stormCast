# StormCast Implementation for Singapore

This repository contains work from a summer internship project adapting NVIDIA's StormCast model for Singapore using SINGV-RCM data.

The project was carried out in the Core Modelling Development branch of the Centre for Climate Research Singapore (CCRS) by Aditya Jayaraj, an undergraduate from the National University of Singapore, under the supervision of Dr Chanh Kieu.

Questions and enquiries may be directed to either `aditya.j@u.nus.edu` or `chanh_kieu@nea.gov.sg/kieucq@gmail.com`.

For a step-by-step example covering preprocessing, training and inference, see [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md).

## Project overview

The training pipeline uses high-resolution SINGV states together with ERA5 background fields:

```text
SINGV(t) + ERA5(t) → SINGV(t + 6 h)
```

StormCast training is performed in two stages:

1. **Regression** — produces the deterministic forecast.
2. **Diffusion** — learns a residual correction to the regression forecast.

The modified inference code can produce both regression-only and regression + diffusion forecasts.

## Repository structure

```text
Singapore-stormCast/
├── preprocessing/
├── stormcast/
├── era5_inference/
├── singv_inference/
├── PIPELINE_GUIDE.md
└── README.md
```

### `preprocessing/`

Contains the preprocessing pipeline used to prepare SINGV and ERA5 data for StormCast.

Main tasks include:

- assembling raw SINGV variables;
- cleaning and regridding SINGV fields;
- masking invalid pressure-level cells;
- downloading and preparing ERA5 background fields;
- constructing 6-hour input-target pairs;
- building training, validation and testing manifests;
- computing SINGV and ERA5 normalisation statistics.

Filesystem locations used by the preprocessing pipeline are configured in:

```text
preprocessing/paths.py
```

### `stormcast/`

Contains the modified StormCast training and inference application based on NVIDIA PhysicsNeMo.

This directory is used to:

- load prepared SINGV and ERA5 data;
- train regression and diffusion models;
- perform validation and generate diagnostics;
- save model checkpoints;
- run autoregressive inference.

Important paths include:

- `stormcast/datasets/singv_dataset_adapter.py`  
  Loads and normalises prepared SINGV and ERA5 samples.

- `stormcast/config/`  
  Contains Hydra configurations for datasets, models, training, sampling and inference.

- `stormcast/train.py`  
  Entry point for regression and diffusion training.

- `stormcast/inference.py`  
  Entry point for model inference.

- `stormcast/utils/`  
  Contains model, training, plotting, checkpointing and scheduling utilities.

The main example configurations are:

```text
stormcast/config/singv_regression_1month.yaml
stormcast/config/singv_diffusion_1month.yaml
stormcast/config/singv_inference_1month.yaml
```

### `era5_inference/` and `singv_inference/`

These directories contain earlier exercises using the original pretrained StormCast model. They are not required for the preprocessing, training and inference pipeline described in [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md).

#### `era5_inference/`

Contains my first experiments with StormCast inference using ERA5 data.

The aim was mainly to understand StormCast's expected inputs, preprocessing steps and inference workflow using relatively clean and familiar data. ERA5 fields were retrieved, interpolated onto the required grid and converted into the format expected by the pretrained StormCast model.

This work was done mainly on Google Colab, so the directory consists mostly of Jupyter notebooks rather than a clean pipeline.

#### `singv_inference/`

Contains the next stage of these experiments: running the original pretrained StormCast model using SINGV data instead of ERA5.

The aim was to learn how to prepare a custom regional dataset for StormCast and to identify the differences between the SINGV fields and the inputs expected by the pretrained model.

This includes collecting and preprocessing SINGV variables, preparing StormCast-compatible inputs and running pretrained-model inference.

These two directories mainly document the exploratory work that preceded the final SINGV retraining pipeline.

## Data and outputs

By default, generated data and model outputs are stored under:

```text
~/scratch/stormcast-data/
```

with the approximate structure:

```text
stormcast-data/
├── assembled/
├── background/
├── manifests/
├── normalisation_stats/
├── prepared/
├── runs/
└── inference/
```

The data root can be changed in the preprocessing and StormCast dataset configurations.

## Current status and future work

The full pipeline from data preparation to training and inference is complete and tested. The regression model produces reasonable forecasts, but the diffusion model currently adds too much small-scale noise.

The main suspected issue is masking. SINGV pressure-level data contain invalid cells below terrain, while the original HRRR-based StormCast setup does not require this treatment. Masking has been implemented, but its handling in the loss and diffusion calculations still needs improvement.

Further work includes:

- improving mask handling throughout training;
- adding land mask and orography as model inputs;
- testing 1-hour surface-only forecasts instead of the current 6-hour forecasts.

The current 6-hour interval is limited by the availability of SINGV pressure-level data. A 1-hour surface-only setup may be better suited to short-lived variables such as precipitation (`pr`) and to the diffusion model.

## Quick start

See [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md) for the full example workflow:

```text
prepare training data
        ↓
prepare validation data
        ↓
compute normalisation statistics
        ↓
train regression model
        ↓
train diffusion model
        ↓
run inference
```
