# StormCast Implementation for Singapore

This repository contains work from a summer internship project focused on adapting NVIDIA's StormCast model to the Singapore domain using SINGV3 data for training, validation, and testing.

The project was carried out in the Core Modelling Development branch of the Centre for Climate Research Singapore by Aditya Jayaraj, an undergraduate student from the National University of Singapore, under the supervision of Dr Chanh Kieu.

Questions and enquiries may be directed to `chanh_kieu@nea.gov.sg`.

## Repository structure

### `pretrained/`

Code for preparing SINGV data and running inference with the original pretrained StormCast model.

### `retraining/`

Data-preparation and job-orchestration code used for model retraining.

This directory converts raw SINGV data into StormCast-ready training samples. It includes:

- variable cleaning and preprocessing;
- pressure-level masking;
- normalization-statistics calculation;
- input-target pair construction;
- dataset assembly;
- PBS job submission and training launch scripts.

This directory prepares the data but does not contain the StormCast training implementation itself.

### `stormcast/`

Repository-managed copy of the NVIDIA PhysicsNeMo StormCast training application.

This directory contains the code used to:

- load prepared datasets;
- construct the regression and diffusion models;
- calculate training losses;
- perform training and validation;
- save checkpoints and diagnostics;
- run model inference.

Project-specific dataset adapters and Hydra configurations are stored here because they are used directly by the StormCast training application.

Important paths include:

- `stormcast/datasets/regression_dataset_adapter.py`  
  Loads the prepared SINGV samples during training and validation.

- `stormcast/config/dataset/`  
  Contains dataset-specific Hydra configurations.

- `stormcast/config/`  
  Contains experiment, model, training, sampler, and inference configurations.

- `stormcast/utils/`  
  Contains the main training, loss, model-input, plotting, checkpointing, and scheduling utilities.

In summary:

- `retraining/` creates model-ready data.
- `stormcast/` consumes that data and trains or runs the model.