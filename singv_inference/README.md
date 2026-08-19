# Pretrained StormCast Inference with SINGV

This directory contains an exploratory pipeline for running the original pretrained StormCast model using SINGV-RCM data.

The workflow:

1. collects the required SINGV variables;
2. converts them into StormCast-compatible inputs;
3. runs pretrained StormCast inference using ERA5 background conditioning.

This was used as an intermediate step before retraining StormCast on SINGV data.

## Configure paths

Filesystem paths are configured in:

```text
paths.py
```

Set:

- `DATA_ROOT` to the location used for generated files;
- `SINGV_ARCHIVE_ROOT` to the SINGV-RCM archive.

By default, generated files are stored under:

```text
~/scratch/pretrained/
├── singv_collated/
├── singv_inputs/
└── singv_forecasts/
```

## Workflow

Run the following commands from `singv_inference/`.

### 1. Collect SINGV data

```bash
python collect_singv.py --datetime YYYY-MM-DDTHH:MM
```

Pressure-level fields are available at:

```text
01:00, 07:00, 13:00, 19:00 UTC
```

Output:

```text
~/scratch/pretrained/singv_collated/
```

### 2. Prepare StormCast input

```bash
python prepare_singv_input.py COLLATED_FILE.nc
```

This converts the SINGV fields into the 99-channel input format expected by pretrained StormCast.

Output:

```text
~/scratch/pretrained/singv_inputs/
```

### 3. Run inference

```bash
python run_forecast.py INPUT_FILE.nc
```

StormCast uses the prepared SINGV state together with ERA5 background conditioning retrieved through ARCO.

Output:

```text
~/scratch/pretrained/singv_forecasts/
```

The forecast length is controlled by `N_STEPS` in `run_forecast.py`.

## Notes

The pretrained StormCast model was developed for a different domain and input format. Some SINGV fields therefore require approximate conversions, including:

- surface pressure;
- pressure-level to hybrid-level conversion;
- composite reflectivity derived from precipitation.

This workflow was intended to test SINGV compatibility with pretrained StormCast rather than to produce physically reliable forecasts.