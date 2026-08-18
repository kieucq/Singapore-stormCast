# Pretrained StormCast Inference

Scripts for collecting SingV3/SINGV-RCM data on NSCC Aspire2A, converting it into the format expected by the pretrained StormCast model, and running inference.

NetCDF outputs are arranged in an `ncview`-friendly format for inspection. Output files are written under:

```text
~/scratch/pretrained/
├── singv_collated/
├── singv_inputs/
└── singv_forecasts/
```

## Workflow

Run the following commands from `pretrained_inference/`.

### 1. Collate SingV3 data

```bash
python collect_singv.py --datetime YYYY-MM-DDTHH:MM
```

Available pressure-level times are:

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

Output:

```text
~/scratch/pretrained/singv_inputs/
```

### 3. Run inference

```bash
python run_forecast.py INPUT_FILE.nc
```

Output:

```text
~/scratch/pretrained/singv_forecasts/
```

The forecast length is controlled by `N_STEPS` in `run_forecast.py`. It is currently set to one step.

## Diagnostics

Run diagnostic scripts from `pretrained_inference/diagnostics/`.

Validate the variables and coordinates in a collated dataset:

```bash
python validate_singv_dataset.py COLLATED_FILE
```

Print variable statistics and pressure-level diagnostics:

```bash
python inspect_singv_dataset.py COLLATED_FILE
```

Compare collated and prepared 2 m temperature fields:

```bash
python compare_t2m_fields.py COLLATED_FILE INPUT_FILE
```

## Notes

This pipeline adapts SingV3 data for a StormCast model pretrained on a different domain and data format. Several fields, including surface pressure, hybrid-level variables, and composite reflectivity, therefore require approximate conversions.
