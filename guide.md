# Introduction to pipeline
The working directory is generally hard-coded to the following structure:
```bash
Singapore-stormCast
├── retraining
└── stormcast
    ├── config
    │   ├── dataset
    │   ├── hydra
    │   ├── inference
    │   ├── model
    │   ├── sampler
    │   └── training
    ├── datasets
    └── utils
```
The `stormcast` directory is copied from the StormCast GitHub repo, whereas the `retraining` directory contains my preprocessing code.

The preprocessing code creates a data storage directory
```bash
/home/users/nus/e1155933/scratch/retraining
```
which henceforth will be abbreviated as
```bash
~/scratch/retraining/
```

The `~/scratch/retraining` directory, once all the code below is executed, will have the structure
```bash
retraining
├── assembled
├── inference
├── logs
├── manifests
├── normalisation_stats
├── prepared
└── runs
```

> For the rest of this guide, you are assumed to be in the `Singapore-stormCast` directory.

# Prepare the training dataset
Dataset preparation is done by the python file `build_dataset.py`.

To build a dataset, supply the following details:
- The dataset split: `training`, `validation` or `testing`
- An inclusive start date
- An inclusive end date

To execute the file, do
```bash
python retraining/build_dataset.py \
	--split training \
	--start-date 1995-01-01 \
	--end-date 1995-01-31
```

This creates a one-month training dataset
```bash
~/scratch/retraining/manifests/training_19950101_19950131.csv
```

The code automatically checks for the raw assembled data at `~/scratch/retraining/assembled` and for the preprocessed data at `~/scratch/retraining/prepared`.

If the required data doesn't exist at these locations, then it is automatically assembled from
```bash
/home/project/13004327/data_service/model_data/V3_Historical/V3-WMC-2/CCRS/ERA5/historical/reanalysis/SINGV-RCM/vn5
```
 and preprocessed.

# Prepare the validation dataset
Similarly, do
```bash
python retraining/build_dataset.py \
	--split validation \
	--start-date 1995-02-01 \
	--end-date 1995-02-07
```
to create a one-week validation dataset
```bash
~/scratch/retraining/manifests/validation_19950201_19950207.csv
```

# Compute normalisation statistics
Computation of normalisation statistics is done by `compute_normalisation_stats.py`.

Run on a **training manifest**:
```bash
python retraining/compute_normalisation_stats.py \
	~/scratch/retraining/manifests/training_19950101_19950131.csv
```

This produces
```bash
~/scratch/retraining/normalisation_stats/training_19950101_19950131_normalisation_stats.csv
```
and
```bash
~/scratch/retraining/normalisation_stats/training_19950101_19950131_normalisation_stats.npz
```

Both files contain essentially the same data. The `npz` file is eventually read by the training code, while the `csv` file is a human-readable version.

# Run model training
Executing training and inference requires inputting numerous parameters to StormCast's training code. These parameters are conveyed via Hydra configuration files.

For this example pipeline, the required Hydra configuration files have been prepared and are located at
```bash
stormcast/config
├── dataset
│   └── singv_1month.yaml
├── singv_diffusion_1month.yaml
├── singv_inference_1month.yaml
└── singv_regression_1month.yaml
```
These files can be inspected to better understand the required parameters. Sample files provided by the StormCast GitHub repo – `regression.yaml`, `regression_lite.yaml`, `diffusion.yaml` and `diffusion_lite.yaml` – are also located in the same `config` directory.

To train a model based on a Hydra configuration, do
```bash
cd stormcast

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name <Hydra config name>
```
in a GPU compute node.

For example,
```bash
cd stormcast

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name singv_regression_1month
```

The training and inference code **requires** the following structure:
```bash
Singapore-stormCast
└── stormcast
	├── train.py
	├── inference.py
    └── config
        └── <Hydra config name>.yaml
```
Furthermore, you **must** be in the `stormcast` directory when you run training or inference.

After running the above regression training and remaining in the `stormcast` directory, do
```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --config-name singv_diffusion_1month
```
to perform diffusion training and to complete the model.

The checkpoints, validation plots and logs of the regression and diffusion training are located at
```bash
~/scratch/retraining/runs/singv-regression-1month
~/scratch/retraining/runs/singv-diffusion-1month
```
respectively.

# Run model inference
Similar to training, one can run inference on an interactive GPU compute node by executing
```bash
python inference.py \
	--config-name singv_inference_1month
```
Once again, note that you must be in the `stormcast` directory.

Inference is computed with `1995-02-01T01:00` as input, the first `datetime` in the validation dataset, and the forecast datetime is `1995-02-01T07:00`.

The following variables are forecasted:
- `pr`
- `tas`
- `ta_1000`
- `ta_850`
- `hus_850`
- `zg_500`

These parameters can be modified in the inference Hydra configuration file `stormcast/config/singv_inference_1month.yaml`.

This inference code has been modified from the original to produce two types of forecasts for each variable:
- One forecast output is produced only by the regression model.
- The other then incorporates the diffusion model's contribution.

The output plots can be accessed at
```bash
~/scratch/retraining/inference/singv-inference-1month/regression-diffusion-comparison
```