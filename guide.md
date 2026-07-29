# Introduction
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
where 
```bash
/scratch/users/nus/e1155933/Singapore-stormCast
```

The code creates a repository
```bash
/home/users/nus/e1155933/scratch/retraining
```
which has the structure
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

The code automatically checks for the raw assembled data at `~/scratch/retraining/assembled` and for the preprocessed data at `/scratch/retraining/prepared`.

If the data doesn't exist, then it is automatically assembled from
```bash
/home/project/13004327/data_service/model_data/V3_Historical/V3-WMC-2/CCRS/ERA5/historical/reanalysis/SINGV-RCM/vn5
```
 and preprocessed.

# Prepare the validation dataset
Similarly, do
```bash
python retraining/build_dataset.py \
	--split validation \
	--start-date 1995-02-01
	--end-date 1995-02-07
```
to create a one-week validation dataset at
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

I.e., identical files are produced. The `npz` file is ultimately read by the later code, whereas the `csv` file is for human reading.

# Run model training
To train a model based on a Hydra configuration, do
```bash
python Singapore-stormCast/stormcast/train.py \
	--config-name <Hydra config name>
```
in an interactive job or to submit a script that runs it to ASPIRE2A. 

For example,
```bash
python Singapore_stormCast/stormcast/train.py \
	--config-name singv_regression_1month.yaml
```

This requires the following structure:
```bash
Singapore-stormCast
└── stormcast
	├── train.py
	├── inference.py
    └── config
        └── <Hydra config name>.yaml
```

# Run model inference
Similar to training, one can run inference on an interactive GPU compute node by executing
```bash
python Singapore-stormCast inference.py \
	<Hydra config name>
```

This once again requires the `stormcast/` repository structure described under the section on model training.