# Preprocessing
## Prepare the dataset
Dataset preparation is done by the python file `build_dataset.py`.

To build a training range, supply the following details:
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

This creates
```bash
~/scratch/retraining/manifests/training_19950101_19950131.csv
```

The code automatically checks for the raw assembled data at `~/scratch/retraining/assembled` and for the preprocessed data at `/scratch/retraining/prepared`.

If the data doesn't exist, then it is automatically assembled from
```bash
/home/project/13004327/data_service/model_data/V3_Historical/V3-WMC-2/CCRS/ERA5/historical/reanalysis/SINGV-RCM/vn5
```
 and preprocessed.

## Compute normalisation statistics
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

