# Description of contents
The first portion of my work was done on Colab notebooks, where I learnt how to use StormCast and worked on processing custom ERA5 data and plugging that into StormCast. Below is a list of each notebook in this folder and the changes between each version.

- `01_earth2studio.ipynb`: Very first notebook sent to Dr Chanh, before internship began. Simple code to run inference on StormCast using Earth2Studio.
- `stormCast_guide.ipynb`: Dr Chanh's guide to show me the next steps.
- `02_t2m_to_stormcast.ipynb`: Initial attempt of plugging in ERA5 data into StormCast using `t2m`: temperature at 2m above surface.
- `03_direct_vars_to_stormcast.ipynb`: All single-level variables are now plugged into StormCast (4/99 variables) with remaining set to zero.
- `04_most_vars_to_stormcast.ipynb`: 98/99 variables can be inputted into StormCast now. Conversion from ERA5's fixed pressure levels to HRRR's hybrid levels is approximated using simple linear interpolation and [sigma level parameters](https://rapidrefresh.noaa.gov/faq/HRRR.faq.html).
- `05_xr_standardisation.ipynb`: Standardise the default data container in `MyLocalData` (ie `self.data`) to be in `xr.DataArray` format. Created `utils.py` to store mappings and helper functions.
- `06_refc.ipynb`: Implemented a proxy to convert total cloud rain water (`tcrw`) to composite reflectivity (`refc`). All 99 variables are now at least approximately plugged into StormCast.
- `07_lead0_rmse_sanity_check.ipynb`: Created helper functions that are used by `build_input_array()` as well as the new `build_era5_truth()`. Computed RMSE between input and 0h lead time inference and verified that all are zero.
- `08_rmse.ipynb`: Notebook now computes RMSE, MAE and bias between ERA5 truth and StormCast inference over Singapore for up to 12h lead time. Also plots RMSE vs lead time for selected variables as well as visuals of the two versions.

- `utils.py`: Utilities file that is compatible with `08_rmse.ipynb`. May not work for older versions.

# Data used
All notebooks until `07_lead0_rmse_sanity_check.ipynb` use ERA5 data from 2025-03-25 00:00, while `08_rmse.ipynb` uses ERA5 data from 2025-11-27 06:00 until 18:00 inclusive. The specific sources and variables are:

1. From ERA5 database on [pressure-level variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download):
   - Geopotential
   - Specific humidity
   - Temperature
   - U-component of wind
   - V-component of wind
2. From ERA5 database on [surface variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download):
   - 10m u-component of wind
   - 10m v-component of wind
   - 2m temperature
   - Mean sea level pressure
   - Surface pressure
   - Total column rain water
  
Latitude range  = [-5, 15]; longitude range = [95, 115]
File type: NetCDF

# Figures folder
The figures folder contains images of important plots that may not be rendered in the notebooks when viewed on GitHub.

