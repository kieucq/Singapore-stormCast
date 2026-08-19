"""
Filesystem paths used by the pretrained SINGV inference workflow.

Edit DATA_ROOT and SINGV_ARCHIVE_ROOT for the local system.
All other paths are derived from these locations.
"""

from pathlib import Path

DATA_ROOT = Path("~/scratch/pretrained").expanduser()

COLLATED_DIR = DATA_ROOT / "singv_collated"
INPUT_DIR = DATA_ROOT / "singv_inputs"
FORECAST_DIR = DATA_ROOT / "singv_forecasts"

SINGV_ARCHIVE_ROOT = Path(
    "/home/project/13004327/data_service/model_data/"
    "V3_Historical/V3-WMC-2/CCRS/ERA5/historical/"
    "reanalysis/SINGV-RCM/vn5"
)