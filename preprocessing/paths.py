from pathlib import Path


# Generated project data
DATA_ROOT = Path("~/scratch/stormcast-data").expanduser()

ASSEMBLED_DIR = DATA_ROOT / "assembled"
PREPARED_DIR = DATA_ROOT / "prepared"
MANIFEST_DIR = DATA_ROOT / "manifests"
NORMALISATION_DIR = DATA_ROOT / "normalisation_stats"
AUDIT_DIR = DATA_ROOT / "audit"

BACKGROUND_RAW_DIR = DATA_ROOT / "background" / "raw"
BACKGROUND_PREPARED_DIR = DATA_ROOT / "background" / "prepared"


# External SINGV archive
SINGV_ARCHIVE_ROOT = Path(
    "/home/project/13004327/data_service/model_data/"
    "V3_Historical/V3-WMC-2/CCRS/ERA5/historical/"
    "reanalysis/SINGV-RCM/vn5"
)