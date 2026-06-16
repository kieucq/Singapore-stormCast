import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import gc
from pathlib import Path

# ── Constants / mappings ─────────────────────────

REFC_ALPHA = 10.0
REFC_TCRW_THRESHOLD = 0.2
REFC_MIN_DBZ = -10.0
REFC_MAX_DBZ = 75.0

STORMCAST_SURFACE_VARS = {"t2m", "u10m", "v10m", "mslp", "refc"}

# ERA5 variable name → StormCast variable name (surface/single-level)
ERA5_SURFACE_MAP = {
    "t2m": "t2m",
    "u10": "u10m",
    "v10": "v10m",
    "msl": "mslp",
}

# ERA5 variable name → StormCast variable prefix (pressure-level)
ERA5_PRESSURE_MAP = {
    "u": "u",
    "v": "v",
    "t": "t",
    "q": "q",
    "z": "Z",   # ERA5 geopotential [m²/s²] → StormCast geopotential height [m]
}

# Mapping of HRRR hybrid level number → approximate sigma value
HRRR_SIGMA = {
     1: 1.0000,  2: 0.9980,  3: 0.9940,  4: 0.9870,  5: 0.9750,
     6: 0.9590,  7: 0.9390,  8: 0.9160,  9: 0.8920, 10: 0.8650,
    11: 0.8350, 13: 0.7660, 15: 0.6850, 20: 0.4565, 25: 0.3078,
    30: 0.2188,
}

# Variable units, used for colourbar labels
UNITS = {
    "t2m": "K",   "u10m": "m/s", "v10m": "m/s", "mslp": "Pa",
    "refc": "dBZ",
}


# ── Basic utilities ──────────────────────────────

def infer_unit(var: str) -> str:
    if var in UNITS:
        return UNITS[var]
    prefix = var[0]
    return {"t": "K", "u": "m/s", "v": "m/s", "q": "kg/kg",
            "Z": "m", "p": "Pa"}.get(prefix, "")

def assert_no_nan(name, field):
    if np.isnan(field).any():
        frac = np.isnan(field).mean()
        raise ValueError(f"{name} contains NaNs after processing. NaN fraction={frac:.6f}")
    
class GridSpec:
    def __init__(self, src_lats, src_lons, target_lats, target_lons, hrrr_y, hrrr_x):
        self.src_lats = src_lats
        self.src_lons = src_lons
        self.target_lats = target_lats
        self.target_lons = target_lons
        self.hrrr_y = hrrr_y
        self.hrrr_x = hrrr_x

    @property
    def ny(self):
        return len(self.hrrr_y)

    @property
    def nx(self):
        return len(self.hrrr_x)
    
def make_grid(ds_surface, hrrr_y, hrrr_x):
    src_lats = ds_surface.latitude.values
    src_lons = ds_surface.longitude.values

    target_lats = np.linspace(src_lats.max(), src_lats.min(), len(hrrr_y))
    target_lons = np.linspace(src_lons.min(), src_lons.max(), len(hrrr_x))

    return GridSpec(
        src_lats=src_lats,
        src_lons=src_lons,
        target_lats=target_lats,
        target_lons=target_lons,
        hrrr_y=hrrr_y,
        hrrr_x=hrrr_x,
    )


# ── Earth2Studio data wrapper ────────────────────

class MyLocalData:
    """
    Wraps an xr.DataArray into the data-source interface expected by Earth2Studio.

    Expected input DataArray dimensions:
        (time, variable, hrrr_y, hrrr_x)

    Expected coordinates:
        time
        variable
        hrrr_y
        hrrr_x
    """

    def __init__(self, data: xr.DataArray):
        expected_dims = ("time", "variable", "hrrr_y", "hrrr_x")

        missing_dims = [dim for dim in expected_dims if dim not in data.dims]
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        data = data.transpose(*expected_dims)

        times = data["time"].values
        variables = data["variable"].values

        if len(np.unique(times)) != len(times):
            raise ValueError(f"Duplicate times found: {times}")

        if len(np.unique(variables)) != len(variables):
            raise ValueError("Duplicate variables found in input DataArray.")

        self.data = data.astype(np.float32)

    def __call__(self, time, variable):
        requested_times = np.atleast_1d(np.array(time, dtype="datetime64[ns]"))

        if isinstance(variable, str):
            requested_variables = [variable]
        else:
            requested_variables = list(variable)

        available_times = self.data["time"].values
        available_variables = self.data["variable"].values

        missing_times = [
            t for t in requested_times
            if t not in available_times
        ]
        if missing_times:
            raise ValueError(
                f"Requested times not found: {missing_times}. "
                f"Available times: {available_times}"
            )

        missing_variables = [
            v for v in requested_variables
            if v not in available_variables
        ]
        if missing_variables:
            raise ValueError(
                f"Requested variables not found: {missing_variables}. "
                f"Available variables include: {list(available_variables)}"
            )

        return self.data.sel(
            time=requested_times,
            variable=requested_variables,
        ).transpose("time", "variable", "hrrr_y", "hrrr_x")



# ── ERA5 validation ──────────────────────────────

def validate_era5_pair(ds_surface, ds_pressure):
    required_surface_vars = ["t2m", "u10", "v10", "msl", "sp", "tcrw"]
    required_pressure_vars = ["z", "q", "t", "u", "v"]

    missing_surface = [v for v in required_surface_vars if v not in ds_surface]
    missing_pressure = [v for v in required_pressure_vars if v not in ds_pressure]

    if missing_surface:
        raise ValueError(f"Missing surface variables: {missing_surface}")

    if missing_pressure:
        raise ValueError(f"Missing pressure variables: {missing_pressure}")

    if not np.array_equal(
        ds_surface.valid_time.values.astype("datetime64[ns]"),
        ds_pressure.valid_time.values.astype("datetime64[ns]"),
    ):
        raise ValueError("Surface and pressure valid_time coordinates do not match.")

    if not np.allclose(ds_surface.latitude.values, ds_pressure.latitude.values):
        raise ValueError("Latitude coordinates do not match.")

    if not np.allclose(ds_surface.longitude.values, ds_pressure.longitude.values):
        raise ValueError("Longitude coordinates do not match.")

    return True

def validate_forecast_window(
    ds_surface,
    start_time,
    nsteps,
    require_gfs_fx_time=False,
):
    start_time = np.datetime64(start_time, "ns")

    available_times = ds_surface.valid_time.values.astype("datetime64[ns]")

    needed_times = np.array(
        [start_time + np.timedelta64(h, "h") for h in range(nsteps + 1)],
        dtype="datetime64[ns]",
    )

    missing_times = [t for t in needed_times if t not in available_times]

    if missing_times:
        raise ValueError(
            "ERA5 files do not cover all required valid times.\n"
            f"start_time: {start_time}\n"
            f"nsteps: {nsteps}\n"
            f"needed range: {needed_times[0]} to {needed_times[-1]}\n"
            f"missing times: {missing_times}\n"
            f"available range: {available_times[0]} to {available_times[-1]}"
        )

    if require_gfs_fx_time:
        hour = int(str(start_time).split("T")[1][:2])

        if hour not in [0, 6, 12, 18]:
            raise ValueError(
                f"STARTING_TIME {start_time} is not valid for GFS_FX. "
                "GFS_FX requires 00Z, 06Z, 12Z, or 18Z."
            )

    return True

def validate_stormcast_input_array(data, variables, ny, nx):
    expected_dims = ("time", "variable", "hrrr_y", "hrrr_x")

    if data.dims != expected_dims:
        raise ValueError(f"Expected dims {expected_dims}, got {data.dims}")

    expected_shape = (1, len(variables), ny, nx)

    if data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")

    is_filled = (data != 0).any(dim=("time", "hrrr_y", "hrrr_x"))

    filled = list(data["variable"].values[is_filled.values])
    missing = list(data["variable"].values[~is_filled.values])

    print(f"Filled {len(filled)}/{len(variables)} variables ({missing} left as zero)")

    if missing:
        raise ValueError(f"Some StormCast input variables were not filled: {missing}")

    return True

def validate_forecast_output(ds_out, nsteps, verbose=False):
    if verbose: 
        print(ds_out)
        print("lead_time:", ds_out.lead_time.values)
        print("n_leads:", ds_out.sizes["lead_time"])

    expected = nsteps + 1
    actual = ds_out.sizes["lead_time"]

    if actual != expected:
        raise ValueError(f"Expected {expected} lead times, got {actual}")

    return True

# ── Interpolation helpers ─────────────────

def hinterp(field, src_lats, src_lons, target_lats, target_lons):
    """
    Using bilinear interpolation, take an input grid src_lats * src_lons and return the
    interpolated gric target_lats * target_lons

    Note that this is an approximation as HRRR uses lambert conformal coordinates while ERA5 uses lat/lon

    Input and output fields are 2D maps/grids/matrices
    """

    field = np.asarray(field, dtype=np.float32)

    # ensures that latitudes are in ascending order
    # this is a requirement for RegularGridInterpolator
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        field = field[::-1, :]

    # our interpolation function. if target point is outside original domain, returns NaN
    fn = RegularGridInterpolator(
        (src_lats, src_lons), field,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # build target grid. lons2d/lats2d returns the lon/lat at the input grid pt [j, i]
    lons2d, lats2d = np.meshgrid(target_lons, target_lats)

    # RegularGridInterpolator wants points as a list, not as a grid
    # ie it wants [(lat0, lon0), (lat0, lon1), ... (lat1, lon0), ...]
    pts = np.stack([lats2d.ravel(), lons2d.ravel()], axis=-1)

    # interpolate and reshape back into 2D map of 32-bit floats
    return fn(pts).reshape(len(target_lats), len(target_lons)).astype(np.float32)


def hinterp_all_levels(
    da_pressure_var,
    src_lats,
    src_lons,
    target_lats,
    target_lons,
    time_index=0,
):
    """
    The xr.DataArray of pressure-level variables downloaded from ERA5 looks something
    like (variable, time, pressure level, latitute, longitude).

    Slice this dataset by variable (eg ds["z"], ds["t"], etc), and submit that slice
    into this function. This function will loop through all the pressure levels in
    ds["z"], ds["t"], etc and regrid (interpolate) each to the target grid.

    Note that although an xr.DataArray is inputted, the output is a numpy array

    Args:
        da_pressure_var:
            ERA5 DataArray with dimensions:
            (valid_time, pressure_level, latitude, longitude)

        src_lats, src_lons:
            Source ERA5 latitude/longitude coordinates.

        target_lats, target_lons:
            Target latitude/longitude coordinates.

        time_index:
            Which valid_time index to use. Defaults to 0.
            (just interpret time as valid_time. the meaning of valid time
             has some significance but it's not important in this context)

    Returns:
        NumPy array with shape:
            (n_pressure_levels, n_target_lats, n_target_lons)
    """
    regridded_levels = []

    n_pressure_levels = len(da_pressure_var["pressure_level"])

    for pressure_index in range(n_pressure_levels):
        # Select one 2D pressure-level slice: (latitude, longitude)
        field_2d = da_pressure_var.isel(
            valid_time=time_index,
            pressure_level=pressure_index,
        ).values.astype(np.float32)

        # Regrid that 2D slice to the target grid: (target_lat, target_lon)
        field_2d_regridded = hinterp(
            field=field_2d,
            src_lats=src_lats,
            src_lons=src_lons,
            target_lats=target_lats,
            target_lons=target_lons,
        )

        regridded_levels.append(field_2d_regridded)
        # regridded_levels is a list of 2D grids

    # Stack all pressure levels into one 3D array:
    # (pressure_level, target_lat, target_lon)
    return np.stack(regridded_levels, axis=0)

def vinterp(field_3d, src_pressure_pa, target_pressure_pa):
    """
    Vertically interpolate a 3D pressure-level field to a 2D target pressure field.

    Args:
        field_3d:
            Array of shape (n_levels, NY, NX).
            Example: ERA5 temperature on pressure levels after horizontal regridding.

        src_pressure_pa:
            1D array of ERA5 pressure levels in Pa, shape (n_levels,).

        target_pressure_pa:
            2D array of target pressures in Pa, shape (NY, NX).
            Example: sigma_level * surface_pressure.

    Returns:
        2D array of interpolated values, shape (NY, NX).
    """

    # 1. Sort pressure levels from low pressure to high pressure.
    # np.searchsorted expects the coordinate array to be sorted ascending.
    sort_order = np.argsort(src_pressure_pa)

    pressure_sorted = src_pressure_pa[sort_order]
    field_sorted = field_3d[sort_order, :, :]

    # 2. For each grid cell, find where the target pressure fits
    # between the available ERA5 pressure levels.

    # searchsorted returns the index where target_pressure_pa should be inserted
    # into the sorted array pressure_sorted to keep it sorted.
    # ie this is literally the index of the pressure level directly above the target pressure
    upper_index = np.searchsorted(pressure_sorted, target_pressure_pa)

    # Prevent indices from going outside the available pressure-level range.
    # forces 1 <= upper_index < len(pressure_sorted)
    # upper_index and lower_index are 2D grids of levels of shape (NY, NX)
    upper_index = np.clip(upper_index, 1, len(pressure_sorted) - 1)
    lower_index = upper_index - 1

    # 3. Build y/x index arrays so we can pick values column-by-column.
    y_indices = np.arange(target_pressure_pa.shape[0])[:, None]
    x_indices = np.arange(target_pressure_pa.shape[1])[None, :]

    # 4. Get the two pressure levels surrounding each target pressure.
    pressure_lower = pressure_sorted[lower_index]
    pressure_upper = pressure_sorted[upper_index]

    # 5. Get the field values at those two surrounding pressure levels.
    field_lower = field_sorted[lower_index, y_indices, x_indices]
    field_upper = field_sorted[upper_index, y_indices, x_indices]

    # 6. Compute linear interpolation weight.
    weight = (target_pressure_pa - pressure_lower) / (
        pressure_upper - pressure_lower
    )

    # 7. Linearly interpolate.
    interpolated = field_lower + weight * (field_upper - field_lower)

    return interpolated.astype(np.float32)


# ── ERA5 → StormCast conversion helpers ──────────

def _put_field(data, var_name, field, time_value=None):
    """
    Put a 2D field into either:
    - input array: (time, variable, hrrr_y, hrrr_x)
    - truth array: (variable, hrrr_y, hrrr_x)
    """
    if "time" in data.dims:
        if time_value is None:
            raise ValueError("time_value is required for data with a time dimension.")
        data.loc[dict(time=time_value, variable=var_name)] = field
    else:
        data.loc[dict(variable=var_name)] = field


# interpolates surface fields and inserts them into the xr.DataArray
def fill_surface_fields(data, ds_surface, grid, time_index=0, time_value=None, verbose=False):
    """Fill t2m, u10m, v10m, mslp."""

    available_vars = set(data["variable"].values)

    for era5_name, sc_name in ERA5_SURFACE_MAP.items():
        if sc_name not in available_vars:
            continue

        raw = ds_surface[era5_name].isel(valid_time=time_index).values.astype(np.float32)

        field = hinterp(
            raw,
            grid.src_lats,
            grid.src_lons,
            grid.target_lats,
            grid.target_lons,
        )

        assert_no_nan(sc_name, field)
        _put_field(data, sc_name, field, time_value=time_value)

        if verbose:
            print(
                f"  {era5_name:4s} → {sc_name:5s} | "
                f"min={field.min():.2f}  max={field.max():.2f}"
            )


def get_surface_pressure(ds_surface, grid, time_index=0, verbose=False):
    """Return surface pressure regridded to the StormCast grid."""

    sp_raw = ds_surface["sp"].isel(valid_time=time_index).values.astype(np.float32)

    sp = hinterp(
        sp_raw,
        grid.src_lats,
        grid.src_lons,
        grid.target_lats,
        grid.target_lons,
    )

    assert_no_nan("sp", sp)

    if verbose:
        print(
            f"\n  sp          | min={sp.min():.1f}  "
            f"max={sp.max():.1f}  mean={sp.mean():.1f} Pa"
        )

    return sp


def fill_hybrid_fields(data, ds_pressure, sp, grid, time_index=0, time_value=None, verbose=False):
    """
    Fill hybrid-level variables:
    u#hl, v#hl, t#hl, q#hl, Z#hl, p#hl.
    """

    era5_p_pa = ds_pressure.pressure_level.values.astype(np.float32) * 100.0 # convert hPa -> Pa
    available_vars = set(data["variable"].values)

    if verbose:
        print("\nRegridding ERA5 pressure-level variables...")

    pressure_3d = {}

    for era5_name in ERA5_PRESSURE_MAP:
        pressure_3d[era5_name] = hinterp_all_levels(
            ds_pressure[era5_name],
            grid.src_lats,
            grid.src_lons,
            grid.target_lats,
            grid.target_lons,
            time_index=time_index,
        )

        if verbose:
            print(f"  {era5_name} pressure levels regridded")

    for level, sigma in HRRR_SIGMA.items():
        p_target = (sigma * sp).astype(np.float32)

        p_name = f"p{level}hl"
        if p_name in available_vars:
            _put_field(data, p_name, p_target, time_value=time_value)

        for era5_name, sc_prefix in ERA5_PRESSURE_MAP.items():
            sc_name = f"{sc_prefix}{level}hl"

            if sc_name not in available_vars:
                continue

            field = vinterp(
                pressure_3d[era5_name],
                era5_p_pa,
                p_target,
            )

            if era5_name == "z":
                field = field / 9.80665

            assert_no_nan(sc_name, field)
            _put_field(data, sc_name, field.astype(np.float32), time_value=time_value)


def fill_refc(data, ds_surface, grid, time_index=0, time_value=None):
    """Fill refc using ERA5 tcrw proxy."""

    available_vars = set(data["variable"].values)

    if "refc" not in available_vars:
        return

    if "tcrw" not in ds_surface:
        return

    tcrw_raw = ds_surface["tcrw"].isel(valid_time=time_index).values.astype(np.float32)

    tcrw = hinterp(
        tcrw_raw,
        grid.src_lats,
        grid.src_lons,
        grid.target_lats,
        grid.target_lons,
    )

    assert_no_nan("tcrw", tcrw)

    refc = era5_tcrw_to_refc(
        tcrw=tcrw,
        alpha=10.0,
        tcrw_threshold=0.2,
        min_dbz=-10.0,
        max_dbz=75.0,
    )

    _put_field(data, "refc", refc, time_value=time_value)

def era5_tcrw_to_refc(
    tcrw,
    alpha=REFC_ALPHA,
    tcrw_threshold=REFC_TCRW_THRESHOLD,
    min_dbz=REFC_MIN_DBZ,
    max_dbz=REFC_MAX_DBZ,
):
    tcrw = np.maximum(np.asarray(tcrw, dtype=np.float32), 0.0)

    active = tcrw >= tcrw_threshold

    rain_rate = alpha * tcrw

    Z_linear = 200.0 * np.maximum(rain_rate, 1e-10) ** 1.6
    refc = 10.0 * np.log10(Z_linear)

    refc = np.where(active, refc, min_dbz)

    return np.clip(refc, min_dbz, max_dbz).astype(np.float32)



# —— Verification / diagnostics helpers ————————————————–

def lead_hour(ds, lead_idx):
    return int(ds.lead_time.values[lead_idx] / np.timedelta64(1, "h"))


def valid_time_for_lead(start_time, ds, lead_idx):
    return pd.Timestamp(start_time) + pd.to_timedelta(ds.lead_time.values[lead_idx])


def get_forecast_field(ds, var, lead_idx):
    return (
        ds[var]
        .isel(time=0, lead_time=lead_idx)
        .values
        .astype(np.float32)
    )


def get_truth_field(truth, var):
    return truth.sel(variable=var).values.astype(np.float32)


def compute_metrics(forecast, truth):
    error = forecast - truth

    return {
        "rmse": float(np.sqrt(np.nanmean(error ** 2))),
        "bias": float(np.nanmean(error)),
        "mae": float(np.nanmean(np.abs(error))),
    }

def compute_verification_metrics(ds_out, start_time, variables, truth_builder):
    records = []

    for lead_idx in range(ds_out.sizes["lead_time"]):
        valid_time = valid_time_for_lead(start_time, ds_out, lead_idx)
        hour = lead_hour(ds_out, lead_idx)

        print(f"Verifying lead {hour}h, valid_time={valid_time}")

        truth = truth_builder(valid_time.to_pydatetime(), variables=variables)

        for var in variables:
            forecast_field = get_forecast_field(ds_out, var, lead_idx)
            truth_field = get_truth_field(truth, var)

            metrics = compute_metrics(forecast_field, truth_field)

            records.append({
                "lead_idx": lead_idx,
                "lead_hour": hour,
                "valid_time": valid_time,
                "variable": var,
                **metrics,
                "unit": infer_unit(var),
            })

            del forecast_field, truth_field
        del truth
    return pd.DataFrame(records)

def build_truth_cache(ds_out, start_time, variables, leads, truth_builder):
    truth_cache = {}

    for lead_idx in leads:
        valid_time = valid_time_for_lead(start_time, ds_out, lead_idx)
        hour = lead_hour(ds_out, lead_idx)

        print(f"Building panel truth for lead {hour}h, valid_time={valid_time}")

        truth_cache[lead_idx] = truth_builder(
            valid_time.to_pydatetime(),
            variables=variables,
        )

    return truth_cache

def get_metric_from_df(df_metrics, var, lead_idx, metric):
    row = df_metrics[
        (df_metrics["variable"] == var)
        & (df_metrics["lead_idx"] == lead_idx)
    ]

    if row.empty:
        return None

    return float(row[metric].iloc[0])



# —— Verification plotting helpers ——————————————————————



def robust_limits(fields, lower=1, upper=99):
    vals = np.concatenate([
        f[np.isfinite(f)].ravel()
        for f in fields
        if np.isfinite(f).any()
    ])

    if vals.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.nanpercentile(vals, [lower, upper])

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    return float(vmin), float(vmax)


def symmetric_robust_limit(fields, percentile=99):
    vals = np.concatenate([
        f[np.isfinite(f)].ravel()
        for f in fields
        if np.isfinite(f).any()
    ])

    if vals.size == 0:
        return 1.0

    vmax = np.nanpercentile(np.abs(vals), percentile)

    if vmax == 0 or not np.isfinite(vmax):
        vmax = np.nanmax(np.abs(vals))

    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    return float(vmax)

# ── 13. Forecast / truth / error panel function ───────────────────────────────




def format_error_title(df_metrics, var, lead_idx, unit):
    rmse = get_metric_from_df(df_metrics, var, lead_idx, "rmse")
    bias = get_metric_from_df(df_metrics, var, lead_idx, "bias")

    if rmse is None or bias is None:
        return "Forecast − proxy truth\nqualitative only"

    return f"Forecast − truth\nRMSE={rmse:.3g} {unit}, Bias={bias:.3g} {unit}"

def field_title(kind, var, hour):
    if var == "refc" and kind == "truth":
        return f"ERA5 tcrw-derived proxy\n{var}, lead {hour}h"

    if kind == "truth":
        return f"ERA5 truth\n{var}, lead {hour}h"

    if kind == "forecast":
        return f"StormCast forecast\n{var}, lead {hour}h"

    raise ValueError(f"Unknown kind: {kind}")

def plot_forecast_truth_error_panel(
    var,
    leads_to_plot,
    ds,
    truth_cache,
    df_metrics,
    save_dir,
    field_percentiles=(1, 99),
    error_percentile=99,
    dpi=160,
):
    if var not in ds.data_vars:
        print(f"Skipping {var}: not found in ds_out")
        return

    unit = infer_unit(var)

    forecast_fields = {}
    truth_fields = {}
    error_fields = {}

    for lead_idx in leads_to_plot:
        forecast = get_forecast_field(ds, var, lead_idx)
        truth = get_truth_field(truth_cache[lead_idx], var)

        forecast_fields[lead_idx] = forecast
        truth_fields[lead_idx] = truth
        error_fields[lead_idx] = forecast - truth

    vmin, vmax = robust_limits(
        list(forecast_fields.values()) + list(truth_fields.values()),
        lower=field_percentiles[0],
        upper=field_percentiles[1],
    )

    err_vmax = symmetric_robust_limit(
        list(error_fields.values()),
        percentile=error_percentile,
    )

    fig, axes = plt.subplots(
        len(leads_to_plot),
        3,
        figsize=(15, 4.2 * len(leads_to_plot)),
        squeeze=False,
    )

    for row, lead_idx in enumerate(leads_to_plot):
        hour = lead_hour(ds, lead_idx)

        panels = [
            {
                "field": truth_fields[lead_idx],
                "title": field_title("truth", var, hour),
                "cmap": None,
                "vmin": vmin,
                "vmax": vmax,
                "label": unit,
            },
            {
                "field": forecast_fields[lead_idx],
                "title": field_title("forecast", var, hour),
                "cmap": None,
                "vmin": vmin,
                "vmax": vmax,
                "label": unit,
            },
            {
                "field": error_fields[lead_idx],
                "title": format_error_title(df_metrics, var, lead_idx, unit),
                "cmap": "RdBu_r",
                "vmin": -err_vmax,
                "vmax": err_vmax,
                "label": f"error [{unit}]",
            },
        ]

        for col, panel in enumerate(panels):
            im = axes[row, col].imshow(
                panel["field"],
                origin="upper",
                cmap=panel["cmap"],
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )

            axes[row, col].set_title(panel["title"])
            axes[row, col].set_xlabel("hrrr_x")
            axes[row, col].set_ylabel("hrrr_y")
            plt.colorbar(im, ax=axes[row, col], label=panel["label"])

    fig.suptitle(
        f"{var}: ERA5 truth vs StormCast forecast vs error",
        fontsize=16,
        y=1.01,
    )

    plt.tight_layout()

    out_path = save_dir / f"{var}_forecast_truth_error_panel.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")

    del forecast_fields, truth_fields, error_fields
    gc.collect()



def plot_error_curves(
    df_metrics,
    variables=None,
    metrics=("rmse", "mae", "bias"),
    save_dir=None,
    include_lead0=True,
    dpi=160,
):
    """
    Plot verification error curves as a function of forecast lead time.

    One figure is produced per variable.
    """

    if variables is None:
        variables = list(df_metrics["variable"].unique())

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    for var in variables:
        sub = df_metrics[df_metrics["variable"] == var].copy()

        if not include_lead0:
            sub = sub[sub["lead_hour"] > 0]

        if sub.empty:
            print(f"Skipping {var}: no metrics found.")
            continue

        unit = sub["unit"].iloc[0]

        plt.figure(figsize=(6, 4))

        for metric in metrics:
            if metric not in sub.columns:
                continue

            plt.plot(
                sub["lead_hour"],
                sub[metric],
                marker="o",
                label=metric.upper(),
            )

        plt.axhline(0, linewidth=1)
        plt.xlabel("Forecast lead time [h]")
        plt.ylabel(f"Error [{unit}]")
        plt.title(f"{var}: error vs forecast lead time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"{var}_error_curve.png"
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out_path}")

        plt.show()