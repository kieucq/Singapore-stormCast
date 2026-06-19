import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

import gc
from pathlib import Path



# CONSTANTS AND VARIABLE MAPPINGS

REFC_RAIN_THRESHOLD_MMHR = 0.1
REFC_MIN_DBZ = -10.0
REFC_MAX_DBZ = 75.0

# SingV3 variable name → StormCast variable name (surface/single-level)
SINGV_TO_SC_SURFACE = {
    "tas": "t2m",
    "uas": "u10m",
    "vas": "v10m",
    "psl": "mslp",
}

# SingV3 variable name → StormCast variable prefix (pressure-level)
SINGV_TO_SC_PRESSURE = {
    "ua":  "u",
    "va":  "v",
    "ta":  "t",
    "hus": "q",
    "zg":  "Z",
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


# GRID AND XR.DATAARRAY HELPERS

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

def make_grid_spec(ds_surface, hrrr_y, hrrr_x):
    src_lats = ds_surface["lat"].values
    src_lons = ds_surface["lon"].values

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

def make_empty_field_array(variables, hrrr_y, hrrr_x):
    """
    Create an empty StormCast-style field array without a time dimension.

    Shape:
        (variable, hrrr_y, hrrr_x)
    """
    data = xr.DataArray(
        np.zeros(
            (len(variables), len(hrrr_y), len(hrrr_x)),
            dtype=np.float32,
        ),
        dims=("variable", "hrrr_y", "hrrr_x"),
        coords={
            "variable": variables,
            "hrrr_y": hrrr_y,
            "hrrr_x": hrrr_x,
        },
        name="singv3_stormcast_fields",
    )

    return data

def add_time_dimension(data, time_value):
    """
    Add a time dimension to a 3D field array so Earth2Studio/StormCast can use it.

    Input:
        (variable, hrrr_y, hrrr_x)

    Output:
        (time, variable, hrrr_y, hrrr_x)
    """
    data_with_time = data.expand_dims(
        time=[np.datetime64(time_value, "ns")]
    )

    return data_with_time.transpose(
        "time", "variable", "hrrr_y", "hrrr_x"
    ).astype(np.float32)



# BASIC NUMERICAL HELPERS

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

def fill_nan_nearest(field: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a 2D field using nearest valid neighbour.
    Useful for tiny edge missing strips in uas/vas and for final cleanup.
    """
    arr = np.asarray(field, dtype=np.float32)

    missing = ~np.isfinite(arr) # ~ is numpy's logical NOT
    # missing is a boolean array with True for non-finite values and False otherwise 

    if not missing.any():
        return arr

    if missing.all():
        raise ValueError("Cannot fill field: all values are NaN.")

    indices = distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )

    filled = arr[tuple(indices)]
    return filled.astype(np.float32)

def _insert_field(data, var_name, field):
    """
    Put a 2D field into a 3D field array:
        (variable, hrrr_y, hrrr_x)
    """
    expected_dims = ("variable", "hrrr_y", "hrrr_x")

    if data.dims != expected_dims:
        raise ValueError(f"Expected dims {expected_dims}, got {data.dims}")

    if var_name not in data["variable"].values:
        raise ValueError(f"{var_name} is not in field array variables.")

    field = np.asarray(field, dtype=np.float32)

    expected_shape = (data.sizes["hrrr_y"], data.sizes["hrrr_x"])

    if field.shape != expected_shape:
        raise ValueError(
            f"{var_name} has shape {field.shape}, expected {expected_shape}"
        )

    data.loc[dict(variable=var_name)] = field



# INTERPOLATION HELPERS

def regrid(field, src_lats, src_lons, target_lats, target_lons):
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


def regrid_all_p_levels(
    da_pressure_var,
    src_lats,
    src_lons,
    target_lats,
    target_lons,
):
    """
    Regrid all pressure levels of a SingV3 pressure-level variable.

    Input dimensions are usually:
        (plev, lat, lon)

    Output:
        NumPy array with shape:
        (n_pressure_levels, n_target_lats, n_target_lons)
    """
    regridded_levels = []

    n_pressure_levels = len(da_pressure_var["plev"])

    for pressure_index in range(n_pressure_levels):
        # Select one 2D pressure-level slice: (lat, lon)
        field_2d = da_pressure_var.isel(plev=pressure_index).values.astype(np.float32)

        # Regrid that 2D slice to the target grid: (target_lat, target_lon)
        field_2d_regridded = regrid(
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
    return np.stack(regridded_levels, axis=0).astype(np.float32, copy=False)

def vinterp(field_3d, src_pressure_pa, target_pressure_pa):
    """
    Vertically interpolate a 3D pressure-level field to a target 2D pressure field.

    First-pass SingV3 pipeline:
    - pressure-level zero masks are kept as actual zero values
    - target pressures outside the available pressure range are clamped
      to the nearest available pressure level
    """

    sort_order = np.argsort(src_pressure_pa)

    pressure_sorted = src_pressure_pa[sort_order]
    field_sorted = field_3d[sort_order, :, :]

    target = np.clip(
        target_pressure_pa,
        pressure_sorted[0],
        pressure_sorted[-1],
    )

    upper_index = np.searchsorted(pressure_sorted, target)
    upper_index = np.clip(upper_index, 1, len(pressure_sorted) - 1)
    lower_index = upper_index - 1

    y_indices = np.arange(target.shape[0])[:, None]
    x_indices = np.arange(target.shape[1])[None, :]

    pressure_lower = pressure_sorted[lower_index]
    pressure_upper = pressure_sorted[upper_index]

    field_lower = field_sorted[lower_index, y_indices, x_indices]
    field_upper = field_sorted[upper_index, y_indices, x_indices]

    weight = (target - pressure_lower) / (pressure_upper - pressure_lower)

    interpolated = field_lower + weight * (field_upper - field_lower)

    return interpolated.astype(np.float32)



# SINGV3 DIAGNOSTICS AND VALIDATION

def print_tgrid_zero_mask_diagnostics(ds_pressure):
    """Print zero-mask diagnostics for the T-grid pressure variables ta, hus, zg."""
    ta_zero  = ds_pressure["ta"]  == 0
    hus_zero = ds_pressure["hus"] == 0
    zg_zero  = ds_pressure["zg"]  == 0

    all_zero          = ta_zero & hus_zero & zg_zero
    any_zero          = ta_zero | hus_zero | zg_zero
    inconsistent_zero = any_zero & ~all_zero

    total                   = int(all_zero.size)
    inconsistent_zero_count = int(inconsistent_zero.sum().item())

    print("\nT-grid zero-mask diagnostics")
    print("----------------------------")
    print(f"total points:              {total}")
    print(f"ta zero count:             {int(ta_zero.sum().item())}")
    print(f"hus zero count:            {int(hus_zero.sum().item())}")
    print(f"zg zero count:             {int(zg_zero.sum().item())}")
    print(f"all three zero count:      {int(all_zero.sum().item())}")
    print(f"any zero count:            {int(any_zero.sum().item())}")
    print(f"inconsistent zero count:   {inconsistent_zero_count}")
    print(f"inconsistent zero pct:     {100 * inconsistent_zero_count / total:.6f}%")

    if inconsistent_zero_count > 0:
        print("\nBreakdown of inconsistent zeroes")
        print("--------------------------------")
        print(f"ta only:                   {int((ta_zero  & ~hus_zero & ~zg_zero).sum().item())}")
        print(f"hus only:                  {int((~ta_zero & hus_zero  & ~zg_zero).sum().item())}")
        print(f"zg only:                   {int((~ta_zero & ~hus_zero & zg_zero ).sum().item())}")
        print(f"ta+hus only:               {int((ta_zero  & hus_zero  & ~zg_zero).sum().item())}")
        print(f"ta+zg only:                {int((ta_zero  & ~hus_zero & zg_zero ).sum().item())}")
        print(f"hus+zg only:               {int((~ta_zero & hus_zero  & zg_zero ).sum().item())}")


def validate_singv_dataset(ds, verbose=True):
    """
    Minimal validation for one collected SingV3 NetCDF file.

    Checks:
    - required variables exist
    - required coordinates exist
    - which variables contain NaNs
    - ta/hus/zg shared zero-mask diagnostics
    """

    required_vars = [
        "tas", "uas", "vas", "psl", "pr",
        "ua", "va", "ta", "hus", "zg",
    ]

    required_coords = ["lat", "lon", "plev"]

    missing_vars = [v for v in required_vars if v not in ds]
    missing_coords = [c for c in required_coords if c not in ds.coords]

    if missing_vars:
        raise ValueError(f"Missing SingV3 variables: {missing_vars}")

    if missing_coords:
        raise ValueError(f"Missing SingV3 coordinates: {missing_coords}")

    if not verbose:
        return True

    print("\nSingV3 dataset validation")
    print("------------------------")

    if "valid_time" in ds:
        print(f"valid_time: {ds['valid_time'].values}")

    print(f"lat shape:  {ds['lat'].shape}")
    print(f"lon shape:  {ds['lon'].shape}")
    print(f"plev:       {ds['plev'].values}")

    print("\nNaN diagnostics")
    print("---------------")

    any_nan = False

    for var in required_vars:
        arr = ds[var].values
        nan_count = int(np.isnan(arr).sum())
        total = arr.size
        nan_pct = 100.0 * nan_count / total

        if nan_count > 0:
            any_nan = True
            print(
                f"{var:4s} contains NaNs: "
                f"{nan_count}/{total} = {nan_pct:.6f}%"
            )

    if not any_nan:
        print("No NaNs found in required variables.")

    print_tgrid_zero_mask_diagnostics(ds)

    return True


def validate_field_array(fields, variables, ny, nx):
    """
    Validate converted SingV3 fields before adding the StormCast time dimension.

    Expected shape:
        (variable, hrrr_y, hrrr_x)
    """
    ZERO_ALLOWED = {"refc"}

    expected_dims = ("variable", "hrrr_y", "hrrr_x")
    if fields.dims != expected_dims:
        raise ValueError(f"Expected dims {expected_dims}, got {fields.dims}")

    expected_shape = (len(variables), ny, nx)
    if fields.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {fields.shape}")

    if np.isnan(fields.values).any():
        nan_frac = np.isnan(fields.values).mean()
        raise ValueError(f"Converted fields contain NaNs. NaN fraction={nan_frac:.6f}")

    is_filled = (fields != 0).any(dim=("hrrr_y", "hrrr_x"))

    filled         = [v for v in variables if is_filled.sel(variable=v).item()]
    zero_by_design = [v for v in variables if not is_filled.sel(variable=v).item() and v in ZERO_ALLOWED]
    missing        = [v for v in variables if not is_filled.sel(variable=v).item() and v not in ZERO_ALLOWED]

    print(
        f"Filled {len(filled)}/{len(variables)} variables "
        f"({zero_by_design} intentionally zero, {missing} unexpectedly zero)"
    )

    if missing:
        raise ValueError(f"Some variables were not filled: {missing}")

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



# SINGV3 -> STORMCAST CONVERSION


def fill_surface_fields(data, ds_surface, grid, verbose=False):
    """
    Interpolates surface fields t2m, u10m, v10m and mslp
    and inserts them into the xr.DataArray
    """

    available_vars = set(data["variable"].values)

    for singv_name, sc_name in SINGV_TO_SC_SURFACE.items():
        if sc_name not in available_vars:
            continue

        raw = ds_surface[singv_name].values.astype(np.float32)
        if singv_name in ["uas", "vas"]:
            raw = fill_nan_nearest(raw)

        field = regrid(
            raw,
            grid.src_lats,
            grid.src_lons,
            grid.target_lats,
            grid.target_lons,
        )

        field = fill_nan_nearest(field)

        assert_no_nan(sc_name, field)
        _insert_field(data, sc_name, field)

        if verbose:
            print(
                f"  {singv_name:4s} → {sc_name:5s} | "
                f"min={field.min():.2f}  max={field.max():.2f}"
            )


def get_surface_pressure(ds_surface, grid, verbose=False):
    """
    Return approximate surface pressure on StormCast grid.

    First-pass approximation:
        surface pressure ≈ psl

    This is not physically exact because psl is mean sea-level pressure,
    but it is acceptable for a first SingV3 pipeline test.
    """

    psl_raw = ds_surface["psl"].values.astype(np.float32)

    sp = regrid(
        psl_raw,
        grid.src_lats,
        grid.src_lons,
        grid.target_lats,
        grid.target_lons,
    )

    sp = fill_nan_nearest(sp)
    assert_no_nan("sp", sp)

    if verbose:
        print(
            f"\n  approx sp=psl | min={sp.min():.1f}  "
            f"mean={sp.mean():.1f}  max={sp.max():.1f} Pa"
        )

    return sp.astype(np.float32)


def fill_hybrid_fields(data, ds_pressure, sp, grid, verbose=False):
    """
    Fill hybrid-level variables:
    u#hl, v#hl, t#hl, q#hl, Z#hl, p#hl.
    """

    singv_p_pa = ds_pressure["plev"].values.astype(np.float32)
    available_vars = set(data["variable"].values)

    if verbose:
        print("\nRegridding SingV3 pressure-level variables...")

    pressure_3d = {}

    # horizontally regrid all singv3 pressure-level variables
    for singv_name in SINGV_TO_SC_PRESSURE:
        da = ds_pressure[singv_name]

        pressure_3d[singv_name] = regrid_all_p_levels(
            da,
            grid.src_lats,
            grid.src_lons,
            grid.target_lats,
            grid.target_lons,
        )

        if verbose:
            arr = pressure_3d[singv_name]
            print(
                f"  {singv_name:3s} regridded | "
                f"min={np.nanmin(arr):.3g}  "
                f"mean={np.nanmean(arr):.3g}  "
                f"max={np.nanmax(arr):.3g}"
            )

    # for each hybrid level, fill u, v, t, q and Z
    for level, sigma in HRRR_SIGMA.items():
        p_target = (sigma * sp).astype(np.float32)

        p_name = f"p{level}hl"
        if p_name in available_vars:
            _insert_field(data, p_name, p_target)

        for singv_name, sc_prefix in SINGV_TO_SC_PRESSURE.items():
            sc_name = f"{sc_prefix}{level}hl"

            if sc_name not in available_vars:
                continue

            field = vinterp(
                pressure_3d[singv_name],
                singv_p_pa,
                p_target,
            )

            if singv_name == "hus":
                field = np.maximum(field, 0.0)

            assert_no_nan(sc_name, field)
            _insert_field(data, sc_name, field.astype(np.float32))

            if verbose:
                print(
                    f"  {singv_name:3s} → {sc_name:5s} | "
                    f"min={field.min():.3g}  mean={field.mean():.3g}  max={field.max():.3g}"
                )

def fill_refc(data, ds_surface, grid, verbose=False):
    """Fill refc using SingV3 pr proxy."""

    available_vars = set(data["variable"].values)

    if "refc" not in available_vars:
        return

    if "pr" not in ds_surface:
        raise ValueError("Cannot derive refc: SingV3 variable 'pr' not found.")

    pr_raw = ds_surface["pr"].values.astype(np.float32)

    pr = regrid(
        pr_raw,
        grid.src_lats,
        grid.src_lons,
        grid.target_lats,
        grid.target_lons,
    )

    assert_no_nan("pr", pr)

    refc = singv_pr_to_refc(
        pr,
        rain_threshold=REFC_RAIN_THRESHOLD_MMHR,
        min_dbz=REFC_MIN_DBZ,
        max_dbz=REFC_MAX_DBZ,
    )

    assert_no_nan("refc", refc)
    _insert_field(data, "refc", refc)

    if verbose:
        print(
            f"  pr → refc | "
            f"pr min={pr.min():.3g} mean={pr.mean():.3g} max={pr.max():.3g} | "
            f"refc min={refc.min():.3g} mean={refc.mean():.3g} max={refc.max():.3g}"
        )

def singv_pr_to_refc(
    pr,
    rain_threshold=REFC_RAIN_THRESHOLD_MMHR,
    min_dbz=REFC_MIN_DBZ,
    max_dbz=REFC_MAX_DBZ,
):
    """
    Approximate composite reflectivity from SingV3 precipitation rate.

    SingV3 pr units:
        kg m-2 s-1

    Since 1 kg m-2 is 1 mm water:
        pr * 3600 = mm/hr
    """
    pr = np.maximum(np.asarray(pr, dtype=np.float32), 0.0)

    R_mmhr = pr * 3600.0
    active = R_mmhr > rain_threshold

    Z_linear = 200.0 * np.maximum(R_mmhr, 1e-10) ** 1.6
    refc = 10.0 * np.log10(Z_linear)

    refc = np.where(active, refc, min_dbz)

    return np.clip(refc, min_dbz, max_dbz).astype(np.float32)




# EARTH2STUDIO ADAPTER

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




# def validate_forecast_output(ds_out, nsteps, verbose=False):
#     if verbose: 
#         print(ds_out)
#         print("lead_time:", ds_out.lead_time.values)
#         print("n_leads:", ds_out.sizes["lead_time"])

#     expected = nsteps + 1
#     actual = ds_out.sizes["lead_time"]

#     if actual != expected:
#         raise ValueError(f"Expected {expected} lead times, got {actual}")

#     return True

# # —— Verification / diagnostics helpers ————————————————–

# def lead_hour(ds, lead_idx):
#     return int(ds.lead_time.values[lead_idx] / np.timedelta64(1, "h"))


# def valid_time_for_lead(start_time, ds, lead_idx):
#     return pd.Timestamp(start_time) + pd.to_timedelta(ds.lead_time.values[lead_idx])


# def get_forecast_field(ds, var, lead_idx):
#     return (
#         ds[var]
#         .isel(time=0, lead_time=lead_idx)
#         .values
#         .astype(np.float32)
#     )


# def get_truth_field(truth, var):
#     return truth.sel(variable=var).values.astype(np.float32)


# def compute_metrics(forecast, truth):
#     error = forecast - truth

#     return {
#         "rmse": float(np.sqrt(np.nanmean(error ** 2))),
#         "bias": float(np.nanmean(error)),
#         "mae": float(np.nanmean(np.abs(error))),
#     }

# def compute_verification_metrics(ds_out, start_time, variables, truth_builder):
#     records = []

#     for lead_idx in range(ds_out.sizes["lead_time"]):
#         valid_time = valid_time_for_lead(start_time, ds_out, lead_idx)
#         hour = lead_hour(ds_out, lead_idx)

#         print(f"Verifying lead {hour}h, valid_time={valid_time}")

#         truth = truth_builder(valid_time.to_pydatetime(), variables=variables)

#         for var in variables:
#             forecast_field = get_forecast_field(ds_out, var, lead_idx)
#             truth_field = get_truth_field(truth, var)

#             metrics = compute_metrics(forecast_field, truth_field)

#             records.append({
#                 "lead_idx": lead_idx,
#                 "lead_hour": hour,
#                 "valid_time": valid_time,
#                 "variable": var,
#                 **metrics,
#                 "unit": infer_unit(var),
#             })

#             del forecast_field, truth_field
#         del truth
#     return pd.DataFrame(records)

# def build_truth_cache(ds_out, start_time, variables, leads, truth_builder):
#     truth_cache = {}

#     for lead_idx in leads:
#         valid_time = valid_time_for_lead(start_time, ds_out, lead_idx)
#         hour = lead_hour(ds_out, lead_idx)

#         print(f"Building panel truth for lead {hour}h, valid_time={valid_time}")

#         truth_cache[lead_idx] = truth_builder(
#             valid_time.to_pydatetime(),
#             variables=variables,
#         )

#     return truth_cache

# def get_metric_from_df(df_metrics, var, lead_idx, metric):
#     row = df_metrics[
#         (df_metrics["variable"] == var)
#         & (df_metrics["lead_idx"] == lead_idx)
#     ]

#     if row.empty:
#         return None

#     return float(row[metric].iloc[0])



# # —— Verification plotting helpers ——————————————————————



# def robust_limits(fields, lower=1, upper=99):
#     vals = np.concatenate([
#         f[np.isfinite(f)].ravel()
#         for f in fields
#         if np.isfinite(f).any()
#     ])

#     if vals.size == 0:
#         return 0.0, 1.0

#     vmin, vmax = np.nanpercentile(vals, [lower, upper])

#     if np.isclose(vmin, vmax):
#         vmin = float(np.nanmin(vals))
#         vmax = float(np.nanmax(vals))

#     if np.isclose(vmin, vmax):
#         vmin -= 1.0
#         vmax += 1.0

#     return float(vmin), float(vmax)


# def symmetric_robust_limit(fields, percentile=99):
#     vals = np.concatenate([
#         f[np.isfinite(f)].ravel()
#         for f in fields
#         if np.isfinite(f).any()
#     ])

#     if vals.size == 0:
#         return 1.0

#     vmax = np.nanpercentile(np.abs(vals), percentile)

#     if vmax == 0 or not np.isfinite(vmax):
#         vmax = np.nanmax(np.abs(vals))

#     if vmax == 0 or not np.isfinite(vmax):
#         vmax = 1.0

#     return float(vmax)

# # ── 13. Forecast / truth / error panel function ───────────────────────────────




# def format_error_title(df_metrics, var, lead_idx, unit):
#     rmse = get_metric_from_df(df_metrics, var, lead_idx, "rmse")
#     bias = get_metric_from_df(df_metrics, var, lead_idx, "bias")

#     if rmse is None or bias is None:
#         return "Forecast − proxy truth\nqualitative only"

#     return f"Forecast − truth\nRMSE={rmse:.3g} {unit}, Bias={bias:.3g} {unit}"

# def field_title(kind, var, hour):
#     if var == "refc" and kind == "truth":
#         return f"ERA5 tcrw-derived proxy\n{var}, lead {hour}h"

#     if kind == "truth":
#         return f"ERA5 truth\n{var}, lead {hour}h"

#     if kind == "forecast":
#         return f"StormCast forecast\n{var}, lead {hour}h"

#     raise ValueError(f"Unknown kind: {kind}")

# def plot_forecast_truth_error_panel(
#     var,
#     leads_to_plot,
#     ds,
#     truth_cache,
#     df_metrics,
#     save_dir,
#     field_percentiles=(1, 99),
#     error_percentile=99,
#     dpi=160,
# ):
#     if var not in ds.data_vars:
#         print(f"Skipping {var}: not found in ds_out")
#         return

#     unit = infer_unit(var)

#     forecast_fields = {}
#     truth_fields = {}
#     error_fields = {}

#     for lead_idx in leads_to_plot:
#         forecast = get_forecast_field(ds, var, lead_idx)
#         truth = get_truth_field(truth_cache[lead_idx], var)

#         forecast_fields[lead_idx] = forecast
#         truth_fields[lead_idx] = truth
#         error_fields[lead_idx] = forecast - truth

#     vmin, vmax = robust_limits(
#         list(forecast_fields.values()) + list(truth_fields.values()),
#         lower=field_percentiles[0],
#         upper=field_percentiles[1],
#     )

#     err_vmax = symmetric_robust_limit(
#         list(error_fields.values()),
#         percentile=error_percentile,
#     )

#     fig, axes = plt.subplots(
#         len(leads_to_plot),
#         3,
#         figsize=(15, 4.2 * len(leads_to_plot)),
#         squeeze=False,
#     )

#     for row, lead_idx in enumerate(leads_to_plot):
#         hour = lead_hour(ds, lead_idx)

#         panels = [
#             {
#                 "field": truth_fields[lead_idx],
#                 "title": field_title("truth", var, hour),
#                 "cmap": None,
#                 "vmin": vmin,
#                 "vmax": vmax,
#                 "label": unit,
#             },
#             {
#                 "field": forecast_fields[lead_idx],
#                 "title": field_title("forecast", var, hour),
#                 "cmap": None,
#                 "vmin": vmin,
#                 "vmax": vmax,
#                 "label": unit,
#             },
#             {
#                 "field": error_fields[lead_idx],
#                 "title": format_error_title(df_metrics, var, lead_idx, unit),
#                 "cmap": "RdBu_r",
#                 "vmin": -err_vmax,
#                 "vmax": err_vmax,
#                 "label": f"error [{unit}]",
#             },
#         ]

#         for col, panel in enumerate(panels):
#             im = axes[row, col].imshow(
#                 panel["field"],
#                 origin="upper",
#                 cmap=panel["cmap"],
#                 vmin=panel["vmin"],
#                 vmax=panel["vmax"],
#             )

#             axes[row, col].set_title(panel["title"])
#             axes[row, col].set_xlabel("hrrr_x")
#             axes[row, col].set_ylabel("hrrr_y")
#             plt.colorbar(im, ax=axes[row, col], label=panel["label"])

#     fig.suptitle(
#         f"{var}: ERA5 truth vs StormCast forecast vs error",
#         fontsize=16,
#         y=1.01,
#     )

#     plt.tight_layout()

#     out_path = save_dir / f"{var}_forecast_truth_error_panel.png"
#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.show()

#     print(f"Saved: {out_path}")

#     del forecast_fields, truth_fields, error_fields
#     gc.collect()



# def plot_error_curves(
#     df_metrics,
#     variables=None,
#     metrics=("rmse", "mae", "bias"),
#     save_dir=None,
#     include_lead0=True,
#     dpi=160,
# ):
#     """
#     Plot verification error curves as a function of forecast lead time.

#     One figure is produced per variable.
#     """

#     if variables is None:
#         variables = list(df_metrics["variable"].unique())

#     if save_dir is not None:
#         save_dir = Path(save_dir)
#         save_dir.mkdir(exist_ok=True)

#     for var in variables:
#         sub = df_metrics[df_metrics["variable"] == var].copy()

#         if not include_lead0:
#             sub = sub[sub["lead_hour"] > 0]

#         if sub.empty:
#             print(f"Skipping {var}: no metrics found.")
#             continue

#         unit = sub["unit"].iloc[0]

#         plt.figure(figsize=(6, 4))

#         for metric in metrics:
#             if metric not in sub.columns:
#                 continue

#             plt.plot(
#                 sub["lead_hour"],
#                 sub[metric],
#                 marker="o",
#                 label=metric.upper(),
#             )

#         plt.axhline(0, linewidth=1)
#         plt.xlabel("Forecast lead time [h]")
#         plt.ylabel(f"Error [{unit}]")
#         plt.title(f"{var}: error vs forecast lead time")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         if save_dir is not None:
#             out_path = save_dir / f"{var}_error_curve.png"
#             plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#             print(f"Saved: {out_path}")

#         plt.show()
