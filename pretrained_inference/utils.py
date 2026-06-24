"""
Utilities for converting collated SingV3 data into inputs for
pretrained StormCast inference.
"""

import numpy as np
import xarray as xr

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

# CONSTANTS AND VARIABLE MAPPINGS

REFC_RAIN_THRESHOLD_MMHR = 0.1
REFC_MIN_DBZ = -10.
REFC_MAX_DBZ = 75.

# STORMCAST SETTINGS

FULL_HYBRID_LEVELS = [
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 13, 15, 20, 25, 30,
]

PRESSURE_HYBRID_LEVELS = [
    1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 13, 15, 20,
]

SURFACE_VARIABLES = [
    "t2m",
    "u10m",
    "v10m",
    "mslp",
    "refc",
]

PROFILE_VARIABLES = [
    "u",
    "v",
    "Z",
    "t",
    "q",
]

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
    def __init__(self, src_lats, src_lons, target_lats, target_lons):
        self.src_lats = src_lats
        self.src_lons = src_lons
        self.target_lats = target_lats
        self.target_lons = target_lons

def make_grid_spec(ds_surface, hrrr_y, hrrr_x):
    src_lats = ds_surface["lat"].values
    src_lons = ds_surface["lon"].values

    if not np.all(np.diff(src_lats) > 0):
        raise ValueError(
            "SingV latitude coordinates must be strictly increasing."
        )

    if not np.all(np.diff(src_lons) > 0):
        raise ValueError(
            "SingV longitude coordinates must be strictly increasing."
        )

    target_lats = np.linspace(src_lats.min(), src_lats.max(), len(hrrr_y))
    target_lons = np.linspace(src_lons.min(), src_lons.max(), len(hrrr_x))

    return GridSpec(
        src_lats=src_lats,
        src_lons=src_lons,
        target_lats=target_lats,
        target_lons=target_lons,
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

def assert_finite(name, field):
    """Raise an error if a processed field contains NaN or infinite values."""
    field = np.asarray(field)
    nonfinite = ~np.isfinite(field)

    if nonfinite.any():
        fraction = float(nonfinite.mean())
        raise ValueError(
            f"{name} contains non-finite values after processing. "
            f"Non-finite fraction={fraction:.6f}"
        )


def fill_nan_nearest(field: np.ndarray) -> np.ndarray:
    """Fill non-finite values in a 2D field using the nearest finite neighbour."""
    arr = np.asarray(field, dtype=np.float32)

    missing = ~np.isfinite(arr) # ~ is numpy's logical NOT
    # missing is a boolean array with True for non-finite values and False otherwise 

    if not missing.any():
        return arr

    if missing.all():
        raise ValueError("Cannot fill field: all values are non-finite.")

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
    Bilinearly interpolate a 2D field from one regular latitude/longitude
    grid onto another regular latitude/longitude grid.

    Both source coordinate arrays must be one-dimensional. This function
    does not perform a map-projection transformation.
    """

    field = np.asarray(field, dtype=np.float32)

    src_lats = np.asarray(src_lats)
    src_lons = np.asarray(src_lons)

    expected_shape = (len(src_lats), len(src_lons))

    if field.shape != expected_shape:
        raise ValueError(
            f"Field shape {field.shape} does not match source grid "
            f"shape {expected_shape}."
        )

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
    Vertically interpolate a three-dimensional pressure-level field onto
    a two-dimensional target-pressure field.

    Subterranean zero masks are expected to have been sanitized before this
    function is called. Target pressures outside the available pressure range
    are clamped to the nearest available pressure level.
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
    Minimal validation for one collated SingV3 NetCDF file.

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

    expected_dims = ("variable", "hrrr_y", "hrrr_x")
    if fields.dims != expected_dims:
        raise ValueError(f"Expected dims {expected_dims}, got {fields.dims}")

    expected_shape = (len(variables), ny, nx)
    if fields.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {fields.shape}")

    nonfinite = ~np.isfinite(fields.values)

    if nonfinite.any():
        fraction = float(nonfinite.mean())
        raise ValueError(
            f"Converted fields contain non-finite values. "
            f"Non-finite fraction={fraction:.6f}"
        )

    is_filled = (fields != 0).any(dim=("hrrr_y", "hrrr_x"))
    missing        = [v for v in variables if not is_filled.sel(variable=v).item()]

    print(
        f"Filled {len(variables) - len(missing)}/{len(variables)} variables "
        f"({missing} unexpectedly zero)"
    )

    if missing:
        raise ValueError(
            f"Some variables were not filled: {missing}"
        )

    return True

def validate_stormcast_input_array(data, variables, ny, nx):
    expected_dims = ("time", "variable", "hrrr_y", "hrrr_x")

    if data.dims != expected_dims:
        raise ValueError(f"Expected dims {expected_dims}, got {data.dims}")

    expected_shape = (1, len(variables), ny, nx)

    if data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
    
    nonfinite = ~np.isfinite(data.values)

    if nonfinite.any():
        fraction = float(nonfinite.mean())
        raise ValueError(
            f"StormCast input contains non-finite values. "
            f"Non-finite fraction={fraction:.6f}"
        )

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

        assert_finite(sc_name, field)
        _insert_field(data, sc_name, field)

        if verbose:
            print(
                f"  {singv_name:4s} → {sc_name:5s} | "
                f"min={field.min():.2f}  max={field.max():.2f}"
            )


def get_surface_pressure(ds_surface, grid, verbose=False):
    """
    Return an approximate surface-pressure field on the target grid.

    Mean sea-level pressure is used as a proxy because true SingV surface
    pressure is unavailable. This approximation is retained only for testing
    the pretrained StormCast compatibility pipeline and is not physically
    consistent over elevated terrain.
    """

    psl_raw = ds_surface["psl"].values.astype(np.float32)

    sp = regrid(
        psl_raw,
        grid.src_lats,
        grid.src_lons,
        grid.target_lats,
        grid.target_lons,
    )

    assert_finite("sp", sp)

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

            assert_finite(sc_name, field)
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

    assert_finite("pr", pr)

    refc = singv_pr_to_refc(
        pr,
        rain_threshold=REFC_RAIN_THRESHOLD_MMHR,
        min_dbz=REFC_MIN_DBZ,
        max_dbz=REFC_MAX_DBZ,
    )

    assert_finite("refc", refc)
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



def sc_to_ncview(data):
    """
    Convert the flat 99-channel StormCast array into an
    ncview-friendly Dataset with 11 named variables.
    """
    ds = xr.Dataset(attrs=dict(data.attrs))

    # Surface variables
    for var in SURFACE_VARIABLES:
        ds[var] = data.sel(
            variable=var,
            drop=True,
        )

        ds[var].attrs["units"] = infer_unit(var)

    # u, v, Z, t and q on 16 hybrid levels
    for prefix in PROFILE_VARIABLES:
        names = [
            f"{prefix}{level}hl"
            for level in FULL_HYBRID_LEVELS
        ]

        ds[prefix] = (
            data
            .sel(variable=names)
            .rename(variable="hybrid_level")
            .assign_coords(
                hybrid_level=FULL_HYBRID_LEVELS
            )
        )

        ds[prefix].attrs["units"] = infer_unit(names[0])

    # p on only 14 hybrid levels
    pressure_names = [
        f"p{level}hl"
        for level in PRESSURE_HYBRID_LEVELS
    ]

    ds["p"] = (
        data
        .sel(variable=pressure_names)
        .rename(variable="p_hybrid_level")
        .assign_coords(
            p_hybrid_level=PRESSURE_HYBRID_LEVELS
        )
    )

    ds["p"].attrs["units"] = "Pa"

    return ds


def ncview_to_sc(ds, variables):
    """
    Convert the ncview-friendly Dataset back into the flat
    StormCast input array.
    """
    fields = {}

    # Surface variables
    for var in SURFACE_VARIABLES:
        fields[var] = ds[var]

    # u, v, Z, t and q
    for prefix in PROFILE_VARIABLES:
        for level in FULL_HYBRID_LEVELS:
            name = f"{prefix}{level}hl"

            fields[name] = ds[prefix].sel(
                hybrid_level=level,
                drop=True,
            )

    # p
    for level in PRESSURE_HYBRID_LEVELS:
        name = f"p{level}hl"

        fields[name] = ds["p"].sel(
            p_hybrid_level=level,
            drop=True,
        )

    # Reassemble using StormCast's exact variable order
    data = xr.concat(
        [
            fields[var].expand_dims(variable=[var])
            for var in variables
        ],
        dim="variable",
    )

    return (
        data
        .transpose(
            "time",
            "variable",
            "hrrr_y",
            "hrrr_x",
        )
        .astype(np.float32)
    )



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



def build_stormcast_input(
    ds_singv,
    variables,
    hrrr_y,
    hrrr_x,
    verbose=True,
):
    """
    Convert one collated SingV3 dataset into a StormCast input array.

    Returns
    -------
    data
        DataArray with dimensions:
        (time, variable, hrrr_y, hrrr_x)

    singv_time
        Actual valid time of the SingV3 source data.
    """
    validate_singv_dataset(
        ds_singv,
        verbose=verbose,
    )

    if "valid_time" not in ds_singv:
        raise ValueError(
            "Collated SingV3 dataset has no valid_time variable."
        )

    singv_time = np.datetime64(
        ds_singv["valid_time"].values,
        "ns",
    )

    grid = make_grid_spec(
        ds_surface=ds_singv,
        hrrr_y=hrrr_y,
        hrrr_x=hrrr_x,
    )

    fields = make_empty_field_array(
        variables=variables,
        hrrr_y=hrrr_y,
        hrrr_x=hrrr_x,
    )

    if verbose:
        print("\nFilling surface fields...")

    fill_surface_fields(
        data=fields,
        ds_surface=ds_singv,
        grid=grid,
        verbose=verbose,
    )

    if verbose:
        print("\nConstructing approximate surface pressure...")

    surface_pressure = get_surface_pressure(
        ds_surface=ds_singv,
        grid=grid,
        verbose=verbose,
    )

    if verbose:
        print("\nFilling hybrid-level fields...")

    fill_hybrid_fields(
        data=fields,
        ds_pressure=ds_singv,
        sp=surface_pressure,
        grid=grid,
        verbose=verbose,
    )

    if verbose:
        print("\nConstructing reflectivity proxy...")

    fill_refc(
        data=fields,
        ds_surface=ds_singv,
        grid=grid,
        verbose=verbose,
    )

    validate_field_array(
        fields=fields,
        variables=variables,
        ny=len(hrrr_y),
        nx=len(hrrr_x),
    )

    data = add_time_dimension(
        data=fields,
        time_value=singv_time,
    )

    data.name = "stormcast_input"

    data.attrs.update(
        {
            "source": "SINGV-RCM ERA5-driven reanalysis (CCRS), vn5",
            "singv_valid_time": str(singv_time),
            "surface_pressure_note": (
                "Mean sea-level pressure psl is used as an "
                "approximation to surface pressure."
            ),
            "vertical_coordinate_note": (
                "StormCast hybrid levels are approximated using "
                "p_level = sigma * surface_pressure."
            ),
            "refc_note": (
                "Composite reflectivity is approximated from "
                "SingV3 precipitation rate pr."
            ),
            "target_lat_min": float(grid.target_lats.min()),
            "target_lat_max": float(grid.target_lats.max()),
            "target_lon_min": float(grid.target_lons.min()),
            "target_lon_max": float(grid.target_lons.max()),
        }
    )

    validate_stormcast_input_array(
        data=data,
        variables=variables,
        ny=len(hrrr_y),
        nx=len(hrrr_x),
    )

    return data, singv_time