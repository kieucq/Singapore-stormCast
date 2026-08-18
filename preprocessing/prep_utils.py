"""
Utilities for the first SINGV -> StormCast retraining preprocessing stage.

This module converts one assembled SINGV state from the native 960 x 960
latitude/longitude grid into a cropped and regridded 624 x 624 state.

This retraining pipeline keeps the native
SINGV pressure levels and does not construct HRRR hybrid levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


# ── Fixed retraining specification ────────────────────────────────────────────

SURFACE_VARIABLES = ["tas", "uas", "vas", "psl", "pr"]
PRESSURE_VARIABLES = ["ta", "ua", "va", "hus", "zg"]

PRESSURE_LEVELS_PA = np.asarray(
    [
        100000,
        92500,
        85000,
        80000,
        75000,
        70000,
        60000,
        50000,
        40000,
        30000,
        20000,
        10000,
        5000,
        1000,
    ],
    dtype=np.float64,
)

EXPECTED_SOURCE_SHAPE = (960, 960)
CROP_PIXELS = 12
CROPPED_SHAPE = (936, 936)
TARGET_SHAPE = (624, 624)
VALID_FRACTION_THRESHOLD = 0.5

# Used only when the source file omits units for these fields.
FALLBACK_UNITS = {
    "tas": "K",
    "uas": "m s-1",
    "vas": "m s-1",
    "psl": "Pa",
    "pr": "kg m-2 s-1",
    "ua": "m s-1",
    "va": "m s-1",
    "ta": "K",
    "hus": "kg kg-1",
    "zg": "m",
}


@dataclass(frozen=True)
class TargetGrid:
    """Coordinates, bounds, and conservative-remapping weights."""

    source_latitude: np.ndarray
    source_longitude: np.ndarray
    target_latitude: np.ndarray
    target_longitude: np.ndarray
    source_latitude_bounds: np.ndarray
    source_longitude_bounds: np.ndarray
    target_latitude_bounds: np.ndarray
    target_longitude_bounds: np.ndarray
    latitude_weights: np.ndarray
    longitude_weights: np.ndarray


# ── Metadata and validation helpers ───────────────────────────────────────────

def channel_names() -> list[str]:
    """
    Return the permanent 75-channel ordering.

    Order:
        5 surface channels,
        then all 14 pressure levels of ta,
        then ua, va, hus, and zg.
    """
    names = list(SURFACE_VARIABLES)

    for variable in PRESSURE_VARIABLES:
        for pressure_pa in PRESSURE_LEVELS_PA.astype(int):
            pressure_hpa = pressure_pa // 100
            names.append(f"{variable}_{pressure_hpa}")

    if len(names) != 75:
        raise RuntimeError(f"Expected 75 channels, constructed {len(names)}")

    return names


def channel_units(ds: xr.Dataset) -> list[str]:
    """Return one units string for every output channel."""
    units: list[str] = []

    for variable in SURFACE_VARIABLES:
        units.append(str(ds[variable].attrs.get("units", FALLBACK_UNITS[variable])))

    for variable in PRESSURE_VARIABLES:
        unit = str(ds[variable].attrs.get("units", FALLBACK_UNITS[variable]))
        units.extend([unit] * len(PRESSURE_LEVELS_PA))

    return units


def validate_source_dataset(ds: xr.Dataset) -> None:
    """Validate the pieces needed for one SINGV training state."""
    required_variables = SURFACE_VARIABLES + PRESSURE_VARIABLES
    required_coordinates = ["lat", "lon", "plev"]

    missing_variables = [name for name in required_variables if name not in ds]
    missing_coordinates = [name for name in required_coordinates if name not in ds.coords]

    if missing_variables:
        raise ValueError(f"Missing required SINGV variables: {missing_variables}")

    if missing_coordinates:
        raise ValueError(f"Missing required SINGV coordinates: {missing_coordinates}")

    source_shape = (ds.sizes["lat"], ds.sizes["lon"])
    if source_shape != EXPECTED_SOURCE_SHAPE:
        raise ValueError(
            f"Expected source grid {EXPECTED_SOURCE_SHAPE}, got {source_shape}"
        )

    actual_levels = np.asarray(ds["plev"].values, dtype=np.float64)
    if actual_levels.shape != PRESSURE_LEVELS_PA.shape or not np.array_equal(
        actual_levels, PRESSURE_LEVELS_PA
    ):
        raise ValueError(
            "Unexpected pressure levels or ordering.\n"
            f"Expected: {PRESSURE_LEVELS_PA.tolist()}\n"
            f"Found:    {actual_levels.tolist()}"
        )

    latitudes = np.asarray(ds["lat"].values, dtype=np.float64)
    longitudes = np.asarray(ds["lon"].values, dtype=np.float64)

    if not np.all(np.diff(latitudes) > 0):
        raise ValueError("Latitude must be strictly increasing.")

    if not np.all(np.diff(longitudes) > 0):
        raise ValueError("Longitude must be strictly increasing.")

    for variable in SURFACE_VARIABLES:
        expected = ("lat", "lon")
        if ds[variable].dims != expected:
            raise ValueError(
                f"{variable} has dimensions {ds[variable].dims}; expected {expected}"
            )

    for variable in PRESSURE_VARIABLES:
        expected = ("plev", "lat", "lon")
        if ds[variable].dims != expected:
            raise ValueError(
                f"{variable} has dimensions {ds[variable].dims}; expected {expected}"
            )


def extract_valid_time(ds: xr.Dataset) -> np.datetime64:
    """Read the scalar valid_time from an assembled SINGV file."""
    if "valid_time" not in ds:
        raise ValueError("Input dataset has no valid_time variable.")

    value = np.asarray(ds["valid_time"].values).squeeze()

    try:
        return np.datetime64(value, "ns")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not decode valid_time value {value!r}") from exc


def crop_source(ds: xr.Dataset) -> xr.Dataset:
    """Crop 12 pixels from every side: 960 x 960 -> 936 x 936."""
    cropped = ds.isel(
        lat=slice(CROP_PIXELS, -CROP_PIXELS),
        lon=slice(CROP_PIXELS, -CROP_PIXELS),
    )

    shape = (cropped.sizes["lat"], cropped.sizes["lon"])
    if shape != CROPPED_SHAPE:
        raise ValueError(f"Expected cropped shape {CROPPED_SHAPE}, got {shape}")

    return cropped


# ── Grid construction ─────────────────────────────────────────────────────────

def centres_to_bounds(centres: np.ndarray) -> np.ndarray:
    """
    Infer cell-edge coordinates from strictly increasing one-dimensional centres.
    """
    centres = np.asarray(centres, dtype=np.float64)

    if centres.ndim != 1 or centres.size < 2:
        raise ValueError("Coordinate centres must be a 1D array with >= 2 entries.")

    spacing = np.diff(centres)
    if not np.all(spacing > 0):
        raise ValueError("Coordinate centres must be strictly increasing.")

    bounds = np.empty(centres.size + 1, dtype=np.float64)
    bounds[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    bounds[0] = centres[0] - 0.5 * spacing[0]
    bounds[-1] = centres[-1] + 0.5 * spacing[-1]

    return bounds


def _linear_overlap_weights(
    source_bounds: np.ndarray,
    target_bounds: np.ndarray,
) -> np.ndarray:
    """
    Calculate 1D overlap weights for longitude-like coordinates.

    Each target row sums to one.
    """
    source_low = source_bounds[:-1][None, :]
    source_high = source_bounds[1:][None, :]
    target_low = target_bounds[:-1][:, None]
    target_high = target_bounds[1:][:, None]

    overlap_low = np.maximum(source_low, target_low)
    overlap_high = np.minimum(source_high, target_high)
    overlap = np.maximum(overlap_high - overlap_low, 0.0)

    target_width = target_high - target_low
    weights = overlap / target_width

    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Longitude overlap weights do not sum to one.")

    return weights


def _spherical_latitude_overlap_weights(
    source_bounds_degrees: np.ndarray,
    target_bounds_degrees: np.ndarray,
) -> np.ndarray:
    """
    Calculate latitude overlap weights using spherical cell-area factors.

    On a latitude/longitude grid, cell area is proportional to:
        sin(latitude_north) - sin(latitude_south)
    """
    source_low = source_bounds_degrees[:-1][None, :]
    source_high = source_bounds_degrees[1:][None, :]
    target_low = target_bounds_degrees[:-1][:, None]
    target_high = target_bounds_degrees[1:][:, None]

    overlap_low = np.maximum(source_low, target_low)
    overlap_high = np.minimum(source_high, target_high)
    has_overlap = overlap_high > overlap_low

    overlap = np.zeros_like(overlap_low, dtype=np.float64)
    overlap[has_overlap] = (
        np.sin(np.deg2rad(overlap_high[has_overlap]))
        - np.sin(np.deg2rad(overlap_low[has_overlap]))
    )

    target_measure = (
        np.sin(np.deg2rad(target_high))
        - np.sin(np.deg2rad(target_low))
    )

    weights = overlap / target_measure

    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Latitude overlap weights do not sum to one.")

    return weights


def make_target_grid(ds_cropped: xr.Dataset) -> TargetGrid:
    """
    Construct a 624 x 624 target grid covering the same cell-edge domain
    as the cropped 936 x 936 source grid.
    """
    source_latitude = np.asarray(ds_cropped["lat"].values, dtype=np.float64)
    source_longitude = np.asarray(ds_cropped["lon"].values, dtype=np.float64)

    source_latitude_bounds = centres_to_bounds(source_latitude)
    source_longitude_bounds = centres_to_bounds(source_longitude)

    target_latitude_bounds = np.linspace(
        source_latitude_bounds[0],
        source_latitude_bounds[-1],
        TARGET_SHAPE[0] + 1,
        dtype=np.float64,
    )
    target_longitude_bounds = np.linspace(
        source_longitude_bounds[0],
        source_longitude_bounds[-1],
        TARGET_SHAPE[1] + 1,
        dtype=np.float64,
    )

    target_latitude = 0.5 * (
        target_latitude_bounds[:-1] + target_latitude_bounds[1:]
    )
    target_longitude = 0.5 * (
        target_longitude_bounds[:-1] + target_longitude_bounds[1:]
    )

    latitude_weights = _spherical_latitude_overlap_weights(
        source_latitude_bounds,
        target_latitude_bounds,
    )
    longitude_weights = _linear_overlap_weights(
        source_longitude_bounds,
        target_longitude_bounds,
    )

    return TargetGrid(
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        source_latitude_bounds=source_latitude_bounds,
        source_longitude_bounds=source_longitude_bounds,
        target_latitude_bounds=target_latitude_bounds,
        target_longitude_bounds=target_longitude_bounds,
        latitude_weights=latitude_weights,
        longitude_weights=longitude_weights,
    )


# ── Regridding methods ────────────────────────────────────────────────────────

def bilinear_regrid(field: np.ndarray, grid: TargetGrid) -> np.ndarray:
    """Bilinearly interpolate one 2D field onto the target cell centres."""
    field = np.asarray(field, dtype=np.float64)

    expected_shape = (
        grid.source_latitude.size,
        grid.source_longitude.size,
    )
    if field.shape != expected_shape:
        raise ValueError(f"Field shape {field.shape}; expected {expected_shape}")

    interpolator = RegularGridInterpolator(
        (grid.source_latitude, grid.source_longitude),
        field,
        method="linear",
        bounds_error=True,
    )

    target_lon_2d, target_lat_2d = np.meshgrid(
        grid.target_longitude,
        grid.target_latitude,
    )
    target_points = np.column_stack(
        [target_lat_2d.ravel(), target_lon_2d.ravel()]
    )

    output = interpolator(target_points).reshape(TARGET_SHAPE)
    return output.astype(np.float32)


def conservative_regrid(field: np.ndarray, grid: TargetGrid) -> np.ndarray:
    """
    Conservatively regrid a cell-average 2D field.

    The method weights source values by the spherical area overlapping each
    target cell. The source and target grids must be rectilinear lat/lon grids
    covering the same outer cell bounds.
    """
    field = np.asarray(field, dtype=np.float64)

    expected_shape = (
        grid.source_latitude.size,
        grid.source_longitude.size,
    )
    if field.shape != expected_shape:
        raise ValueError(f"Field shape {field.shape}; expected {expected_shape}")

    # Separable 2D area-weighted remapping:
    # (target_y, source_y) @ (source_y, source_x)
    # @ (source_x, target_x)
    output = (
        grid.latitude_weights
        @ field
        @ grid.longitude_weights.T
    )

    return output.astype(np.float32)


def mask_aware_bilinear_regrid(
    field: np.ndarray,
    source_valid: np.ndarray,
    grid: TargetGrid,
) -> np.ndarray:
    """
    Bilinearly regrid using only valid source values.

    Computes:
        R(mask * field) / R(mask)

    Invalid output cells remain NaN until the conservative valid-fraction
    threshold is applied.
    """
    field = np.asarray(field, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)

    if field.shape != source_valid.shape:
        raise ValueError(
            f"Field shape {field.shape} and mask shape {source_valid.shape} differ."
        )

    numerator_source = np.where(source_valid, field, 0.0)
    denominator_source = source_valid.astype(np.float64)

    numerator = bilinear_regrid(numerator_source, grid).astype(np.float64)
    denominator = bilinear_regrid(denominator_source, grid).astype(np.float64)

    output = np.full(TARGET_SHAPE, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=output,
        where=denominator > 1e-12,
    )

    return output.astype(np.float32)


# ── One-state preprocessing ───────────────────────────────────────────────────

def preprocess_one_state(
    ds: xr.Dataset,
    *,
    valid_fraction_threshold: float = VALID_FRACTION_THRESHOLD,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Convert one assembled SINGV file into a 75-channel 624 x 624 state.

    The returned Dataset is not normalized. Pressure-level cells that fail
    the valid-fraction threshold remain NaN. They will be excluded from
    normalization statistics and filled only after normalization in the
    eventual training dataset loader.
    """
    if not 0.0 <= valid_fraction_threshold <= 1.0:
        raise ValueError("valid_fraction_threshold must lie between 0 and 1.")

    validate_source_dataset(ds)
    valid_time = extract_valid_time(ds)
    ds_cropped = crop_source(ds)
    grid = make_target_grid(ds_cropped)

    if verbose:
        print("\nGrid")
        print("----")
        print(f"Source:  {EXPECTED_SOURCE_SHAPE[0]} x {EXPECTED_SOURCE_SHAPE[1]}")
        print(f"Cropped: {CROPPED_SHAPE[0]} x {CROPPED_SHAPE[1]}")
        print(f"Target:  {TARGET_SHAPE[0]} x {TARGET_SHAPE[1]}")
        print(
            f"Latitude:  {grid.target_latitude[0]:.6f} "
            f"to {grid.target_latitude[-1]:.6f}"
        )
        print(
            f"Longitude: {grid.target_longitude[0]:.6f} "
            f"to {grid.target_longitude[-1]:.6f}"
        )

    state_fields: list[np.ndarray] = []

    # Surface fields
    if verbose:
        print("\nSurface fields")
        print("--------------")

    for variable in SURFACE_VARIABLES:
        source = np.asarray(ds_cropped[variable].values, dtype=np.float32)

        if not np.isfinite(source).all():
            fraction = float((~np.isfinite(source)).mean())
            raise ValueError(
                f"{variable} contains non-finite values after cropping; "
                f"fraction={fraction:.8f}"
            )

        if variable == "pr":
            negative_count = int((source < 0.0).sum())
            if negative_count:
                minimum = float(source.min())
                print(
                    f"Warning: pr contains {negative_count} negative values "
                    f"(minimum={minimum:.6g}); clipping them to zero."
                )
                source = np.maximum(source, 0.0)

            target = conservative_regrid(source, grid)
            method = "conservative"
        else:
            target = bilinear_regrid(source, grid)
            method = "bilinear"

        if not np.isfinite(target).all():
            raise ValueError(f"{variable} became non-finite after {method} regridding.")

        state_fields.append(target)

        if verbose:
            print(
                f"{variable:4s} | {method:12s} | "
                f"min={target.min():.6g} "
                f"mean={target.mean():.6g} "
                f"max={target.max():.6g}"
            )

    # Source mask from ta. Below-ground cells are represented by zero in the
    # assembled SINGV files, while decoded fill values appear as NaN.
    ta_source = np.asarray(ds_cropped["ta"].values, dtype=np.float32)
    source_valid = np.isfinite(ta_source) & (ta_source > 0.0)

    if source_valid.shape != (
        len(PRESSURE_LEVELS_PA),
        CROPPED_SHAPE[0],
        CROPPED_SHAPE[1],
    ):
        raise ValueError(f"Unexpected ta mask shape {source_valid.shape}")

    pressure_valid_fraction = np.empty(
        (len(PRESSURE_LEVELS_PA), *TARGET_SHAPE),
        dtype=np.float32,
    )
    pressure_valid = np.empty(
        (len(PRESSURE_LEVELS_PA), *TARGET_SHAPE),
        dtype=bool,
    )

    if verbose:
        print("\nPressure-level validity")
        print("-----------------------")

    for level_index, pressure_pa in enumerate(PRESSURE_LEVELS_PA.astype(int)):
        valid_fraction = conservative_regrid(
            source_valid[level_index].astype(np.float32),
            grid,
        )
        target_valid = valid_fraction >= valid_fraction_threshold

        pressure_valid_fraction[level_index] = valid_fraction
        pressure_valid[level_index] = target_valid

        if verbose:
            print(
                f"{pressure_pa // 100:4d} hPa | "
                f"source valid={source_valid[level_index].mean():.4f} | "
                f"target valid={target_valid.mean():.4f}"
            )

    # Pressure fields, grouped by variable and then pressure level.
    if verbose:
        print("\nPressure fields")
        print("---------------")

    for variable in PRESSURE_VARIABLES:
        source_3d = np.asarray(ds_cropped[variable].values, dtype=np.float32)

        if source_3d.shape != source_valid.shape:
            raise ValueError(
                f"{variable} shape {source_3d.shape}; expected {source_valid.shape}"
            )

        for level_index, pressure_pa in enumerate(PRESSURE_LEVELS_PA.astype(int)):
            source_2d = source_3d[level_index]

            # The ta mask is deliberately copied to every pressure variable.
            source_2d_valid = source_valid[level_index]

            # A non-finite value inside a ta-valid cell indicates a genuine
            # inconsistency that should not be hidden.
            inconsistent = source_2d_valid & ~np.isfinite(source_2d)
            if inconsistent.any():
                raise ValueError(
                    f"{variable} at {pressure_pa // 100} hPa contains "
                    f"{int(inconsistent.sum())} non-finite values inside "
                    "the ta-valid atmosphere."
                )

            target_2d = mask_aware_bilinear_regrid(
                source_2d,
                source_2d_valid,
                grid,
            )

            # The conservative validity decision is authoritative.
            target_2d = np.where(
                pressure_valid[level_index],
                target_2d,
                np.nan,
            ).astype(np.float32)

            valid_values = target_2d[pressure_valid[level_index]]
            if valid_values.size == 0:
                raise ValueError(
                    f"No valid target cells for {variable} at "
                    f"{pressure_pa // 100} hPa."
                )

            if not np.isfinite(valid_values).all():
                raise ValueError(
                    f"{variable} at {pressure_pa // 100} hPa contains "
                    "non-finite values inside the target-valid region."
                )

            state_fields.append(target_2d)

            if verbose:
                print(
                    f"{variable}_{pressure_pa // 100:<4d} | "
                    f"min={valid_values.min():.6g} "
                    f"mean={valid_values.mean():.6g} "
                    f"max={valid_values.max():.6g}"
                )

    state = np.stack(state_fields, axis=0).astype(np.float32)
    names = channel_names()
    units = channel_units(ds)

    if state.shape != (75, *TARGET_SHAPE):
        raise ValueError(
            f"Expected state shape {(75, *TARGET_SHAPE)}, got {state.shape}"
        )

    # Validation: no infinities; NaNs only in masked pressure cells.
    if np.isinf(state).any():
        raise ValueError("Processed state contains infinite values.")

    if not np.isfinite(state[: len(SURFACE_VARIABLES)]).all():
        raise ValueError("A processed surface channel contains NaN or infinity.")

    repeated_pressure_valid = np.concatenate(
        [pressure_valid] * len(PRESSURE_VARIABLES),
        axis=0,
    )
    pressure_state = state[len(SURFACE_VARIABLES) :]

    if np.isnan(pressure_state[repeated_pressure_valid]).any():
        raise ValueError("NaN found inside a valid pressure-level target cell.")

    if not np.isnan(pressure_state[~repeated_pressure_valid]).all():
        raise ValueError("A masked pressure-level target cell was not stored as NaN.")

    output = xr.Dataset(
        data_vars={
            "state": (
                ("time", "channel", "y", "x"),
                state[None, ...],
                {
                    "description": (
                        "Unnormalized SINGV state for six-hour StormCast "
                        "regression training."
                    ),
                },
            ),
            "pressure_valid": (
                ("time", "plev", "y", "x"),
                pressure_valid[None, ...].astype(np.uint8),
                {
                    "description": (
                        "1 where at least the configured fraction of the "
                        "target cell is valid atmosphere; 0 otherwise."
                    ),
                    "valid_fraction_threshold": float(valid_fraction_threshold),
                },
            ),
            "pressure_valid_fraction": (
                ("time", "plev", "y", "x"),
                pressure_valid_fraction[None, ...],
                {
                    "description": (
                        "Conservatively remapped fraction of each target cell "
                        "covered by ta-valid source atmosphere."
                    ),
                    "units": "1",
                },
            ),
            "channel_units": (
                ("channel",),
                np.asarray(units, dtype=object),
                {
                    "description": "Physical unit corresponding to each state channel."
                },
            ),
        },
        coords={
            "time": np.asarray([valid_time], dtype="datetime64[ns]"),
            "channel": np.asarray(names, dtype=object),
            "plev": PRESSURE_LEVELS_PA.astype(np.float64),
            "y": np.arange(TARGET_SHAPE[0], dtype=np.int32),
            "x": np.arange(TARGET_SHAPE[1], dtype=np.int32),
            "latitude": ("y", grid.target_latitude.astype(np.float64)),
            "longitude": ("x", grid.target_longitude.astype(np.float64)),
            "latitude_bounds": (
                ("y_bounds",),
                grid.target_latitude_bounds.astype(np.float64),
            ),
            "longitude_bounds": (
                ("x_bounds",),
                grid.target_longitude_bounds.astype(np.float64),
            ),
        },
        attrs={
            "source": "SINGV-RCM ERA5-driven reanalysis (CCRS), vn5",
            "processing_stage": "prepared",
            "source_grid_shape": f"{EXPECTED_SOURCE_SHAPE[0]}x{EXPECTED_SOURCE_SHAPE[1]}",
            "crop_pixels_each_side": CROP_PIXELS,
            "cropped_grid_shape": f"{CROPPED_SHAPE[0]}x{CROPPED_SHAPE[1]}",
            "target_grid_shape": f"{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}",
            "surface_regridding": (
                "bilinear for tas/uas/vas/psl; conservative for pr"
            ),
            "pressure_regridding": "ta-mask-aware bilinear",
            "valid_fraction_regridding": "spherical-area conservative",
            "valid_fraction_threshold": float(valid_fraction_threshold),
            "normalization": "none",
            "masked_pressure_storage": "NaN",
            "channel_order": (
                "surface variables in SURFACE_VARIABLES order; then each "
                "pressure variable in PRESSURE_VARIABLES order over all "
                "14 pressure levels"
            ),
        },
    )

    output["plev"].attrs.update(
        {
            "standard_name": "air_pressure",
            "units": "Pa",
            "positive": "down",
        }
    )
    output["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )
    output["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )

    return output