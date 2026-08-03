"""
SINGV and ERA5 dataset adapter for StormCast training.

The adapter reads SINGV input-target pairs and corresponding ERA5
backgrounds from a CSV manifest. It applies separate training-set
normalization statistics to SINGV and ERA5, fills masked SINGV cells
with zero, and returns samples in the format expected by the
StormCast trainer.


Dataset parameters
------------------
data_root
    Dataset root used to resolve relative paths stored inside the manifests.
    Defaults to ``~/scratch/retraining``.

train_manifest
    Pair manifest used when ``train=True``.

validation_manifest
    Pair manifest used when ``train=False``.

normalisation_path
    NPZ file produced by ``compute_normalisation_stats.py``.

background_normalisation_path
    NPZ file containing training-set ERA5 background statistics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .dataset import StormCastDataset


EXPECTED_COLUMNS = ("input_time", "input_file", "background_time", "background_file", "target_time", "target_file")
EXPECTED_STATE_DIMS = ("time", "channel", "y", "x")
EXPECTED_BACKGROUND_DIMS = ("time", "channel", "y", "x")
DEFAULT_DATA_ROOT = "~/scratch/retraining"


@dataclass(frozen=True)
class PairRecord:
    """One input-target pair."""

    input_time: datetime
    background_time: datetime
    target_time: datetime

    input_path: Path
    background_path: Path
    target_path: Path


def _resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Expand ``~`` and resolve an absolute or root-relative path."""

    path = Path(value).expanduser()
    if root is not None and not path.is_absolute():
        path = root / path
    return path.resolve()


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    """Convert an xarray string coordinate to Python strings."""

    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )


def _parse_time(value: str, row_number: int, column: str) -> datetime:
    """Parse one ISO-format timestamp from the manifest."""

    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"Manifest row {row_number} has invalid {column!r} value {value!r}."
        ) from error


def _read_manifest(path: Path, data_root: Path) -> tuple[PairRecord, ...]:
    """Read and validate an input-target pair manifest."""

    records: list[PairRecord] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")

        missing = [
            column for column in EXPECTED_COLUMNS if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"Manifest {path} is missing required column(s): " + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):
            values = {
                column: (row.get(column) or "").strip() for column in EXPECTED_COLUMNS
            }
            empty = [column for column, value in values.items() if not value]
            if empty:
                raise ValueError(
                    f"Manifest row {row_number} has empty column(s): "
                    + ", ".join(empty)
                )

            input_time = _parse_time(
                values["input_time"],
                row_number,
                "input_time",
            )

            background_time = _parse_time(
                values["background_time"],
                row_number,
                "background_time",
            )

            target_time = _parse_time(
                values["target_time"],
                row_number,
                "target_time",
            )

            if background_time != input_time:
                raise ValueError(
                    f"Manifest row {row_number} has background_time "
                    f"{background_time.isoformat()}, but input_time is "
                    f"{input_time.isoformat()}."
                )

            if target_time <= input_time:
                raise ValueError(
                    f"Manifest row {row_number} must have target_time "
                    "after input_time."
                )

            records.append(
                PairRecord(
                    input_time=input_time,
                    background_time=background_time,
                    target_time=target_time,
                    input_path=_resolve_path(
                        values["input_file"],
                        data_root,
                    ),
                    background_path=_resolve_path(
                        values["background_file"],
                        data_root,
                    ),
                    target_path=_resolve_path(
                        values["target_file"],
                        data_root,
                    ),
                )
            )

    if not records:
        raise ValueError(f"Manifest contains no data rows: {path}")
    
    lead_times = {
        record.target_time - record.input_time
        for record in records
    }

    if len(lead_times) != 1:
        raise ValueError(
            f"Manifest contains inconsistent forecast intervals: {path}"
        )

    return tuple(records)


class RegressionDatasetAdapter(StormCastDataset):
    """Load normalized SINGV input-target pairs for StormCast."""

    lead_time_steps = 0

    def __init__(self, params: Any, train: bool) -> None:
        """Initialize either the training or validation split."""

        data_root = _resolve_path(getattr(params, "data_root", DEFAULT_DATA_ROOT))
        manifest_name = "train_manifest" if train else "validation_manifest"

        try:
            manifest_path = _resolve_path(
                getattr(params, manifest_name),
                data_root,
            )

            normalisation_path = _resolve_path(
                params.normalisation_path,
                data_root,
            )

            background_normalisation_path = _resolve_path(
                params.background_normalisation_path,
                data_root,
            )

        except AttributeError as error:
            raise ValueError(
                f"Missing required dataset parameter: {error.name}"
            ) from error

        self._records = _read_manifest(manifest_path, data_root)

        # SINGV statistics and metadata.
        self._load_normalisation(normalisation_path)
        self._load_metadata(self._records[0].input_path)

        # ERA5 statistics and metadata.
        self._load_background_normalisation(
            background_normalisation_path
        )
        self._load_background_metadata(
            self._records[0].background_path,
            self._records[0].background_time,
        )

    def _load_normalisation(self, path: Path) -> None:
        """Load channel statistics and image metadata."""

        with np.load(path) as data:
            required = {"channels", "mean", "std", "image_shape"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    f"{path} is missing required array(s): " + ", ".join(missing)
                )

            self._channels = _decode_strings(data["channels"])
            self._mean = np.asarray(data["mean"], dtype=np.float32)
            self._std = np.asarray(data["std"], dtype=np.float32)
            image_shape = np.asarray(data["image_shape"], dtype=np.int64)

        if not self._channels:
            raise ValueError(f"{path}: no channels were found.")

        expected_shape = (len(self._channels),)
        if self._mean.shape != expected_shape or self._std.shape != expected_shape:
            raise ValueError(
                f"{path}: mean and std must both have shape {expected_shape}."
            )

        if image_shape.shape != (2,):
            raise ValueError(f"{path}: image_shape must have shape (2,).")

        self._image_shape = (int(image_shape[0]), int(image_shape[1]))
        self._state_shape = (len(self._channels), *self._image_shape)

        if not np.all(np.isfinite(self._mean)):
            raise ValueError(f"{path}: means must all be finite.")
        if not np.all(np.isfinite(self._std)) or np.any(self._std <= 0):
            raise ValueError(
                f"{path}: standard deviations must be finite and positive."
            )

        self._mean_3d = self._mean[:, None, None]
        self._std_3d = self._std[:, None, None]

    def _load_background_normalisation(self, path: Path) -> None:
        """Load ERA5 background normalization statistics."""

        with np.load(path) as data:
            required = {"channels", "mean", "std", "image_shape"}
            missing = sorted(required.difference(data.files))

            if missing:
                raise ValueError(
                    f"{path} is missing required array(s): "
                    + ", ".join(missing)
                )

            self._background_channels = _decode_strings(
                data["channels"]
            )
            self._background_mean = np.asarray(
                data["mean"],
                dtype=np.float32,
            )
            self._background_std = np.asarray(
                data["std"],
                dtype=np.float32,
            )
            image_shape = np.asarray(
                data["image_shape"],
                dtype=np.int64,
            )

        if not self._background_channels:
            raise ValueError(
                f"{path}: no ERA5 background channels were found."
            )

        if len(self._background_channels) != 26:
            raise ValueError(
                f"{path}: expected 26 ERA5 background channels, got "
                f"{len(self._background_channels)}."
            )

        expected_shape = (len(self._background_channels),)

        if (
            self._background_mean.shape != expected_shape
            or self._background_std.shape != expected_shape
        ):
            raise ValueError(
                f"{path}: ERA5 mean and std must both have shape "
                f"{expected_shape}."
            )

        if image_shape.shape != (2,):
            raise ValueError(
                f"{path}: image_shape must have shape (2,)."
            )

        background_image_shape = (
            int(image_shape[0]),
            int(image_shape[1]),
        )

        if background_image_shape != self._image_shape:
            raise ValueError(
                f"{path}: ERA5 image shape {background_image_shape} "
                f"does not match SINGV image shape "
                f"{self._image_shape}."
            )

        if not np.all(np.isfinite(self._background_mean)):
            raise ValueError(
                f"{path}: ERA5 means must all be finite."
            )

        if (
            not np.all(np.isfinite(self._background_std))
            or np.any(self._background_std <= 0)
        ):
            raise ValueError(
                f"{path}: ERA5 standard deviations must be "
                "finite and positive."
            )

        self._background_shape = (
            len(self._background_channels),
            *self._image_shape,
        )

        self._background_mean_3d = (
            self._background_mean[:, None, None]
        )
        self._background_std_3d = (
            self._background_std[:, None, None]
        )

    def _load_metadata(self, path: Path) -> None:
        """Validate the first prepared state and load its coordinate grids."""

        with xr.open_dataset(path) as dataset:
            required = {"state", "channel", "latitude", "longitude"}
            missing = sorted(required.difference(dataset.variables))
            if missing:
                raise ValueError(f"{path} is missing: " + ", ".join(missing))

            state = dataset["state"]
            if tuple(state.dims) != EXPECTED_STATE_DIMS or state.sizes["time"] != 1:
                raise ValueError(
                    f"{path}: state must have dimensions {EXPECTED_STATE_DIMS} "
                    "with one time step."
                )

            if _decode_strings(dataset["channel"].values) != self._channels:
                raise ValueError(
                    f"{path}: channel names or order do not match the "
                    "normalization file."
                )

            state_shape = tuple(state.isel(time=0).shape)
            if state_shape != self._state_shape:
                raise ValueError(
                    f"{path}: expected state shape {self._state_shape}, "
                    f"got {state_shape}."
                )

            latitude = np.asarray(dataset["latitude"].values, dtype=np.float32)
            longitude = np.asarray(dataset["longitude"].values, dtype=np.float32)

            if latitude.ndim == 1 and longitude.ndim == 1:
                longitude, latitude = np.meshgrid(longitude, latitude)

        if latitude.shape != self._image_shape or longitude.shape != self._image_shape:
            raise ValueError(
                f"{path}: latitude and longitude must both have shape "
                f"{self._image_shape}."
            )
        if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
            raise ValueError(f"{path}: coordinates contain non-finite values.")

        self._latitude = latitude
        self._longitude = longitude

    def _load_state(self, path: Path) -> np.ndarray:
        """Load one prepared state as ``(channel, y, x)``."""

        with xr.open_dataset(path) as dataset:
            state = np.asarray(
                dataset["state"].isel(time=0).values,
                dtype=np.float32,
            )

        if state.shape != self._state_shape:
            raise ValueError(
                f"{path}: expected state shape {self._state_shape}, got {state.shape}."
            )
        if np.isinf(state).any():
            raise ValueError(f"{path}: state contains infinite values.")

        return state

    def _load_background(
        self,
        path: Path,
        expected_time: datetime,
    ) -> np.ndarray:
        """Load one prepared ERA5 background."""

        with xr.open_dataset(path) as dataset:
            required = {"background", "channel", "time"}
            missing = sorted(
                required.difference(dataset.variables)
            )

            if missing:
                raise ValueError(
                    f"{path} is missing: " + ", ".join(missing)
                )

            variable = dataset["background"]

            if tuple(variable.dims) != EXPECTED_BACKGROUND_DIMS:
                raise ValueError(
                    f"{path}: background must have dimensions "
                    f"{EXPECTED_BACKGROUND_DIMS}, got "
                    f"{tuple(variable.dims)}."
                )

            if variable.sizes["time"] != 1:
                raise ValueError(
                    f"{path}: expected exactly one ERA5 time step, "
                    f"got {variable.sizes['time']}."
                )

            channels = _decode_strings(
                dataset["channel"].values
            )

            if channels != self._background_channels:
                raise ValueError(
                    f"{path}: ERA5 channel names or order do not "
                    "match the ERA5 normalization file."
                )

            normalization = str(
                dataset.attrs.get("normalization", "")
            ).lower()

            if normalization != "none":
                raise ValueError(
                    f"{path}: expected normalization='none', got "
                    f"{dataset.attrs.get('normalization')!r}."
                )

            file_time = np.asarray(
                dataset["time"].values
            ).reshape(-1)[0]

            expected_time_np = np.datetime64(
                expected_time.isoformat(),
                "ns",
            )
            file_time_np = np.datetime64(file_time, "ns")

            if file_time_np != expected_time_np:
                raise ValueError(
                    f"{path}: file time {file_time_np} does not "
                    f"match manifest background_time "
                    f"{expected_time_np}."
                )

            background = np.asarray(
                variable.isel(time=0).values,
                dtype=np.float32,
            )

        if background.shape != self._background_shape:
            raise ValueError(
                f"{path}: expected ERA5 background shape "
                f"{self._background_shape}, got "
                f"{background.shape}."
            )

        if not np.all(np.isfinite(background)):
            locations = np.argwhere(~np.isfinite(background))
            first = tuple(int(value) for value in locations[0])
            channel_name = self._background_channels[first[0]]

            raise ValueError(
                f"{path}: ERA5 contains a non-finite value at "
                f"(channel={first[0]} [{channel_name}], "
                f"y={first[1]}, x={first[2]})."
            )

        return background

    def _load_background_metadata(
        self,
        path: Path,
        expected_time: datetime,
    ) -> None:
        """Validate the first ERA5 file and its coordinate grid."""

        # Validate the data, timestamp, channels and dimensions.
        self._load_background(path, expected_time)

        with xr.open_dataset(path) as dataset:
            required = {"latitude", "longitude"}
            missing = sorted(
                required.difference(dataset.variables)
            )

            if missing:
                raise ValueError(
                    f"{path} is missing: " + ", ".join(missing)
                )

            latitude = np.asarray(
                dataset["latitude"].values,
                dtype=np.float32,
            )
            longitude = np.asarray(
                dataset["longitude"].values,
                dtype=np.float32,
            )

            if latitude.ndim == 1 and longitude.ndim == 1:
                longitude, latitude = np.meshgrid(
                    longitude,
                    latitude,
                )

        if (
            latitude.shape != self._image_shape
            or longitude.shape != self._image_shape
        ):
            raise ValueError(
                f"{path}: ERA5 latitude and longitude must "
                f"both have shape {self._image_shape}."
            )

        if (
            not np.all(np.isfinite(latitude))
            or not np.all(np.isfinite(longitude))
        ):
            raise ValueError(
                f"{path}: ERA5 coordinates contain "
                "non-finite values."
            )

        if not np.allclose(
            latitude,
            self._latitude,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"{path}: ERA5 latitude grid does not match "
                "the SINGV grid."
            )

        if not np.allclose(
            longitude,
            self._longitude,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"{path}: ERA5 longitude grid does not match "
                "the SINGV grid."
            )

    def _normalize_and_fill(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize one state and fill invalid cells with zero."""

        valid = np.isfinite(state)
        normalized = (state - self._mean_3d) / self._std_3d
        normalized[~valid] = 0.0
        return normalized, valid.astype(np.float32)

    def _normalize_background(
        self,
        background: np.ndarray,
    ) -> np.ndarray:
        """Normalize one complete ERA5 background."""

        normalized = (
            background - self._background_mean_3d
        ) / self._background_std_3d

        if not np.all(np.isfinite(normalized)):
            raise ValueError(
                "Normalized ERA5 background contains "
                "non-finite values."
            )

        return normalized.astype(
            np.float32,
            copy=False,
        )

    def __len__(self) -> int:
        """Return the number of input-target pairs."""

        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one normalized StormCast sample."""

        record = self._records[index]
        input_state, input_mask = self._normalize_and_fill(
            self._load_state(record.input_path)
        )
        target_state, target_mask = self._normalize_and_fill(
            self._load_state(record.target_path)
        )

        background = self._normalize_background(
            self._load_background(
                record.background_path,
                record.background_time,
            )
        )

        return {
            "background": background,
            "state": (input_state, target_state),
            "input_mask": input_mask,
            "target_mask": target_mask,
        }

    def background_channels(self) -> list[str]:
        """Return the ERA5 background channel names."""

        return list(self._background_channels)

    def state_channels(self) -> list[str]:
        """Return the SINGV state channel names."""

        return list(self._channels)

    def image_shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""

        return self._image_shape

    def latitude(self) -> np.ndarray:
        """Return the two-dimensional latitude grid."""

        return self._latitude

    def longitude(self) -> np.ndarray:
        """Return the two-dimensional longitude grid."""

        return self._longitude

    def _state_statistics_for(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return broadcastable statistics for a 3-D or 4-D state array."""

        if x.ndim == 3:
            channel_axis = 0
            shape = (len(self._channels), 1, 1)
        elif x.ndim == 4:
            channel_axis = 1
            shape = (1, len(self._channels), 1, 1)
        else:
            raise ValueError(f"Expected a 3-D or 4-D state array, got shape {x.shape}.")

        if x.shape[channel_axis] != len(self._channels):
            raise ValueError(
                f"Expected {len(self._channels)} channels, got shape {x.shape}."
            )

        return self._mean.reshape(shape), self._std.reshape(shape)

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert physical state values to normalized values."""

        x = np.asarray(x, dtype=np.float32)
        mean, std = self._state_statistics_for(x)
        return (x - mean) / std

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized values back to physical units."""

        x = np.asarray(x, dtype=np.float32)
        mean, std = self._state_statistics_for(x)
        return x * std + mean

    def _background_statistics_for(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return broadcastable ERA5 statistics."""

        num_channels = len(self._background_channels)

        if x.ndim == 3:
            channel_axis = 0
            shape = (num_channels, 1, 1)

        elif x.ndim == 4:
            channel_axis = 1
            shape = (1, num_channels, 1, 1)

        else:
            raise ValueError(
                "Expected a 3-D or 4-D ERA5 array, "
                f"got shape {x.shape}."
            )

        if x.shape[channel_axis] != num_channels:
            raise ValueError(
                f"Expected {num_channels} ERA5 channels, "
                f"got shape {x.shape}."
            )

        return (
            self._background_mean.reshape(shape),
            self._background_std.reshape(shape),
        )


    def normalize_background(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Convert physical ERA5 values to normalized values."""

        x = np.asarray(x, dtype=np.float32)
        mean, std = self._background_statistics_for(x)

        return (x - mean) / std


    def denormalize_background(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Convert normalized ERA5 values to physical units."""

        x = np.asarray(x, dtype=np.float32)
        mean, std = self._background_statistics_for(x)

        return x * std + mean

    def index_for_time(self, input_time: datetime) -> int:
        """Return the manifest index corresponding to an input time."""

        for index, record in enumerate(self._records):
            if record.input_time == input_time:
                return index

        raise ValueError(
            f"No dataset sample begins at {input_time.isoformat()}."
        )

    def input_time(self, index: int) -> datetime:
        """Return the input time for one sample."""

        return self._records[index].input_time

    def target_time(self, index: int) -> datetime:
        """Return the target time for one sample."""

        return self._records[index].target_time