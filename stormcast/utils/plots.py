# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping

from matplotlib import pyplot as plt
import numpy as np


def _normalize_backgrounds(background):
    """Ensure background inputs are handled uniformly."""
    if background is None:
        return {}
    if isinstance(background, Mapping):
        return background
    if isinstance(background, (list, tuple)):
        return {f"background_{idx}": arr for idx, arr in enumerate(background)}
    return {"background": background}


def _masked_field(field, mask=None):
    """Return a 2-D masked array with invalid cells hidden."""

    field = np.asarray(field, dtype=np.float32)

    if field.ndim != 2:
        raise ValueError(f"Expected a 2-D field, got shape {field.shape}.")

    invalid = ~np.isfinite(field)

    if mask is not None:
        mask = np.asarray(mask)

        if mask.shape != field.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match field shape {field.shape}."
            )

        invalid |= mask <= 0

    return np.ma.array(field, mask=invalid)


def _plot_field(
    fig,
    ax,
    field,
    latitude,
    longitude,
    *,
    cmap,
    vmin,
    vmax,
):
    """Plot one field with geographical coordinates where available."""

    if latitude is not None and longitude is not None:
        latitude = np.asarray(latitude)
        longitude = np.asarray(longitude)

        if latitude.shape != field.shape or longitude.shape != field.shape:
            raise ValueError(
                "Latitude, longitude, and field must have matching shapes: "
                f"{latitude.shape}, {longitude.shape}, {field.shape}."
            )

        image = ax.pcolormesh(
            longitude,
            latitude,
            field,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Force conventional geographical orientation:
        # west to east from left to right, south to north from bottom to top.
        ax.set_xlim(float(np.nanmin(longitude)), float(np.nanmax(longitude)))
        ax.set_ylim(float(np.nanmin(latitude)), float(np.nanmax(latitude)))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
    else:
        image = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return image


def validation_plot(
    generated,
    truth,
    input_state,
    variable,
    background=None,
    *,
    input_mask=None,
    target_mask=None,
    latitude=None,
    longitude=None,
):
    """Produce a physical-unit validation plot during training.

    Args:
        generated: Generated output in physical units.
        truth: Ground truth in physical units.
        input_state: Input state in physical units, or None.
        variable: Variable name for the title.
        background: Optional background channel(s).
        input_mask: Validity mask for the input state.
        target_mask: Validity mask for generated output and truth.
        latitude: Two-dimensional latitude grid.
        longitude: Two-dimensional longitude grid.

    Returns:
        matplotlib figure.
    """

    backgrounds = _normalize_backgrounds(background)

    generated = _masked_field(generated, target_mask)
    truth = _masked_field(truth, target_mask)

    input_field = (
        None
        if input_state is None
        else _masked_field(input_state, input_mask)
    )

    truth_values = truth.compressed()
    if truth_values.size == 0:
        raise ValueError(f"No valid truth values available for {variable}.")

    # Use one common scale so generated, truth, and input remain comparable.
    vmin = float(truth_values.min())
    vmax = float(truth_values.max())

    if vmin == vmax:
        padding = max(abs(vmin) * 1.0e-6, 1.0e-6)
        vmin -= padding
        vmax += padding

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    num_panels = 3 + len(backgrounds)

    fig, axes = plt.subplots(
        1,
        num_panels,
        sharex=True,
        sharey=True,
        figsize=(5 * num_panels, 5),
        squeeze=False,
    )
    axes = axes[0]

    _plot_field(
        fig,
        axes[0],
        generated,
        latitude,
        longitude,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"Generated: {variable}")

    _plot_field(
        fig,
        axes[1],
        truth,
        latitude,
        longitude,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Truth")

    if input_field is None:
        axes[2].set_title("Input: none")
        axes[2].axis("off")
    else:
        _plot_field(
            fig,
            axes[2],
            input_field,
            latitude,
            longitude,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[2].set_title("Input")

    for index, (name, background_field) in enumerate(backgrounds.items()):
        ax = axes[3 + index]
        background_field = _masked_field(background_field)

        background_values = background_field.compressed()
        if background_values.size == 0:
            ax.set_title(f"Background: {name} (no valid data)")
            ax.axis("off")
            continue

        background_min = float(background_values.min())
        background_max = float(background_values.max())

        _plot_field(
            fig,
            ax,
            background_field,
            latitude,
            longitude,
            cmap=cmap,
            vmin=background_min,
            vmax=background_max,
        )
        ax.set_title(f"Background: {name}")

    fig.tight_layout()
    return fig


color_limits = {
    "u10m": (-5, 5),
    "v10": (-5, 5),
    "t2m": (260, 310),
    "tcwv": (0, 60),
    "msl": (0.1, 0.3),
    "refc": (-10, 30),
}


def inference_plot(
    background,
    state_pred,
    state_true,
    plot_var_background,
    plot_var_state,
    initial_time,
    lead_time,
    *,
    prediction_mask=None,
    truth_mask=None,
    background_mask=None,
    latitude=None,
    longitude=None,
):
    """Plot one inference prediction in physical units.

    State prediction and truth share the same colour scale. Invalid state
    cells are masked. Coordinates are used when available so that the map
    has the correct geographical orientation.
    """

    state_pred = _masked_field(state_pred, prediction_mask)
    state_true = _masked_field(state_true, truth_mask)

    # Masked-array subtraction automatically uses the union of both masks.
    state_error = state_pred - state_true

    truth_values = state_true.compressed()
    if truth_values.size == 0:
        raise ValueError(
            f"No valid truth values available for {plot_var_state}."
        )

    state_min = float(truth_values.min())
    state_max = float(truth_values.max())

    if state_min == state_max:
        padding = max(abs(state_min) * 1.0e-6, 1.0e-6)
        state_min -= padding
        state_max += padding

    error_values = state_error.compressed()
    max_error = (
        float(np.max(np.abs(error_values)))
        if error_values.size > 0
        else 1.0
    )
    max_error = max(max_error, 1.0e-6)

    state_cmap = plt.get_cmap("viridis").copy()
    state_cmap.set_bad(color="lightgray")

    error_cmap = plt.get_cmap("RdBu_r").copy()
    error_cmap.set_bad(color="lightgray")

    has_background = background is not None
    num_panels = 4 if has_background else 3

    fig, axes = plt.subplots(
        1,
        num_panels,
        sharex=True,
        sharey=True,
        figsize=(5 * num_panels, 5),
        squeeze=False,
    )
    axes = axes[0]

    _plot_field(
        fig,
        axes[0],
        state_pred,
        latitude,
        longitude,
        cmap=state_cmap,
        vmin=state_min,
        vmax=state_max,
    )
    axes[0].set_title(
        f"Predicted: {plot_var_state}\n"
        f"Initial time: {initial_time}\n"
        f"Lead time: {lead_time} h"
    )

    _plot_field(
        fig,
        axes[1],
        state_true,
        latitude,
        longitude,
        cmap=state_cmap,
        vmin=state_min,
        vmax=state_max,
    )
    axes[1].set_title(f"Truth: {plot_var_state}")

    _plot_field(
        fig,
        axes[2],
        state_error,
        latitude,
        longitude,
        cmap=error_cmap,
        vmin=-max_error,
        vmax=max_error,
    )
    axes[2].set_title(f"Error: {plot_var_state}")

    if has_background:
        background = _masked_field(background, background_mask)
        background_values = background.compressed()

        if background_values.size == 0:
            axes[3].set_title(
                f"Background: {plot_var_background}\n(no valid data)"
            )
            axes[3].axis("off")
        else:
            background_min = float(background_values.min())
            background_max = float(background_values.max())

            if background_min == background_max:
                padding = max(
                    abs(background_min) * 1.0e-6,
                    1.0e-6,
                )
                background_min -= padding
                background_max += padding

            _plot_field(
                fig,
                axes[3],
                background,
                latitude,
                longitude,
                cmap=state_cmap,
                vmin=background_min,
                vmax=background_max,
            )
            axes[3].set_title(
                f"Background: {plot_var_background}"
            )

    fig.tight_layout()
    return fig
