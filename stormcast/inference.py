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

import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import hydra
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig
from physicsnemo.core import Module

from datasets import dataset_classes
from utils.io import (
    init_inference_results_zarr,
    write_inference_results_zarr,
    save_inference_results_netcdf,
)
from utils.nn import build_network_condition_and_target, diffusion_model_forward
from utils.plots import inference_plot


@hydra.main(version_base=None, config_path="config", config_name="stormcast_inference")
def main(cfg: DictConfig):
    # Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    seed = int(cfg.inference.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    initial_time = datetime.fromisoformat(cfg.inference.initial_time)
    n_steps = cfg.inference.n_steps

    # Dataset prep
    dataset_cls = dataset_classes[cfg.dataset.name]
    dataset = dataset_cls(cfg.dataset, train=False)

    background_channels = dataset.background_channels()
    state_channels = dataset.state_channels()

    invariant_array = dataset.get_invariants()

    if invariant_array is None:
        invariant_tensor = None
    else:
        invariant_tensor = torch.as_tensor(
            invariant_array,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

    if len(cfg.inference.output_state_channels) == 0:
        output_state_channels = state_channels.copy()
    else:
        output_state_channels = cfg.inference.output_state_channels

    vardict_state: dict[str, int] = {
        state_channel: i for i, state_channel in enumerate(state_channels)
    }

    vardict_background = {
        background_channel: i
        for i, background_channel in enumerate(background_channels)
    }

    latitude_getter = getattr(dataset, "latitude", None)
    longitude_getter = getattr(dataset, "longitude", None)

    latitude = (
        np.asarray(latitude_getter(), dtype=np.float32)
        if callable(latitude_getter)
        else None
    )
    longitude = (
        np.asarray(longitude_getter(), dtype=np.float32)
        if callable(longitude_getter)
        else None
    )

    # Load pretrained models
    if "regression" in cfg.model.diffusion_conditions:
        regression_model = (
            Module.from_checkpoint(cfg.inference.regression_checkpoint)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
    else:
        regression_model = None

    diffusion_model = (
        Module.from_checkpoint(cfg.inference.diffusion_checkpoint)
        .eval()
        .requires_grad_(False)
        .to(device)
    )

    # initialize zarr
    (
        group,
        target_group,
        edm_prediction_group,
        noedm_prediction_group,
    ) = init_inference_results_zarr(
        dataset, cfg.inference.rundir, output_state_channels, n_steps
    )

    start_index = dataset.index_for_time(initial_time)

    if start_index + n_steps > len(dataset):
        raise ValueError(
            "Requested forecast extends beyond the dataset manifest."
        )

    state_pred = None
    forecast_mask = None
    val_times = []

    with torch.no_grad():
        for i in range(n_steps):
            data_index = start_index + i

            if i > 0 and dataset.input_time(data_index) != val_times[-1]:
                raise ValueError(
                    "Selected manifest rows are not contiguous in time."
                )
                
            data = dataset[data_index]

            background = torch.as_tensor(
                data["background"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            input_state = torch.as_tensor(
                data["state"][0],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            target_state = torch.as_tensor(
                data["state"][1],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            input_mask = torch.as_tensor(
                data["input_mask"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            target_mask = torch.as_tensor(
                data["target_mask"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            if state_pred is None:
                state_pred = input_state
                forecast_mask = input_mask.clone() # input mask is used throughout autoregressive rollout

            lead_time_label = data.get("lead_time_label")
            if lead_time_label is not None:
                lead_time_label = torch.as_tensor(
                    lead_time_label,
                    dtype=torch.int64,
                    device=device,
                ).unsqueeze(0)

            # Build the diffusion condition and generate the regression forecast.
            (condition, _, regression_output) = build_network_condition_and_target(
                background,
                (state_pred, state_pred),
                invariant_tensor,
                lead_time_label=lead_time_label,
                regression_net=regression_model,
                condition_list=cfg.model.diffusion_conditions,
                regression_condition_list=cfg.model.regression_conditions,
                regression_mask=forecast_mask,
            )

            if regression_output is None:
                regression_output = torch.zeros_like(state_pred)

            # build_network_condition_and_target() has already applied forecast_mask
            # to the regression output used for diffusion conditioning.

            # The future validity mask is unavailable during a real forecast,
            # so retain the initial input mask throughout the rollout.
            state_pred_noedm = regression_output.masked_fill(
                forecast_mask == 0,
                0.0,
            )

            diffusion_correction = diffusion_model_forward(
                diffusion_model,
                condition,
                regression_output.shape,
                sampler_args=dict(cfg.sampler.args),
                lead_time_label=lead_time_label,
            )

            state_pred = regression_output + diffusion_correction.float()
            state_pred = state_pred.masked_fill(
                forecast_mask == 0,
                0.0,
            )

            state_pred_edm = state_pred.clone()

            denorm_pred_edm = dataset.denormalize_state(
                state_pred_edm.cpu().numpy()
            )[0]

            denorm_pred_noedm = dataset.denormalize_state(
                state_pred_noedm.cpu().numpy()
            )[0]

            denorm_target = dataset.denormalize_state(
                target_state.cpu().numpy()
            )[0]

            prediction_valid = forecast_mask.cpu().numpy()[0].astype(bool)
            target_valid = target_mask.cpu().numpy()[0].astype(bool)

            denorm_pred_edm[~prediction_valid] = np.nan
            denorm_pred_noedm[~prediction_valid] = np.nan
            denorm_target[~target_valid] = np.nan

            write_inference_results_zarr(
                denorm_pred_edm,
                denorm_pred_noedm,
                denorm_target,
                edm_prediction_group,
                noedm_prediction_group,
                target_group,
                output_state_channels,
                vardict_state,
                i,
            )

            valid_time = dataset.target_time(data_index)
            val_times.append(valid_time)

            forecast_hour = int(
                (valid_time - initial_time).total_seconds() / 3600
            )

            # Variables to plot from this forecast.
            plot_state_variables = list(
                cfg.inference.get("plot_state_variables", [])
            )

            # Keep compatibility with the original single-variable config.
            if not plot_state_variables:
                plot_var_state = cfg.inference.get(
                    "plot_var_state",
                    None,
                )

                if plot_var_state is None:
                    raise ValueError(
                        "Set either inference.plot_state_variables or "
                        "inference.plot_var_state."
                    )

                plot_state_variables = [plot_var_state]

            unknown_variables = [
                variable
                for variable in plot_state_variables
                if variable not in vardict_state
            ]

            if unknown_variables:
                raise ValueError(
                    f"Unknown plotting variables: {unknown_variables}. "
                    f"Available variables: {state_channels}"
                )

            plot_var_background = cfg.inference.get(
                "plot_var_background",
                None,
            )

            background_plot = None
            background_mask_plot = None

            if background_channels and plot_var_background is not None:
                if plot_var_background not in vardict_background:
                    raise ValueError(
                        f"Unknown background plotting variable "
                        f"{plot_var_background!r}. Available variables: "
                        f"{background_channels}"
                    )

                denormalize_background = getattr(
                    dataset,
                    "denormalize_background",
                    None,
                )

                if not callable(denormalize_background):
                    raise AttributeError(
                        "The dataset provides background channels but does not "
                        "implement denormalize_background()."
                    )

                background_arr = denormalize_background(
                    background.detach().float().cpu().numpy()[0]
                )

                varidx_background = vardict_background[
                    plot_var_background
                ]
                background_plot = background_arr[varidx_background]

                background_mask_data = data.get("background_mask")

                if background_mask_data is not None:
                    if torch.is_tensor(background_mask_data):
                        background_mask_arr = (
                            background_mask_data.detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                    else:
                        background_mask_arr = np.asarray(
                            background_mask_data,
                            dtype=np.float32,
                        )

                    if background_mask_arr.ndim == 3:
                        background_mask_plot = background_mask_arr[
                            varidx_background
                        ]
                    elif background_mask_arr.ndim == 2:
                        background_mask_plot = background_mask_arr
                    else:
                        raise ValueError(
                            "background_mask must have shape (C, H, W) "
                            f"or (H, W), got {background_mask_arr.shape}."
                        )

            # The forecast has already been generated for all variables.
            # Only the plotting step is repeated here.
            for plot_var_state in plot_state_variables:
                varidx_state = vardict_state[plot_var_state]

                fig = inference_plot(
                    background_plot,
                    denorm_pred_edm[varidx_state],
                    denorm_target[varidx_state],
                    plot_var_background,
                    plot_var_state,
                    initial_time,
                    forecast_hour,
                    prediction_mask=prediction_valid[varidx_state],
                    truth_mask=target_valid[varidx_state],
                    background_mask=background_mask_plot,
                    latitude=latitude,
                    longitude=longitude,
                )

                fig.savefig(
                    f"{cfg.inference.rundir}/"
                    f"out_{forecast_hour}h_{plot_var_state}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

    save_inference_results_netcdf(
        ds_out_path=cfg.inference.rundir,
        zarr_group=group,
        vertical_vars=cfg.inference.save_vertical_vars,
        level_names=cfg.inference.save_vertical_levels,
        horizontal_vars=cfg.inference.save_horizontal_vars,
        val_times=val_times,
    )


if __name__ == "__main__":
    main()
