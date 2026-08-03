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
import importlib
import pkgutil

from .dataset import StormCastDataset


# Find StormCastDataset implementations in this package and register them
# using "<module>.<class>" keys.
dataset_classes = {}

for module_info in pkgutil.iter_modules(__path__):
    mod_name = module_info.name

    if mod_name == "dataset":
        continue

    module = importlib.import_module(
        f"{__name__}.{mod_name}"
    )

    for name, member in module.__dict__.items():
        if (
            name != "StormCastDataset"
            and isinstance(member, type)
            and issubclass(member, StormCastDataset)
        ):
            dataset_classes[f"{mod_name}.{name}"] = member