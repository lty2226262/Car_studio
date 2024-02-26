# 2023 Tianyu LIU [tliubk@connect.ust.hk] Copyright. All rights reserved.
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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def get_image_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read an image from the given path and return a boolean tensor
    """
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil = pil_image.resize(newsize, resample=Image.NEAREST)
    image_tensor = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(image_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return image_tensor