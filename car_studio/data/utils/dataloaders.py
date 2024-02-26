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

# for multithreading
import concurrent.futures
import multiprocessing
# for multithreading
import random
from abc import abstractmethod
from typing import Dict, Optional, Sized, Tuple, Union

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import EvalDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import track
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class CustomRandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.
    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __iter__(self):
        return self

    def __next__(self):
        # choose a random image index
        obj_or_img_idx = random.randint(0, self.__len__() - 1)
        ray_bundle, batch = self.get_data_from_image_idx(obj_or_img_idx)
        return ray_bundle, batch
    
    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        obj_or_img_idx = image_idx
        batch = self.input_dataset[obj_or_img_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        true_img_idx = int(batch['image_idx'])
        img_whole_coords = self.cameras.get_image_coords(index=true_img_idx)
        xmin, ymin, xmax, ymax = batch['xyxy_mask'].to(torch.long)
        img_coords = img_whole_coords[ymin:ymax+1, xmin:xmax+1,:]
        ray_bundle = self.cameras.generate_rays(camera_indices=true_img_idx,
                                                coords=img_coords)

        scene_box = Tensor([-1., -1., -1., 1., 1., 1.]).to(self.cameras.device)

        t_min, t_max = intersect_aabb(ray_bundle.origins,
                                        ray_bundle.directions,
                                        scene_box)
        t_min = t_min.unsqueeze(-1)
        t_max = t_max.unsqueeze(-1)

        ray_bundle.nears = t_min
        ray_bundle.fars = t_max

        if self.kwargs['collate_fn'] is not None:
            batch = self.kwargs['collate_fn']([batch])
        return ray_bundle, batch

class CustomFixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        raise StopIteration
    
    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        true_img_idx = int(batch['image_idx'])
        img_whole_coords = self.cameras.get_image_coords(index=true_img_idx)
        xmin, ymin, xmax, ymax = batch['xyxy_mask'].to(torch.long)
        img_coords = img_whole_coords[ymin:ymax+1, xmin:xmax+1,:]
        ray_bundle = self.cameras.generate_rays(camera_indices=true_img_idx,
                                                coords=img_coords)

        scene_box = Tensor([-1., -1., -1., 1., 1., 1.]).to(self.cameras.device)

        t_min, t_max = intersect_aabb(ray_bundle.origins,
                                        ray_bundle.directions,
                                        scene_box)
        t_min = t_min.unsqueeze(-1)
        t_max = t_max.unsqueeze(-1)

        ray_bundle.nears = t_min
        ray_bundle.fars = t_max

        if self.kwargs['collate_fn'] is not None:
            batch = self.kwargs['collate_fn']([batch])
        return ray_bundle, batch
