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

import random
from typing import Dict, Optional, Union

import torch
from jaxtyping import Int
from nerfstudio.data.pixel_samplers import PixelSampler
from torch import Tensor


class CustomPixelSampler(PixelSampler):
    """Custom Pixel Sampler inhereit from PixelSampler"""
    def __init__(self, num_rays_per_batch: int, n_instance_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        super().__init__(num_rays_per_batch, keep_full_image, **kwargs)
        self.n_instance_per_batch = n_instance_per_batch

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if \
            key not in ["image_idx", "patch", "object_id"] and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch * self.n_instance_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        collated_batch["original_indices"] = c # to save memory for encoder

        to_process_keys = ["patch", "object_id"]

        for key in to_process_keys:
            assert key in batch.keys(), f'key {key} not in the batch'
            c_set, inverse_indices = torch.unique(c, return_inverse=True)
            key_set = {}
            for idx, i in enumerate(c_set):
                first_idx = ((inverse_indices == idx).nonzero(as_tuple=True)[0])[0]
                frame_index = c[first_idx].item()
                new_value = batch[key][frame_index] if isinstance(batch[key],
                                                              list) else batch[key][frame_index, ...]
                key_set[i.item()] = new_value
            # assert c.max() == c.min()
            # frame_index = c[0].item()
            # new_value = batch[key][frame_index] if isinstance(batch[key],
            #                                                   list) else batch[key][frame_index, ...]
            collated_batch[key] = key_set

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
    
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels from one possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """

        assert isinstance(mask, torch.Tensor)
        select_frames = random.sample(range(0,mask.shape[0]),
                                      k = self.n_instance_per_batch)
        indices_list = []
        for select_frame in select_frames:
            nonzero_indices = torch.nonzero(mask[select_frame, ..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            indices_frame = torch.full((batch_size, 1), select_frame, device=nonzero_indices.device)
            indices = nonzero_indices[chosen_indices]
            indices = torch.concat([indices_frame, indices], dim=-1)
            indices_list.append(indices)
        final_indices = torch.concat(indices_list, dim=0)

        return final_indices

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch
