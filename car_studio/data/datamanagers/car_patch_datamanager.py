# 2023 Tianyu LIU [tliubk@connect.ust.hk] Copyright. All rights reserved.#

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

"""
Datamanager.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, List, Literal, Optional,
                    Tuple, Type, Union)

import torch
import torchvision.transforms as T
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager, VanillaDataManagerConfig)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import (EquirectangularPixelSampler,
                                            PatchPixelSampler, PixelSampler)
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.rich_utils import CONSOLE
from torch.nn import Parameter
from typing_extensions import TypeVar

from car_studio.data.custom_pixel_samplers import CustomPixelSampler
from car_studio.data.utils.dataloaders import (
    CustomFixedIndicesEvalDataloader, CustomRandIndicesEvalDataloader)
from car_studio.model_components.custom_ray_generators import \
    CustomRayGenerator


def custom_variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for data in batch:
        image = data.pop('image')
        if 'corresponding_image' in data.keys():
            image = data.pop('corresponding_image')
            src_img_to_vis = image.detach().clone()
            src_img_shape = src_img_to_vis.shape[:2]
            src_img_canvas = torch.zeros((900, 1600, 3))
            src_img_canvas[:src_img_shape[0], :src_img_shape[1], :] = src_img_to_vis

        mask = data.pop('mask')
        src_img = data.pop('src_img')
        xyxy_msk = data.pop('xyxy_mask').to(torch.int)
        xmin, ymin, xmax, ymax = xyxy_msk
        image_canvas = torch.zeros((900, 1600, 3))
        image_canvas[ymin:ymax+1, xmin:xmax+1, :] = src_img[ymin:ymax+1, xmin:xmax+1,:]
        mask_canvas = torch.zeros((900, 1600, 1), dtype=torch.bool)
        mask_canvas[ymin:ymax+1, xmin:xmax+1, :] = True
        patch_mask_canvas = torch.zeros((900, 1600, 1), dtype=torch.bool)
        patch_mask_canvas[ymin:ymax+1, xmin:xmax+1, :] = mask
        data['mask'] = mask_canvas
        data['image'] = image_canvas
        data['patch'] = img_transform(image.permute(2,0,1)).permute(1,2,0)
        data['patch_mask'] = patch_mask_canvas

    new_batch = nerfstudio_collate(batch)

    return new_batch

def variable_whole_img_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the whole img dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for data in batch:
        image = data.pop('image')
        mask = data.pop('mask')
        src_img = data.pop('src_img')
        xyxy_msk = data.pop('xyxy_mask').to(torch.int)
        xmin, ymin, xmax, ymax = xyxy_msk
        image_canvas = torch.zeros((900, 1600, 3))
        image_canvas[ymin:ymax+1, xmin:xmax+1, :] = src_img[ymin:ymax+1, xmin:xmax+1,:]
        mask_canvas = torch.zeros((900, 1600, 1), dtype=torch.bool)
        mask_canvas[ymin:ymax+1, xmin:xmax+1, :] = True
        patch_mask_canvas = torch.zeros((900, 1600, 1), dtype=torch.bool)
        src_img_canvas = torch.zeros((900, 1600, 3))
        src_img_canvas[:src_img.shape[0], :src_img.shape[1], :] = src_img
        patch_mask_canvas[ymin:ymax+1, xmin:xmax+1, :] = mask
        data['patch'] = img_transform(image.permute(2,0,1)).permute(1,2,0)
        data['patch_mask'] = patch_mask_canvas
        data['xyxy_msk'] = xyxy_msk
        data['src_img'] = src_img_canvas

    new_batch = nerfstudio_collate(batch)

    return new_batch

@dataclass
class CarPatchDataManagerConfig(VanillaDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: CarPatchDataManager)
    """Target class to instantiate."""
    collate_fn: Callable[[Any], Any] = staticmethod(custom_variable_res_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    collate_whole_fn: Callable[[Any], Any] = staticmethod(variable_whole_img_res_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as 
    for data from Record3D."""
    n_instance_per_batch: int = 3
    """Specifies the number of instances sampled within a batch"""
    

TDataset_type = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

class CarPatchDataManager(DataManager, Generic[TDataset_type]):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: CarPatchDataManagerConfig
    train_dataset: TDataset_type
    eval_dataset: TDataset_type
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: CarPatchDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.dataset_type: Type[TDataset_type] = kwargs.get("_dataset_type", getattr(TDataset_type, "__default__"))
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        super().__init__()

    def __class_getitem__(cls, item):
        return type(
            cls.__name__,
            (cls,),
            {"__module__": cls.__module__, "__init__": functools.partialmethod(cls.__init__, _dataset_type=item)},
        )

    def create_train_dataset(self) -> TDataset_type:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset_type:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.eval_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: TDataset_type, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return CustomPixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        assert self.config.train_num_images_to_sample_from < len(self.train_dataset), \
            f'please keep the train_num_images_to_sample_from smaller than {len(self.train_dataset)}'
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch,
                                                           self.config.n_instance_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = CustomRayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        assert self.config.eval_num_images_to_sample_from <= len(self.eval_dataset), \
            f'please keep the eval_num_images_to_sample_from smaller than {len(self.eval_dataset)}\n. \
                Be careful that if you set to the max number it will avoid random selecting'
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch,
                                                          1)
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )
        self.eval_ray_generator = CustomRayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.eval_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = CustomFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            collate_fn=self.config.collate_whole_fn,
        )
        self.eval_dataloader = CustomRandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            collate_fn=self.config.collate_whole_fn,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            assert image_idx == int(batch['image_idx']), 'the idx does not consistent'
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups
    