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

"""
A pipeline that train the carfnerf for stage two.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional, Type

import torch
import torch.distributed as dist
import tqdm
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from car_studio.data.datamanagers.car_patch_datamanager import (
    CarPatchDataManager, CarPatchDataManagerConfig)
from car_studio.models.car_nerf import CarNerfModelConfig
from car_studio.pipelines.base_stage_two import (BaseStageTwoPipeline,
                                                 BaseStageTwoPipelineConfig)


@dataclass
class CarNerfStageTwoPipelineConfig(BaseStageTwoPipelineConfig):
    """codenerf stage two pipeline"""

    _target: Type = field(default_factory=lambda: CarNerfStageTwoPipeline)
    """target class to instantiate"""
    latent_dim: int = 512

class CarNerfStageTwoPipeline(BaseStageTwoPipeline):
    """
    CarNerf pipeline, init latents from pre-train networks
    """

    @profiler.time_function
    def init_latent_codes(self, device) -> Dict:
        """ init latent codes, 
        """
        
        assert self.config.load_pretrain_model_path is not None
        CONSOLE.print('using encoder initialization for car nerf.')
        train_dataset_object_ids = self.datamanager.train_dataset.metadata['object_ids']
        val_dataset_object_ids = self.datamanager.eval_dataset.metadata['object_ids']
        all_objects = set(train_dataset_object_ids + val_dataset_object_ids)
        result = {}
        idx = 0
        with torch.no_grad():
            loaded_state = torch.load(self.config.load_pretrain_model_path, map_location="cpu")
            self.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            for camera_ray_bundle, batch in self.datamanager.eval_dataloader:
                if idx >= len(self.datamanager.eval_dataloader):
                    break
                assert camera_ray_bundle.camera_indices is not None
                image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
                assert image_idx == int(batch['image_idx']), 'the idx does not consistent'
                patch = batch['patch'][0].permute(2,0,1).unsqueeze(0).to(self._model.device)
                obj_id = batch['object_id'][0]
                latents = self._model.get_latents(patch).detach().clone().to(device)
                if obj_id in result.keys():
                    result[obj_id].append(latents)
                else:
                    result[obj_id] = list([latents])
                idx += 1
            for key, value in result.items():
                values = torch.cat(value, dim=0)
                value_mean = torch.mean(values, dim=0)
                result[key] = value_mean.unsqueeze(0)
                self.model.register_buffer('latent_vectors_' + key, result[key])
        loaded_state = torch.load(self.config.load_pretrain_model_path)
        self.load_pipeline(loaded_state['pipeline'], loaded_state['step'])
        return result
            

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
