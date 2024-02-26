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
class CodeNerfStageOnePipelineConfig(BaseStageTwoPipelineConfig):
    """codenerf stage two pipeline"""

    _target: Type = field(default_factory=lambda: CodeNerfStageOnePipeline)
    """target class to instantiate"""
    latent_dim: int = 512

class CodeNerfStageOnePipeline(BaseStageTwoPipeline):
    """
    CodeNerf pipeline, init latents from pre-train networks
    """
    
    def init_latent_codes(self, device) -> Dict:
        """ init latent codes, 
        """
        
        if self.config.load_pretrain_model_path is not None:
            raise NotImplementedError
        else:
            CONSOLE.print('using empty initialization for code nerf.')
            train_dataset_object_ids = self.datamanager.train_dataset.metadata['object_ids']
            val_dataset_object_ids = self.datamanager.eval_dataset.metadata['object_ids']
            all_objects = set(train_dataset_object_ids + val_dataset_object_ids)
            result = {}
            for iter in all_objects:
                result[iter] = torch.zeros((1, self.config.latent_dim), 
                                           device=device,
                                           requires_grad=True)
            for key, value in result.items():
                self.model.register_buffer('latent_vectors_' + key, value)
        return result
            

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
