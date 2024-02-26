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

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (FieldHead, FieldHeadNames,
                                                     RGBFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor

from car_studio.field_components.resnetfc import ResnetFC


class PixelResnetField(Field):
    """PixelResnet Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        resnet_blocks: Number of layers for resnet blocks.
        latent_width: Width of input latent codes.
        linear_witdh: Width of linear layers.
        combine_layer_blocks: Number of layers for combine layers.
    """
    def __init__(self, position_encoding: Encoding = Identity(in_dim=3),
                  direction_encoding: Encoding = Identity(in_dim=3),
                  resnet_blocks: int = 5,
                  latent_width: int = 512,
                  linear_width: int = 512,
                  combine_layer_blocks: int = 3,
                  out_width: int = 4,
                  field_heads: Optional[Tuple[FieldHead]] = (RGBFieldHead(),),
    )->None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.latent_dim = latent_width
        self.resnet_blocks = resnet_blocks
        self.linear_width = linear_width
        self.combine_layer_blocks = combine_layer_blocks
        self.field_heads = torch.nn.ModuleList(field_heads)
        self.out_width = out_width
        resnet_in_width = position_encoding.get_out_dim() + 3   # pe + viewdirs

        self.resnet_fc = ResnetFC(d_in=resnet_in_width,
                                  d_out=out_width,
                                  n_blocks=resnet_blocks,
                                  d_latent=latent_width,
                                  d_hidden=linear_width,
                                  combine_layer_blocks=combine_layer_blocks)
        
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        latents = ray_samples.metadata['latents']

        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        viewdirs = ray_samples.frustums.directions

        zx = torch.cat((latents, encoded_xyz, viewdirs), dim=-1)
        
        base_mlp_out = self.resnet_fc(in_tensor=zx)
        density = torch.relu(base_mlp_out[..., 3:4])

        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            outputs[field_head.field_head_name] = torch.sigmoid(density_embedding[..., :3])
        return outputs
