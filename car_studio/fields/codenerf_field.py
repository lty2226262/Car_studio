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
from nerfstudio.field_components.field_heads import (DensityFieldHead,
                                                     FieldHead, FieldHeadNames,
                                                     RGBFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor, nn


class CodeNerfField(Field):
    """Code Nerf Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """
    def __init__(self, position_encoding: Encoding = Identity(in_dim=3),
                  direction_encoding: Encoding = Identity(in_dim=3),
                  base_mlp_num_layers: int = 3,
                  base_mlp_layer_width: int = 256,
                  head_mlp_num_layers: int = 1,
                  head_mlp_layer_width: int = 256,
                  latent_width: int = 512,
                  field_heads: Optional[Tuple[FieldHead]] = (RGBFieldHead(),),
                  use_integrated_encoding: bool = False,
    )->None:
        super().__init__()
        self.base_mlp_num_layers = base_mlp_num_layers
        self.head_mlp_num_layers = head_mlp_num_layers
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding       
        self.latent_width = int(latent_width / 2)
        self.encoding_xyz = MLP(in_dim=position_encoding.get_out_dim(),
                                num_layers=1,
                                layer_width=base_mlp_layer_width,
                                out_activation=nn.ReLU())
        self.use_integrated_encoding = use_integrated_encoding
        for i in range(base_mlp_num_layers):
            layer = MLP(in_dim=self.latent_width,
                        num_layers=1,
                        layer_width=base_mlp_layer_width,
                        out_activation=nn.ReLU())
            setattr(self, f'shape_latent_layer_{i+1}', layer)
            layer = MLP(in_dim=base_mlp_layer_width,
                        num_layers=1,
                        layer_width=base_mlp_layer_width,
                        out_activation=nn.ReLU())
            setattr(self, f'shape_layer_{i+1}', layer)
        self.encoding_shape = nn.Linear(base_mlp_layer_width, base_mlp_layer_width)
        self.field_output_density = DensityFieldHead(in_dim=base_mlp_layer_width)
        self.encoding_viewdir = MLP(in_dim=(direction_encoding.get_out_dim() + self.latent_width),
                                    num_layers=1,
                                    layer_width=base_mlp_layer_width,
                                    out_activation=nn.ReLU())
        for i in range(head_mlp_num_layers):
            layer = MLP(in_dim=self.latent_width,
                        num_layers=1,
                        layer_width=head_mlp_layer_width,
                        out_activation=nn.ReLU())
            setattr(self, f'texture_latent_layer_{i+1}', layer)
            layer = MLP(in_dim=head_mlp_layer_width,
                        num_layers=1,
                        layer_width=head_mlp_layer_width,
                        out_activation=nn.ReLU())
            setattr(self, f'texture_layer_{i+1}', layer)
        
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(head_mlp_layer_width)
        
        
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        latents = ray_samples.metadata['latents']
        latent_dim = latents.shape[-1]
        assert self.latent_width * 2 == latent_dim
        shape_latent = latents[..., :self.latent_width]

        assert not self.use_integrated_encoding
        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)

        y = self.encoding_xyz(encoded_xyz)
        for i in range(self.base_mlp_num_layers):
            z = getattr(self, f'shape_latent_layer_{i+1}')(shape_latent)
            y = y + z
            y = getattr(self, f'shape_layer_{i+1}')(y)
        base_mlp_out = self.encoding_shape(y)
        density = self.field_output_density(y)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        latents = ray_samples.metadata['latents']
        latent_dim = latents.shape[-1]
        assert self.latent_width * 2 == latent_dim
        texture_latent = latents[..., self.latent_width:]

        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            y = torch.cat([density_embedding, encoded_dir], dim=-1)
            y = self.encoding_viewdir(y)
            for i in range(self.head_mlp_num_layers):
                z = getattr(self, f'texture_latent_layer_{i+1}')(texture_latent)
                y = y + z
                y = getattr(self, f'texture_layer_{i+1}')(y)
            outputs[field_head.field_head_name] = field_head(y)
        return outputs
