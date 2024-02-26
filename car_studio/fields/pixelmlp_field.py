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
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from torch import Tensor, nn


class PixelMLPField(NeRFField):
    """Pixel MLP Field

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
                  base_mlp_num_layers: int = 8,
                  base_mlp_layer_width: int = 256,
                  head_mlp_num_layers: int = 1,
                  head_mlp_layer_width: int = 128,
                  latent_width: int = 512,
                  skip_connections: Tuple[int] = (4,),
                  field_heads: Optional[Tuple[FieldHead]] = (RGBFieldHead(),),
                  use_integrated_encoding: bool = False,
                  spatial_distortion: Optional[SpatialDistortion] = None,
    )->None:
        super().__init__(position_encoding,
                        direction_encoding,
                        base_mlp_num_layers,
                        base_mlp_layer_width,
                        head_mlp_num_layers,
                        head_mlp_layer_width,
                        skip_connections,
                        field_heads,
                        use_integrated_encoding,
                        spatial_distortion)
        self.latent_width = latent_width

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim() + latent_width,
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU()
        )

        self.mlp_head = MLP(
            in_dim=base_mlp_layer_width + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=base_mlp_layer_width)
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(head_mlp_layer_width)  # type: ignore
        
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        latents = ray_samples.metadata['latents']
        assert not self.use_integrated_encoding
        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        base_mlp_in = torch.cat((encoded_xyz, latents), dim=-1)
        base_mlp_out = self.mlp_base(base_mlp_in)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
