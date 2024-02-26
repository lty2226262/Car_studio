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

import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (FieldHead, FieldHeadNames,
                                                     RGBFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from torch import Tensor, nn


class AutoRFField(NeRFField):
    """AutoRF Field

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
                  feat_mlp_layer_width: int = 128,
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
        self.fc_geo = MLP(in_dim=512, out_dim=feat_mlp_layer_width,
                            activation=nn.ReLU(),
                            num_layers=1, layer_width=0)
        self.fc_z_geo = nn.Linear(feat_mlp_layer_width, 
                                  base_mlp_layer_width)
        self.fc_tex = MLP(in_dim=512, out_dim=feat_mlp_layer_width,
                                    activation=nn.ReLU(),
                                    num_layers=1, layer_width=0)        
        self.fc_z_tex = MLP(in_dim=feat_mlp_layer_width, out_dim=base_mlp_layer_width,
                        activation=nn.ReLU(),
                        num_layers=1, layer_width=0)
        self.head_mlp_feat = MLP(in_dim=base_mlp_layer_width, 
                                 out_dim=base_mlp_layer_width,
                                 num_layers=1,
                                 layer_width=0
                                 )
        self.head_fc_dir = MLP(in_dim=self.direction_encoding.get_out_dim(),
                               out_dim=base_mlp_layer_width,
                               num_layers=1,
                               layer_width=0
                               )
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.mlp_base = None

        self.blocks = nn.ModuleList([
            nn.Linear(base_mlp_layer_width,
                      base_mlp_layer_width) for i in range(base_mlp_num_layers - 1)
        ])
        self.skips = skip_connections
        n_skips = sum([i in skip_connections for i in range(base_mlp_num_layers - 1)])

        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(feat_mlp_layer_width, base_mlp_layer_width) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(self.position_encoding.get_out_dim(),
                          base_mlp_layer_width) for i in range(n_skips)
            ])

        self.fc_in = nn.Linear(self.position_encoding.get_out_dim(),
                              base_mlp_layer_width)
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        latents = ray_samples.metadata['latents']
        z_geo = self.fc_geo(latents)
        assert not self.use_integrated_encoding
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        encoded_xyz = self.position_encoding(positions)
        base_mlp_xyz =  self.fc_in(encoded_xyz) + self.fc_z_geo(z_geo)
        base_mlp_out = F.relu(base_mlp_xyz)
        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            base_mlp_out = F.relu(layer(base_mlp_out))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                base_mlp_out = base_mlp_out + self.fc_z_skips[skip_idx](z_geo)
                base_mlp_out = base_mlp_out + self.fc_p_skips[skip_idx](encoded_xyz)
                skip_idx += 1

        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        latents = ray_samples.metadata['latents']
        z_tex = self.fc_tex(latents)

        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)

        density_embedding = self.head_mlp_feat(density_embedding)
        density_embedding = density_embedding + \
                            self.fc_z_tex(z_tex) + \
                            self.head_fc_dir(encoded_dir)

        outputs = {}
        for field_head in self.field_heads:
            mlp_out = self.mlp_head(density_embedding)
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
