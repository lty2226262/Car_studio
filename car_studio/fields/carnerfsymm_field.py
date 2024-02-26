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

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import (Encoding, Identity,
                                                   SHEncoding)
from nerfstudio.field_components.field_heads import (DensityFieldHead,
                                                     FieldHead, FieldHeadNames,
                                                     RGBFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class CarNeRFSymmField(Field):
    """CarNeRFSymm Field

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
    def __init__(self, 
                 aabb:Tensor,
                 feature_encoding: Encoding = Identity(in_dim=3),
                 direction_encoding: Encoding = Identity(in_dim=3),
                 density_encoding: Encoding = Identity(in_dim=3),
                 color_encoding: Encoding = Identity(in_dim=3),
                 appearance_dim: int = 27,
                 head_mlp_num_layers: int = 2,
                 head_mlp_layer_width: int = 128,
                 use_integrated_encoding: bool = False,
                 spatial_distortion: Optional[SpatialDistortion] = None,
    )->None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.density_encoding = density_encoding
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.color_encoding = color_encoding

        self.mlp_head = MLP(
            in_dim=appearance_dim + 3 + \
                self.direction_encoding.get_out_dim() + \
                self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )
        self.B = nn.Linear(in_features=self.color_encoding.get_out_dim(), out_features=appearance_dim, bias=False)
        

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(),
                                             activation=nn.Sigmoid())

        # self.use_integrated_encoding = use_integrated_encoding
        # self.spatial_distortion = spatial_distortion

        # self.fc_geo = MLP(in_dim=512, out_dim=self.density_encoding.get_out_dim(),
        #                     activation=nn.ReLU(),
        #                     num_layers=1,
        #                     layer_width=0,
        #                     out_activation=nn.ReLU())
        # self.appearance_dim = appearance_dim        
        # self.density_head = DensityFieldHead(in_dim= appearance_dim)
        
        # self.fc_tex = MLP(in_dim=512,
        #                   out_dim=self.density_encoding.get_out_dim(),
        #                   activation=nn.ReLU(),
        #                   num_layers=1,
        #                   layer_width=0,
        #                   out_activation=nn.ReLU())
        # self.B = nn.Linear(in_features=self.density_encoding.get_out_dim() + \
        #                    self.density_encoding.get_out_dim(),
        #                    out_features=appearance_dim,
        #                    bias=False)
        # self.fc_z_tex = MLP(in_dim=self.density_encoding.get_out_dim(),
        #                     out_dim=self.appearance_dim,
        #                     activation=nn.ReLU(),
        #                     num_layers=1,
        #                     layer_width=0,
        #                     out_activation=nn.ReLU())
        # self.mlp_feat = MLP(
        #     in_dim=appearance_dim,
        #     out_dim=appearance_dim,
        #     num_layers=1,
        #     layer_width=appearance_dim,
        #     activation=nn.ReLU(),
        #     out_activation=nn.ReLU())
        
        # self.mlp_head = MLP(
        #     in_dim=appearance_dim * 3,
        #     out_dim=appearance_dim,
        #     num_layers=head_mlp_num_layers,
        #     layer_width=appearance_dim,
        #     activation=nn.ReLU(),
        #     out_activation=nn.ReLU(),
        # )
        # self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(),
        #                                      activation=nn.Sigmoid())
        # self.head_dir_encoded = MLP(
        #     in_dim=self.direction_encoding.n_output_dims,
        #     out_dim=self.appearance_dim,
        #     num_layers=1,
        #     layer_width=self.appearance_dim,
        #     activation=nn.ReLU(),
        #     out_activation=nn.ReLU(),
        # )
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor]:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        density = self.density_encoding(positions)
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc
        # latents = ray_samples.metadata['latents']
        # geo_latents = self.fc_geo(latents)
        # positions = ray_samples.frustums.get_positions()
        # normalized_pos = SceneBox.get_normalized_positions(positions, self.aabb)
        
        # density = self.density_encoding(normalized_pos)
        # density_cat = torch.cat([geo_latents, density],dim=-1)

        # density_embed = self.B(density_cat)

        # density_enc = self.density_head(density_embed)
        # return density_enc, density_embed, geo_latents

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None,
        geo_embedding: Optional[Tensor] = None
    ) -> Tuple[Tensor]:
    # ) -> Dict[FieldHeadNames, Tensor]:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        rgb_features = self.color_encoding(positions)
        rgb_features = self.B(rgb_features)

        d_encoded = self.direction_encoding(d)
        rgb_features_encoded = self.feature_encoding(rgb_features)

        # if self.use_sh:
        #     sh_mult = self.sh(d)[:, :, None]
        #     rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
        #     rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        # else:
        out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
        rgb = self.field_output_rgb(out)

        return rgb
        # latents = ray_samples.metadata['latents']
        # z_tex = self.fc_tex(latents)
        # # geo_latents = geo_embedding

        # dir = ray_samples.frustums.directions
        # encoded_dir = self.direction_encoding(dir.reshape(-1, 3)).reshape(*dir.shape[:-1],-1).to(torch.float32)
        # density_embedding = self.mlp_feat(density_embedding)
        # density_embedding = density_embedding + \
        #                     self.fc_z_tex(z_tex) + \
        #                     self.head_dir_encoded(encoded_dir)
        # rgb = self.field_output_rgb(density_embedding)
        # # # positions = ray_samples.frustums.get_positions()
        
        # # # rgb_features = self.color_encoding(positions)
        # # # rgb_features = self.B(torch.cat([geo_latents,
        # # #                                  rgb_features], dim=-1))
        
        # # rgb_features = self.mlp_feat(density_embedding) # appearance_dim
        # # tex_features = self.fc_z_tex(z_tex) # self.density_encoding.get_out_dim()
        
        # # d_encoded = torch.cat([
        # #     , # self.density_encoding.get_out_dim()
        # #     dir],dim=-1)
        # # dir_features = self.head_dir_encoded(d_encoded)
        # # # rgb_features_encoded = self.feature_encoding(rgb_features.reshape(-1, 
        

        # # out = self.mlp_head(torch.cat([
        # #                             #   geo_latents,
        # #                               rgb_features,
        # #                               tex_features,
        # #                             #   rgb_features,
        # #                             #   rgb_features_encoded,
        # #                               dir_features], dim=-1))
        # # rgb = self.field_output_rgb(out)
        
        # return rgb
        

    def forward(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        # if compute_normals:
        #     with torch.enable_grad():
        #         density, density_embedding = self.get_density(ray_samples)
        # else:
        density= self.get_density(ray_samples)
        rgb = self.get_outputs(ray_samples, None)
        # rgb = self.get_outputs(ray_samples, None)
        return {FieldHeadNames.DENSITY: density,
                FieldHeadNames.RGB: rgb}

#             density, density_embedding = self.get_density(ray_samples)

#         field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
# ````        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
# ````
#         if compute_normals:
#             with torch.enable_grad():
#                 normals = self.get_normals()
#             field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
#         return field_outputs