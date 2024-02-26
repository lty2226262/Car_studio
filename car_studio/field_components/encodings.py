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

from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from nerfstudio.field_components.encodings import Encoding
from torch import Tensor, nn


class TriplaneZSymmEncoding(Encoding):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        assert self.resolution % 2 == 0
        
        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((4, self.num_components, self.resolution, int(self.resolution / 2)))
        )   # half x-y, another half x-y, x-z, y-z

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs num_components featuresize"]:
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        xy_plane_coef = torch.cat([self.plane_coef[0], self.plane_coef[1]], dim=-1).unsqueeze(0)
        xz_plane_coef_half = self.plane_coef[2]
        xz_plane_coef = torch.cat([xz_plane_coef_half, torch.flip(xz_plane_coef_half, [-1])], dim=-1).unsqueeze(0)
        yz_plane_coef_half = self.plane_coef[3]
        yz_plane_coef = torch.cat([yz_plane_coef_half, torch.flip(yz_plane_coef_half, [-1])], dim=-1).unsqueeze(0)
        
        plane_coef = torch.cat([xy_plane_coef, xz_plane_coef, yz_plane_coef], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        # if self.reduce == "product":
        #     plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        # else:
        #     plane_features = plane_features.sum(0).squeeze(-1).T
        reshaped_plane_features = plane_features.reshape(3, *original_shape[:-1], self.num_components)
        result_plane_features = torch.cat(torch.split(reshaped_plane_features, 1), dim=-1).squeeze(0)
        return result_plane_features

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution

