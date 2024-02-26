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

import torch
from jaxtyping import Int
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.math import intersect_aabb
from torch import Tensor


class CustomRayGenerator(RayGenerator):
    """ray generator with a specific camera pose"""

    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        c = ray_indices[:, 0] # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        camera_opt_to_camera = self.pose_optimizer(c)

        # set bounding box for each object
        scene_box = Tensor([-1., -1., -1., 1., 1., 1.]).to(self.cameras.device)

        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=camera_opt_to_camera,
            disable_distortion=True,
            keep_shape=False,
        )
        with torch.no_grad():
            t_min, t_max = intersect_aabb(ray_bundle.origins,
                                          ray_bundle.directions,
                                          scene_box)
            t_min = t_min.reshape([-1, 1])
            t_max = t_max.reshape([-1, 1])

            ray_bundle.nears = t_min
            ray_bundle.fars = t_max

        return ray_bundle
