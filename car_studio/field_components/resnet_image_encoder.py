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
import torch.nn.functional as F
from nerfstudio.field_components.base_field_component import FieldComponent
from torchvision.models import resnet34


class ResnetImageEncoder(FieldComponent):
    """Resnet image encoder for extract latent codes"""
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)

    def forward(self, in_tensor):
        """
        Extract feature pyramid from image. See Section 4.1., Section B.1 in the
        Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.
        x : B x 3 x 128 x 128
        """

        x = self.resnet.conv1(in_tensor)  # x : B x 64 x 64 x 64
        x = self.resnet.bn1(x)  # x : B x 64 x 64 x 64
        feats1 = self.resnet.relu(x)  # feats1 : B x 64 x 64 x 64

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))  # x : B x 64 x 32 x 32
        feats3 = self.resnet.layer2(feats2)  # x : B x 128 x 16 x 16
        feats4 = self.resnet.layer3(feats3)  # x : B x 256 x 8 x 8

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i, _ in enumerate(latents):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",
                align_corners=True,  # latents[0~3]: B x [64, 64, 128, 256] x 64 x 64
            )

        latents = torch.cat(latents, dim=1)  # B x 512 x 64 x 64
        return F.max_pool2d(latents, kernel_size=latents.size()[2:])[:, :, 0, 0]  # B x 512