# 2023 Tianyu LIU [tliubk@connect.ust.hk] Copyright. All rights reserved.
# modified from https://github.com/sxyu/pixel-nerf/blob/master/src/model/resnetfc.py
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
from torch import Tensor, nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx

class ResnetFC(FieldComponent):
    """Resnet image encoder for extract latent codes"""
    def __init__(self, d_in: int, d_out:int = 4, n_blocks=5, d_latent=0,
                 d_hidden=128, beta=0.0, combine_layer_blocks=3,):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        self.lin_in = nn.Linear(d_in, d_hidden)
        nn.init.constant_(self.lin_in.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode='fan_in')

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a= 0, mode='fan_in')

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.combine_layer = combine_layer_blocks

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]
        )

        n_lin_z = min(combine_layer_blocks, n_blocks)
        self.lin_z = nn.ModuleList(
            [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
        )
        for i in range(n_lin_z):
            nn.init.constant_(self.lin_z[i].bias, 0.0)
            nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, in_tensor: Tensor):
        zx = in_tensor
        assert zx.size(-1) == self.d_latent + self.d_in
        z = zx[..., :self.d_latent]
        x = zx[..., self.d_latent:]

        x = self.lin_in(x)

        for blkid in range(self.n_blocks):
            if self.d_latent > 0 and blkid < self.combine_layer:
                tz = self.lin_z[blkid](z)
                x = x + tz

            x = self.blocks[blkid](x)
        out = self.lin_out(self.activation(x))
        return out