# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any, List

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "SCN",
    "scn_x2", "scn_x3", "scn_x4",

]

from config import in_channels, upscale_factor


class _ScaleWiseBlock(nn.Module):
    """Scale-wise Convolution Block"""

    def __init__(self, channels: int, width_multiplier: int):
        super(_ScaleWiseBlock, self).__init__()
        midden_channels = int(channels * width_multiplier)

        self.body = nn.Sequential(
            nn.Conv2d(channels, midden_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(midden_channels, channels, (3, 3), (1, 1), (1, 1)),
        )

        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1), (1, 1), (0, 0)),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
        )

        self.up = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1), (1, 1), (0, 0)),
            nn.UpsamplingBilinear2d(scale_factor=2.0),
        )

    def forward(self, x_list: List[Tensor]) -> List[Tensor]:
        body_outputs = [self.body(x) for x in x_list]
        down_outputs = [body_outputs[0]] + [self.down(out) for out in body_outputs[:-1]]
        up_outputs = [self.up(out) for out in body_outputs[1:]] + [body_outputs[-1]]

        out = [x + r + d + u for x, r, d, u in zip(x_list, body_outputs, down_outputs, up_outputs)]

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class SCN(nn.Module):
    def __init__(self, in_channels: int, upscale_factor: int, num_blocks: int = 16):
        super(SCN, self).__init__()
        out_channels = int(in_channels ** upscale_factor)
        self.num_scale = int(math.log(upscale_factor, 2))

        # Skip layer
        self.skip = nn.Conv2d(in_channels, out_channels, (5, 5), (1, 1), (2, 2))

        # First layer
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), (1, 1), (1, 1))

        # Head layer
        self.head = nn.UpsamplingBilinear2d(scale_factor=0.5)

        # Feature trunk
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ScaleWiseBlock(32, 4))
        self.trunk = nn.Sequential(*trunk)

        # Trunk final layer
        self.conv2 = nn.Conv2d(32, out_channels, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        self.upsampling = nn.PixelShuffle(upscale_factor)

        # Initialize all layer
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        out = self.conv1(x)

        out_list = [out]
        for _ in range(4):
            out_list.append(self.head(out_list[-1]))

        out = self.trunk(out_list)
        out = self.conv2(out[0])
        out = torch.add(out, skip)
        out = self.upsampling(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data = torch.mul(0.1, module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def scn_x2(**kwargs: Any) -> SCN:
    model = SCN(upscale_factor=2, **kwargs)

    return model


def scn_x3(**kwargs: Any) -> SCN:
    model = SCN(upscale_factor=3, **kwargs)

    return model


def scn_x4(**kwargs: Any) -> SCN:
    model = SCN(upscale_factor=4, **kwargs)

    return model


x = torch.randn([1, 3, 24, 24])
print(x.shape)
model = scn_x4(in_channels=3)
y = model(x)
print(y)