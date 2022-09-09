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
from typing import Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "MSRN",
    "msrn_x2", "msrn_x3", "msrn_x4", "msrn_x8",

]


class _MSRB(nn.Module):
    """Multi-scale Residual Block"""

    def __init__(self, channels: int):
        super(_MSRB, self).__init__()
        up_channels = int(2 * channels)
        confusion_channels = int(4 * channels)

        self.conv_3_1 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))
        self.conv_5_1 = nn.Conv2d(channels, channels, (5, 5), (1, 1), (2, 2))
        self.conv_3_2 = nn.Conv2d(up_channels, up_channels, (3, 3), (1, 1), (1, 1))
        self.conv_5_2 = nn.Conv2d(up_channels, up_channels, (5, 5), (1, 1), (2, 2))

        self.confusion = nn.Conv2d(confusion_channels, channels, (1, 1), (1, 1), (0, 0))
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out_3_1 = self.conv_3_1(x)
        out_3_1 = self.relu(out_3_1)
        out_5_1 = self.conv_5_1(x)
        out_5_1 = self.relu(out_5_1)
        out1 = torch.cat((out_3_1, out_5_1), 1)

        out_3_2 = self.conv_3_2(out1)
        out_3_2 = self.relu(out_3_2)
        out_5_2 = self.conv_5_2(out1)
        out_5_2 = self.relu(out_5_2)
        out2 = torch.cat((out_3_2, out_5_2), 1)

        out = self.confusion(out2)
        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class MSRN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int, num_blocks: int = 8):
        super(MSRN, self).__init__()
        self.num_blocks = num_blocks

        # First layer
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))

        # Feature trunk
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_MSRB(64))
        self.trunk = nn.Sequential(*trunk)

        # Trunk final layer
        self.conv2 = nn.Conv2d(576, 64, (1, 1), (1, 1), (0, 0))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif upscale_factor == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv4 = nn.Conv2d(64, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out1 = out

        msrb_out = []
        for i in range(self.num_blocks):
            out1 = self.trunk[i](out1)
            msrb_out.append(out1)
        msrb_out.append(out)
        out = torch.cat(msrb_out, 1)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsampling(out)
        out = self.conv4(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data = torch.mul(0.1, module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def msrn_x2(**kwargs: Any) -> MSRN:
    model = MSRN(upscale_factor=2, **kwargs)

    return model


def msrn_x3(**kwargs: Any) -> MSRN:
    model = MSRN(upscale_factor=3, **kwargs)

    return model


def msrn_x4(**kwargs: Any) -> MSRN:
    model = MSRN(upscale_factor=4, **kwargs)

    return model


def msrn_x8(**kwargs: Any) -> MSRN:
    model = MSRN(upscale_factor=8, **kwargs)

    return model
