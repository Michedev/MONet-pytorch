from typing import List, Literal, Union, Dict

import torch
from torch import nn as nn


class BroadcastDecoderNet(nn.Module):
    def __init__(
            self,
            w_broadcast: int,
            h_broadcast: int,
            net: Union[torch.nn.Sequential, torch.nn.Module]
            ):
        super().__init__()
        self.w_broadcast = w_broadcast
        self.h_broadcast = h_broadcast
        self.net = net

        ys = torch.linspace(-1, 1, self.h_broadcast)
        xs = torch.linspace(-1, 1, self.w_broadcast)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        batch_size = z.shape[0]
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).expand(batch_size, z.shape[1], self.h_broadcast, self.w_broadcast)
        coord_map = self.coord_map_const.expand(batch_size, 2, self.h_broadcast, self.w_broadcast)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.net(inp)
        return result
