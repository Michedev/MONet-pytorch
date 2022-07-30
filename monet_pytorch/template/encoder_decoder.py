from typing import List, Literal, Union, Dict

import torch
from torch import nn as nn
from monet_pytorch.nn_utils import get_activation_module
from monet_pytorch.template.sequential_cnn import make_sequential_cnn_from_config


class EncoderNet(nn.Module):

    def __init__(
            self,
            width: int,
            height: int,
            input_channels: int,
            activations: Literal['relu', 'leakyrelu', 'elu'],
            channels: List[int],
            batchnorms: List[bool],
            bn_affines: List[bool],
            kernels: List[int],
            strides: List[int],
            paddings: List[int],
            mlp_hidden_size: int,
            mlp_output_size: int,
            ):
        super().__init__()
        self.width = width
        self.height = height
        self.input_channels = input_channels
        self.activations = activations
        self.channels = channels
        self.batchnorms = batchnorms
        self.bn_affines = bn_affines
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_output_size = mlp_output_size
        self.convs, params = make_sequential_cnn_from_config(self.input_channels, self.channels, self.kernels,
                                                             self.batchnorms, self.bn_affines, self.paddings,
                                                             self.strides, self.activations, return_params=True)
        width = self.width
        height = self.height
        for kernel, stride, padding in zip(params['kernels'], params['strides'], params['paddings']):
            width = (width + 2 * padding - kernel) // stride + 1
            height = (height + 2 * padding - kernel) // stride + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.channels[-1] * width * height, self.mlp_hidden_size),
            get_activation_module(self.activations, try_inplace=True),
            nn.Linear(self.mlp_hidden_size, self.mlp_output_size)
        )

    def forward(self, x):
        x = self.convs(x).flatten(1)
        x = self.mlp(x)
        return x


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
