from typing import Union

import torch
from torch import nn as nn
from torch.nn import LogSoftmax

from monet_pytorch.unet import UNet


class AttentionNet(nn.Module):
    def __init__(self, unet: Union[torch.nn.Module, UNet]):
        super().__init__()
        self.unet = unet
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, x, log_scope):
        inp = torch.cat((x, log_scope), 1)
        logits = self.unet(inp)
        log_alpha = self.log_softmax(logits)
        log_mask = log_scope + log_alpha[:, 0:1]  # scope * alpha ~= log_scope + log_alpha
        log_scope += log_alpha[:, 1:2]  # scope * (1-alpha) ~= log_scope * log(1-alpha)
        return log_mask, log_scope
