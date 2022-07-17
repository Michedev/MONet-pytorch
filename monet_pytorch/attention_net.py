import torch
from torch import nn as nn
from torch.nn import LogSoftmax

from monet_pytorch.unet import UNet


class AttentionNet(nn.Module):
    def __init__(self, input_channels: int, num_blocks: int, filters: int = 32, mlp_size: int = 128):
        super().__init__()
        self.unet = UNet(input_channels, num_blocks, filters, mlp_size)
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, x, log_scope):
        inp = torch.cat((x, log_scope), 1)
        logits = self.unet(inp)
        #       logits = self.last_conv(logits)
        log_alpha = self.log_softmax(logits)
        log_mask = log_scope + log_alpha[:, 0:1]  # scope * alpha ~= log_scope + log_alpha
        log_scope += log_alpha[:, 1:2]  # scope * (1-alpha) ~= log_scope * log(1-alpha)
        return log_mask, log_scope
