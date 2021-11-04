import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import numpy as np


class DepthConv(nn.Module):
    @staticmethod
    def fd(p0, k):
        return torch.exp(-8.3 * torch.abs(p0 - k))

    @staticmethod
    def functional(data, depth, weight, bias=None, stride=1, padding=0, dilation=0):
        batch_size = data.shape[0]
        kernel_size = (weight.shape[2], weight.shape[3])
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        data_height = data.shape[2]
        data_width = data.shape[3]
        output_height = int(int(data_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
        output_width = int(int(data_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1

        unfold = nn.Unfold(kernel_size, dilation=dilation, padding=padding, stride=stride)
        fold = nn.Fold((output_height, output_width), 1, dilation=dilation, padding=padding, stride=stride)

        # (batch, in_channels * kernel_size, blocks)
        blocks = unfold(data)

        # (batch, 1 * kernel_size, blocks)
        d_blocks = unfold(depth)
        d_kernel_center = d_blocks.shape[1] // 2
        for i in range(batch_size):
            d_blocks[i, :, :] = DepthConv.fd(d_blocks[i, d_kernel_center, :], d_blocks[i, :, :])

        # (batch, out_channels, blocks)
        output = torch.zeros((batch_size, weight.shape[0], blocks.shape[2]))
        
        for i in range(batch_size):
            for channel in range(weight.shape[0]):
                this_block = blocks[i, :, :]
                this_weight = weight[channel].reshape(-1, 1).squeeze()
                this_depth = d_blocks[i, :, :].repeat(data.shape[1], 1)

                result = torch.matmul(this_weight, torch.mul(this_block, this_depth))
                output[i, channel] = result

        return fold(output)


    def __init__(self, stride=1, padding=0, dilation=0, bias=None):
        super(DepthConv, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # TODO: groups

    def forward(self, data, depth, weight, bias=None):
        self.save_for_backward(data, depth, weight, bias)

    def backward(self):
        pass
