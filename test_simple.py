import torch
import torch.nn.functional as F
import torch.nn as nn
from depthconv import DepthConvFunction

import time

# TODO:
# - Depth
# - Let bias be None

if __name__ == '__main__':
    print()

    batch_size = 1
    in_channels = 3
    out_channels = 3

    #device = torch.device('cpu')
    device = torch.device('cuda', 0)

    images = torch.rand((batch_size, in_channels, 512, 512)).to(device)
    depths = torch.ones((batch_size, 1, 512, 512)).to(device)
    weights = nn.Parameter(torch.ones(out_channels, in_channels, 3, 3).to(device))
    bias = nn.Parameter(torch.ones(out_channels).cuda())
    stride = 1
    padding = 0
    dilation = 1
    # TODO: groups

    # CUDA warmup...
    _ = F.conv2d(images, weights, bias=bias, stride=stride, padding=padding, dilation=dilation)

    # PyTorch implementation
    start = time.time()
    torch_output = F.conv2d(images, weights, bias=bias, stride=stride, padding=padding, dilation=dilation)
    print(f'Torch: {time.time() - start}')

    # My implementation
    start = time.time()
    my_output = DepthConvFunction.apply(images, depths, weights, bias, padding, stride, dilation)
    print(f'This implementation: {time.time() - start}')

    print()
    print(torch_output.shape)
    print(my_output.shape)
    print(f'Are the two outputs equal? {torch.allclose(torch_output.cpu(), my_output.cpu())}')
    print()
