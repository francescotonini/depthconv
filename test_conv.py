import torch
import torch.nn.functional as F
from conv import DepthConv

import time

if __name__ == '__main__':
    batch_size = 12
    in_channels = 3
    out_channels = 12

    #device = torch.device('cpu')
    device = torch.device('cuda', 0)

    images = torch.rand((batch_size, in_channels, 512, 512)).to(device)
    depths = torch.ones((batch_size, 1, 512, 512)).to(device)
    weights = torch.rand((out_channels, in_channels, 3, 3)).to(device)
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    # TODO: groups

    # CUDA warmup...
    _ = F.conv2d(images, weights, bias=bias, stride=stride, padding=padding, dilation=dilation)

    start = time.time()
    torch_output = F.conv2d(images, weights, bias=bias, stride=stride, padding=padding, dilation=dilation)
    print(f'Torch: {time.time() - start}')

    start = time.time()
    my_output = DepthConv.functional(images, depths, weights, bias=bias, stride=stride, padding=padding, dilation=dilation)
    print(f'This implementation: {time.time() - start}')

    # print(torch_output.shape)
    # print(my_output.shape)

    print(f'Are the two outputs equal? {torch.allclose(torch_output.cpu(), my_output.cpu())}')
