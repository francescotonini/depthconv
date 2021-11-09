import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.autograd as autograd
import depthavgpooling_cuda


class DepthAvgPoolingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, depth, kernel_size=1, stride=1, padding=0):
        ctx.save_for_backward(input)
        ctx.depth = depth
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.depth_weight_count = torch.zeros(input.shape).cuda()

        (kernel_h, kernel_w) = _pair(kernel_size)
        (stride_h, stride_w) = _pair(stride)
        (padding_h, padding_w) = _pair(padding)

        return depthavgpooling_cuda.forward(input, depth, ctx.depth_weight_count, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        depth = ctx.depth
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        depth_weight_count = ctx.depth_weight_count

        (kernel_h, kernel_w) = _pair(kernel_size)
        (stride_h, stride_w) = _pair(stride)
        (padding_h, padding_w) = _pair(padding)

        grad_input = depthavgpooling_cuda.backward(input, depth, depth_weight_count, grad_output, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w)

        return grad_input, None, None, None, None

class DepthAvgPooling(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(DepthAvgPooling, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input, depth):
        return DepthAvgPoolingFunction.apply(input, depth, self.kernel_size, self.stride, self.padding)
