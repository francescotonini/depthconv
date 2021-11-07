import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.autograd as autograd
import depthconv_cuda


class DepthConvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, depth, weight, bias=None, padding=0, stride=1, dilation=1):
        ctx.save_for_backward(input, depth, weight, bias)
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        
        (padding_w, padding_h) = padding = _pair(padding)
        (stride_w, stride_h) = stride = _pair(stride)
        (dilation_w, dilation_h) = dilation = _pair(dilation)

        return depthconv_cuda.forward(input, depth, weight, bias, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w)

    @staticmethod
    def backward(ctx, grad_output):
        input, depth, weight, bias = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        dilation = ctx.dilation

        (padding_w, padding_h) = padding = _pair(padding)
        (stride_w, stride_h) = stride = _pair(stride)
        (dilation_w, dilation_h) = dilation = _pair(dilation)

        grad_input, grad_weight, grad_bias = depthconv_cuda.backward(input, depth, weight, bias, grad_output, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w)

        return grad_input, None, grad_weight, grad_bias, None, None, None, None

class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=None):
        super(DepthConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        (self.kernel_height, self.kernel_width) = _pair(kernel_size)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        self.weights = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_height, self.kernel_width).cuda())
        self.bias = nn.Parameter(torch.empty(out_channels).cuda())

    def forward(self, input, depth):
        return DepthConvFunction.apply(input, depth, self.weights, self.bias, self.padding, self.stride, self.dilation)
