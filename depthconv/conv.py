import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.autograd as autograd
import depthconv_cuda


class DepthConvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, depth, weight, bias=None, stride=1, padding=0, dilation=1, has_bias=True):
        ctx.save_for_backward(input, weight, bias)
        ctx.depth = depth
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.has_bias = True

        (stride_w, stride_h) = stride = _pair(stride)
        (padding_w, padding_h) = padding = _pair(padding)
        (dilation_w, dilation_h) = dilation = _pair(dilation)

        return depthconv_cuda.forward(input, depth, weight, bias, stride_h, stride_w, padding_h, padding_w,
                                      dilation_h, dilation_w, has_bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        depth = ctx.depth
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        has_bias = ctx.has_bias
        (stride_w, stride_h) = stride = _pair(stride)
        (padding_w, padding_h) = padding = _pair(padding)
        (dilation_w, dilation_h) = dilation = _pair(dilation)

        grad_input, grad_weight, grad_bias = depthconv_cuda.backward(input, depth, weight, bias, grad_output, stride_h,
                                                                     stride_w, padding_h, padding_w, dilation_h,
                                                                     dilation_w, has_bias)

        return grad_input, None, grad_weight, grad_bias, None, None, None, None

class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=31, stride=1, padding=0, dilation=1, bias=False):
        super(DepthConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        (self.kernel_height, self.kernel_width) = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

        self.weights = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_height, self.kernel_width).cuda())
        self.bias = nn.Parameter(torch.empty(out_channels).cuda())

    def forward(self, input, depth):
        return DepthConvFunction.apply(input, depth, self.weights, self.bias, self.stride, self.padding,
                                       self.dilation, self.has_bias)
