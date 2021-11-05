import torch
import torch.nn as nn
import torch.autograd as autograd
import conv_cuda


class DepthConvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, depth, weights, bias, stride, padding, dilation):
        ctx.save_for_backward(input, depth, weights, bias, autograd.Variable(torch.Tensor([stride, padding, dilation])).cuda())

        return conv_cuda.forward(input, depth, weights, bias, stride, stride, padding, padding, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        input, depth, weights, bias, (stride, padding, dilation) = ctx.saved_tensors

        grad_input, grad_weight, grad_bias = conv_cuda.backward(input, depth, grad_output, weights, stride, stride, padding, padding, dilation)

        return grad_input, grad_weight, grad_bias, autograd.Variable(torch.zeros(6))

class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=None):
        super(DepthConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        self.weights = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).cuda())
        self.bias = nn.Parameter(torch.empty(out_channels).cuda())

        # TODO: initialize weights?

    def forward(self, input, depth):
        return DepthConvFunction.apply(input, depth, self.weights, self.bias, self.stride, self.padding, self.dilation)
