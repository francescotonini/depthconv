from torch.utils.cpp_extension import load
depthconv = load('depthconv_cuda', sources=['depthconv_cuda.cpp', 'depthconv_cuda_kernel.cu'], verbose=True)
help(depthconv)
