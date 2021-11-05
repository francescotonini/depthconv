from torch.utils.cpp_extension import load
depthconv = load('conv_cuda', sources=['conv_cuda.cpp'], verbose=True)
help(depthconv)
