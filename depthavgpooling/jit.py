from torch.utils.cpp_extension import load
depthavgpooling = load('depthavgpooling_cuda', sources=['depthavgpooling_cuda.cpp', 'depthavgpooling_cuda_kernel.cu'], verbose=True)
help(depthavgpooling)
