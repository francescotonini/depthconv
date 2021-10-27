import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'


setup(
    name='depthconv_cuda',
    ext_modules=[
        CUDAExtension('depthconv_cuda', [
            'depthconv_cuda.c',
            'depthconv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
