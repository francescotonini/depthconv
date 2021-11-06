from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthconv_cuda',
    ext_modules=[
        CUDAExtension('depthconv_cuda', [
            'depthconv_cuda.cpp',
            'depthconv_cuda_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
