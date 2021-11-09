from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthavgpooling_cuda',
    ext_modules=[
        CUDAExtension('depthavgpooling_cuda', [
            'depthavgpooling_cuda.cpp',
            'depthavgpooling_cuda_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
