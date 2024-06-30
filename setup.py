from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bin_cuda',
    ext_modules=[
        CUDAExtension('bin_cuda', [
            'bin_cuda.cpp',
            'bin_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
