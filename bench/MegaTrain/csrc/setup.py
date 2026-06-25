from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='infinity_memory_ops',
    ext_modules=[
        CUDAExtension(
            'infinity_memory_ops',
            ['memory_ops.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
