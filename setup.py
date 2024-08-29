from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_ops',
      ext_modules=[cpp_extension.CppExtension('custom_ops', ['custom_ops.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
