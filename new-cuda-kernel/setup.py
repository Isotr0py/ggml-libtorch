from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ggml_cuda",
    ext_modules=[
        CUDAExtension(
            "ggml_cuda",
            [
                "ggml_cuda.cpp",
                "gguf_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
