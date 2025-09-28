# GGUF Kernel from vLLM
- A minimal version GGUF hf-kernel ported from [vLLM](https://github.com/vllm-project/vllm) for development
- Source: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gguf

## Local Installation
### Installing build2cmake
Make sure [`Rust`](https://www.rust-lang.org/tools/install) installed, then run:
```
cargo install build2cmake
```

### Generate Python project with `build2cmake` and install

Use `build2cmake` to generate the CMake/Python project for this kernel as follows:

```bash
build2cmake generate-torch build.toml -f
# generate MMQ kernel instances from template
python ggml/kernel_instances/generate_mmq.py
```

Set the environment variable `CUDA_HOME` to the installation path of CUDA Toolkit, and make sure that the nvcc compiler is in your `PATH`, e.g.:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

Then use `pip` to build and install the kernel to your python environment:

```bash
pip install wheel # Needed once to enable bdist_wheel.
pip install --no-build-isolation -e .
```

## Build and test kernels with Nix (only for development)
```bash
nix develop -L --extra-experimental-features nix-command --extra-experimental-features flakes

pytest tests
```

## Run a full build for kernels (for release)
```bash
nix run .#build-and-copy  --extra-experimental-features nix-command --extra-experimental-features flakes --max-jobs 8 -j 8 -L
```

## Use kernel from hf_hub
- :construction: Under pushing to hf_hub :construction:
