import argparse
import json
import time
from functools import cache

import pandas as pd
import torch
from gguf import GGMLQuantizationType
from utils import get_gguf_sample_tensors, seed_everything

QUANT_TYPES_MAP = {
    "Q2_K": GGMLQuantizationType.Q2_K,
    "Q3_K": GGMLQuantizationType.Q3_K,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q6_K": GGMLQuantizationType.Q6_K,
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q5_0": GGMLQuantizationType.Q5_0,
    "Q8_0": GGMLQuantizationType.Q8_0,
}

DTYPES_MAP = {
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
}


@cache
def get_kernel_ops(use_remote: bool, revision: str = "main"):
    if use_remote:
        from kernels import get_kernel

        ops = get_kernel("Isotr0py/ggml", revision=revision)
        print(f"Using remote kernels from revision: {revision}.")
    else:
        print(f"Using local kernels.")
        import ggml as ops
    return ops


@torch.inference_mode()
def main(
    num_tokens: int,
    hidden_size: int,
    quant_type: GGMLQuantizationType,
    dtype: torch.dtype,
    seed: int = 0,
    do_profile: bool = False,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
    use_remote: bool = False,
    revision: str = "main",
) -> None:
    ops = get_kernel_ops(use_remote, revision=revision)

    seed_everything(seed)
    torch.set_default_device("cuda")

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w = [
        torch.tensor(tensor.data, device="cuda")
        for tensor in get_gguf_sample_tensors(
            hidden_size=hidden_size, quant_type=quant_type
        )
    ]

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            for tensor in w:
                ops.ggml_mul_mat_a8(
                    tensor,
                    x,
                    quant_type,
                    tensor.size(0),
                )

        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return (end_time - start_time) / num_iters

    # Warmup.
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=num_warmup_iters, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=num_iters, profile=False)

    quant_name = [
        name for name, qtype in QUANT_TYPES_MAP.items() if qtype == quant_type
    ][0]
    print(f"{quant_name} Kernel running time: {latency * 1e3 :.3f} ms")
    return latency * 1e3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the MMQ kernel for matrix multiplication with quantized weights"
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, choices=[256, 1024], default=1024)
    parser.add_argument(
        "--quant-dtype", type=str, choices=QUANT_TYPES_MAP.keys(), default="Q4_0"
    )
    parser.add_argument("--dtype", type=str, choices=DTYPES_MAP.keys(), default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--use-remote", action="store_true", help="Use remote kernels")
    parser.add_argument(
        "--revision", type=str, default="main", help="The branch, tag or commit to use for remote kernel."
    )
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations. "
        "If --profile is set, this number is ignored",
    )

    args = parser.parse_args()
    print(args)

    dtype = DTYPES_MAP[args.dtype]
    result = {"Quantization": [], "Time (ms)": []}
    for quant_name, quant_type in QUANT_TYPES_MAP.items():
        latency = main(
            num_tokens=args.num_tokens,
            hidden_size=args.hidden_size,
            quant_type=quant_type,
            dtype=dtype,
            seed=args.seed,
            do_profile=args.profile,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            use_remote=args.use_remote,
            revision=args.revision,
        )
        result["Quantization"].append(quant_name)
        result["Time (ms)"].append(latency)

    kernel_mode = "remote" if args.use_remote else "local"
    result_df = pd.DataFrame(result)
    result_df.to_csv(f"benchmark_results_{kernel_mode}_HD{args.hidden_size}xB{args.num_tokens}.csv", index=False)
