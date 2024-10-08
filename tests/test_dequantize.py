from pathlib import Path
from typing import List

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor, dequantize
from huggingface_hub import snapshot_download
import custom_ops as ops


# GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")
GGUF_SAMPLE = "./samples"


def get_gguf_sample_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> List[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


DTYPES = [torch.float]
DEVICE = [torch.device("cpu")]
# Hidden_size for testing, must match the sample file in HF repo,
# we have `hidden_size = 256, 1024` for test in HF repo currently.
HIDDEN_SIZES = [256, 1024]
NUM_TOKENS = [7, 83, 128, 2048]  # Arbitrary values for testing
SEEDS = [0]
QUANT_TYPES = [
    # i-matrix
    # GGMLQuantizationType.IQ1_M,
    # GGMLQuantizationType.IQ1_S,
    # GGMLQuantizationType.IQ2_S,
    # GGMLQuantizationType.IQ2_XS,
    # GGMLQuantizationType.IQ3_S,
    # GGMLQuantizationType.IQ3_XXS,
    # GGMLQuantizationType.IQ4_NL,
    # GGMLQuantizationType.IQ4_XS,
    # k-quants
    # GGMLQuantizationType.Q2_K,
    # GGMLQuantizationType.Q3_K,
    # GGMLQuantizationType.Q4_K,
    # GGMLQuantizationType.Q5_K,
    # GGMLQuantizationType.Q6_K,
    # standard quantization
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICE)
@torch.inference_mode()
def test_dequantize(hidden_size: int, dtype: torch.dtype, device: torch.device,
                    quant_type: GGMLQuantizationType):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        shape = map(int, shape_str.split("x"))

        ref_output = torch.tensor(dequantize(tensor.data, quant_type),
                                  device=device).to(dtype)
        output = ops.ggml_dequantize(torch.tensor(tensor.data, device="cpu"),
                                     quant_type, *list(shape)).to(dtype)

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)