import torch

try:
    from ._ops import ops
except ImportError as e:
    # Fallback for local development.
    try:
        import _ggml

        ops = torch.ops._ggml
    except ImportError:
        raise e


def ggml_dequantize(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
) -> torch.Tensor:
    """Dequantize the GGML tensor."""
    return ops.ggml_dequantize(W, quant_type, m, n)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """Mulmat with MMVQ kernel, require batch_size==1."""
    batch = X.size(0)
    assert batch == 1, "Batch size must be 1 for MMVQ kernel"
    return ops.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """Mulmat through MMQ kernel for arbitrary batch size."""
    return ops.ggml_mul_mat_a8(W, X, quant_type, row)
