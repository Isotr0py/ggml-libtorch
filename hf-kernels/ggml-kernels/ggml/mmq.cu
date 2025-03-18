#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "mmq.cuh"


// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}


torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);
    mmq_args<scalar_t> kernel_args;
    kernel_args = {
        (char*)W.data_ptr(), (char*)quant_X.data_ptr(),
        (scalar_t*)Y.data_ptr(), col, row, col/QK4_0, padded, batch, row
    };
    switch (type) {
      case GGML_TYPE_Q4_0:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q4_0>(kernel_args, stream);
        break;
    //   case 3:
    //     ggml_mul_mat_q4_1_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 6:
    //     ggml_mul_mat_q5_0_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 7:
    //     ggml_mul_mat_q5_1_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 8:
    //     ggml_mul_mat_q8_0_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 10:
    //     ggml_mul_mat_q2_K_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 11:
    //     ggml_mul_mat_q3_K_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 12:
    //     ggml_mul_mat_q4_K_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 13:
    //     ggml_mul_mat_q5_K_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    //   case 14:
    //     ggml_mul_mat_q6_K_q8_1_cuda(
    //         (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
    //         (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
    //     break;
    }
  });
  return Y;
}
