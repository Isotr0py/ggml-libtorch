#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "cuda_utils.h"
#include "dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "mmq.cuh"


cuda_device_info get_cuda_info() {
    int id;
    // CUDA_CHECK(cudaGetDevice(&id));
    cudaGetDevice(&id);

    cudaDeviceProp prop;
    // CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
    cudaGetDeviceProperties(&prop, id);

    cuda_device_info info;
    info.cc = prop.major*100 + prop.minor * 10;
    info.nsm = prop.multiProcessorCount;
    info.smpb = prop.sharedMemPerBlock;
    info.smpbo = prop.sharedMemPerBlockOptin;
    info.vmm = prop.managedMemory;
    info.vmm_granularity = prop.managedMemory ? prop.managedMemory : 0;
    info.total_vram = prop.totalGlobalMem;

    return info;
}


int64_t ggml_get_block_size(int64_t type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return QK4_0;
        case GGML_TYPE_Q4_1:    return QK4_1;
        case GGML_TYPE_Q5_0:    return QK5_0;
        case GGML_TYPE_Q5_1:    return QK5_1;
        case GGML_TYPE_Q8_0:    return QK8_0;
        case GGML_TYPE_Q8_1:    return QK8_1;
        case GGML_TYPE_Q2_K:    return QK_K;
        case GGML_TYPE_Q3_K:    return QK_K;
        case GGML_TYPE_Q4_K:    return QK_K;
        case GGML_TYPE_Q5_K:    return QK_K;
        case GGML_TYPE_Q6_K:    return QK_K;
        case GGML_TYPE_IQ2_XXS: return QK_K;
        case GGML_TYPE_IQ2_XS:  return QK_K;
        case GGML_TYPE_IQ2_S:   return QK_K;
        case GGML_TYPE_IQ3_XXS: return QK_K;
        case GGML_TYPE_IQ3_S:   return QK_K;
        case GGML_TYPE_IQ1_S:   return QK_K;
        case GGML_TYPE_IQ1_M:   return QK_K;
        case GGML_TYPE_IQ4_NL:  return QK4_NL;
        case GGML_TYPE_IQ4_XS:  return QK_K;
        default: return 0; // unsupported type
    }
}


static int mmq_need_sum(int64_t type_x) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return true;
        case GGML_TYPE_Q5_0:
            return false;
        case GGML_TYPE_Q5_1:
            return true;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return false;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return true;
        case GGML_TYPE_Q6_K:
            return false;
        default:
            break;
    }
    return false;
}


template <typename scalar_t, bool need_sum>
static __global__ void quantize_mmq_q8_1(
    const scalar_t * __restrict__ x, void * __restrict__ vy, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded) {

    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = kx1*blockIdx.z + blockIdx.y;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*(gridDim.y*gridDim.x*blockDim.x/(4*QK8_1)); // first block of channel
    const int64_t ib  = ib0 + (ix0 / (4*QK8_1))*kx1 + blockIdx.y;              // block index in channel
    const int64_t iqs = ix0 % (4*QK8_1);                                       // quant index in block

    const float xi = ix0 < kx0 ? static_cast<float>(x[ix1*kx0 + ix0]) : 0.0f;
    float amax = fabsf(xi);

    float sum = xi;

  #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
      if (need_sum) {
        sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
      }
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs % QK8_1 != 0) {
        return;
    }

    if (need_sum) {
        y[ib].ds[iqs/QK8_1] = make_half2(d, sum);
    } else {
        ((float *) y[ib].ds)[iqs/QK8_1] = d;
    }
}


template <typename scalar_t>
void quantize_mmq_q8_1_cuda(
    const scalar_t * x, void * vy, const int64_t kx0, const int64_t kx1,
    const int64_t type_x, cudaStream_t stream) {
  
    // different from original ggml implementation, kx_padded
    // and channels is computed inside the function here
    const int64_t kx0_padded = (kx0 + 512 - 1) / 512 * 512;
    const int channels = 1;

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, channels);
    if (mmq_need_sum(type_x)) {
        quantize_mmq_q8_1<scalar_t, true><<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
    } else {
        quantize_mmq_q8_1<scalar_t, false><<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
    }
}


torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row) {
  int64_t x_ndim = X.dim();
  TORCH_CHECK(
      x_ndim == 2 || x_ndim == 3,
      "X must have shape [num_tokens, hidden_size] or [batch_size, num_tokens, hidden_size]");

  int col = X.sizes()[x_ndim - 1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());

  at::Tensor Y;
  int batch;
  if (x_ndim == 2) {
    batch = X.sizes()[0];
    Y = torch::empty({batch, row}, options);
  }
  else if (x_ndim == 3) {
    batch = X.sizes()[0] * X.sizes()[1];
    Y = torch::empty({X.sizes()[0], X.sizes()[1], row}, options);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_mmq_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, type, stream);

    const int64_t stride00 = col / ggml_get_block_size(type);
    mmq_args<scalar_t> kernel_args;
    kernel_args = {
        (char*)W.data_ptr(), (char*)quant_X.data_ptr(),
        (scalar_t*)Y.data_ptr(), col, row, stride00, padded, batch, col, row
    };

    switch (type) {
      case GGML_TYPE_Q4_0:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q4_0>(kernel_args, stream);
        break;
      case GGML_TYPE_Q4_1:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q4_1>(kernel_args, stream);
        break;
      case GGML_TYPE_Q5_0:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q5_0>(kernel_args, stream);
        break;
      case GGML_TYPE_Q5_1:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q5_1>(kernel_args, stream);
        break;
      case GGML_TYPE_Q8_0:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q8_0>(kernel_args, stream);
        break;
      case GGML_TYPE_Q2_K:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q2_K>(kernel_args, stream);
        break;
      case GGML_TYPE_Q3_K:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q3_K>(kernel_args, stream);
        break;
      case GGML_TYPE_Q4_K:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q4_K>(kernel_args, stream);
        break;
      case GGML_TYPE_Q5_K:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q5_K>(kernel_args, stream);
        break;
      case GGML_TYPE_Q6_K:
        mul_mat_q_case<scalar_t, GGML_TYPE_Q6_K>(kernel_args, stream);
        break;
    }
  });
  return Y;
}
