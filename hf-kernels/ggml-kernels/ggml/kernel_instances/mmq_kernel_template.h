#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "../cuda_compat.h"
#include "../cuda_utils.h" 
#include "../dispatch_utils.h"

#include "../ggml-common.h"
#include "../vecdotq.cuh"
#include "../mmq.cuh"

// Template for mul_mat_q_case instantiation
#define DECLARE_MUL_MAT_Q_CASE(QTYPE) \
    template <typename scalar_t> \
    void mul_mat_q_case_##QTYPE(const mmq_args<scalar_t> & args, cudaStream_t stream);

// Template for mul_mat_q_case implementation
#define IMPLEMENT_MUL_MAT_Q_CASE(QTYPE) \
    template <typename scalar_t> \
    void mul_mat_q_case_##QTYPE(const mmq_args<scalar_t> & args, cudaStream_t stream) { \
        const int nsm = get_cuda_info().nsm; \
        const int cc  = get_cuda_info().cc; \
        \
        const int mmq_x_max = get_mmq_x_max_host(cc); \
        const int mmq_y = get_mmq_y_host(cc, mmq_x_max); \
        const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y; \
        \
        int mmq_x_best  = 0; \
        int nwaves_best = INT_MAX; \
        \
        for (int mmq_x = 8; mmq_x <= mmq_x_max && nwaves_best > 1; mmq_x += 8) { \
            const int block_num_x = (args.ne11 + mmq_x - 1) / mmq_x; \
            const int nwaves = (block_num_x*block_num_y + nsm - 1) / nsm; \
            \
            if (nwaves < nwaves_best) { \
                mmq_x_best  = mmq_x; \
                nwaves_best = nwaves; \
            } \
        } \
        \
        switch (mmq_x_best) { \
            case   8: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,   8, 4>(args, stream); \
                break; \
            case  16: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  16, 8>(args, stream); \
                break; \
            case  24: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  24, 8>(args, stream); \
                break; \
            case  32: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  32, 8>(args, stream); \
                break; \
            case  40: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  40, 8>(args, stream); \
                break; \
            case  48: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  48, 8>(args, stream); \
                break; \
            case  56: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  56, 8>(args, stream); \
                break; \
            case  64: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  64, 8>(args, stream); \
                break; \
            case  72: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  72, 8>(args, stream); \
                break; \
            case  80: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  80, 8>(args, stream); \
                break; \
            case  88: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  88, 8>(args, stream); \
                break; \
            case  96: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE,  96, 8>(args, stream); \
                break; \
            case 104: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE, 104, 8>(args, stream); \
                break; \
            case 112: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE, 112, 8>(args, stream); \
                break; \
            case 120: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE, 120, 8>(args, stream); \
                break; \
            case 128: \
                launch_mul_mat_q<scalar_t, GGML_TYPE_##QTYPE, 128, 8>(args, stream); \
                break; \
            default: \
                assert(false); \
                break; \
        } \
    }

// Explicit template instantiations
#define INSTANTIATE_MUL_MAT_Q_CASE(QTYPE) \
    template void mul_mat_q_case_##QTYPE<float>(const mmq_args<float> & args, cudaStream_t stream); \
    template void mul_mat_q_case_##QTYPE<at::Half>(const mmq_args<at::Half> & args, cudaStream_t stream);
