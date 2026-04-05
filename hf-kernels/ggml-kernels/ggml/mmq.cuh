#pragma once
#include "mma.cuh"

#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)
#define MMQ_NWARPS 8

// ----- Vector dot product of K-quants MMQ -----
#define VDR_Q4_0_Q8_1_MMQ  4
#define VDR_Q4_1_Q8_1_MMQ  4
#define VDR_Q5_0_Q8_1_MMQ  4
#define VDR_Q5_1_Q8_1_MMQ  4
#define VDR_Q8_0_Q8_1_MMQ  8
#define VDR_Q2_K_Q8_1_MMQ  2
#define VDR_Q3_K_Q8_1_MMQ  2
#define VDR_Q4_K_Q8_1_MMQ  8
#define VDR_Q5_K_Q8_1_MMQ  8
#define VDR_Q6_K_Q8_1_MMQ  8

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        const float2 dm2f = __half22float2(dm2[i0/(QI8_1/2)]);
        int sumi_d = 0;
        int sumi_m = 0;

        const int vi0 = v[i0/(QI8_1/2)];
#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            const int vi = (vi0 >> (2*(i % (QI8_1/2)))) & 0x03030303;
            sumi_d = __dp4a(vi,         u[i], sumi_d); // SIMD dot product
            sumi_m = __dp4a(0x01010101, u[i], sumi_m);
        }

        sumf_d += dm2f.x * sumi_d;
        sumf_m += dm2f.y * sumi_m;
    }

    return d8*(sumf_d - sumf_m);
#endif
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            const int vi = __vsubss4((v[i/2] >> (4*(i%2))) & 0x0F0F0F0F, 0x04040404);
            sumi_sc = __dp4a(vi, u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
#endif
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = __dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = __dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = __dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = __dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
#endif
}

// -----------------------
#define  MMQ_MAX_BATCH_SIZE 64 // max batch size to use MMQ kernels when tensor cores are available

static constexpr int get_mmq_x_max_host(const int cc) {
    return cc >= 700 && cc < 1000000 ? MMQ_MAX_BATCH_SIZE : 64;
}
    
// Round rows to this value for --split-mode row:
static constexpr int get_mmq_y_host(const int cc) {
    return cc >= 700 ? 128 : 64;
}


typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int & kbx0, const int & i_max, const int & stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0);
template <typename scalar_t>
using mmq_write_back_t = void (*)(const float * __restrict__ sum, scalar_t * __restrict__ dst, const int & stride, const int & i_max, const int & j_max);
struct block_q8_1_mmq {
    half2  ds[4];
    int8_t qs[4*QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};
// get_mmq_x_max_host is in common.cuh so that it can be used to determine the correct way to round for --split-mode row
static constexpr __device__ int get_mmq_x_max_device() {
#if defined(USE_ROCM)
    return 64;
#else
#if __CUDA_ARCH__ >= 700
#ifdef CUDA_USE_TENSOR_CORES
    return MMQ_MAX_BATCH_SIZE;
#else
    return 128;
#endif // CUDA_USE_TENSOR_CORES
#else
    return 64;
#endif // __CUDA_ARCH__ >= 700
#endif // defined(USE_ROCM)
}
// get_mmq_y_host is in common.cuh so that it can be used to determine the correct way to round for --split-mode row
static constexpr __device__ int get_mmq_y_device() {
#if defined(USE_ROCM)
    return 128;
#else
#if __CUDA_ARCH__ >= 700
    return 128;
#else
    return 64;
#endif // __CUDA_ARCH__ >= 700
#endif // defined(USE_ROCM)
}
#define MMQ_DP4A_TXS_Q4_0 tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_0 + mmq_y/QI4_0, 0}
#define MMQ_DP4A_TXS_Q4_1 tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_1 + mmq_y/QI4_1, 0}
#define MMQ_DP4A_TXS_Q5_0 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_0 + mmq_y/QI5_0, 0}
#define MMQ_DP4A_TXS_Q5_1 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_1 + mmq_y/QI5_1, 0}
#define MMQ_DP4A_TXS_Q8_0 tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI8_0 + mmq_y/QI8_0, 0}
#define MMQ_DP4A_TXS_Q2_K tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE       + mmq_y,       0}
#define MMQ_DP4A_TXS_Q3_K tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI3_K + mmq_y/QI3_K, mmq_y*WARP_SIZE/4 + mmq_y/4}
#define MMQ_DP4A_TXS_Q4_K tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_K + mmq_y/QI4_K, mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q5_K tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_K + mmq_y/QI5_K, mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q6_K tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI6_K + mmq_y/QI6_K, mmq_y*WARP_SIZE/8 + mmq_y/8}
static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    return type == GGML_TYPE_Q4_0 ? MMQ_DP4A_TXS_Q4_0 :
        type == GGML_TYPE_Q4_1 ? MMQ_DP4A_TXS_Q4_1 :
        type == GGML_TYPE_Q5_0 ? MMQ_DP4A_TXS_Q5_0 :
        type == GGML_TYPE_Q5_1 ? MMQ_DP4A_TXS_Q5_1 :
        type == GGML_TYPE_Q8_0 ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_Q2_K ? MMQ_DP4A_TXS_Q2_K :
        type == GGML_TYPE_Q3_K ? MMQ_DP4A_TXS_Q3_K :
        type == GGML_TYPE_Q4_K ? MMQ_DP4A_TXS_Q4_K :
        type == GGML_TYPE_Q5_K ? MMQ_DP4A_TXS_Q5_K :
        type == GGML_TYPE_Q6_K ? MMQ_DP4A_TXS_Q6_K :
        tile_x_sizes{0, 0, 0};
}
#define MMQ_MMA_TILE_X_K_Q4_0 (1*WARP_SIZE + WARP_SIZE/QI4_0               + 4)
#define MMQ_MMA_TILE_X_K_Q4_1 (1*WARP_SIZE + WARP_SIZE/QI4_1               + 4)
#define MMQ_MMA_TILE_X_K_Q5_0 (2*WARP_SIZE + WARP_SIZE/QI5_0               + 4)
#define MMQ_MMA_TILE_X_K_Q5_1 (2*WARP_SIZE + WARP_SIZE/QI5_1               + 4)
#define MMQ_MMA_TILE_X_K_Q8_0 (1*WARP_SIZE + WARP_SIZE/QI8_0               + 0)
#define MMQ_MMA_TILE_X_K_Q2_K (1*WARP_SIZE + WARP_SIZE                     + 4)
#define MMQ_MMA_TILE_X_K_Q3_K (2*WARP_SIZE + WARP_SIZE/QI3_K + WARP_SIZE/4 + 2)
#define MMQ_MMA_TILE_X_K_Q4_K (1*WARP_SIZE + WARP_SIZE/QI4_K + WARP_SIZE/8 + 7)
#define MMQ_MMA_TILE_X_K_Q5_K (2*WARP_SIZE + WARP_SIZE/QI5_K + WARP_SIZE/8 + 7)
#define MMQ_MMA_TILE_X_K_Q6_K (2*WARP_SIZE + WARP_SIZE/QI6_K + WARP_SIZE/8 + 7)
static_assert(MMQ_MMA_TILE_X_K_Q4_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q4_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q5_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q5_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q4_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q5_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q6_K % 8 == 4, "Wrong padding.");
static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? MMQ_MMA_TILE_X_K_Q4_0 :
        type == GGML_TYPE_Q4_1 ? MMQ_MMA_TILE_X_K_Q4_1 :
        type == GGML_TYPE_Q5_0 ? MMQ_MMA_TILE_X_K_Q5_0 :
        type == GGML_TYPE_Q5_1 ? MMQ_MMA_TILE_X_K_Q5_1 :
        type == GGML_TYPE_Q8_0 ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q2_K ? MMQ_MMA_TILE_X_K_Q2_K :
        type == GGML_TYPE_Q3_K ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_Q4_K ? MMQ_MMA_TILE_X_K_Q4_K :
        type == GGML_TYPE_Q5_K ? MMQ_MMA_TILE_X_K_Q5_K :
        type == GGML_TYPE_Q6_K ? MMQ_MMA_TILE_X_K_Q6_K :
        0;
}
#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)
#define MMQ_NWARPS 8
static int mmq_get_granularity_host(const int mmq_x, const int cc) {
    return cc >= 750 && mmq_x >= 48 ? 16 : 8;
}
#ifdef INT8_MMA_AVAILABLE
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}
#else
static constexpr __device__ int mmq_get_granularity_device(const int /* mmq_x */) {
    return 8;
}
#endif // INT8_MMA_AVAILABLE
// ------------------------------------------------------------
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q4_0 + threadIdx.x] = get_int_from_uint8(bxi->qs, kqsx);
#else
        x_qs[i*(WARP_SIZE + 1)       + threadIdx.x] = get_int_from_uint8(bxi->qs, kqsx);
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q4_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));
            int u[2*VDR_Q4_0_Q8_1_MMQ];
#pragma unroll
            for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI4_0) % WARP_SIZE];
            }
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE + 1) + k0], u, x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + k0/QI4_0],
                y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
    mma_A A[ntx];
    float dA[ntx][mma_C::ne/2];
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_A::ne; ++l) {
            const int i     = i0 + n*mma_A::I + mma_A::get_i(l);
            const int k     = k0              + mma_A::get_k(l) % QI4_0;
            const int shift =                4*(mma_A::get_k(l) / QI4_0);
            A[n].x[l] = __vsubss4((x_qs[i*MMQ_MMA_TILE_X_K_Q4_0 + k] >> shift) & 0x0F0F0F0F, 0x08080808);
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q4_0 + k0/QI4_0];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B  B;
        float dB[mma_C::ne/2];
        B.load(y_qs + j0*MMQ_TILE_Y_K + (2*k0) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
            mma_C C;
            C.mma_K8(A[n], B);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += dA[n][l/2]*dB[l%2]*C.x[l];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI4_1;
    const int kqsx = threadIdx.x % QI4_1;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbx;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q4_1 + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
#else
        x_qs[i*(WARP_SIZE + 1)       + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + threadIdx.y * QI4_1 + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q4_1       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + kbxd] = bxi->dm;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));
            int u[2*VDR_Q4_1_Q8_1_MMQ];
#pragma unroll
            for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI4_1) % WARP_SIZE];
            }
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE + 1) + k0], u, x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + k0/QI4_1],
                y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_A_I16K4 mma_A_K4;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
    mma_A A[ntx];
    half2 dmA[ntx][mma_C::ne/2];
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
        ((mma_A_K4 *) &A[n])[0].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q4_1 + k0, MMQ_MMA_TILE_X_K_Q4_1);
        A[n].x[2]  = (A[n].x[0] >> 4) & 0x0F0F0F0F;
        A[n].x[3]  = (A[n].x[1] >> 4) & 0x0F0F0F0F;
        A[n].x[0] &= 0x0F0F0F0F;
        A[n].x[1] &= 0x0F0F0F0F;
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
            dmA[n][l] = x_dm[i*MMQ_MMA_TILE_X_K_Q4_1 + k0/QI4_1];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B B;
        half2 dsB[mma_C::ne/2];
        B.load(y_qs + j0*MMQ_TILE_Y_K + (2*k0) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dsB[l] = y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            mma_C C;
            C.mma_K8(A[n], B);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                const half2 dmA_dsB = dmA[n][l/2]*dsB[l%2];
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += __low2float(dmA_dsB)*C.x[l] + __high2float(dmA_dsB);
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI5_0;
    const int kqsx = threadIdx.x % QI5_0;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbx;
        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_0));
        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16
        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q5_0 + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q5_0 + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + threadIdx.y * QI5_0 + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q5_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI5_0) + i/QI5_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, QR5_0*VDR_Q5_0_Q8_1_MMQ>
                (&x_qs[i*(2*WARP_SIZE + 1) + 2*k0], &y_qs[j*MMQ_TILE_Y_K + (2*k0) % WARP_SIZE],
                 x_df[i*(WARP_SIZE/QI5_0) + i/QI5_0 + k0/QI5_0], y_df[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    mma_A A[ntx];
    float dA[ntx][mma_C::ne/2];
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
        A[n].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q5_0 + QR5_1*k0, MMQ_MMA_TILE_X_K_Q5_0);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + mma_C::get_i(2*l) + n*mma_C::I;
            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q5_0 + k0/QI5_0];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B B;
        float dB[mma_C::ne/2];
        B.load(y_qs + j0*MMQ_TILE_Y_K + (2*k0) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dB[l] = y_df[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            mma_C C;
            C.mma_K8(A[n], B);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += dA[n][l/2]*dB[l%2]*C.x[l];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI5_1;
    const int kqsx = threadIdx.x % QI5_1;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbx;
        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_1));
        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28
        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q5_1 + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q5_1 + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + threadIdx.y * QI5_1 + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q5_1       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + kbxd] = bxi->dm;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                (&x_qs[i*(2*WARP_SIZE + 1) + 2*k0], &y_qs[j*MMQ_TILE_Y_K + (2*k0) % WARP_SIZE],
                 x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + k0/QI5_1], y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
    mma_A A[ntx];
    half2 dmA[ntx][mma_C::ne/2];
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
        A[n].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q5_1 + QR5_1*k0, MMQ_MMA_TILE_X_K_Q5_1);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + mma_C::get_i(2*l) + n*mma_C::I;
            dmA[n][l] = x_dm[i*MMQ_MMA_TILE_X_K_Q5_1 + k0/QI5_1];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B B;
        half2 dsB[mma_C::ne/2];
        B.load(y_qs + j0*MMQ_TILE_Y_K + (2*k0) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dsB[l] = y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            mma_C C;
            C.mma_K8(A[n], B);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                const half2 dmA_dsB = dmA[n][l/2]*dsB[l%2];
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += __low2float(dmA_dsB)*C.x[l] + __high2float(dmA_dsB);
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI8_0;
    const int kqsx = threadIdx.x % QI8_0;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x] = get_int_from_int8(bxi->qs, kqsx);
#else
        x_qs[i*(WARP_SIZE + 1)       + threadIdx.x] = get_int_from_int8(bxi->qs, kqsx);
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + threadIdx.y * QI8_0 + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0         + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0], x_df[i*(WARP_SIZE/QI8_0) + i/QI8_0 + k0/QI8_0],
                y_df[j*MMQ_TILE_Y_K + k0/QI8_1]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    mma_A A[ntx];
    float dA[ntx][mma_C::ne/2];
    const int i0 = (threadIdx.y/ntx)*rows_per_warp;
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
        A[n].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);
            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B B;
        float dB[mma_C::ne/2];
        B.load(y_qs + j0*MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dB[l] = y_df[j*MMQ_TILE_Y_K + k0/QI8_1];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            mma_C C;
            C.mma_K8(A[n], B);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += C.x[l]*dA[n][l/2]*dB[l%2];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI2_K;
    const int kqsx = threadIdx.x % QI2_K;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride + kbx;
        const int x_ql_0 = get_int_from_uint8(bxi->qs, kqsx);
#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = kbx*QI2_K + (kqsx/8)*8 + l*2 + (kqsx % 8)/4;
            int x_qs_k = ((x_ql_0 >> (2*l)) & 0x03030303) << (2*(kqsx % 4));
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 1, WARP_SIZE);
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 2, WARP_SIZE);
            if (kqsx % QR2_K != 0) {
                continue;
            }
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i*(WARP_SIZE + 1)       + k] = x_qs_k;
#endif // INT8_MMA_AVAILABLE
        }
        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE
#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + threadIdx.x] = x_dm_ik;
#else
        x_dm[i*(WARP_SIZE + 1)       + threadIdx.x] = x_dm_ik;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq(
                &x_qs[i*(WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + (QR2_K*k0) % WARP_SIZE],
                &x_dm[i*(WARP_SIZE + 1) + k0], y_df[j*MMQ_TILE_Y_K + ((QR2_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
    mma_A   A[ntx][2];
    float  dA[ntx][mma_C::ne/2][2];
    float  mA[ntx][mma_C::ne/2][2];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_A::ne; ++l) {
            const int i = i0 + n*mma_A::I + mma_A::get_i(l);
            const int shift = 2*mma_A::get_k(l);
            A[n][0].x[l] = (x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k0 + 0] >> shift) & 0x03030303;
            A[n][1].x[l] = (x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k0 + 1] >> shift) & 0x03030303;
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
#pragma unroll
            for (int kdm = 0; kdm < 2; ++kdm) {
                const float2 dm = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + k0 + kdm]);
                dA[n][l][kdm] = dm.x;
                mA[n][l][kdm] = dm.y;
            }
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B B[2];
        float dB[mma_C::ne/2];
        B[0].load(y_qs + j0*MMQ_TILE_Y_K + (QR2_K*k0 + 0)        % WARP_SIZE, MMQ_TILE_Y_K);
        B[1].load(y_qs + j0*MMQ_TILE_Y_K + (QR2_K*k0 + mma_B::K) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dB[l] = y_df[j*MMQ_TILE_Y_K + ((4*k0)/QI8_1) % (WARP_SIZE/QI8_1)];
        }
        mma_C Cm[2];
        mma_A A1;
        A1.x[0] = 0x01010101;
        A1.x[1] = 0x01010101;
        Cm[0].mma_K4(A1, B[0]);
        Cm[1].mma_K4(A1, B[1]);
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
            mma_C Cd[2];
            Cd[0].mma_K4(A[n][0], B[0]);
            Cd[1].mma_K4(A[n][1], B[1]);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += (
                    Cd[0].x[l]*dA[n][l/2][0] + Cd[1].x[l]*dA[n][l/2][1] - Cm[0].x[l]*mA[n][l/2][0] - Cm[1].x[l]*mA[n][l/2][1])*dB[l%2];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
    int   * x_sc = (int   *) (x_df + WARP_SIZE/QI3_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = threadIdx.x / QI3_K;
    const int kqsx = threadIdx.x % QI3_K;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + kbx;
        const int x_ql_0 = get_int_from_uint8(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_from_uint8(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));
#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = kbx*(QR3_K*QI3_K) + (kqsx/8)*32 + l*8 + kqsx % 8;
            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;
            int x_qs_k = (x_ql_k | x_qh_k) << (4*(k%2));
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 1, WARP_SIZE);
            if (kqsx % 2 != 0) {
                continue;
            }
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k/2] = x_qs_k;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k/2] = x_qs_k;
#endif // INT8_MMA_AVAILABLE
        }
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + threadIdx.y * QI3_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI3_K) + i/QI3_K + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/4)) / (QI3_K/4);
        const int ksc = threadIdx.x % (QI3_K/4);
        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;
        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;
        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);
#ifdef INT8_MMA_AVAILABLE
        x_sc[i*MMQ_MMA_TILE_X_K_Q3_K + threadIdx.x % (WARP_SIZE/4)] = sc;
#else
        x_sc[i*(WARP_SIZE/4) + i/4   + threadIdx.x % (WARP_SIZE/4)] = sc;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int kbx  = k0 / QI3_K;
            const int ky  = (k0 % QI3_K) * QR3_K;
            const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q3_K_q8_1_impl_mmq(
                &x_qs[i*(2*WARP_SIZE + 1) + 2*k0], &y_qs[j*MMQ_TILE_Y_K + (k0*QR3_K) % WARP_SIZE], scales,
                x_df[i*(WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[j*MMQ_TILE_Y_K + ((k0*QR3_K) % WARP_SIZE)/QI8_1]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * x_sc = (const int   *) x_df + WARP_SIZE/QI3_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
    mma_A   A[ntx][2];
    int   scA[ntx][mma_C::ne/2][2];
    float  dA[ntx][mma_C::ne/2];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_A::ne; ++l) {
            const int i = i0 + n*mma_A::I + mma_A::get_i(l);
            const int k = QR3_K*k0 + mma_A::get_k(l);
            A[n][0].x[l] = (x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k/2 + 0]          >> (4*(k%2))) & 0x0F0F0F0F;
            A[n][1].x[l] = (x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k/2 + mma_A::K/2] >> (4*(k%2))) & 0x0F0F0F0F;
            A[n][0].x[l] = __vsubss4(A[n][0].x[l], 0x04040404);
            A[n][1].x[l] = __vsubss4(A[n][1].x[l], 0x04040404);
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
            const int kbx  = k0 / QI3_K;
            const int ky  = (k0 % QI3_K) * QR3_K;
            const int8_t * sc = ((const int8_t *) (x_sc + i*MMQ_MMA_TILE_X_K_Q3_K + kbx*4)) + ky/4;
            scA[n][l][0] = sc[0];
            scA[n][l][1] = sc[1];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/QI3_K];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        mma_B B[2];
        float dB[mma_C::ne/2];
        B[0].load(y_qs + j0*MMQ_TILE_Y_K + (QR3_K*k0 + 0)        % WARP_SIZE, MMQ_TILE_Y_K);
        B[1].load(y_qs + j0*MMQ_TILE_Y_K + (QR3_K*k0 + mma_B::K) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);
            dB[l] = y_df[j*MMQ_TILE_Y_K + ((4*k0)/QI8_1) % (WARP_SIZE/QI8_1)];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            mma_C C[2];
            C[0].mma_K4(A[n][0], B[0]);
            C[1].mma_K4(A[n][1], B[1]);
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += (C[0].x[l]*scA[n][l/2][0] + C[1].x[l]*scA[n][l/2][1])*dA[n][l/2]*dB[l%2];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + WARP_SIZE);
    int   * x_sc = (int   *) (x_dm + WARP_SIZE/QI4_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = 0;           // threadIdx.x / QI4_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI4_K
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + kbx;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q4_K + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
#else
        x_qs[i*(WARP_SIZE + 1)       + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + threadIdx.y * QI4_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q4_K       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI4_K) + i/QI4_K + kbxd] = bxi->dm;
#endif // INT8_MMA_AVAILABLE
    }
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / (QI4_K/8);
        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE/8);
        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits
#ifdef INT8_MMA_AVAILABLE
        x_sc[i*MMQ_MMA_TILE_X_K_Q4_K + ksc] = scales8;
#else
        x_sc[i*(WARP_SIZE/8) + i/8   + ksc] = scales8;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]) + 2*((k0 % 16) / 8);
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_K_q8_1_impl_mmq(
                &x_qs[i*(WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + (QR4_K*k0) % WARP_SIZE], sc, sc+8,
                x_dm[i*(WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[j*MMQ_TILE_Y_K + ((QR4_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + WARP_SIZE;
    const int   * x_sc = (const int   *) x_dm + WARP_SIZE/QI4_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
    mma_A   A[ntx][2];
    int   scA[ntx][mma_C::ne/2][2];
    int    mA[ntx][mma_C::ne/2][2];
    half2 dmA[ntx][mma_C::ne/2];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q4_K_Q8_1_MMQ; kvdr += 8) {
            A[n][kvdr/4 + 0].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q4_K + k0, MMQ_MMA_TILE_X_K_Q4_K);
#pragma unroll
            for (int l = 0; l < mma_A::ne; ++l) {
                A[n][kvdr/4 + 1].x[l]  = (A[n][kvdr/4 + 0].x[l] >> 4) & 0x0F0F0F0F;
                A[n][kvdr/4 + 0].x[l] &= 0x0F0F0F0F;
            }
        }
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q4_K_Q8_1_MMQ; kvdr += 4) {
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);
                const uint8_t * sc = ((const uint8_t *) &x_sc[i*MMQ_MMA_TILE_X_K_Q4_K + k0/16]) + 2 * ((k0 % 16) / 8);
                const uint8_t *  m = sc + 8;
                scA[n][l][kvdr/4] = sc[kvdr/4];
                mA[n][l][kvdr/4]  =  m[kvdr/4];
            }
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);
            dmA[n][l] = x_dm[i*MMQ_MMA_TILE_X_K_Q4_K + k0/QI4_K];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float tmpd[ntx][mma_C::ne] = {{0.0f}};
        float tmpm[ntx][mma_C::ne] = {{0.0f}};
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q4_K_Q8_1_MMQ; kvdr += 4) {
            mma_B   B;
            half2 dsB[mma_C::ne/2];
            B.load(y_qs + j0*MMQ_TILE_Y_K + (2*k0 + 2*kvdr) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);
                dsB[l] = y_ds[j*MMQ_TILE_Y_K + ((2*k0 + 2*kvdr)/QI8_1) % (WARP_SIZE/QI8_1)];
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma_K8(A[n][kvdr/4], B);
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    tmpd[n][l] += (C.x[l]*scA[n][l/2][kvdr/4]) *  __low2float(dsB[l%2]);
                    tmpm[n][l] += mA[n][l/2][kvdr/4]           * __high2float(dsB[l%2]);
                }
            }
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += __low2float(dmA[n][l/2])*tmpd[n][l] - __high2float(dmA[n][l/2])*tmpm[n][l];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + WARP_SIZE*2);
    int   * x_sc = (int   *) (x_dm + WARP_SIZE/QI5_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = 0;           // threadIdx.x / QI5_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI5_K
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + kbx;
        const int ky = QR5_K*kqsx;
        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;
        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;
        const int kq0 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + (QI5_K/4);
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q5_K + kq0] = ql0 | qh0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q5_K + kq1] = ql1 | qh1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kq0] = ql0 | qh0;
        x_qs[i*(2*WARP_SIZE + 1)     + kq1] = ql1 | qh1;
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + threadIdx.y * QI5_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q5_K       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI5_K) + i/QI5_K + kbxd] = bxi->dm;
#endif // INT8_MMA_AVAILABLE
    }
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / (QI5_K/8);
        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE/8);
        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits
#ifdef INT8_MMA_AVAILABLE
        x_sc[i*MMQ_MMA_TILE_X_K_Q5_K + ksc] = scales8;
#else
        x_sc[i*(WARP_SIZE/8) + i/8   + ksc] = scales8;
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]) + 2 * ((k0 % 16) / 8);
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q5_K_q8_1_impl_mmq(
                &x_qs[i*(QR5_K*WARP_SIZE + 1) + QR5_K*k0], &y_qs[j*MMQ_TILE_Y_K + (QR5_K*k0) % WARP_SIZE], sc, sc+8,
                x_dm[i*(WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[j*MMQ_TILE_Y_K + ((QR5_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + WARP_SIZE*2;
    const int   * x_sc = (const int   *) x_dm + WARP_SIZE/QI5_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
    mma_A   A[ntx][2];
    int   scA[ntx][mma_C::ne/2][2];
    int    mA[ntx][mma_C::ne/2][2];
    half2 dmA[ntx][mma_C::ne/2];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q5_K_Q8_1_MMQ; kvdr += 4) {
            A[n][kvdr/4].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q5_K + (QR5_K*k0 + QR5_K*kvdr), MMQ_MMA_TILE_X_K_Q5_K);
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
                const uint8_t * sc = ((const uint8_t *) &x_sc[i*MMQ_MMA_TILE_X_K_Q5_K + k0/16]) + 2 * ((k0 % 16) / 8);
                const uint8_t *  m = sc + 8;
                scA[n][l][kvdr/4] = sc[kvdr/4];
                mA[n][l][kvdr/4]  =  m[kvdr/4];
            }
        }
    #pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
            dmA[n][l] = x_dm[i*MMQ_MMA_TILE_X_K_Q5_K + k0/QI5_K];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float tmpd[ntx][mma_C::ne] = {{0.0f}};
        float tmpm[ntx][mma_C::ne] = {{0.0f}};
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q5_K_Q8_1_MMQ; kvdr += 4) {
            mma_B   B;
            half2 dsB[mma_C::ne/2];
            B.load(y_qs + j0*MMQ_TILE_Y_K + (2*k0 + 2*kvdr) % WARP_SIZE, MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);
                dsB[l] = y_ds[j*MMQ_TILE_Y_K + ((2*k0 + 2*kvdr)/QI8_1) % (WARP_SIZE/QI8_1)];
            }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma_K8(A[n][kvdr/4], B);
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    tmpd[n][l] += (C.x[l]*scA[n][l/2][kvdr/4]) *  __low2float(dsB[l%2]);
                    tmpm[n][l] += mA[n][l/2][kvdr/4]           * __high2float(dsB[l%2]);
                }
            }
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += __low2float(dmA[n][l/2])*tmpd[n][l] - __high2float(dmA[n][l/2])*tmpm[n][l];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
    int   * x_sc = (int   *) (x_df + WARP_SIZE/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // INT8_MMA_AVAILABLE
    const int kbx  = 0;           // threadIdx.x / QI6_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI6_K
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + kbx;
        const int ky = QR6_K*kqsx;
        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;
        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;
        const int kq0 = ky - ky % QI6_K + threadIdx.x % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + threadIdx.x % (QI6_K/2) + (QI6_K/2);
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*WARP_SIZE + 1)     + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // INT8_MMA_AVAILABLE
    }
    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + threadIdx.y * QI6_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + kbxd;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q6_K       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / 4;
#ifdef INT8_MMA_AVAILABLE
        x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + threadIdx.x % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, threadIdx.x % (QI6_K/8));
#else
        x_sc[i*(WARP_SIZE/8) + i/8   + threadIdx.x % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, threadIdx.x % (QI6_K/8));
#endif // INT8_MMA_AVAILABLE
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/8]);
            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q6_K_q8_1_impl_mmq(
                &x_qs[i*(QR6_K*WARP_SIZE + 1) + QR6_K*k0], &y_qs[j*MMQ_TILE_Y_K + (QR6_K*k0) % WARP_SIZE], sc,
                x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + ((QR6_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * x_sc = (const int   *) x_df + WARP_SIZE/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);
    mma_A   A[ntx][4];
    int   scA[ntx][mma_C::ne/2][4];
    float  dA[ntx][mma_C::ne/2];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q6_K_Q8_1_MMQ; kvdr += 4) {
            A[n][kvdr/2 + 0].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (QR6_K*k0 + QR6_K*kvdr + 0),        MMQ_MMA_TILE_X_K_Q6_K);
            A[n][kvdr/2 + 1].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (QR6_K*k0 + QR6_K*kvdr + mma_A::K), MMQ_MMA_TILE_X_K_Q6_K);
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
                const int8_t * sc = ((const int8_t *) &x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + k0/8]);
                scA[n][l][kvdr/2 + 0] = sc[kvdr/2 + 0];
                scA[n][l][kvdr/2 + 1] = sc[kvdr/2 + 1];
            }
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);
            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q6_K + k0/QI6_K];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float tmp[ntx][mma_C::ne] = {{0.0f}};
#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q6_K_Q8_1_MMQ; kvdr += 4) {
            mma_B B[2];
            float dB[mma_C::ne/2];
            const int k0B = (2*k0 + 2*kvdr) % WARP_SIZE;
            B[0].load(y_qs + j0*MMQ_TILE_Y_K + 0        + k0B, MMQ_TILE_Y_K);
            B[1].load(y_qs + j0*MMQ_TILE_Y_K + mma_B::K + k0B, MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);
                dB[l] = y_df[j*MMQ_TILE_Y_K + ((2*k0 + 2*kvdr)/QI8_1) % (WARP_SIZE/QI8_1)];
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma_K4(A[n][kvdr/2 + 0], B[0]);
                C[1].mma_K4(A[n][kvdr/2 + 1], B[1]);
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][kvdr/2 + 0] + C[1].x[l]*scA[n][l/2][kvdr/2 + 1])*dB[l%2];
                }
            }
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    // NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
template<typename scalar_t, int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
    const float * __restrict__ sum, scalar_t * __restrict__ dst, const int & stride, const int & i_max, const int & j_max) {
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) {
            return;
        }
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (need_check && i > i_max) {
                continue;
            }
            dst[j*stride + i] = sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}
template<typename scalar_t, int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
    const float * __restrict__ sum, scalar_t * __restrict__ dst, const int & stride, const int & i_max, const int & j_max) {
    typedef mma_int_C_I16J8 mma_C;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.
    const int i0 = (threadIdx.y / ntx) * (ntx*mma_C::I);
#ifdef INT8_MMA_AVAILABLE
    static_assert(nwarps*mma_C::I == mmq_y, "nwarps*mma_C::I != mmq_y");
#endif // INT8_MMA_AVAILABLE

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * mma_C::J + mma_C::get_j(l);
                if (j > j_max) {
                    continue;
                }
                const int i = i0 + n*mma_C::I + mma_C::get_i(l);
                if (need_check && i > i_max) {
                    continue;
                }
                dst[j*stride + i] = sum[(j0/mma_C::J + n)*mma_C::ne + l];
            }
        }
    }
}
// -------------------------------------------------------------------------------------------------------------------------------------
template <int mmq_x, int mmq_y, int nwarps, bool need_check, ggml_type type>
struct mmq_type_traits;
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q4_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q4_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_0> {
    static constexpr int              vdr          = VDR_Q5_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q5_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_1> {
    static constexpr int              vdr          = VDR_Q5_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q5_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q8_0> {
    static constexpr int              vdr          = VDR_Q8_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q8_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q2_K> {
    static constexpr int              vdr          = VDR_Q2_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q3_K> {
    static constexpr int              vdr          = VDR_Q3_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q3_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr          = VDR_Q4_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q4_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_K> {
    static constexpr int              vdr          = VDR_Q5_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q5_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};
template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_K> {
    static constexpr int              vdr          = VDR_Q6_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <typename scalar_t, ggml_type type, int mmq_x, int nwarps, bool need_check, bool fixup>
static __device__ void mul_mat_q_process_tile(
    const char * __restrict__ x, const char * __restrict__ yc, scalar_t * __restrict__ dst, float * __restrict__ tmp_fixup,
    const int & ne00, const int & ne01, const int & stride01, const int & ne10, const int & ne11, const int & stride11, const int & ne0,
    const int & it, const int & jt, const int & kb0_start, const int & kb0_stop) {
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              qr         = ggml_cuda_type_traits<type>::qr;
    constexpr int              qi         = ggml_cuda_type_traits<type>::qi;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr int              vdr        = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vdr;
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::load_tiles;
    extern __shared__ char data_mul_mat_q[];
    int * tile_y = (int *) data_mul_mat_q;
    int * tile_x = tile_y + GGML_PAD(mmq_x*(WARP_SIZE + WARP_SIZE/QI8_1), nwarps*WARP_SIZE);
#ifdef INT8_MMA_AVAILABLE
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t<scalar_t> write_back = mmq_write_back_mma<scalar_t, mmq_x, mmq_y, nwarps, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t<scalar_t> write_back = mmq_write_back_dp4a<scalar_t, mmq_x, mmq_y, nwarps, need_check>;
#endif // INT8_MMA_AVAILABLE
    constexpr int blocks_per_warp = WARP_SIZE / qi;
    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};
    const int tile_x_max_i = ne01 - it*mmq_y - 1;
    const int tile_y_max_j = ne11 - jt*mmq_x - 1;
    const int * y = (const int *) yc + jt*(mmq_x*sizeof(block_q8_1_mmq)/sizeof(int));
    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_warp) {
        load_tiles(x, tile_x, stride01*it*mmq_y + kb0, tile_x_max_i, stride01);
#pragma unroll
        for (int kr = 0; kr < qr; ++kr) {
            const int * by0 = y + stride11*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + kr*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
                tile_y[l] = by0[l];
            }
            __syncthreads();
// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k0 = kr*WARP_SIZE/qr; k0 < (kr+1)*WARP_SIZE/qr; k0 += vdr) {
                vec_dot(tile_x, tile_y, sum, k0);
            }
            __syncthreads();
        }
    }
    if (fixup) {
        float * fixup_dst = tmp_fixup + blockIdx.x*(mmq_x*mmq_y);
#ifdef INT8_MMA_AVAILABLE
        mmq_write_back_mma<float, mmq_x, mmq_y, nwarps, false>(sum, fixup_dst, mmq_y, mmq_y - 1, mmq_x - 1);
#else
        mmq_write_back_dp4a<float, mmq_x, mmq_y, nwarps, false>(sum, fixup_dst, mmq_y, mmq_y - 1, mmq_x - 1);
#endif
    } else {
        write_back(sum, dst + jt*mmq_x*ne0 + it*mmq_y, ne0, tile_x_max_i, tile_y_max_j);
    }
}
// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598
template <typename scalar_t, ggml_type type, int mmq_x, int nwarps, bool need_check>
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE*nwarps, 2)
#else
#if __CUDA_ARCH__ >= 700
    __launch_bounds__(WARP_SIZE*nwarps, 1)
#else
    __launch_bounds__(WARP_SIZE*nwarps, type == GGML_TYPE_Q2_K ? 1 : 2)
#endif // __CUDA_ARCH__ >= 700
#endif // defined(USE_ROCM)
static __global__ void mul_mat_q(
    const char * __restrict__ x, const char * __restrict__ yc, scalar_t * __restrict__ dst, float * __restrict__ tmp_fixup,
    const int ne00, const int ne01, const int stride01, const int ne10, const int ne11, const int stride11, const int ne0) {
    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        // NO_DEVICE_CODE;
        return;
    }
    constexpr int qk    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi    = ggml_cuda_type_traits<type>::qi;
    constexpr int mmq_y = get_mmq_y_device();
    // On AMD or old CUDA the performance with stream-k was worse, use conventional tiling instead:
#if (defined(USE_ROCM)) || __CUDA_ARCH__ < 700
    {
        constexpr bool fixup = false;
        mul_mat_q_process_tile<scalar_t, type, mmq_x, nwarps, need_check, fixup>
            (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
                blockIdx.x, blockIdx.y, 0, ne00/qk);
        return;
    }
#endif // (defined(USE_ROCM)) || __CUDA_ARCH__ < 700
    const     int64_t blocks_per_ne00 = ne00 / qk;
    constexpr int     blocks_per_warp = WARP_SIZE / qi;
    const int ntx = (ne11 + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (ne01 + mmq_y - 1) / mmq_y; // Number of tiles y
    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc      = (int64_t) blockIdx.x     *blocks_per_ne00*ntx*nty / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*blocks_per_ne00*ntx*nty / gridDim.x;

    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_warp;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_warp;
    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        const int jt =  kbc /    (blocks_per_ne00*nty);                    // j index of current tile.
        const int it = (kbc - jt*(blocks_per_ne00*nty)) / blocks_per_ne00; // i index of current tile.
        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<scalar_t, type, mmq_x, nwarps, need_check, fixup>
            (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
             it, jt, kb0_start, kb0_stop);
        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;
        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }
    if (kbc >= kbc_stop) {
        return;
    }
    const int jt =  kbc /    (blocks_per_ne00*nty);
    const int it = (kbc - jt*(blocks_per_ne00*nty)) / blocks_per_ne00;
    constexpr bool fixup = true; // Last index writes it data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<scalar_t, type, mmq_x, nwarps, need_check, fixup>
        (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
            it, jt, kb0_start, kb0_stop);
}
template <typename scalar_t, ggml_type type, int mmq_x, int nwarps, bool need_check>
static __global__ void mul_mat_q_stream_k_fixup(
    scalar_t * __restrict__ dst, const float * __restrict__ tmp_last_tile, const int ne00, const int ne01, const int ne11, const int ne0, const int block_num_mmq) {
    constexpr int     mmq_y           = get_mmq_y_device();
    constexpr int     qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int     qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int     blocks_per_warp = WARP_SIZE / qi;
    const     int64_t blocks_per_ne00 = ne00 / qk;
    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};
    const int ntx = (ne11 + mmq_x - 1) / mmq_x;
    const int nty = (ne01 + mmq_y - 1) / mmq_y;
    bool any_fixup = false;
    const int bidx_start = (int)((int64_t)(blockIdx.y*nty + blockIdx.x)     * block_num_mmq / (gridDim.y*gridDim.x));
    const int bidx_stop  = min(block_num_mmq, (int)((int64_t)(blockIdx.y*nty + blockIdx.x + 1) * block_num_mmq / (gridDim.y*gridDim.x) + 1));
    for (int bidx = bidx_start; bidx < bidx_stop; ++bidx) {
        int64_t kbc      = (int64_t) bidx     *blocks_per_ne00*ntx*nty / block_num_mmq;
        int64_t kbc_stop = (int64_t)(bidx + 1)*blocks_per_ne00*ntx*nty / block_num_mmq;

        kbc      -= (kbc      % blocks_per_ne00) % blocks_per_warp;
        kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_warp;
        // Skip fixup tile if the MMQ CUDA block never wrote anything to it:
        if (kbc == kbc_stop || kbc_stop % blocks_per_ne00 == 0) {
            continue;
        }
        const int jt =  kbc_stop /    (blocks_per_ne00*nty);
        const int it = (kbc_stop - jt*(blocks_per_ne00*nty)) / blocks_per_ne00;
        // Skip fixup tile if it's unrelated to the output tile assigned to this CUDA block:
        if (it != blockIdx.x || jt != blockIdx.y) {
            continue;
        }
        any_fixup = true;
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
            }
        }
    }
    if (!any_fixup) {
        return;
    }
    dst += blockIdx.y*mmq_x*ne0 + blockIdx.x*mmq_y;
    const int i_max = ne01 - blockIdx.x*mmq_y - 1;
    const int j_max = ne11 - blockIdx.y*mmq_x - 1;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) {
            return;
        }
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (need_check && i > i_max) {
                continue;
            }
            dst[j*ne0 + i] += sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}
template <typename scalar_t>
struct mmq_args {
    const char * x; const char * y; scalar_t * dst;
    int64_t ne00; int64_t ne01; int64_t stride01;
    int64_t ne10; int64_t ne11; int64_t stride11;
    int64_t ne0;
};
template<ggml_type type>
static int mmq_get_shmem(const int mmq_x, const int mmq_y, const int cc) {
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const int shmem_x = cc >= 750 ? mmq_y*mmq_tile_x_k*sizeof(int) : txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);
    const int shmem_y = mmq_x*sizeof(block_q8_1_mmq);
    return shmem_x + GGML_PAD(shmem_y, MMQ_NWARPS*WARP_SIZE*sizeof(int));
}
template <typename scalar_t, ggml_type type, int mmq_x>
static void launch_mul_mat_q(const mmq_args<scalar_t> & args, cudaStream_t stream) {
    const cuda_device_info cuda_info = get_cuda_info();
    const int cc    = cuda_info.cc;
    const int nsm   = cuda_info.nsm;
    const int mmq_y = get_mmq_y_host(cc);
    const dim3 block_dims(WARP_SIZE, MMQ_NWARPS, 1);
    const int shmem = mmq_get_shmem<type>(mmq_x, mmq_y, cc);

    const int nty = (args.ne01 + mmq_y - 1) / mmq_y;
    const int ntx = (args.ne11 + mmq_x - 1) / mmq_x;
    const dim3 block_nums_xy_tiling(nty, ntx, 1);
    
    const bool use_stream_k = cc >= 700 && cc < 1000000;
    if (!use_stream_k) {
        if (args.ne01 % mmq_y == 0) {
            constexpr bool need_check = false;
            CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
            mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, shmem, stream>>>
                (args.x, args.y, args.dst, nullptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        } else {
            constexpr bool need_check = true;
            CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
            mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, shmem, stream>>>
                (args.x, args.y, args.dst, nullptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        }
        return;
    }
    const dim3 block_nums_mmq(nsm, 1, 1);
    float * tmp_fixup = nullptr;
#if CUDART_VERSION >= 11020
    CUDA_CHECK(cudaMallocAsync(&tmp_fixup, block_nums_mmq.x * mmq_x * mmq_y * sizeof(float), stream));
#else
    CUDA_CHECK(cudaMalloc(&tmp_fixup, block_nums_mmq.x * mmq_x * mmq_y * sizeof(float)));
#endif
    if (args.ne01 % mmq_y == 0) {
        constexpr bool need_check = false;
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_mmq, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, tmp_fixup, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        mul_mat_q_stream_k_fixup<scalar_t, type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, 0, stream>>>
            (args.dst, tmp_fixup, args.ne00, args.ne01, args.ne11, args.ne0, block_nums_mmq.x);
    } else {
        constexpr bool need_check = true;
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        mul_mat_q<scalar_t, type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_mmq, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, tmp_fixup, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        mul_mat_q_stream_k_fixup<scalar_t, type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, 0, stream>>>
            (args.dst, tmp_fixup, args.ne00, args.ne01, args.ne11, args.ne0, block_nums_mmq.x);
    }

#if CUDART_VERSION >= 11020
    CUDA_CHECK(cudaFreeAsync(tmp_fixup, stream));
#else
    CUDA_CHECK(cudaFree(tmp_fixup));
#endif
}
template <typename scalar_t, ggml_type type>
void mul_mat_q_case(const mmq_args<scalar_t> & args, cudaStream_t stream);
#define DECL_MMQ_CASE(scalar, type) \
    template void mul_mat_q_case<scalar, type>(const mmq_args<scalar> & args, cudaStream_t stream)
// fp32 kernel
extern DECL_MMQ_CASE(float, GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(float, GGML_TYPE_Q6_K);
// fp16 kernel
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q6_K);

// bf16 kernel
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(c10::BFloat16, GGML_TYPE_Q6_K);
