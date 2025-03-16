// copied from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmq.cu
template <typename scalar_t, int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE_GGUF / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE_GGUF];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE_GGUF/QI8_1];

    float sum[mmq_y/WARP_SIZE_GGUF][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr && ib0 + ir * blocks_per_warp/qr < blocks_per_row_x; ++ir) {
            const int kqs = ir*WARP_SIZE_GGUF + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses
                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];
                const int index_y = (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF/QI8_1)) % mmq_x;
                const int kby = threadIdx.x % (WARP_SIZE_GGUF/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE_GGUF/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE_GGUF/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2float(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE_GGUF/qr; k < (ir+1)*WARP_SIZE_GGUF/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
                        sum[i/WARP_SIZE_GGUF][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + threadIdx.y;
        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            const int row_dst = row_dst_0 + threadIdx.x + i;
            if (row_dst >= nrows_dst) {
                continue;
            }
            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE_GGUF][j/nwarps];
        }
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_0  64
#define  MMQ_Y_Q4_0  128
#define NWARPS_Q4_0  8
#else
#define  MMQ_X_Q4_0 4
#define  MMQ_Y_Q4_0 32
#define NWARPS_Q4_0 4
#endif


template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI4_0) + mmq_y/QI4_0];
    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE_GGUF/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI4_0) + i / QI4_0 + kbxd] = __half2float(bxi->d);
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_0, 2)
#endif
mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_0;
    const int mmq_y  =  MMQ_Y_Q4_0;
    const int nwarps = NWARPS_Q4_0;

    mul_mat_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int mmq_x  =  MMQ_X_Q4_0;
    int mmq_y  =  MMQ_Y_Q4_0;
    int nwarps = NWARPS_Q4_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_1 64
#define  MMQ_Y_Q4_1 128
#define NWARPS_Q4_1 8
#else
#define  MMQ_X_Q4_1 4
#define  MMQ_Y_Q4_1 32
#define NWARPS_Q4_1 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE_GGUF) +     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI4_1) + mmq_y/QI4_1];
    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_1, 2)
#endif
mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_1;
    const int mmq_y  =  MMQ_Y_Q4_1;
    const int nwarps = NWARPS_Q4_1;

    mul_mat_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int mmq_x  =  MMQ_X_Q4_1;
    int mmq_y  =  MMQ_Y_Q4_1;
    int nwarps = NWARPS_Q4_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_0 64
#define  MMQ_Y_Q5_0 128
#define NWARPS_Q5_0 8
#else
#define  MMQ_X_Q5_0 4
#define  MMQ_Y_Q5_0 32
#define NWARPS_Q5_0 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;
        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI5_0) + i / QI5_0 + kbxd] = __half2float(bxi->d);
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_0, 2)
#endif
mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_0;
    const int mmq_y  =  MMQ_Y_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    mul_mat_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q5_0;
    const int mmq_y  =  MMQ_Y_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_1 64
#define  MMQ_Y_Q5_1 128
#define NWARPS_Q5_1 8
#else
#define  MMQ_X_Q5_1 4
#define  MMQ_Y_Q5_1 32
#define NWARPS_Q5_1 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE_GGUF/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_1, 2)
#endif
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int mmq_y  =  MMQ_Y_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    mul_mat_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int mmq_y  =  MMQ_Y_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q8_0 64
#define  MMQ_Y_Q8_0 128
#define NWARPS_Q8_0 8
#else
#define  MMQ_X_Q8_0 4
#define  MMQ_Y_Q8_0 32
#define NWARPS_Q8_0 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI8_0) + i / QI8_0 + kbxd] = __half2float(bxi->d);
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q8_0, 2)
#endif
mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int mmq_y  =  MMQ_Y_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    mul_mat_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int mmq_y  =  MMQ_Y_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q2_K 64
#define  MMQ_Y_Q2_K 128
#define NWARPS_Q2_K 8
#else
#define  MMQ_X_Q2_K 4
#define  MMQ_Y_Q2_K 32
#define NWARPS_Q2_K 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4);

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/4)) / (QI2_K/4);
        x_sc[i * (WARP_SIZE_GGUF/4) + i / 4 + k % (WARP_SIZE_GGUF/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q2_K, 2)
#endif
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int mmq_y  =  MMQ_Y_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    mul_mat_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int mmq_y  =  MMQ_Y_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q3_K 64
#define  MMQ_Y_Q3_K 128
#define NWARPS_Q3_K 8
#else
#define  MMQ_X_Q3_K 4
#define  MMQ_Y_Q3_K 32
#define NWARPS_Q3_K 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE_GGUF/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI3_K) + i / QI3_K + kbxd] = __half2float(bxi->d);
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE_GGUF/2);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/2)) / (QI3_K/2);
        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE_GGUF/2) + i / 2 + k % (WARP_SIZE_GGUF/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE_GGUF/4) + i / 4 + k % (WARP_SIZE_GGUF/4)] = sc;
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q3_K, 2)
#endif
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const int mmq_x  =  MMQ_X_Q3_K;
    const int mmq_y  =  MMQ_Y_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    mul_mat_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q3_K;
    const int mmq_y  =  MMQ_Y_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_K 64
#define  MMQ_Y_Q4_K 128
#define NWARPS_Q4_K 8
#else
#define  MMQ_X_Q4_K 4
#define  MMQ_Y_Q4_K 32
#define NWARPS_Q4_K 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE_GGUF/8);
        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales8;
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_K, 2)
#endif
mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int mmq_y  =  MMQ_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    mul_mat_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int mmq_y  =  MMQ_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_K 64
#define  MMQ_Y_Q5_K 128
#define NWARPS_Q5_K 8
#else
#define  MMQ_X_Q5_K 4
#define  MMQ_Y_Q5_K 32
#define NWARPS_Q5_K 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE_GGUF/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales8;
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_K, 2)
#endif
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    mul_mat_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q6_K 64
#define  MMQ_Y_Q6_K 128
#define NWARPS_Q6_K 8
#else
#define  MMQ_X_Q6_K 4
#define  MMQ_Y_Q6_K 32
#define NWARPS_Q6_K 4
#endif

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI6_K) + i / QI6_K + kbxd] = __half2float(bxi->d);
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / 4;

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + k % (WARP_SIZE_GGUF/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q6_K, 2)
#endif
mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    mul_mat_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}
