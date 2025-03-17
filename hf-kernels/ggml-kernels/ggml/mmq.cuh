// copied from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmq.cu

#define MMQ_TILE_Y_K (WARP_SIZE_GGUF + WARP_SIZE_GGUF/QI8_1)

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};


// tiles loading function
template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE_GGUF + 1) + threadIdx.x] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));

            int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE_GGUF];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI4_0) % WARP_SIZE_GGUF];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE_GGUF + 1) + k0], u, x_df[i*(WARP_SIZE_GGUF/QI4_0) + i/QI4_0 + k0/QI4_0],
                y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI4_1;
    const int kqsx = threadIdx.x % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE_GGUF + 1) + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + threadIdx.y * QI4_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE_GGUF/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));

            int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE_GGUF];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI4_1) % WARP_SIZE_GGUF];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE_GGUF + 1) + k0], u, x_dm[i*(WARP_SIZE_GGUF/QI4_1) + i/QI4_1 + k0/QI4_1],
                y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

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

        x_qs[i * (2*WARP_SIZE_GGUF + 1) + 2*threadIdx.x+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_qs[i * (2*WARP_SIZE_GGUF + 1) + 2*threadIdx.x+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + threadIdx.y * QI5_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_0_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const int   * y_qs  = (const int   *) y + 4;
    const float * y_df  = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));
            const int index_bx = i*(WARP_SIZE_GGUF/QI5_0) + i/QI5_0 + k0/QI5_0;

            int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE_GGUF];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI5_0) % WARP_SIZE_GGUF];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
                (&x_qs[i*(2*WARP_SIZE_GGUF + 1) + 2*k0], u, x_dmf[index_bx], y_df[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

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

        x_qs[i * (2*WARP_SIZE_GGUF + 1) + 2*threadIdx.x+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_qs[i * (2*WARP_SIZE_GGUF + 1) + 2*threadIdx.x+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + threadIdx.y * QI5_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE_GGUF/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_1_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const int   * y_qs  = (const int   *) y + 4;
    const half2 * y_ds  = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));
            const int index_bx = i*(WARP_SIZE_GGUF/QI5_1) + i/QI5_1 + k0/QI5_1;

            int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE_GGUF];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI5_1) % WARP_SIZE_GGUF];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                (&x_qs[i*(2*WARP_SIZE_GGUF + 1) + 2*k0], u, x_dm[index_bx], y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI8_0;
    const int kqsx = threadIdx.x % QI8_0;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE_GGUF + 1) + threadIdx.x] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + threadIdx.y * QI8_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const int   * y_qs  = (const int   *) y + 4;
    const float * y_df  = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE_GGUF + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0], x_dmf[i*(WARP_SIZE_GGUF/QI8_0) + i/QI8_0 + k0/QI8_0],
                y_df[j*MMQ_TILE_Y_K + k0/QI8_1]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

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
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 1, WARP_SIZE_GGUF);
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 2, WARP_SIZE_GGUF);

            if (kqsx % QR2_K != 0) {
                continue;
            }

            x_qs[i*(WARP_SIZE_GGUF + 1) + k] = x_qs_k;
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

        x_dm[i*(WARP_SIZE_GGUF + 1) + threadIdx.x] = x_dm_ik;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q2_K_q8_1_impl_mmq(
                &x_qs[i*(WARP_SIZE_GGUF + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + (QR2_K*k0) % WARP_SIZE_GGUF],
                &x_dm[i*(WARP_SIZE_GGUF + 1) + k0], y_df[j*MMQ_TILE_Y_K + ((QR2_K*k0) % WARP_SIZE_GGUF)/QI8_1]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

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
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 1, WARP_SIZE_GGUF);

            if (kqsx % 2 != 0) {
                continue;
            }

            x_qs[i*(2*WARP_SIZE_GGUF + 1) + k/2] = x_qs_k;
        }
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI3_K;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + threadIdx.y * QI3_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE_GGUF/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE_GGUF/4)) / (QI3_K/4);

        const int ksc = threadIdx.x % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE_GGUF/4) + i / 4 + threadIdx.x % (WARP_SIZE_GGUF/4)] = sc;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const int kbx  = k0 / QI3_K;
            const int ky  = (k0 % QI3_K) * QR3_K;

            const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE_GGUF/4) + i/4 + kbx*4)) + ky/4;

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q3_K_q8_1_impl_mmq(
                &x_qs[i*(2*WARP_SIZE_GGUF + 1) + 2*k0], &y_qs[j*MMQ_TILE_Y_K + (k0*QR3_K) % WARP_SIZE_GGUF], scales,
                x_df[i*(WARP_SIZE_GGUF/QI3_K) + i/QI3_K + kbx], y_df[j*MMQ_TILE_Y_K + ((k0*QR3_K) % WARP_SIZE_GGUF)/QI8_1]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

    const int kbx  = 0;           // threadIdx.x / QI4_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI4_K

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE_GGUF + 1) + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + threadIdx.y * QI4_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE_GGUF/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE_GGUF/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE_GGUF/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales8;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + k0/16]) + 2*((k0 % 16) / 8);

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q4_K_q8_1_impl_mmq(
                &x_qs[i*(WARP_SIZE_GGUF + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + (QR4_K*k0) % WARP_SIZE_GGUF], sc, sc+8,
                x_dm[i*(WARP_SIZE_GGUF/QI4_K) + i/QI4_K], &y_ds[j*MMQ_TILE_Y_K + ((QR4_K*k0) % WARP_SIZE_GGUF)/QI8_1]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

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

        x_qs[i * (2*WARP_SIZE_GGUF + 1) + kq0] = ql0 | qh0;
        x_qs[i * (2*WARP_SIZE_GGUF + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + threadIdx.y * QI5_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE_GGUF/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE_GGUF/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE_GGUF/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales8;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const int   * y_qs  = (const int   *) y + 4;
    const half2 * y_ds  = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + k0/16]) + 2 * ((k0 % 16) / 8);

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q5_K_q8_1_impl_mmq(
                &x_qs[i*(QR5_K*WARP_SIZE_GGUF + 1) + QR5_K*k0], &y_qs[j*MMQ_TILE_Y_K + (QR5_K*k0) % WARP_SIZE_GGUF], sc, sc+8,
                x_dm[i*(WARP_SIZE_GGUF/QI5_K) + i/QI5_K], &y_ds[j*MMQ_TILE_Y_K + ((QR5_K*k0) % WARP_SIZE_GGUF)/QI8_1]);
        }
    }
}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

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

        x_qs[i * (2*WARP_SIZE_GGUF + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i * (2*WARP_SIZE_GGUF + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI6_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + threadIdx.y * QI6_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE_GGUF/8)) / 4;

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + threadIdx.x % (WARP_SIZE_GGUF/8)] = get_int_from_int8(bxi->scales, threadIdx.x % (QI6_K/8));
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const float * x_dmf = (const float *) x_dm;
    const int   * y_qs  = (const int   *) y + 4;
    const float * y_df  = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = i0 + threadIdx.x;

            const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + k0/8]);

            sum[j0/nwarps*mmq_y/WARP_SIZE_GGUF + i0/WARP_SIZE_GGUF] += vec_dot_q6_K_q8_1_impl_mmq(
                &x_qs[i*(QR6_K*WARP_SIZE_GGUF + 1) + QR6_K*k0], &y_qs[j*MMQ_TILE_Y_K + (QR6_K*k0) % WARP_SIZE_GGUF], sc,
                x_dmf[i*(WARP_SIZE_GGUF/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + ((QR6_K*k0) % WARP_SIZE_GGUF)/QI8_1]);
        }
    }
}


// utility functions

template <typename scalar_t, int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(const float * __restrict__ sum, scalar_t * __restrict__ dst, const int & ne0, const int & ne1) {
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = blockIdx.y*mmq_x + j0 + threadIdx.y;

        if (j >= ne1) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE_GGUF) {
            const int i = blockIdx.x*mmq_y + i0 + threadIdx.x;

            if (need_check && i >= ne0) {
                continue;
            }

            dst[j*ne0 + i] = sum[(j0/nwarps) * (mmq_y/WARP_SIZE_GGUF) + i0/WARP_SIZE_GGUF];
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------
typedef void (*load_tiles_mmq_t)(
    const void * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride);
typedef void (*vec_dot_mmq_t)(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0);
// typedef void (*mmq_write_back_t)(const float * __restrict__ sum, (typename scalar_t) * __restrict__ dst, const int & ne0, const int & ne1);

template <int mmq_x, int mmq_y, int nwarps, bool need_check, ggml_type type>
struct mmq_type_traits;

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_0> {
    static constexpr int              vdr          = VDR_Q5_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_1> {
    static constexpr int              vdr          = VDR_Q5_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q8_0> {
    static constexpr int              vdr          = VDR_Q8_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q8_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q2_K> {
    static constexpr int              vdr          = VDR_Q2_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q3_K> {
    static constexpr int              vdr          = VDR_Q3_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr          = VDR_Q4_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_K> {
    static constexpr int              vdr          = VDR_Q5_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_K> {
    static constexpr int              vdr          = VDR_Q6_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

static bool mmq_need_sum(const ggml_type type_x) {
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
    }
    return false;
}


#if __CUDA_ARCH__ >= CC_VOLTA || defined(USE_ROCM)
static constexpr __device__ int get_mmq_y_device(int mmq_x) {
    return mmq_x >= 32 ? 128 : 64;
}
#else
static constexpr __device__ int get_mmq_y_device(int mmq_x) {
    return 64;
}
#endif // __CUDA_ARCH__ >= CC_VOLTA

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

#define TILE_X_SIZES_Q4_0 tile_x_sizes{mmq_y*WARP_SIZE_GGUF   + mmq_y, mmq_y*WARP_SIZE_GGUF/QI4_0 + mmq_y/QI4_0, 0}
#define TILE_X_SIZES_Q4_1 tile_x_sizes{mmq_y*WARP_SIZE_GGUF   + mmq_y, mmq_y*WARP_SIZE_GGUF/QI4_1 + mmq_y/QI4_1, 0}
#define TILE_X_SIZES_Q5_0 tile_x_sizes{mmq_y*WARP_SIZE_GGUF*2 + mmq_y, mmq_y*WARP_SIZE_GGUF/QI5_0 + mmq_y/QI5_0, 0}
#define TILE_X_SIZES_Q5_1 tile_x_sizes{mmq_y*WARP_SIZE_GGUF*2 + mmq_y, mmq_y*WARP_SIZE_GGUF/QI5_1 + mmq_y/QI5_1, 0}
#define TILE_X_SIZES_Q8_0 tile_x_sizes{mmq_y*WARP_SIZE_GGUF   + mmq_y, mmq_y*WARP_SIZE_GGUF/QI8_0 + mmq_y/QI8_0, 0}
#define TILE_X_SIZES_Q2_K tile_x_sizes{mmq_y*WARP_SIZE_GGUF   + mmq_y, mmq_y*WARP_SIZE_GGUF       + mmq_y,       0}
#define TILE_X_SIZES_Q3_K tile_x_sizes{mmq_y*WARP_SIZE_GGUF*2 + mmq_y, mmq_y*WARP_SIZE_GGUF/QI3_K + mmq_y/QI3_K, mmq_y*WARP_SIZE_GGUF/4 + mmq_y/4}
#define TILE_X_SIZES_Q4_K tile_x_sizes{mmq_y*WARP_SIZE_GGUF   + mmq_y, mmq_y*WARP_SIZE_GGUF/QI4_K + mmq_y/QI4_K, mmq_y*WARP_SIZE_GGUF/8 + mmq_y/8}
#define TILE_X_SIZES_Q5_K tile_x_sizes{mmq_y*WARP_SIZE_GGUF*2 + mmq_y, mmq_y*WARP_SIZE_GGUF/QI5_K + mmq_y/QI5_K, mmq_y*WARP_SIZE_GGUF/8 + mmq_y/8}
#define TILE_X_SIZES_Q6_K tile_x_sizes{mmq_y*WARP_SIZE_GGUF*2 + mmq_y, mmq_y*WARP_SIZE_GGUF/QI6_K + mmq_y/QI6_K, mmq_y*WARP_SIZE_GGUF/8 + mmq_y/8}

#define GET_TILE_X_SIZES_BODY                           \
    return type == GGML_TYPE_Q4_0 ? TILE_X_SIZES_Q4_0 : \
        type == GGML_TYPE_Q4_1 ? TILE_X_SIZES_Q4_1 :    \
        type == GGML_TYPE_Q5_0 ? TILE_X_SIZES_Q5_0 :    \
        type == GGML_TYPE_Q5_1 ? TILE_X_SIZES_Q5_1 :    \
        type == GGML_TYPE_Q8_0 ? TILE_X_SIZES_Q8_0 :    \
        type == GGML_TYPE_Q2_K ? TILE_X_SIZES_Q2_K :    \
        type == GGML_TYPE_Q3_K ? TILE_X_SIZES_Q3_K :    \
        type == GGML_TYPE_Q4_K ? TILE_X_SIZES_Q4_K :    \
        type == GGML_TYPE_Q5_K ? TILE_X_SIZES_Q5_K :    \
        type == GGML_TYPE_Q6_K ? TILE_X_SIZES_Q6_K :    \
        tile_x_sizes{0, 0, 0}

static tile_x_sizes get_tile_x_sizes_host(const ggml_type type, const int mmq_y) {
    GET_TILE_X_SIZES_BODY;
}

template <int mmq_y>
static constexpr __device__ tile_x_sizes get_tile_x_sizes_device(ggml_type type) {
    GET_TILE_X_SIZES_BODY;
}

// Some variables different from llama.cpp for better comprehension:
// ne00 -> ncols_x
// ne01 -> nrows_x
// ne10 -> ncols_y
// ne11 -> nrows_y
// stride01 -> stride_col_x
// stride11 -> stride_row_y
// ne0 -> nrows_dst
template <typename scalar_t, ggml_type type, int mmq_x, int nwarps, bool need_check>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {

    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              qr         = ggml_cuda_type_traits<type>::qr;
    constexpr int              qi         = ggml_cuda_type_traits<type>::qi;
    constexpr int              mmq_y      = get_mmq_y_device(mmq_x);
    constexpr int              vdr        = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vdr;
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::load_tiles;

    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_dp4a;
    // constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, nwarps, need_check>;

    constexpr tile_x_sizes txs = get_tile_x_sizes_device<mmq_y>(type);

    extern __shared__ char data_mul_mat_q[];
    int   * tile_x_qs = (int   *)  data_mul_mat_q;
    half2 * tile_x_dm = (half2 *) (tile_x_qs + txs.qs);
    int   * tile_x_sc = (int   *) (tile_x_dm + txs.dm);
    int   * tile_y    = (int   *) (tile_x_sc + txs.sc); // [mmq_x * (WARP_SIZE_GGUF + WARP_SIZE_GGUF/QI8_1)]

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_warp = WARP_SIZE_GGUF / qi;

    const int & ne1 = nrows_y;

    const int tile_x_max_i = nrows_x - blockIdx.x*mmq_y - 1;

    const int * y = (const int *) vy + blockIdx.y*(mmq_x*sizeof(block_q8_1_mmq)/sizeof(int));

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE_GGUF)] = {0.0f};

    for (int kb0 = 0; kb0 < blocks_per_row_x; kb0 += blocks_per_warp) {

        load_tiles(vx, tile_x_qs, tile_x_dm, tile_x_sc, stride_col_x*blockIdx.x*mmq_y + kb0, tile_x_max_i, stride_col_x);

    #pragma unroll
        for (int kr = 0; kr < qr && kb0 + kr * blocks_per_warp/qr < blocks_per_row_x; ++kr) {
            const int * by0 = y + stride_row_y*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + kr*sizeof(block_q8_1_mmq)/sizeof(int));
    #pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE_GGUF) {
                int l = l0 + threadIdx.y*WARP_SIZE_GGUF + threadIdx.x;

                tile_y[l] = by0[l];
            }

            __syncthreads();

    // #pragma unroll // unrolling this loop causes too much register pressure
            for (int k0 = kr*WARP_SIZE_GGUF/qr; k0 < (kr+1)*WARP_SIZE_GGUF/qr; k0 += vdr) {
                vec_dot(tile_x_qs, tile_x_dm, tile_x_sc, tile_y, sum, k0);
            }

            __syncthreads();
        }
    }

    // write_back(sum, dst, nrows_dst, ne1);
    mmq_write_back_dp4a<scalar_t, mmq_x, mmq_y, nwarps, need_check>(sum, dst, nrows_dst, ne1);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_0, 2)
#endif
mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_0;
    const int nwarps = NWARPS_Q4_0;

    mul_mat_q<scalar_t, GGML_TYPE_Q4_0, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q4_0;
    const int mmq_y  =  MMQ_Y_Q4_0;
    const int nwarps = NWARPS_Q4_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_1, 2)
#endif
mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_1;
    const int nwarps = NWARPS_Q4_1;

    mul_mat_q<scalar_t, GGML_TYPE_Q4_1, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q4_1;
    const int mmq_y  =  MMQ_Y_Q4_1;
    const int nwarps = NWARPS_Q4_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_0, 2)
#endif
mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    mul_mat_q<scalar_t, GGML_TYPE_Q5_0, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_1, 2)
#endif
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    mul_mat_q<scalar_t, GGML_TYPE_Q5_1, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q8_0, 2)
#endif
mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    mul_mat_q<scalar_t, GGML_TYPE_Q8_0, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q2_K, 2)
#endif
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    mul_mat_q<scalar_t, GGML_TYPE_Q2_K, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q3_K, 2)
#endif
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    mul_mat_q<scalar_t, GGML_TYPE_Q3_K, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_K, 2)
#endif
mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    mul_mat_q<scalar_t, GGML_TYPE_Q4_K, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_K, 2)
#endif
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    mul_mat_q<scalar_t, GGML_TYPE_Q5_K, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
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

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q6_K, 2)
#endif
mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_x, const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    mul_mat_q<scalar_t, GGML_TYPE_Q6_K, mmq_x, nwarps, need_check>(vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, const int stride_col_x,
    const int ncols_y, const int nrows_y, const int stride_row_y, const int nrows_dst, cudaStream_t stream) {
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
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, stride_col_x, ncols_y, nrows_y, stride_row_y, nrows_dst);
    }
}
