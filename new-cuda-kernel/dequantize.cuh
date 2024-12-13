#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = __int2half_rn(vui & 0xF);
    v.y = __int2half_rn(vui >> 4);

    v = __hsub2(v, __floats2half2_rn(8.0f, 8.0f));
    v = __hmul2(v, {d, d});
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = __int2half_rn(vui & 0xF);
    v.y = __int2half_rn(vui >> 4);

    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = __int2half_rn((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = __int2half_rn((x[ib].qs[iqs] >>  4) | xh_1);

    v = __hsub2(v, __floats2half2_rn(16.0f, 16.0f));
    v = __hmul2(v, {d, d});
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = __int2half_rn((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = __int2half_rn((x[ib].qs[iqs] >>  4) | xh_1);

    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = __int2half_rn(x[ib].qs[iqs + 0]);
    v.y = __int2half_rn(x[ib].qs[iqs + 1]);

    v = __hmul2(v, {d, d});
}