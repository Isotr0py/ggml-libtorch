#include <torch/library.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>
#include <iostream>

#include "ggml-common.hpp"


static torch::Tensor dequantize_row_q4_0(const torch::Tensor &W, const torch::Tensor &output) {
    assert(W.is_contiguous());
    int64_t nb = W.size(0);
    const block_q4_0 * x = (const block_q4_0 *) W.data_ptr();
    float * y = (float*) output.data_ptr();

    static const int qk = QK4_0;
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
    return output;
};

static torch::Tensor dequantize_row_q4_1(const torch::Tensor &W, const torch::Tensor &output) {
    assert(W.is_contiguous());
    int64_t nb = W.size(0);
    const block_q4_1 * x = (const block_q4_1 *) W.data_ptr();
    float * y = (float*) output.data_ptr();

    static const int qk = QK4_1;
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;
        const float m = x[i].m;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
    return output;
}

static torch::Tensor dequantize_row_q5_0(const torch::Tensor &W, const torch::Tensor &output) {
    assert(W.is_contiguous());
    int64_t nb = W.size(0);
    const block_q5_0 * x = (const block_q5_0 *) W.data_ptr();
    float * y = (float*) output.data_ptr();

    static const int qk = QK5_0;
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
    return output;
}


static torch::Tensor dequantize_row_q5_1(const torch::Tensor &W, const torch::Tensor &output) {
    assert(W.is_contiguous());
    int64_t nb = W.size(0);
    const block_q5_1 * x = (const block_q5_1 *) W.data_ptr();
    float * y = (float*) output.data_ptr();

    static const int qk = QK5_1;
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;
        const float m = x[i].m;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
    return output;
}

static torch::Tensor dequantize_row_q8_0(const torch::Tensor &W, const torch::Tensor &output) {
    assert(W.is_contiguous());
    int64_t nb = W.size(0);
    const block_q8_0 * x = (const block_q8_0 *) W.data_ptr();
    float * y = (float*) output.data_ptr();

    static const int qk = QK8_0;
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequantize_row_q4_0", &dequantize_row_q4_0, "dequantize Q4_0 tensor");
  m.def("dequantize_row_q4_1", &dequantize_row_q4_1, "dequantize Q4_1 tensor");
  m.def("dequantize_row_q5_0", &dequantize_row_q5_0, "dequantize Q5_0 blocks");
  m.def("dequantize_row_q5_1", &dequantize_row_q5_1, "dequantize Q5_1 blocks");
  m.def("dequantize_row_q8_0", &dequantize_row_q8_0, "dequantize Q8_0 blocks");
}