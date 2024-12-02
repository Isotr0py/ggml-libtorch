#include <torch/library.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>
#include <iostream>

#include "ggml-quants.hpp"


static torch::Tensor ggml_dequantize(const torch::Tensor W, int type, int64_t m, int64_t n) {
    assert(W.is_contiguous());
    int64_t k = m * n;
    torch::Tensor output = torch::empty({m, n});

    switch (type) {
        case 2:
            dequantize_row_q4_0((block_q4_0 *)W.data_ptr(), (float*) output.data_ptr(), k);
            break;
        case 3:
            dequantize_row_q4_1((block_q4_1 *)W.data_ptr(), (float*) output.data_ptr(), k);
            break;
        case 6:
            dequantize_row_q5_0((block_q5_0 *)W.data_ptr(), (float*) output.data_ptr(), k);
            break;
        case 7:
            dequantize_row_q5_1((block_q5_1 *)W.data_ptr(), (float*) output.data_ptr(), k);
            break;
        case 8:
            dequantize_row_q8_0((block_q8_0 *)W.data_ptr(), (float*) output.data_ptr(), k);
            break;
        default:
            break;
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "dequantize GGML tensor");
}