#pragma once

#include <torch/torch.h>

// CUDA forward declarations
torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row);
