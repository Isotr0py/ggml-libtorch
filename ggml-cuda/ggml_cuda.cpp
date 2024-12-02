#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "Deequantize Kernel (CUDA)");
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "MMQ Kernel (CUDA)");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "MMVQ Kernel (CUDA)");
}