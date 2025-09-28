#include "mmq_kernel_template.h"

DECL_MMQ_CASE(float, GGML_TYPE_Q2_K);
DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q2_K);
DECL_MMQ_CASE(nv_bfloat16, GGML_TYPE_Q2_K);