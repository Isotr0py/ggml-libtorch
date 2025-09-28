#include "mmq_kernel_template.h"

DECL_MMQ_CASE(float, GGML_TYPE_Q8_0);
DECL_MMQ_CASE(c10::Half, GGML_TYPE_Q8_0);
DECL_MMQ_CASE(nv_bfloat16, GGML_TYPE_Q8_0);