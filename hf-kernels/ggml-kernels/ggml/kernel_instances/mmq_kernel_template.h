#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "../cuda_compat.h"
#include "../cuda_utils.h" 
#include "../dispatch_utils.h"

#include "../ggml-common.h"
#include "../vecdotq.cuh"
#include "../mmq.cuh"
