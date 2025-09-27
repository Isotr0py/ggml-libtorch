#pragma once

#include <stdio.h>

#if defined(__HIPCC__)
  #define HOST_DEVICE_INLINE __host__ __device__
  #define DEVICE_INLINE __device__
  #define HOST_INLINE __host__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
  #define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #define DEVICE_INLINE __device__ __forceinline__
  #define HOST_INLINE __host__ __forceinline__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
  #define HOST_INLINE inline
#endif

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

struct cuda_device_info {
    int     cc;                 // compute capability
    int     nsm;                // number of streaming multiprocessors
    size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    bool    vmm;                // virtual memory support
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
};

cuda_device_info get_cuda_info() {
    int id;
    // CUDA_CHECK(cudaGetDevice(&id));
    cudaGetDevice(&id);

    cudaDeviceProp prop;
    // CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
    cudaGetDeviceProperties(&prop, id);

    cuda_device_info info;
    info.cc = prop.major*100 + prop.minor * 10;
    info.nsm = prop.multiProcessorCount;
    info.smpb = prop.sharedMemPerBlock;
    info.smpbo = prop.sharedMemPerBlockOptin;
    info.vmm = prop.managedMemory;
    info.vmm_granularity = prop.managedMemory ? prop.managedMemory : 0;
    info.total_vram = prop.totalGlobalMem;

    return info;
}

namespace cuda_utils {

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

};  // namespace cuda_utils