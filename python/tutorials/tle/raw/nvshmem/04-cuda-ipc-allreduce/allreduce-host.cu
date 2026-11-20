#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kMaxBlocks = 36;
constexpr int kMaxRanks = 8;

struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t end[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t epoch[kMaxBlocks];
};

int as_int(cudaError_t result) { return static_cast<int>(result); }

} // namespace

extern "C" size_t tle_ipc_handle_size() { return sizeof(cudaIpcMemHandle_t); }

extern "C" size_t tle_ipc_signal_size() { return sizeof(Signal); }

extern "C" int tle_ipc_allocate(void **pointer, unsigned char *handle,
                                size_t bytes) {
  void *allocation = nullptr;
  cudaError_t result = cudaMalloc(&allocation, bytes);
  if (result != cudaSuccess)
    return as_int(result);

  result = cudaMemset(allocation, 0, bytes);
  if (result != cudaSuccess) {
    cudaFree(allocation);
    return as_int(result);
  }
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    cudaFree(allocation);
    return as_int(result);
  }

  cudaIpcMemHandle_t ipc_handle;
  result = cudaIpcGetMemHandle(&ipc_handle, allocation);
  if (result != cudaSuccess) {
    cudaFree(allocation);
    return as_int(result);
  }

  std::memcpy(handle, &ipc_handle, sizeof(ipc_handle));
  *pointer = allocation;
  return as_int(cudaSuccess);
}

extern "C" int tle_ipc_export_pointer(const void *pointer,
                                      unsigned char *handle, size_t *offset) {
  void *allocation_base = nullptr;
  const CUresult driver_result = cuPointerGetAttribute(
      &allocation_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
      reinterpret_cast<CUdeviceptr>(pointer));
  if (driver_result != CUDA_SUCCESS)
    return as_int(cudaErrorInvalidDevicePointer);

  cudaIpcMemHandle_t ipc_handle;
  cudaError_t result = cudaIpcGetMemHandle(&ipc_handle, allocation_base);
  if (result != cudaSuccess)
    return as_int(result);

  const size_t pointer_offset = static_cast<const char *>(pointer) -
                                static_cast<const char *>(allocation_base);
  std::memcpy(handle, &ipc_handle, sizeof(ipc_handle));
  *offset = pointer_offset;
  return as_int(cudaSuccess);
}

extern "C" int tle_ipc_open(const unsigned char *handle, void **pointer) {
  cudaIpcMemHandle_t ipc_handle;
  std::memcpy(&ipc_handle, handle, sizeof(ipc_handle));
  return as_int(cudaIpcOpenMemHandle(pointer, ipc_handle,
                                     cudaIpcMemLazyEnablePeerAccess));
}

extern "C" int tle_ipc_close(void *pointer) {
  return as_int(cudaIpcCloseMemHandle(pointer));
}

extern "C" int tle_ipc_free(void *pointer) { return as_int(cudaFree(pointer)); }

extern "C" const char *tle_ipc_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}
