// Copyright 2026, The FlagOS Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK_RET(stmt, code)                                           \
  do {                                                                       \
    cudaError_t result_ = (stmt);                                            \
    if (result_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "[%s:%d] CUDA failed: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(result_));                             \
      return (code);                                                         \
    }                                                                        \
  } while (0)

__global__ void publish_ready_kernel(uint64_t *slot, uint64_t epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    *slot = epoch;
    __threadfence_system();
  }
}

static int launch_publish_ready(uint64_t *slot, uint64_t epoch,
                                cudaStream_t stream) {
  publish_ready_kernel<<<1, 1, 0, stream>>>(slot, epoch);
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::fprintf(stderr, "[%s:%d] publish ready launch failed: %s\n",
                 __FILE__, __LINE__, cudaGetErrorString(result));
    return -1;
  }
  return 0;
}

extern "C" int tle_ar_rmsnorm_workspace_create(
    int capacity_tokens,
    int hidden_size,
    void **input_unicast,
    void **ready_unicast,
    cudaStream_t *stream,
    int *mype,
    int *npes,
    int *mype_in_node,
    int *npes_in_node) {
  if (capacity_tokens <= 0 || hidden_size != 2048 || input_unicast == nullptr ||
      ready_unicast == nullptr || stream == nullptr || mype == nullptr ||
      npes == nullptr || mype_in_node == nullptr || npes_in_node == nullptr) {
    return -1;
  }

  *mype = nvshmem_my_pe();
  *npes = nvshmem_n_pes();
  *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  *npes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

  if (*npes <= 0 || *npes != *npes_in_node) {
    return -2;
  }

  CUDA_CHECK_RET(cudaSetDevice(*mype_in_node), -3);

  int least_priority = 0;
  int greatest_priority = 0;
  CUDA_CHECK_RET(cudaDeviceGetStreamPriorityRange(&least_priority,
                                                  &greatest_priority),
                 -3);
  CUDA_CHECK_RET(cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking,
                                              greatest_priority),
                 -3);

  const size_t bytes = static_cast<size_t>(capacity_tokens) *
                       static_cast<size_t>(hidden_size) * sizeof(__nv_bfloat16);
  *input_unicast = nvshmem_malloc(bytes);
  *ready_unicast =
      nvshmem_calloc(static_cast<size_t>(*npes), sizeof(uint64_t));

  if (*input_unicast == nullptr || *ready_unicast == nullptr) {
    if (*ready_unicast != nullptr) {
      nvshmem_free(*ready_unicast);
      *ready_unicast = nullptr;
    }
    if (*input_unicast != nullptr) {
      nvshmem_free(*input_unicast);
      *input_unicast = nullptr;
    }
    cudaStreamDestroy(*stream);
    *stream = nullptr;
    return -4;
  }

  nvshmem_barrier_all();
  return 0;
}

extern "C" void *tle_ar_rmsnorm_peer_workspace_ptr(void *workspace, int peer) {
  return nvshmem_ptr(workspace, peer);
}

extern "C" int tle_ar_rmsnorm_sync_ready(void *ready_unicast,
                                         uint64_t epoch,
                                         int rank,
                                         int npes,
                                         cudaStream_t stream) {
  if (ready_unicast == nullptr || stream == nullptr || epoch == 0 || rank < 0 ||
      rank >= npes || npes <= 0) {
    return -1;
  }

  uint64_t *ready_slots = static_cast<uint64_t *>(ready_unicast);
  int rc = launch_publish_ready(ready_slots + rank, epoch, stream);
  if (rc != 0) {
    return -2;
  }

  for (int peer = 0; peer < npes; ++peer) {
    if (peer == rank) {
      continue;
    }
    nvshmemx_signal_op_on_stream(ready_slots + rank, epoch,
                                 NVSHMEM_SIGNAL_SET, peer, stream);
  }

  for (int src = 0; src < npes; ++src) {
    nvshmemx_signal_wait_until_on_stream(ready_slots + src, NVSHMEM_CMP_GE,
                                         epoch, stream);
  }

  return 0;
}

extern "C" int tle_ar_rmsnorm_workspace_destroy(void *input_unicast,
                                                void *ready_unicast,
                                                cudaStream_t stream) {
  if (stream == nullptr) {
    return -1;
  }

  CUDA_CHECK_RET(cudaStreamSynchronize(stream), -2);
  nvshmem_barrier_all();

  if (ready_unicast != nullptr) {
    nvshmem_free(ready_unicast);
  }
  if (input_unicast != nullptr) {
    nvshmem_free(input_unicast);
  }

  CUDA_CHECK_RET(cudaStreamDestroy(stream), -3);
  return 0;
}
