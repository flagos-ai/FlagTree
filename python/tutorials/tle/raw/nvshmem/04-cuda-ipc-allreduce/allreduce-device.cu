#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

namespace {

constexpr int kMaxBlocks = 36;
constexpr int kMaxRanks = 8;

using Flag = uint32_t;

struct Signal {
  alignas(128) Flag start[kMaxBlocks][kMaxRanks];
  alignas(128) Flag end[kMaxBlocks][kMaxRanks];
  alignas(128) Flag epoch[kMaxBlocks];
};

struct __align__(16) RankData {
  const void *ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal *signals[kMaxRanks];
};

template <typename T, int Size> struct __align__(alignof(T) * Size) Array {
  T data[Size];
  using type = T;
  static constexpr int size = Size;
};

template <typename T> struct Packed {
  using Value = Array<T, 16 / sizeof(T)>;
  using Accumulator = Array<float, 16 / sizeof(T)>;
};

__device__ __forceinline__ void store_flag_volatile(Flag *address, Flag value) {
  asm volatile("st.volatile.global.u32 [%1], %0;" : : "r"(value), "l"(address));
}

__device__ __forceinline__ Flag load_flag_volatile(Flag *address) {
  Flag value;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ void store_flag_release(Flag *address, Flag value) {
  asm volatile("st.release.sys.global.u32 [%1], %0;"
               :
               : "r"(value), "l"(address));
}

__device__ __forceinline__ Flag load_flag_acquire(Flag *address) {
  Flag value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address));
  return value;
}

template <int WorldSize>
__device__ __forceinline__ void barrier_start(const RankSignals &signals,
                                              Signal *self_signal, int rank) {
  const Flag flag = self_signal->epoch[blockIdx.x] + 1;
  if (threadIdx.x < WorldSize) {
    Flag *remote = &signals.signals[threadIdx.x]->start[blockIdx.x][rank];
    Flag *local = &self_signal->start[blockIdx.x][threadIdx.x];
    store_flag_volatile(remote, flag);
    while (load_flag_volatile(local) != flag) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    self_signal->epoch[blockIdx.x] = flag;
}

template <int WorldSize, bool FinalSync = false>
__device__ __forceinline__ void barrier_end(const RankSignals &signals,
                                            Signal *self_signal, int rank) {
  __syncthreads();
  const Flag flag = self_signal->epoch[blockIdx.x] + 1;
  if (threadIdx.x < WorldSize) {
    Flag *remote = &signals.signals[threadIdx.x]->end[blockIdx.x][rank];
    Flag *local = &self_signal->end[blockIdx.x][threadIdx.x];
    if constexpr (FinalSync) {
      store_flag_volatile(remote, flag);
      while (load_flag_volatile(local) != flag) {
      }
    } else {
      store_flag_release(remote, flag);
      while (load_flag_acquire(local) != flag) {
      }
    }
  }
  if constexpr (!FinalSync)
    __syncthreads();
  if (threadIdx.x == 0)
    self_signal->epoch[blockIdx.x] = flag;
}

__device__ __forceinline__ float scalar_to_float(half value) {
  return __half2float(value);
}

__device__ __forceinline__ float scalar_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T scalar_from_float(float);

template <> __device__ __forceinline__ half scalar_from_float(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 scalar_from_float(float value) {
  return __float2bfloat16(value);
}

template <typename T, int Size>
__device__ __forceinline__ Array<float, Size> upcast(Array<T, Size> value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else {
    Array<float, Size> result;
#pragma unroll
    for (int i = 0; i < Size; ++i)
      result.data[i] = scalar_to_float(value.data[i]);
    return result;
  }
}

template <typename Output>
__device__ __forceinline__ Output downcast(Array<float, Output::size> value) {
  if constexpr (std::is_same<typename Output::type, float>::value) {
    return value;
  } else {
    Output result;
#pragma unroll
    for (int i = 0; i < Output::size; ++i)
      result.data[i] = scalar_from_float<typename Output::type>(value.data[i]);
    return result;
  }
}

template <int WorldSize, typename Value, typename Accumulator>
__device__ __forceinline__ Value packed_reduce(const Value *const *pointers,
                                               int index) {
  Accumulator sum = upcast(pointers[0][index]);
#pragma unroll
  for (int peer = 1; peer < WorldSize; ++peer) {
    const Accumulator value = upcast(pointers[peer][index]);
#pragma unroll
    for (int element = 0; element < Accumulator::size; ++element)
      sum.data[element] += value.data[element];
  }
  return downcast<Value>(sum);
}

template <typename T, int WorldSize>
__device__ __forceinline__ void
ipc_allreduce_oneshot_impl(T *output, const int64_t *input_pointer_table,
                           const int64_t *signal_pointer_table, int rank,
                           int numel) {
  using Value = typename Packed<T>::Value;
  using Accumulator = typename Packed<T>::Accumulator;

  RankData data;
  RankSignals signals;
#pragma unroll
  for (int peer = 0; peer < WorldSize; ++peer) {
    data.ptrs[peer] = reinterpret_cast<const void *>(input_pointer_table[peer]);
    signals.signals[peer] =
        reinterpret_cast<Signal *>(signal_pointer_table[peer]);
  }

  Signal *self_signal = signals.signals[rank];
  barrier_start<WorldSize>(signals, self_signal, rank);

  const int packed_count = numel / Value::size;
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const Value *inputs[WorldSize];
#pragma unroll
  for (int peer = 0; peer < WorldSize; ++peer)
    inputs[peer] = reinterpret_cast<const Value *>(data.ptrs[peer]);

  Value *packed_output = reinterpret_cast<Value *>(output);
  for (int index = thread; index < packed_count; index += stride)
    packed_output[index] =
        packed_reduce<WorldSize, Value, Accumulator>(inputs, index);

  barrier_end<WorldSize, true>(signals, self_signal, rank);
}

template <typename Value>
__device__ __forceinline__ Value *temporary_buffer(Signal *signal) {
  return reinterpret_cast<Value *>(signal + 1);
}

template <typename T, int WorldSize>
__device__ __forceinline__ void
ipc_allreduce_twoshot_impl(T *output, const int64_t *input_pointer_table,
                           const int64_t *signal_pointer_table, int rank,
                           int numel) {
  using Value = typename Packed<T>::Value;
  using Accumulator = typename Packed<T>::Accumulator;

  RankData data;
  RankSignals signals;
#pragma unroll
  for (int peer = 0; peer < WorldSize; ++peer) {
    data.ptrs[peer] = reinterpret_cast<const void *>(input_pointer_table[peer]);
    signals.signals[peer] =
        reinterpret_cast<Signal *>(signal_pointer_table[peer]);
  }

  const int packed_count = numel / Value::size;
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int part = packed_count / WorldSize;
  const int start = rank * part;
  const int end = rank == WorldSize - 1 ? packed_count : start + part;
  const int largest_part = part + packed_count % WorldSize;

  const Value *inputs[WorldSize];
  Value *temporaries[WorldSize];
#pragma unroll
  for (int i = 0; i < WorldSize; ++i) {
    const int target = (rank + i) % WorldSize;
    inputs[i] = reinterpret_cast<const Value *>(data.ptrs[target]);
    temporaries[i] = temporary_buffer<Value>(signals.signals[target]);
  }

  Signal *self_signal = signals.signals[rank];
  Value *temporary_output = temporaries[0];
  barrier_start<WorldSize>(signals, self_signal, rank);

  for (int index = start + thread; index < end; index += stride)
    temporary_output[index - start] =
        packed_reduce<WorldSize, Value, Accumulator>(inputs, index);

  barrier_end<WorldSize>(signals, self_signal, rank);

  Value *packed_output = reinterpret_cast<Value *>(output);
  for (int index = thread; index < largest_part; index += stride) {
#pragma unroll
    for (int i = 0; i < WorldSize; ++i) {
      const int source_rank = (rank + i) % WorldSize;
      if (source_rank == WorldSize - 1 || index < part)
        packed_output[source_rank * part + index] = temporaries[i][index];
    }
  }
}

} // namespace

#define DEFINE_IPC_ALLREDUCE(ALGORITHM, NAME, TYPE, WORLD_SIZE)                \
  extern "C" __device__ __attribute__((always_inline)) void                    \
      ipc_allreduce_##ALGORITHM##_##NAME##_##WORLD_SIZE(                       \
          __attribute__((address_space(1))) TYPE *output,                      \
          __attribute__((address_space(1)))                                    \
          const int64_t *input_pointer_table,                                  \
          __attribute__((address_space(1)))                                    \
          const int64_t *signal_pointer_table,                                 \
          int rank, int numel) {                                               \
    ipc_allreduce_##ALGORITHM##_impl<TYPE, WORLD_SIZE>(                        \
        output, input_pointer_table, signal_pointer_table, rank, numel);       \
  }

#define DEFINE_FOR_WORLD_SIZE(WORLD_SIZE)                                      \
  DEFINE_IPC_ALLREDUCE(oneshot, fp16, half, WORLD_SIZE)                        \
  DEFINE_IPC_ALLREDUCE(twoshot, fp16, half, WORLD_SIZE)                        \
  DEFINE_IPC_ALLREDUCE(oneshot, bf16, __nv_bfloat16, WORLD_SIZE)               \
  DEFINE_IPC_ALLREDUCE(twoshot, bf16, __nv_bfloat16, WORLD_SIZE)               \
  DEFINE_IPC_ALLREDUCE(oneshot, fp32, float, WORLD_SIZE)                       \
  DEFINE_IPC_ALLREDUCE(twoshot, fp32, float, WORLD_SIZE)

DEFINE_FOR_WORLD_SIZE(2)
DEFINE_FOR_WORLD_SIZE(4)
DEFINE_FOR_WORLD_SIZE(6)
DEFINE_FOR_WORLD_SIZE(8)

#undef DEFINE_FOR_WORLD_SIZE
#undef DEFINE_IPC_ALLREDUCE
