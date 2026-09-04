#include <cuda_fp16.h>
#include <stdint.h>

using global_half_ptr = __attribute__((address_space(1))) const half *;
using shared_half_ptr = __attribute__((address_space(3))) half *;

static __attribute__((device, always_inline)) uint16_t
load_global_f16_bits(global_half_ptr ptr) {
  uint16_t bits;
  asm volatile("ld.global.u16 %0, [%1];" : "=h"(bits) : "l"(ptr));
  return bits;
}

static __attribute__((device, always_inline)) bool
is_aligned_16(global_half_ptr ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

static __attribute__((device, always_inline)) bool
is_aligned_16(shared_half_ptr ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

static __attribute__((device, always_inline)) void
copy_global_to_shared_16_async(shared_half_ptr dst, global_half_ptr src,
                               uint32_t valid_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(dst), "l"(src), "r"(valid_bytes)
               : "memory");
}

static __attribute__((device, always_inline)) void commit_async_copies() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

static __attribute__((device, always_inline)) void wait_for_async_copies() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// Fill the two memrefs supplied by tle_raw.call and return their descriptors.
// output_indices=[0, 1] tells TLE-Raw which input tensors they alias.
__attribute__((device)) auto
LoadTiles(shared_half_ptr a_allocated, shared_half_ptr a_aligned,
          const int64_t a_offset, const int64_t a_size0, const int64_t a_size1,
          const int64_t a_stride0, const int64_t a_stride1,
          shared_half_ptr b_allocated, shared_half_ptr b_aligned,
          const int64_t b_offset, const int64_t b_size0, const int64_t b_size1,
          const int64_t b_stride0, const int64_t b_stride1, global_half_ptr a,
          global_half_ptr b, const int M, const int N, const int K,
          const int stride_am, const int stride_ak, const int stride_bk,
          const int stride_bn, const int pid_m, const int pid_n, const int k) {
  uint32_t tid;
  uint32_t num_threads;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
  asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(num_threads));
  const int64_t k_begin = static_cast<int64_t>(k) * a_size1;

  constexpr int64_t async_elements = 8;
  bool use_a_async =
      a_stride1 == 1 && stride_ak == 1 && a_size1 % async_elements == 0 &&
      a_stride0 % async_elements == 0 && a_offset % async_elements == 0 &&
      stride_am % async_elements == 0 && is_aligned_16(a) &&
      is_aligned_16(a_aligned + a_offset);

  const int64_t b_tile_col_begin = (static_cast<int64_t>(pid_n) * b_size1) % N;
  bool use_b_async =
      b_stride1 == 1 && stride_bn == 1 && b_size1 % async_elements == 0 &&
      b_stride0 % async_elements == 0 && b_offset % async_elements == 0 &&
      stride_bk % async_elements == 0 &&
      b_tile_col_begin % async_elements == 0 &&
      b_tile_col_begin + b_size1 <= N && is_aligned_16(b) &&
      is_aligned_16(b_aligned + b_offset);

  // use_a_async = false;
  // use_b_async = false;
  if (use_a_async) {
    const int64_t vectors = a_size0 * a_size1 / async_elements;
    for (int64_t vector = tid; vector < vectors; vector += num_threads) {
      const int64_t linear = vector * async_elements;
      const int64_t row = linear / a_size1;
      const int64_t col = linear % a_size1;
      const int64_t global_row =
          (static_cast<int64_t>(pid_m) * a_size0 + row) % M;
      const int64_t global_k = k_begin + col;
      const int64_t remaining = static_cast<int64_t>(K) - global_k;
      const uint32_t valid_elements =
          remaining <= 0
              ? 0
              : static_cast<uint32_t>(
                    remaining < async_elements ? remaining : async_elements);
      global_half_ptr src = valid_elements == 0 ? a
                                                : a + global_row * stride_am +
                                                      global_k * stride_ak;
      shared_half_ptr dst =
          a_aligned + a_offset + row * a_stride0 + col * a_stride1;
      copy_global_to_shared_16_async(dst, src, valid_elements * sizeof(half));
    }
  } else {
    for (int64_t linear = tid; linear < a_size0 * a_size1;
         linear += num_threads) {
      const int64_t row = linear / a_size1;
      const int64_t col = linear % a_size1;
      const int64_t global_row =
          (static_cast<int64_t>(pid_m) * a_size0 + row) % M;
      const int64_t global_k = k_begin + col;
      half value{};
      if (global_k < K) {
        global_half_ptr src = a + global_row * stride_am + global_k * stride_ak;
        union {
          uint16_t bits;
          half value;
        } loaded{load_global_f16_bits(src)};
        value = loaded.value;
      }
      a_aligned[a_offset + row * a_stride0 + col * a_stride1] = value;
    }
  }

  if (use_b_async) {
    const int64_t vectors = b_size0 * b_size1 / async_elements;
    for (int64_t vector = tid; vector < vectors; vector += num_threads) {
      const int64_t linear = vector * async_elements;
      const int64_t row = linear / b_size1;
      const int64_t col = linear % b_size1;
      const int64_t global_k = k_begin + row;
      const int64_t global_col = b_tile_col_begin + col;
      const uint32_t valid_bytes = global_k < K ? sizeof(half) * 8 : 0;
      global_half_ptr src =
          valid_bytes == 0 ? b
                           : b + global_k * stride_bk + global_col * stride_bn;
      shared_half_ptr dst =
          b_aligned + b_offset + row * b_stride0 + col * b_stride1;
      copy_global_to_shared_16_async(dst, src, valid_bytes);
    }
  } else {
    for (int64_t linear = tid; linear < b_size0 * b_size1;
         linear += num_threads) {
      const int64_t row = linear / b_size1;
      const int64_t col = linear % b_size1;
      const int64_t global_k = k_begin + row;
      const int64_t global_col =
          (static_cast<int64_t>(pid_n) * b_size1 + col) % N;
      half value{};
      if (global_k < K) {
        global_half_ptr src = b + global_k * stride_bk + global_col * stride_bn;
        union {
          uint16_t bits;
          half value;
        } loaded{load_global_f16_bits(src)};
        value = loaded.value;
      }
      b_aligned[b_offset + row * b_stride0 + col * b_stride1] = value;
    }
  }

  if (use_a_async || use_b_async) {
    commit_async_copies();
    wait_for_async_copies();
  }

  // All threads must observe complete tiles before the returned tensor values
  // are consumed by tl.dot.
  asm volatile("bar.sync 0;" ::: "memory");

  struct MemRef2D {
    shared_half_ptr allocated;
    shared_half_ptr aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
  };
  struct {
    MemRef2D a;
    MemRef2D b;
  } result{
      {a_allocated,
       a_aligned,
       a_offset,
       {a_size0, a_size1},
       {a_stride0, a_stride1}},
      {b_allocated,
       b_aligned,
       b_offset,
       {b_size0, b_size1},
       {b_stride0, b_stride1}},
  };
  return result;
}
