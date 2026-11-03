// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Private compiler primitive: logical element indices, arbitrary UB strides.
#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void structured_gather(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *indices,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ int32_t, 1> *addresses) {
  using Bits =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  const int64_t n = indices->sizes[0];
  auto src_ptr = reinterpret_cast<__ubuf__ Bits *>(src->aligned + src->offset);
  auto dst_ptr = reinterpret_cast<__ubuf__ Bits *>(dst->aligned + dst->offset);
  auto idx_ptr = indices->aligned + indices->offset;
  auto address_ptr = addresses->aligned + addresses->offset;

  // This branch also preserves holes in a strided destination. The 910B
  // vector ISA has no general scatter exposed by the CANN SDK. The common
  // packed case below has no data-length scalar loop, even for strided src.
  if (indices->strides[0] != 1 || addresses->strides[0] != 1 ||
      dst->strides[0] != 1 || !isAddress32ByteAligned(dst_ptr) ||
      !isAddress32ByteAligned(idx_ptr) ||
      !isAddress32ByteAligned(address_ptr)) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < n; ++i)
      dst_ptr[i * dst->strides[0]] =
          src_ptr[static_cast<int64_t>(idx_ptr[i * indices->strides[0]]) *
                  src->strides[0]];
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // All arithmetic acts on logical indices; padding is never a data element.
  const int32_t scale = src->strides[0] * sizeof(T);
  const int32_t base =
      static_cast<int32_t>(reinterpret_cast<intptr_t>(src_ptr));
  vector_eltwise_vs_1d<VectorOpTy::VMULS>(indices, scale, addresses);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(addresses, base, addresses);
  INTRINSIC(pipe_barrier, PIPE_V);
  auto addr_ptr = reinterpret_cast<__ubuf__ uint32_t *>(addresses->aligned +
                                                        addresses->offset);
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vgather, dst_ptr, addr_ptr, 0, 8, 1);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

#define REGISTER_STRUCTURED_GATHER(SUFFIX, TYPE)                               \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_flagtree_stride_gather_##SUFFIX(                     \
          memref_t<__ubuf__ TYPE, 1> *src,                                     \
          memref_t<__ubuf__ int32_t, 1> *indices,                              \
          memref_t<__ubuf__ TYPE, 1> *dst,                                     \
          memref_t<__ubuf__ int32_t, 1> *addresses) {                          \
    structured_gather(src, indices, dst, addresses);                           \
  }

REGISTER_STRUCTURED_GATHER(half, half)
REGISTER_STRUCTURED_GATHER(bf16, bfloat16_t)
REGISTER_STRUCTURED_GATHER(float, float)
REGISTER_STRUCTURED_GATHER(int16, int16_t)
REGISTER_STRUCTURED_GATHER(int32, int32_t)

template <typename T>
__aiv__ __attribute__((always_inline)) void
structured_split(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *lhs,
                 memref_t<__ubuf__ T, 1> *rhs,
                 memref_t<__ubuf__ int32_t, 1> *addresses) {
  using Bits =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  const int64_t n = lhs->sizes[0];
  auto in = reinterpret_cast<__ubuf__ Bits *>(src->aligned + src->offset);
  auto a = reinterpret_cast<__ubuf__ Bits *>(lhs->aligned + lhs->offset);
  auto b = reinterpret_cast<__ubuf__ Bits *>(rhs->aligned + rhs->offset);
  bool alignedOutputs = isAddress32ByteAligned(a) && isAddress32ByteAligned(b);
  if (src->strides[0] == 1 && lhs->strides[0] == 1 && rhs->strides[0] == 1 &&
      isAddress32ByteAligned(in) && alignedOutputs) {
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, 2 * n);
    INTRINSIC(vreducev2, a, in, in, 1, 1, 1, 8, 0);
    INTRINSIC(vreducev2, b, in, in, 1, 1, 2, 8, 0);
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }
  if (lhs->strides[0] == 1 && rhs->strides[0] == 1 &&
      addresses->strides[0] == 1 && alignedOutputs &&
      isAddress32ByteAligned(addresses->aligned + addresses->offset)) {
    const int32_t scale = src->strides[0] * sizeof(T);
    const int32_t base = static_cast<int32_t>(reinterpret_cast<intptr_t>(in));
    auto addr = addresses->aligned + addresses->offset;
    arange_1d_core<int32_t>(addr, n, base, 2 * scale);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, n);
    INTRINSIC(vgather, a, reinterpret_cast<__ubuf__ uint32_t *>(addr), 0, 8, 1);
    INTRINSIC_NO_ARGS(set_mask_norm);
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vs_1d<VectorOpTy::VADDS>(addresses, scale, addresses);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, n);
    INTRINSIC(vgather, b, reinterpret_cast<__ubuf__ uint32_t *>(addr), 0, 8, 1);
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < n; ++i) {
    a[i * lhs->strides[0]] = in[2 * i * src->strides[0]];
    b[i * rhs->strides[0]] = in[(2 * i + 1) * src->strides[0]];
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

#define REGISTER_STRUCTURED_SPLIT(SUFFIX, TYPE)                                \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_flagtree_split_##SUFFIX(                             \
          memref_t<__ubuf__ TYPE, 1> *src, memref_t<__ubuf__ TYPE, 1> *lhs,    \
          memref_t<__ubuf__ TYPE, 1> *rhs,                                     \
          memref_t<__ubuf__ int32_t, 1> *addresses) {                          \
    structured_split(src, lhs, rhs, addresses);                                \
  }

REGISTER_STRUCTURED_SPLIT(half, half)
REGISTER_STRUCTURED_SPLIT(bf16, bfloat16_t)
REGISTER_STRUCTURED_SPLIT(float, float)
REGISTER_STRUCTURED_SPLIT(int16, int16_t)
REGISTER_STRUCTURED_SPLIT(int32, int32_t)

// Legalize a compact sub-block row into aligned UB rows before GM DMA. Both
// workspaces are allocated by the compiler before UB planning, at most 8 KiB
// together. No scalar GM writes or per-element scalar copy is used.
// Byte-counted DMA preserves the columns owned by adjacent programs, including
// masked tails.
template <typename T>
__aiv__ __attribute__((always_inline)) void
short_row_store(memref_t<__ubuf__ T, 2> *src, memref_t<__gm__ T, 2> *dst,
                memref_t<__ubuf__ int32_t, 1> *addresses,
                memref_t<__ubuf__ int32_t, 1> *scratch) {
  constexpr int blockElements = 32 / sizeof(T);
  constexpr int shift = sizeof(T) == 4 ? 3 : 4;
  const int64_t rows = src->sizes[0], cols = src->sizes[1];
  if (rows == 0 || cols == 0)
    return;
  auto in = src->aligned + src->offset;
  auto out = dst->aligned + dst->offset;
  const int64_t srcStride = src->strides[0], dstStride = dst->strides[0];
  // The operation is scheduled as PIPE_V. Internal MTE3 work is drained before
  // returning, so subsequent users and UB reuse observe the completed store.
  if (src->strides[1] == 1 && isAddress32ByteAligned(in) &&
      ((srcStride >= blockElements && (srcStride * sizeof(T)) % 32 == 0) ||
       (srcStride == cols && dstStride == cols))) {
    INTRINSIC(set_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
    if (srcStride == cols && dstStride == cols) {
      store_ubuf_to_gm_intrin_core(in, 0, out, 0, 1, rows * cols * sizeof(T), 0,
                                   0);
    } else {
      // Split long row counts to respect the DMA instruction's burst limit.
      for (int64_t row = 0; row < rows; row += INTRIN_MAX_BURST_CNT) {
        int64_t n = min(rows - row, int64_t(INTRIN_MAX_BURST_CNT));
        store_ubuf_to_gm_intrin_core(
            in, row * srcStride, out, row * dstStride, n, cols * sizeof(T),
            srcStride * sizeof(T) / 32 - 1, (dstStride - cols) * sizeof(T));
      }
    }
    INTRINSIC(set_flag, PIPE_MTE3, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_MTE3, PIPE_V, LIB_EVENT_ID0);
    return;
  }
  const int64_t chunkRows = min(rows, addresses->sizes[0] / blockElements);
  const int64_t n = chunkRows * blockElements;
  auto addrPtr = addresses->aligned + addresses->offset;
  auto workPtr = scratch->aligned + scratch->offset;
  auto unsignedAddr = reinterpret_cast<__ubuf__ uint32_t *>(addrPtr);
  // Duplicate the final valid column into padding. No address refers to a
  // masked-out element or reads beyond the last source row. These are packed
  // compiler-owned workspaces, so use SIMD directly. The low-bit extraction
  // must use unsigned shifts: signed VSHR extends the sign bit and would create
  // negative addresses in padding, even when those lanes are not stored to GM.
  arange_1d_core<int32_t>(addrPtr, n, 0, 1);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vshr, workPtr, addrPtr, int32_t(shift), 1, 1, 1, 8, 8, 0);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmuls, workPtr, workPtr, int32_t(srcStride * sizeof(T)), 1, 1, 1, 8,
            8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vshl, unsignedAddr, unsignedAddr, uint32_t(32 - shift), 1, 1, 1, 8,
            8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vshr, unsignedAddr, unsignedAddr, uint32_t(32 - shift), 1, 1, 1, 8,
            8, 0);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmins, addrPtr, addrPtr, int32_t(cols - 1), 1, 1, 1, 8, 8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmuls, addrPtr, addrPtr, int32_t(src->strides[1] * sizeof(T)), 1, 1,
            1, 8, 8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vadd, addrPtr, addrPtr, workPtr, 1, 1, 1, 1, 8, 8, 8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vadds, addrPtr, addrPtr, int32_t(reinterpret_cast<intptr_t>(in)), 1,
            1, 1, 8, 8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC_NO_ARGS(set_mask_norm);
  using Bits =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  for (int64_t row = 0; row < rows; row += chunkRows) {
    int64_t activeRows = min(rows - row, chunkRows);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, activeRows * blockElements);
    INTRINSIC(vgather, reinterpret_cast<__ubuf__ Bits *>(workPtr),
              reinterpret_cast<__ubuf__ uint32_t *>(addrPtr), 0, 8, 1);
    INTRINSIC_NO_ARGS(set_mask_norm);
    INTRINSIC(set_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
    store_ubuf_to_gm_intrin_core(
        reinterpret_cast<__ubuf__ T *>(workPtr), 0, out, row * dstStride,
        activeRows, cols * sizeof(T), 0, (dstStride - cols) * sizeof(T));
    INTRINSIC(set_flag, PIPE_MTE3, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_MTE3, PIPE_V, LIB_EVENT_ID0);
    if (row + chunkRows < rows) {
      INTRINSIC_NO_ARGS(set_mask_count);
      INTRINSIC(set_vector_mask, 0, n);
      INTRINSIC(vadds, addrPtr, addrPtr,
                int32_t(chunkRows * srcStride * sizeof(T)), 1, 1, 1, 8, 8);
      INTRINSIC(pipe_barrier, PIPE_V);
      INTRINSIC_NO_ARGS(set_mask_norm);
    }
  }
}

#define REGISTER_SHORT_ROW_STORE(SUFFIX, TYPE)                                 \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_flagtree_short_row_store_##SUFFIX(                   \
          memref_t<__ubuf__ TYPE, 2> *src, memref_t<__gm__ TYPE, 2> *dst,      \
          memref_t<__ubuf__ int32_t, 1> *addresses,                            \
          memref_t<__ubuf__ int32_t, 1> *scratch) {                            \
    short_row_store(src, dst, addresses, scratch);                             \
  }

REGISTER_SHORT_ROW_STORE(half, half)
REGISTER_SHORT_ROW_STORE(bf16, bfloat16_t)
REGISTER_SHORT_ROW_STORE(float, float)
REGISTER_SHORT_ROW_STORE(int16, int16_t)
REGISTER_SHORT_ROW_STORE(int32, int32_t)

// The symmetric load preserves DMA even when the consumer needs packed rows.
// Its single, preplanned workspace is at most 4 KiB; no scalar GM access is
// introduced by CANN's choice of the output buffer's physical row stride.
template <typename T>
__aiv__ __attribute__((always_inline)) void
short_row_load(memref_t<__gm__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
               memref_t<__ubuf__ int32_t, 1> *scratch) {
  constexpr int blockElements = 32 / sizeof(T);
  int64_t rows = src->sizes[0], cols = src->sizes[1];
  if (rows == 0 || cols == 0)
    return;
  auto in = src->aligned + src->offset;
  auto out = dst->aligned + dst->offset;
  int64_t srcStride = src->strides[0], dstStride = dst->strides[0];
  INTRINSIC(set_mov_pad_val, 0);
  if (dstStride >= blockElements && dstStride % blockElements == 0) {
    INTRINSIC(set_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
    for (int64_t row = 0; row < rows; row += INTRIN_MAX_BURST_CNT) {
      int64_t n = min(rows - row, int64_t(INTRIN_MAX_BURST_CNT));
      load_gm_to_ubuf_intrin_core(
          in, row * srcStride, out, row * dstStride, n, cols * sizeof(T), 0,
          (srcStride - cols) * sizeof(T), dstStride / blockElements - 1);
    }
    INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
    return;
  }
  auto work =
      reinterpret_cast<__ubuf__ T *>(scratch->aligned + scratch->offset);
  int64_t chunkRows = min(rows, scratch->sizes[0] / 8);
  for (int64_t row = 0; row < rows; row += chunkRows) {
    int64_t n = min(rows - row, chunkRows);
    INTRINSIC(set_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
    load_gm_to_ubuf_intrin_core(in, row * srcStride, work, 0, n,
                                cols * sizeof(T), 0,
                                (srcStride - cols) * sizeof(T), 0);
    INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
    memref_t<__ubuf__ T, 2> packedSource{
        work, work, 0, {n, cols}, {blockElements, 1}};
    auto packedDest = *dst;
    packedDest.offset += row * dstStride;
    packedDest.sizes[0] = n;
    vreducev2_2d_with_pattern_mode<T, PatternMode::ALL_ELEMENTS>(&packedSource,
                                                                 &packedDest);
  }
}

#define REGISTER_SHORT_ROW_LOAD(SUFFIX, TYPE)                                  \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_flagtree_short_row_load_##SUFFIX(                    \
          memref_t<__gm__ TYPE, 2> *src, memref_t<__ubuf__ TYPE, 2> *dst,      \
          memref_t<__ubuf__ int32_t, 1> *scratch) {                            \
    short_row_load(src, dst, scratch);                                         \
  }

REGISTER_SHORT_ROW_LOAD(half, half)
REGISTER_SHORT_ROW_LOAD(bf16, bfloat16_t)
REGISTER_SHORT_ROW_LOAD(float, float)
REGISTER_SHORT_ROW_LOAD(int16, int16_t)
REGISTER_SHORT_ROW_LOAD(int32, int32_t)

// These conversions serve byte permutations only: word values are always in
// [-128,127]. Half represents that range exactly, so no saturation/overflow
// conversion or CANN's sub-block 2D cast is required.
template <bool EXTEND>
__aiv__ __attribute__((always_inline)) void storage_cast(
    memref_t<__ubuf__ typename std::conditional<EXTEND, int8_t, int16_t>::type,
             1> *src,
    memref_t<__ubuf__ typename std::conditional<EXTEND, int16_t, int8_t>::type,
             1> *dst,
    memref_t<__ubuf__ half, 1> *scratch) {
  auto in = src->aligned + src->offset;
  auto out = dst->aligned + dst->offset;
  auto tmp = scratch->aligned + scratch->offset;
  const int64_t n = dst->sizes[0];
  if (src->strides[0] != 1 || dst->strides[0] != 1 ||
      scratch->strides[0] != 1 ||
      ((reinterpret_cast<intptr_t>(in) | reinterpret_cast<intptr_t>(out) |
        reinterpret_cast<intptr_t>(tmp)) &
       31)) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < n; ++i)
      out[i * dst->strides[0]] = in[i * src->strides[0]];
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  if constexpr (EXTEND)
    INTRINSIC(vconv_s82f16, tmp, in, 1, 1, 1, 8, 4);
  else
    INTRINSIC(vconv_s162f16r, tmp, in, 1, 1, 1, 8, 8);
  INTRINSIC(pipe_barrier, PIPE_V);
  if constexpr (EXTEND)
    INTRINSIC(vconv_f162s16z, out, tmp, 1, 1, 1, 8, 8);
  else
    INTRINSIC(vconv_f162s8z, out, tmp, 1, 1, 1, 4, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_flagtree_byte_to_word(memref_t<__ubuf__ int8_t, 1> *src,
                                          memref_t<__ubuf__ int16_t, 1> *dst,
                                          memref_t<__ubuf__ half, 1> *scratch) {
  storage_cast<true>(src, dst, scratch);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_flagtree_word_to_byte(memref_t<__ubuf__ int16_t, 1> *src,
                                          memref_t<__ubuf__ int8_t, 1> *dst,
                                          memref_t<__ubuf__ half, 1> *scratch) {
  storage_cast<false>(src, dst, scratch);
}

// Late compiler optimization: reuse the two existing private index buffers as
// absolute address tables. The compiler proves exclusive use and fixed packed
// physical layouts before selecting this ABI. No new UB storage is required.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_flagtree_prepare_addresses(
    memref_t<__ubuf__ int32_t, 1> *table_a,
    memref_t<__ubuf__ int32_t, 1> *table_b, int32_t scale, int32_t base_a,
    int32_t base_b) {
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VMULS>(table_a, scale, table_b);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(table_b, base_a, table_a);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(table_b, base_b, table_b);
  INTRINSIC(pipe_barrier, PIPE_V);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
addressed_gather(memref_t<__ubuf__ int32_t, 1> *table,
                 memref_t<__ubuf__ T, 1> *dst) {
  using Bits =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  auto out = reinterpret_cast<__ubuf__ Bits *>(dst->aligned + dst->offset);
  auto addresses =
      reinterpret_cast<__ubuf__ uint32_t *>(table->aligned + table->offset);
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, dst->sizes[0]);
  INTRINSIC(vgather, out, addresses, 0, 8, 1);
  INTRINSIC_NO_ARGS(set_mask_norm);
}
#define REGISTER_ADDRESSED_GATHER(SUFFIX, TYPE)                                \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_flagtree_addressed_gather_##SUFFIX(                  \
          memref_t<__ubuf__ int32_t, 1> *table,                                \
          memref_t<__ubuf__ TYPE, 1> *dst) {                                   \
    addressed_gather(table, dst);                                              \
  }
REGISTER_ADDRESSED_GATHER(half, half)
REGISTER_ADDRESSED_GATHER(bf16, bfloat16_t)
REGISTER_ADDRESSED_GATHER(float, float)
REGISTER_ADDRESSED_GATHER(int16, int16_t)
REGISTER_ADDRESSED_GATHER(int32, int32_t)
