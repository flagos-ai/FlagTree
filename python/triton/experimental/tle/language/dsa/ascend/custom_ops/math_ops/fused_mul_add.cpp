// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// The vector fused multiply-add instructions, which HIVM does not expose as
// ops: vmla / vmadd / vmaddrelu / vaxpy. All four are Level 2 count mode with
// a single repeat and the default strides {blkStride=1, repStride=8}; the
// mixed fp16-source / fp32-dst forms halve the source repeat stride exactly
// as the CANN implementations do for sizeof(T) != sizeof(U).
// Every destination here is read-modify-write, so its incoming value is part
// of the computation.
// Pipeline barriers are the caller's responsibility; these ops add none.

// MulAddDstImpl (MulAddDst Level 2, CANN 9.0
// dav_c220/kernel_operator_vec_binary_impl.h): dst = src0 * src1 + dst,
// where dst holds the addend.

#define REGISTER_CIFACE_MUL_ADD_DST(TYPE_NAME, T, U, SRC_REP_STRIDE)           \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_mul_add_dst_##TYPE_NAME(                             \
          memref_t<__ubuf__ U, 1> *src0, memref_t<__ubuf__ U, 1> *src1,        \
          uint32_t count, memref_t<__ubuf__ T, 1> *dst) {                      \
    set_mask_count();                                                          \
    set_vector_mask(0, count);                                                 \
    vmla(dst->aligned + dst->offset, src0->aligned + src0->offset,             \
         src1->aligned + src1->offset, 1, 1, 1, 1, 8, SRC_REP_STRIDE,          \
         SRC_REP_STRIDE);                                                      \
    set_mask_norm();                                                           \
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));     \
  }

REGISTER_CIFACE_MUL_ADD_DST(half, half, half, 8)
REGISTER_CIFACE_MUL_ADD_DST(float, float, float, 8)
REGISTER_CIFACE_MUL_ADD_DST(fmix, float, half, 4)

// FusedMulAddImpl / MulAddReluImpl (Level 2), the `vmadd` / `vmaddrelu`
// instructions: dst = src0 * dst + src1, optionally followed by relu. Here
// dst is both a multiplicand and the result, so its incoming value matters
// for a different reason than in mul_add_dst; src1 is the addend.
#define REGISTER_CIFACE_FMA3(OP_NAME, INTRINSIC, T)                            \
  extern "C" __aiv__                                                           \
      __attribute__((always_inline)) void _mlir_ciface_custom_##OP_NAME(       \
          memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,        \
          uint32_t count, memref_t<__ubuf__ T, 1> *dst) {                      \
    set_mask_count();                                                          \
    set_vector_mask(0, count);                                                 \
    INTRINSIC(dst->aligned + dst->offset, src0->aligned + src0->offset,        \
              src1->aligned + src1->offset, 1, 1, 1, 1, 8, 8, 8);              \
    set_mask_norm();                                                           \
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));     \
  }

REGISTER_CIFACE_FMA3(fused_mul_add_half, vmadd, half)
REGISTER_CIFACE_FMA3(fused_mul_add_float, vmadd, float)
REGISTER_CIFACE_FMA3(mul_add_relu_half, vmaddrelu, half)
REGISTER_CIFACE_FMA3(mul_add_relu_float, vmaddrelu, float)

// AxpyImpl (Level 2), the `vaxpy` instruction: dst = src * scalar + dst.
// dst is read-modify-write and holds the addend. The scalar arrives as f32
// from the frontend and is narrowed to the source dtype, matching how
// compare_scalar handles its scalar operand. The mixed form keeps the
// halved source repeat stride that AxpyImpl uses for sizeof(T) > sizeof(U).
#define REGISTER_CIFACE_AXPY(TYPE_NAME, T, U, SRC_REP_STRIDE)                  \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_axpy_##TYPE_NAME(memref_t<__ubuf__ U, 1> *src,       \
                                           float scalar, uint32_t count,       \
                                           memref_t<__ubuf__ T, 1> *dst) {     \
    set_mask_count();                                                          \
    set_vector_mask(0, count);                                                 \
    vaxpy(dst->aligned + dst->offset, src->aligned + src->offset,              \
          static_cast<U>(scalar), 1, 1, 1, 8, SRC_REP_STRIDE);                 \
    set_mask_norm();                                                           \
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));     \
  }

REGISTER_CIFACE_AXPY(half, half, half, 8)
REGISTER_CIFACE_AXPY(float, float, float, 8)
REGISTER_CIFACE_AXPY(fmix, float, half, 4)
