/* Copyright 2025- FlagOS Contributors
 * Portions Copyright 2019-2021 Canaan Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Ported from nncase NTT (ntt/include/nncase/ntt/arch/riscv64/), which FTT
 * already derives from -- see ftt/NTT-LICENSE and the design notes in
 * ftt/include/ftt/tensor.h.
 *
 * NTT's vector layer is fixed-length RVV: `riscv_rvv_vector_bits(VLEN)` turns
 * the sizeless RVV types into ordinary value types that can live in structs
 * and be passed by value.  That is what lets a vector be a tensor *element*
 * rather than a special case in every kernel.
 *
 * C908 is VLEN=128 with Zvfh, so one native vector is 4 f32 / 8 f16 lanes.
 */
#pragma once
#include "../../compiler_defs.h"

#if FTT_ARCH_RISCV64

#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

#ifndef FTT_VLEN
#define FTT_VLEN 128
#endif

namespace ftt {
namespace riscv64 {

// Fixed-length aliases when the compiler was told the vector width
// (-mrvv-vector-bits=zvl), otherwise the plain sizeless types.
//
// The fixed form is what NTT relies on: sizeless RVV types cannot be struct
// members or template arguments, so only the fixed form lets a vector be a
// tensor *element*.  The SDK's ggml-cpu flags do not pass that option, so the
// header stays usable either way and FTT_RVV_FIXED says which one is active.
#if defined(__riscv_v_fixed_vlen)
#define FTT_RVV_FIXED 1
using f32xN = vfloat32m1_t __attribute__((riscv_rvv_vector_bits(FTT_VLEN)));
using f16xN = vfloat16m1_t __attribute__((riscv_rvv_vector_bits(FTT_VLEN)));
// Half-width f16: FTT_VLEN/32 lanes, the widening source for an f32 accumulator.
using f16xH = vfloat16mf2_t __attribute__((riscv_rvv_vector_bits(FTT_VLEN / 2)));
using i32xN = vint32m1_t __attribute__((riscv_rvv_vector_bits(FTT_VLEN)));
#else
#define FTT_RVV_FIXED 0
using f32xN = vfloat32m1_t;
using f16xN = vfloat16m1_t;
using f16xH = vfloat16mf2_t;
using i32xN = vint32m1_t;
#endif

inline constexpr size_t f32_lanes = FTT_VLEN / 32;
inline constexpr size_t f16_lanes = FTT_VLEN / 16;

// The vl every operation below runs at.  Fixed, so the compiler can hoist the
// vsetvli out of loops instead of re-issuing it per call.
inline constexpr size_t f32_vl = f32_lanes;
inline constexpr size_t f16_vl = f16_lanes;

FTT_HOST_DEVICE inline f32xN load_f32(const float *p) noexcept {
    return __riscv_vle32_v_f32m1(p, f32_vl);
}
FTT_HOST_DEVICE inline void store_f32(float *p, f32xN v) noexcept {
    __riscv_vse32_v_f32m1(p, v, f32_vl);
}
FTT_HOST_DEVICE inline f16xN load_f16(const _Float16 *p) noexcept {
    return __riscv_vle16_v_f16m1(p, f16_vl);
}
// Load f32_lanes f16 values, ready to widen into an f32 accumulator.
FTT_HOST_DEVICE inline f16xH load_f16_half(const _Float16 *p) noexcept {
    return __riscv_vle16_v_f16mf2(p, f32_vl);
}
FTT_HOST_DEVICE inline void store_f16(_Float16 *p, f16xN v) noexcept {
    __riscv_vse16_v_f16m1(p, v, f16_vl);
}

FTT_HOST_DEVICE inline f32xN splat_f32(float s) noexcept {
    return __riscv_vfmv_v_f_f32m1(s, f32_vl);
}
FTT_HOST_DEVICE inline f32xN add(f32xN a, f32xN b) noexcept {
    return __riscv_vfadd_vv_f32m1(a, b, f32_vl);
}
FTT_HOST_DEVICE inline f32xN sub(f32xN a, f32xN b) noexcept {
    return __riscv_vfsub_vv_f32m1(a, b, f32_vl);
}
FTT_HOST_DEVICE inline f32xN mul(f32xN a, f32xN b) noexcept {
    return __riscv_vfmul_vv_f32m1(a, b, f32_vl);
}
// acc + a*b, one rounding (the fused form), matching NTT's mul_add.
FTT_HOST_DEVICE inline f32xN mul_add(f32xN a, f32xN b, f32xN acc) noexcept {
    return __riscv_vfmacc_vv_f32m1(acc, a, b, f32_vl);
}
// f16 inputs widened into an f32 accumulator -- the shape every dot product
// wants, and the one C908 issues in a single vfwmacc (2 cycles, probe m0366).
FTT_HOST_DEVICE inline f32xN widen_mul_add(f16xH a, f16xH b, f32xN acc) noexcept {
    return __riscv_vfwmacc_vv_f32m1(acc, a, b, f32_vl);
}

// Ordered reduction with an explicit seed, so the summation order is a
// property of the call site rather than of the compiler.
FTT_HOST_DEVICE inline float reduce_sum(f32xN v, float seed = 0.0f) noexcept {
    auto s = __riscv_vfmv_v_f_f32m1(seed, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(v, s, f32_vl));
}
FTT_HOST_DEVICE inline float reduce_max(f32xN v, float seed) noexcept {
    auto s = __riscv_vfmv_v_f_f32m1(seed, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(v, s, f32_vl));
}


// ---------------------------------------------------------------------------
// Widening dot product, m2 form.
//
// This is the shape ggml's own __riscv_zvfh path uses (vec.cpp, ggml_vec_dot_f16):
// accumulate f16xf16 into an f32m2 accumulator, then fold the two m1 halves with
// one vfadd before a single vfredusum seeded at 0.  Keeping that exact structure
// is what makes the result bitwise equal to the reference, so the summation
// order is part of the contract, not an implementation detail.
//
// Using m2 also doubles the work per vfwmacc, which is why it beats the m1 form
// on C908 (vfwmacc is 2 cycles either way, probe m0366).
FTT_HOST_DEVICE inline float widening_dot_f16(const _Float16 *x, const _Float16 *y,
                                              int n) noexcept {
    int vl = __riscv_vsetvlmax_e32m2();
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m2_t sum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
    vfloat16m1_t ax{};
    vfloat16m1_t ay{};
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m1(n - i);
        ax = __riscv_vle16_v_f16m1_tu(ax, x + i, vl);
        ay = __riscv_vle16_v_f16m1_tu(ay, y + i, vl);
        sum = __riscv_vfwmacc_vv_f32m2_tu(sum, ax, ay, vl);
    }
    vl = __riscv_vsetvlmax_e32m1();
    auto folded = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(sum, 0),
                                         __riscv_vget_v_f32m2_f32m1(sum, 1), vl);
    return __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(folded, seed, vl));
}

} // namespace riscv64
} // namespace ftt

#endif // FTT_ARCH_RISCV64
