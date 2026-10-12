/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "compiler_defs.h"

// Arch-neutral scalar math and dtype conversion.
//
// Arch-specific overrides belong in `arch/<name>/primitive_ops.h`; this header
// holds the forms every arch can share.  Conversions are explicit because the
// Qwen3 decode path stores bf16 and accumulates in fp32, and silent promotion
// is how reduction-order bugs get introduced.

namespace ftt {

FTT_DEVICE_INLINE float to_float(float value) { return value; }

// Integer and narrow-float loads reach generated kernels as indices, masks and
// quantized payloads. A template rather than one overload per type: the
// generated code names the destination type nowhere, it relies on the loaded
// pointer's own element type.
template <class T> FTT_DEVICE_INLINE float to_float(T value) {
    return static_cast<float>(value);
}

FTT_DEVICE_INLINE float to_float(bfloat16_t value) {
#if defined(FTT_DEVICE_PASS)
    return static_cast<float>(value);
#else
    // Host path: widen the stored bits.  bf16 is the top 16 bits of fp32.
    unsigned int bits = static_cast<unsigned int>(value.bits) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}

FTT_DEVICE_INLINE bfloat16_t to_bfloat16(float value) {
#if defined(FTT_DEVICE_PASS)
    return static_cast<bfloat16_t>(value);
#else
    unsigned int bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even on the discarded low half.
    const unsigned int rounded =
        bits + 0x7fffu + ((bits >> 16) & 1u);
    return bfloat16_t{static_cast<unsigned short>(rounded >> 16)};
#endif
}

FTT_DEVICE_INLINE float rsqrt(float value) {
#if defined(FTT_DEVICE_PASS)
    return __frsqrt_rn(value);
#else
    return 1.0f / __builtin_sqrtf(value);
#endif
}

FTT_DEVICE_INLINE float silu(float value) {
#if defined(FTT_DEVICE_PASS)
    return value / (1.0f + __expf(-value));
#else
    return value / (1.0f + __builtin_expf(-value));
#endif
}

// Store-side conversion.  Generated kernels compute in fp32 and store into a
// buffer whose element type the buffer plan fixed; `store_as` lets the
// generated line name the destination through the pointer it already has
// rather than repeating the dtype.
//
//     destination[offset] = ftt::store_as(destination, value);
//
// The bf16 specialization goes through `to_bfloat16` so the rounding stays
// round-to-nearest-even rather than whatever a C++ narrowing cast would do.
template <class T> FTT_DEVICE_INLINE T store_as(const T *, float value) {
    return static_cast<T>(value);
}

FTT_DEVICE_INLINE bfloat16_t store_as(const bfloat16_t *, float value) {
    return to_bfloat16(value);
}

// Integer counterpart of `store_as`: narrow an index/length computed in
// `dim_t` to the destination buffer's own integer type, named through the
// pointer.
//
//     destination[offset] = ftt::store_index(destination, value);
//
// `decltype(destination[0])` would be the obvious spelling, but subscripting
// yields a *reference* type, so a `static_cast` to it binds a reference
// instead of converting. Going through the pointer's element type avoids that.
template <class T, class TValue>
FTT_DEVICE_INLINE T store_index(const T *, TValue value) {
    return static_cast<T>(value);
}

FTT_DEVICE_INLINE float sigmoid(float value) {
#if defined(FTT_DEVICE_PASS)
    return 1.0f / (1.0f + __expf(-value));
#else
    return 1.0f / (1.0f + __builtin_expf(-value));
#endif
}

FTT_DEVICE_INLINE float exp(float value) {
#if defined(FTT_DEVICE_PASS)
    return __expf(value);
#else
    return __builtin_expf(value);
#endif
}

FTT_DEVICE_INLINE float log(float value) {
#if defined(FTT_DEVICE_PASS)
    return __logf(value);
#else
    return __builtin_logf(value);
#endif
}

FTT_DEVICE_INLINE float sqrt(float value) { return __builtin_sqrtf(value); }

// The index/predicate algebra the generated kernels inherit from the shared
// TIR renderers reaches these through `codegen/cpp/expressions.py`.  They are
// templates so an index expression keeps its own integer type rather than
// silently widening or narrowing through a fixed signature.
template <class T> FTT_DEVICE_INLINE T max(T left, T right) {
    return left < right ? right : left;
}

template <class T> FTT_DEVICE_INLINE T min(T left, T right) {
    return right < left ? right : left;
}

template <class T> FTT_DEVICE_INLINE T abs(T value) {
    return value < T(0) ? static_cast<T>(-value) : value;
}

// Ceiling division over non-negative operands, matching `tl.cdiv`.
template <class T> FTT_DEVICE_INLINE T cdiv(T value, T divisor) {
    return (value + divisor - T(1)) / divisor;
}

FTT_DEVICE_INLINE long long llrint(float value) {
    return static_cast<long long>(__builtin_rintf(value));
}

// Wide load of a contiguous run of elements.
//
// A packed GEMV weight stores `Count` consecutive K values for one N. Loading
// them one at a time issues `Count` narrow loads; loading them as a single
// 16-byte transaction is measurably faster on BI-V150 (340 vs 248 GB/s on a
// pure streaming probe, 41% vs 30% of the 819 GB/s peak).
//
// `run_t` is deliberately not a vector type with arithmetic: generated kernels
// convert each element to fp32 and accumulate in a scalar, matching the
// Triton path's rounding. This only widens the *load*.
//
// The pointer must be 16-byte aligned, which the buffer plan guarantees for
// these packed weights (their alignment_bytes is 128 or more). An unaligned
// pointer would fault, so callers that cannot prove alignment must use the
// element-at-a-time path.
// Indexing must not go through memory. An `operator[]` that reinterprets the
// raw storage as `const T *` forces the compiler to give the vector an
// address, and inside a large kernel that becomes a real spill-and-reload:
// the wide load is still issued, then immediately re-read two bytes at a
// time, so the SASS ends up with *more* narrow loads than the scalar version
// (measured: 1829 -> 2655 `ml_lsa_load_ushort` on the Qwen3 package).
//
// Extracting from the 32-bit words with shifts keeps everything in registers.
// This is bf16-specific, which is why `at()` is specialized rather than
// generic: bf16 is the top half of an fp32, so widening is a shift, not a
// conversion.
template <int Count> struct bf16_run_t {
    static_assert(Count % 8 == 0,
                  "A wide bf16 run must be a whole number of 16-byte loads.");
    static constexpr int words = Count / 2;
    unsigned int word[words];

    FTT_DEVICE_INLINE float at(int index) const {
        const unsigned int packed = word[index >> 1];
        const unsigned int bits =
            (index & 1) ? (packed & 0xffff0000u) : (packed << 16);
        float value;
        __builtin_memcpy(&value, &bits, sizeof(value));
        return value;
    }
};

// Load `Count` contiguous bf16 as 16-byte transactions.
//
// This is the second half of the physical-order story. Walking the payload in
// physical order makes adjacent *threads* read adjacent bytes, which fixes
// coalescing; giving each thread a contiguous *chunk* is what lets the
// hardware issue wide loads instead of one 2-byte load per element. Measured
// over an identical 1.07 GB span: 1 element per thread per step reaches
// 95.9 GB/s, 8 elements per thread (one 16-byte load) reaches 376.1 GB/s --
// 3.9x, and the only difference is the load width.
//
// The pointer must be 16-byte aligned. The packed weights are (their run bases
// are multiples of 32 bytes), but a caller that cannot prove it must use the
// element-at-a-time path instead; an unaligned wide load faults.
template <int Count>
FTT_DEVICE_INLINE bf16_run_t<Count> load_run(const bfloat16_t *source) {
    bf16_run_t<Count> value;
    FTT_UNROLL
    for (int i = 0; i < Count / 8; ++i) {
        float4 wide = reinterpret_cast<const float4 *>(source)[i];
        __builtin_memcpy(&value.word[i * 4], &wide, sizeof(wide));
    }
    return value;
}

// Argmax reduction step, matching `torch.argmax` exactly.
//
// Three rules, all of which the generated kernels depend on:
//   * NaN precedes every numeric value.
//   * A tie -- including two NaNs and two equal infinities -- selects the
//     lower *logical* index, not the lower owner id. A distributed argmax
//     merges partials in owner order, so breaking ties by arrival order would
//     make the result depend on the mesh layout.
//   * `best` is updated in place so a caller can fold a sequence of
//     candidates without naming an accumulator type.
// `TCandidate` is separate from `TIndex` so a candidate index of a different
// integer width (an `int` partial read beside a `dim_t` accumulator) converts
// at the call rather than failing to deduce.
template <class TIndex, class TCandidate>
FTT_DEVICE_INLINE void argmax_combine(float &best, TIndex &best_index,
                                      float value, TCandidate candidate) {
    const TIndex index = static_cast<TIndex>(candidate);
    const bool best_nan = best != best;
    const bool value_nan = value != value;
    const bool tied = (best == value) || (best_nan && value_nan);
    const bool keep_best = (best_nan && !value_nan)
                           || (!value_nan && best > value)
                           || (tied && best_index < index);
    if (!keep_best) {
        best = value;
        best_index = index;
    }
}

} // namespace ftt
