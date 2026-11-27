/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "ftt.h"

// Native GGML super-blocks remain bytes in the checkpoint and runtime ABI.
// Read little-endian fields explicitly: rows are only two-byte aligned and
// this implementation also serves host-side numerical regression tests.
namespace ftt {
namespace quantized {

enum class encoding { iq4_xs, q4_k };

FTT_DEVICE_INLINE unsigned int load_u16(const unsigned char *p) {
    return static_cast<unsigned int>(p[0]) |
           (static_cast<unsigned int>(p[1]) << 8);
}

FTT_DEVICE_INLINE float load_f16(const unsigned char *p) {
    const unsigned int half = load_u16(p);
    const unsigned int sign = (half & 0x8000u) << 16;
    unsigned int exponent = (half >> 10) & 31u;
    unsigned int mantissa = half & 1023u;
    unsigned int bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 113;
            while ((mantissa & 1024u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            bits = sign | (exponent << 23) | ((mantissa & 1023u) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

template <encoding Format> FTT_DEVICE_INLINE constexpr int block_bytes() {
    return Format == encoding::iq4_xs ? 136 : 144;
}

// Feature is relative to the selected complete packed row, never a byte
// offset into an unquantized view of that row.
template <encoding Format>
FTT_DEVICE_INLINE float decode(const unsigned char *row, dim_t feature) {
    const unsigned char *block = row + (feature / 256) * block_bytes<Format>();
    const int group = static_cast<int>((feature % 256) / 32);
    const int lane = static_cast<int>(feature % 32);
    const float d = load_f16(block);
    if constexpr (Format == encoding::iq4_xs) {
        const unsigned int low = (block[4 + group / 2] >> (4 * (group % 2))) & 15;
        const unsigned int high = (load_u16(block + 2) >> (2 * group)) & 3;
        const int scale = static_cast<int>(low | (high << 4)) - 32;
        const unsigned int packed = block[8 + group * 16 + lane % 16];
        const unsigned int code = (packed >> (4 * (lane / 16))) & 15;
        constexpr signed char values[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113,
        };
        return (d * static_cast<float>(scale)) * static_cast<float>(values[code]);
    } else {
        const unsigned char *scales = block + 4;
        const unsigned int a = scales[group % 4];
        const unsigned int b = scales[4 + group % 4];
        const unsigned int c = scales[8 + group % 4];
        const unsigned int scale = group < 4 ? (a & 63) : ((c & 15) | ((a >> 2) & 48));
        const unsigned int minimum = group < 4 ? (b & 63) : ((c >> 4) | ((b >> 2) & 48));
        const unsigned int packed = block[16 + (group / 2) * 32 + lane];
        const unsigned int code = (packed >> (4 * (group % 2))) & 15;
        return (d * static_cast<float>(scale)) * static_cast<float>(code)
             - load_f16(block + 2) * static_cast<float>(minimum);
    }
}

namespace detail {

// Assemble bytes without imposing alignment on the packed row. Named words
// keep the low/high nibble cache in registers rather than an indexed array.
FTT_DEVICE_INLINE unsigned int load_u32(const unsigned char *p) {
    return load_u16(p) | (load_u16(p + 2) << 16);
}

// Construct once inside the valid complete256-block prefix guard. Offsets used
// below are multiples of four and each span stays inside that complete prefix;
// row/block strides136 preserve the base alignment. This does not upgrade the
// public alignment-one UINT8 ABI or infer allocation bounds from address space.
class iq4_row_reader {
    const unsigned char *row_;
#if defined(FTT_ARCH_NVIDIA) && defined(FTT_DEVICE_PASS)
    unsigned long long global_address_ = 0;
    bool global_words_;
#endif
public:
    FTT_DEVICE_INLINE explicit iq4_row_reader(const unsigned char *row) : row_(row)
#if defined(FTT_ARCH_NVIDIA) && defined(FTT_DEVICE_PASS)
        , global_words_(arch::try_aligned_global_u32_address(row, global_address_))
#endif
    {}

    FTT_DEVICE_INLINE bool uses_global_words() const {
#if defined(FTT_ARCH_NVIDIA) && defined(FTT_DEVICE_PASS)
        return global_words_;
#else
        return false;
#endif
    }

    FTT_DEVICE_INLINE unsigned int word(dim_t byte_offset) const {
#if defined(FTT_ARCH_NVIDIA) && defined(FTT_DEVICE_PASS)
        if (global_words_)
            return arch::load_global_u32(global_address_ + static_cast<size_type>(byte_offset));
#endif
        return load_u32(row_ + byte_offset);
    }
};

FTT_DEVICE_INLINE int iq4_value(unsigned int code) {
    const unsigned int word = code < 8
        ? (code < 4 ? 0xbfad9881u : 0xf6eaddcfu)
        : (code < 12 ? 0x26190d01u : 0x71594535u);
    const unsigned int value = (word >> (8 * (code & 3))) & 255;
    return static_cast<int>(value) - static_cast<int>((value & 128) << 1);
}

#if defined(FTT_ARCH_NVIDIA)
template <int Nibble>
FTT_DEVICE_INLINE unsigned int iq4_lut4(unsigned int packed) {
    static_assert(Nibble == 0 || Nibble == 4, "IQ4 has low and high nibble passes.");
    const unsigned int q = (packed >> Nibble) & 0x0f0f0f0fu;
    // Compact the four code nibbles, then clear every PRMT sign-selector bit.
    const unsigned int pairs = (q | (q >> 4)) & 0x00ff00ffu;
    const unsigned int selector = (pairs | (pairs >> 8)) & 0x7777u;
    const unsigned int high = (q >> 3) & 0x01010101u;
    const unsigned int mask = high * 255u;
    const unsigned int lo = arch::byte_permute_default(0xbfad9881u, 0xf6eaddcfu, selector);
    const unsigned int hi = arch::byte_permute_default(0x26190d01u, 0x71594535u, selector);
    return (lo & ~mask) | (hi & mask);
}

template <int Nibble, class T>
FTT_DEVICE_INLINE float iq4_nibble_dot_grouped(const T *source, unsigned int word0,
                                             unsigned int word1, unsigned int word2,
                                             unsigned int word3, float factor,
                                             float accumulator) {
    // One decoded quad dies after four ascending scalar updates. All low quads
    // are called before any high quad; no source vector load or extra accumulator.
    FTT_NO_UNROLL
    for (int quad = 0; quad < 4; ++quad) {
        const unsigned int packed = quad < 2
            ? (quad == 0 ? word0 : word1) : (quad == 2 ? word2 : word3);
        const unsigned int levels = iq4_lut4<Nibble>(packed);
        const T *values = source + quad * 4;
        float weight = factor * static_cast<float>(arch::signed_byte_extract<0>(levels));
        accumulator += weight * to_float(values[0]);
        weight = factor * static_cast<float>(arch::signed_byte_extract<1>(levels));
        accumulator += weight * to_float(values[1]);
        weight = factor * static_cast<float>(arch::signed_byte_extract<2>(levels));
        accumulator += weight * to_float(values[2]);
        weight = factor * static_cast<float>(arch::signed_byte_extract<3>(levels));
        accumulator += weight * to_float(values[3]);
    }
    return accumulator;
}
#endif

template <int Nibble, class T>
FTT_DEVICE_INLINE float iq4_nibble_dot(const T *source, unsigned int word0,
                                     unsigned int word1, unsigned int word2,
                                     unsigned int word3, float factor,
                                     float accumulator) {
#if defined(FTT_ARCH_NVIDIA) && defined(FTT_DEVICE_PASS)
    return iq4_nibble_dot_grouped<Nibble>(source, word0, word1, word2, word3, factor, accumulator);
#else
    // Explicit register selection avoids an indexed local cache. Do not
    // unroll the recurrence: it is reused by every native dot call body.
    FTT_NO_UNROLL
    for (int lane = 0; lane < 16; ++lane) {
        const unsigned int packed = lane < 8
            ? (lane < 4 ? word0 : word1) : (lane < 12 ? word2 : word3);
        const unsigned int code = (packed >> (8 * (lane & 3) + Nibble)) & 15;
        const float weight = factor * static_cast<float>(iq4_value(code));
        accumulator += weight * to_float(source[lane]);
    }
    return accumulator;
#endif
}

} // namespace detail

template <encoding Format, class T>
FTT_DEVICE_INLINE float dot(const T *source, const unsigned char *row, dim_t k) {
    float accumulator = 0.0f;
    dim_t feature = 0;
    if constexpr (Format == encoding::iq4_xs) {
        // k<=0 must touch neither pointer. A complete prefix also avoids
        // signed overflow in the block bound/increment and any tail preload.
        const dim_t complete = k > 0 ? k - k % 256 : 0;
        if (complete > 0) {
        const detail::iq4_row_reader reader(row);
        FTT_NO_UNROLL
        for (; feature < complete; feature += 256) {
            const dim_t block_offset = (feature / 256) * 136;
            const unsigned char *block = row + block_offset;
            const float d = load_f16(block);
            const unsigned int high = load_u16(block + 2);
            const unsigned int low = reader.word(block_offset + 4);
            FTT_NO_UNROLL
            for (int group = 0; group < 8; ++group) {
                const unsigned int scale_bits = ((low >> (4 * group)) & 15)
                    | (((high >> (2 * group)) & 3) << 4);
                const int scale = static_cast<int>(scale_bits) - 32;
                const float factor = d * static_cast<float>(scale);
                const dim_t payload = block_offset + 8 + group * 16;
                const unsigned int word0 = reader.word(payload);
                const unsigned int word1 = reader.word(payload + 4);
                const unsigned int word2 = reader.word(payload + 8);
                const unsigned int word3 = reader.word(payload + 12);
                const T *values = source + feature + group * 32;
                // GGML stores low features0..15 then high features16..31.
                // Reuse every payload byte without interleaving those halves
                // or moving the factor into the activation product.
                accumulator = detail::iq4_nibble_dot<0>(values, word0, word1, word2, word3, factor, accumulator);
                accumulator = detail::iq4_nibble_dot<4>(values + 16, word0, word1, word2, word3, factor, accumulator);
            }
        }
    }
    }
    // Q4 and the IQ4 partial block retain the exact scalar access/arithmetic
    // path. Full reductions stay FP32, with the target's original contraction.
    FTT_NO_UNROLL
    for (; feature < k; ++feature)
        accumulator += decode<Format>(row, feature) * to_float(source[feature]);
    return accumulator;
}

} // namespace quantized
} // namespace ftt
