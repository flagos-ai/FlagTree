// Adapted from llama.cpp 5ce06fc7 vecdotq.cuh; see LICENSE and adaptation.json.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
static __device__ __forceinline__ int get_int_b4(const uint8_t *x, int i) {
    return ((const int *) x)[i];
}
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
    const uint32_t * table32 = (const uint32_t *) table;

    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
    // However, for the result we need ints with all even/odd 4 bit indices in q4.
    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
alignas(4) static __device__ __constant__ int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// Original vec_dot_iq4_xs_q8_1 word/LUT/scale decoder, adapted to one
// virtual FP32 coordinate instead of Q8_1/DP4A group contraction.
static __device__ __forceinline__ float vec_dot_iq4_xs_fp32(
    int aux_q4, uint32_t header, uint32_t scales_l, int t) {
    const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
    const int group=t/32;
    const int ls = ((scales_l >> (4*group)) & 0x0F) | (((header >> (16+2*group)) & 0x03) << 4);
    const int word = (t%32)<16 ? v.x : v.y;
    const int cv = int(int8_t(uint32_t(word) >> (8*(t%4))));
    const float d = __half2float(__ushort_as_half(uint16_t(header)));
    return __fmul_rn(__fmul_rn(d, float(ls-32)), float(cv));
}
