//===------------------------ mxfp_bf16.c ---------------------------------===//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation
// tx::FP8E5M2ToBF16Op/tx::FP8E4M3ToBF16Op/tx::FP4E2M1ToBF16Op
//
//===----------------------------------------------------------------------===//
#include "tx81.h"
#include <stdio.h>


/**
 * Converts an array of FP8 (E5M2) values to BF16 format
 *
 * @param src         Input array of FP8 values (E5M2 format)
 * @param dst         Output array for BF16 values (must be pre-allocated)
 * @param elem_count  Number of elements to convert
 *
 * FP8 (E5M2) format:
 *   [S][EEEEE][MM]
 *   1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits
 *
 * BF16 output format:
 *   [S][EEEEEEEE][MMMMMMM]
 *   1 sign bit, 8 exponent bits (bias=127), 7 mantissa bits
 */
void __FP8E5M2_BF16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  INTRNISIC_RUN_SWITCH;
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  for (uint32_t i = 0; i < elem_count; i++) {
    // Extract FP8 components
    uint8_t fp8 = src[i];
    uint8_t sign = fp8 & 0x80;            // Isolate sign bit (10000000)
    uint8_t exponent = (fp8 >> 2) & 0x1F; // Extract 5-bit exponent (01111100)
    uint8_t mantissa = fp8 & 0x03;        // Extract 2-bit mantissa (00000011)

    // Handle special cases
    if (exponent == 0) {
      // Handle FP8 E5M2 subnormal/zero (per OCP MX Spec 5.3.1: E=0 is subnormal/zero)      
      static const uint16_t subNormLut[] = {0x0000, 0x3780, 0x3800, 0x3840};

      // Reconstruct BF16 format (per OCP MX Spec 5.3.1 and BF16 definition):
      // [15]    - Sign bit (from FP8 sign)
      // [14:7]  - 8-bit exponent (BF16 exponent)
      // [6:0]   - 7-bit mantissa (BF16 mantissa)
      dst[i] = (uint16_t)(sign << 8) | subNormLut[mantissa];
      continue;
    }

    if (exponent == 0x1F) {
      // NaN/Infinity: Preserve sign and mantissa, set max exponent
      dst[i] = (sign << 8) | (0x1F << 10) | (mantissa << 7);
      continue;
    }

    // Convert exponent from FP8 (bias=15) to BF16 (bias=127)
    // Formula: E_bf16 = E_fp8 + (127 - 15) = E_fp8 + 112
    uint16_t bf16_exponent = (uint16_t)(exponent + 112) << 7;

    // Reconstruct BF16 format:
    // [15]    - Sign bit
    // [14:7]  - 8-bit exponent
    // [6:0]   - 7-bit mantissa (FP8's 2-bit mantissa becomes bits [6:5])
    dst[i] = (sign << 8) |    // Sign bit at bit 15
             bf16_exponent |  // Exponent at bits 14-7
             (mantissa << 5); // Mantissa at bits 6-5 (bits 4-0 zero)
  }
}

/**
 * Converts an array of FP8 (E4M3) values to BF16 format
 *
 * @param src         Input array of FP8 values (E4M3 format)
 * @param dst         Output array for BF16 values (must be pre-allocated)
 * @param elem_count  Number of elements to convert
 *
 * FP8 (E4M3) format:
 *   [S][EEEE][MMM]
 *   1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
 *
 * Note: E4M3 has no infinities. Exponent=15 (0xF) represents NaNs
 *
 * BF16 output format:
 *   [S][EEEEEEEE][MMMMMMM]
 *   1 sign bit, 8 exponent bits (bias=127), 7 mantissa bits
 */
void __FP8E4M3_BF16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  INTRNISIC_RUN_SWITCH;
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  for (uint32_t i = 0; i < elem_count; i++) {
    // Extract FP8 components
    uint8_t fp8 = src[i];
    uint8_t sign = fp8 & 0x80;            // Isolate sign bit (10000000)
    uint8_t exponent = (fp8 >> 3) & 0x0F; // Extract 4-bit exponent (00001111)
    uint8_t mantissa = fp8 & 0x07;        // Extract 3-bit mantissa (00000111)

    // Handle special cases
    if (exponent == 0) {
      // Denormal/subnormal: Flush to zero (preserving sign only)
      // E4M3 denormals are not supported in this implementation
      dst[i] = sign << 8;
      continue;
    }

    if (exponent == 0x0F) {
      // NaN case (E4M3 has no infinities)
      // Set BF16 exponent to all 1s (0xFF) and preserve mantissa
      // Shift mantissa to top 3 bits of BF16 mantissa field
      dst[i] = (sign << 8) | (0xFF << 7) | (mantissa << 4);
      continue;
    }

    // Convert exponent from FP8 (bias=7) to BF16 (bias=127)
    // Formula: E_bf16 = E_fp8 + (127 - 7) = E_fp8 + 120
    uint8_t bf16_exponent = exponent + 120;

    // Reconstruct BF16 format:
    // [15]    - Sign bit
    // [14:7]  - 8-bit exponent
    // [6:0]   - 7-bit mantissa
    // Shift FP8 mantissa to bits [6:4] of BF16 mantissa field
    dst[i] = (sign << 8) |          // Sign bit at position 15
             (bf16_exponent << 7) | // Exponent at bits 14-7
             (mantissa << 4);       // Mantissa at bits 6-4 (bits 3-0 zero)
  }
}

/**
 * Converts an array of FP8 (E4M3FN) values to BF16 format (supports subnormal numbers)
 *
 * @param src         Input array of FP8 values (E4M3FN: Finite Normal + Subnormal)
 * @param dst         Output array for BF16 values (must be pre-allocated)
 * @param elem_count  Number of elements to convert
 *
 * Based on: OCP Microscaling Formats (MX) Specification v1.0
 * - FP8 E4M3 format: 1 sign bit (S) + 4 exponent bits (E, bias=7) + 3 mantissa bits (M)
 * - Subnormal numbers (E=0): v = (-1)^S × 2^(1-bias) × (M/8) (Section 5.3.1)
 * - NaN (E=0xF): No infinities defined; E=0xF represents NaNs (Table 2)
 * - BF16 format: 1 sign bit + 8 exponent bits (bias=127) + 7 mantissa bits (IEEE 754 compatible)
 */
void __FP8E4M3FN_BF16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
    // Memory mapping wrapper (retained as original)
    src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
    dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

    for (uint32_t i = 0; i < elem_count; i++) {
        uint8_t fp8 = src[i];

        // 1. Extract FP8 E4M3FN core fields (per Section 5.3.1)
        uint8_t sign_bit = (fp8 >> 7) & 0x01;  // Sign bit: 0=positive, 1=negative
        uint8_t exponent = (fp8 >> 3) & 0x0F;  // 4-bit exponent (E: 0~0xF)
        uint8_t mantissa = fp8 & 0x07;         // 3-bit mantissa (M: 0~7)

        // 2. Handle special case: NaN (E=0xF, explicitly no infinities in E4M3 per Table 2)
        if (fp8 == 0x7F || fp8 == 0xFF) { // Positive or negative NaN
            // BF16 NaN rule: all-1s exponent (0xFF) + non-zero mantissa (preserve E4M3 mantissa)
            uint16_t bf16_sign = (uint16_t)sign_bit << 15;  // Sign bit at BF16 bit 15
            uint16_t bf16_exponent = 0xFF << 7;             // Exponent bits (BF16 bits 14~7) all 1s
            uint16_t bf16_mantissa = (uint16_t)mantissa << 4;// E4M3 mantissa occupies top 3 bits of BF16 mantissa (bits 6~4)
            dst[i] = bf16_sign | bf16_exponent | bf16_mantissa;
            continue;
        }

        // 3. Handle subnormal numbers (E=0, per Section 5.3.1 formula)
        if (exponent == 0) {

          static const int denormsAndZeroLut[8] = {
              0x0000, 0x3b00, 0x3b80, 0x3bc0, 0x3c00, 0x3c20, 0x3c40, 0x3c60};
          dst[i] = (sign_bit << 15) | denormsAndZeroLut[mantissa];

          continue;
        }

        // 4. Handle normal numbers (E=1~14, per Section 5.3.1 formula)
        // 4.1 Exponent conversion: E_BF16 = (E_FP8 - bias_FP8) + bias_BF16 = E + (127-7) = E + 120
        uint8_t bf16_exponent = exponent + 120;
        // (Note: E=1→121, E=14→134, all within BF16 normal exponent range [1,254], no overflow handling needed)

        // 4.2 Reconstruct BF16 format (sign + exponent + mantissa)
        uint16_t bf16_sign = (uint16_t)sign_bit << 15;        // BF16 bit 15: sign
        uint16_t bf16_exp_bits = (uint16_t)bf16_exponent << 7;// BF16 bits 14~7: exponent
        uint16_t bf16_mant_bits = (uint16_t)mantissa << 4;   // BF16 bits 6~4: E4M3 mantissa (lower 4 bits zero-padded)
        dst[i] = bf16_sign | bf16_exp_bits | bf16_mant_bits;

    }

}


/**
 * Converts packed FP4 (E2M1) values to BF16 format
 *
 * @param src         Input array of packed FP4 values (2 values per byte)
 * @param dst         Output array for BF16 values (must be pre-allocated)
 * @param elem_count  Number of FP4 elements (not bytes)
 *
 * FP4 (E2M1) format (per element):
 *   [S][EE][M]
 *   1 sign bit (bit3), 2 exponent bits (bit2-1), 1 mantissa bit (bit0)
 *
 * Storage format:
 *   Each byte contains two FP4 values:
 *   [S1 E1 E0 M1] [S0 E1 E0 M0] (high nibble first, corrected symmetry)
 */
void __FP4E2M1_BF16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  INTRNISIC_RUN_SWITCH;
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);


  // Constants matching Python logic
  const uint16_t to_bias = 127;       // BF16 exponent bias
  const uint16_t to_m_bits = 7;       // BF16 mantissa bits
  const uint16_t to_point5 = 16128;   // BF16 value for ±0.5 (subnormal mapping)
  const uint16_t bias_offset = (to_bias - 1) << to_m_bits;  // (127-1) <<7 = 126<<7


  for (uint32_t i = 0; i < (elem_count + 1) / 2; i++) {
    uint8_t byte = src[i];

    // Process high nibble (first element, bits 7-4)
    if (2 * i + 1 < elem_count) {
      uint8_t elem = (byte >> 4) & 0x0F;  // Extract 4-bit FP4 element
      uint16_t sign = elem & 0x08;         // Sign bit (bit3 of element)
      uint16_t exp = (elem >> 1) & 0x03;   // Exponent bits (bit2-1 of element)
      uint16_t mant = elem & 0x01;         // Mantissa bit (bit0 of element)

      uint16_t result;
      if (exp == 0) {
        // Subnormal or zero (exp=0)
        if (mant == 1) {
          // Subnormal: map to ±0.5 (to_point5 + sign)
          result = (sign << 12) | to_point5;  // Sign shifted to BF16 bit15 (3+12=15)
        } else {
          // Zero: only preserve sign
          result = sign << 12;
        }
      } else {
        // Normal number: base mantissa + bias offset
        uint16_t abs = elem & 0x07;
        uint16_t base = (abs << (to_m_bits - 1)) | (sign << 12);  // Mantissa shifted to BF16 bit6 (7-1=6)
        result = base + bias_offset;
      }
      dst[2 * i  + 1] = result;
    }

    // Process low nibble (second element, bits 3-0) if needed
    {
      uint8_t elem = byte & 0x0F;         // Extract 4-bit FP4 element
      uint16_t sign = elem & 0x08;         // Sign bit (bit3 of element)
      uint16_t exp = (elem >> 1) & 0x03;   // Exponent bits (bit2-1 of element)
      uint16_t mant = elem & 0x01;         // Mantissa bit (bit0 of element)

      uint16_t result;
      if (exp == 0) {
        if (mant == 1) {
          result = (sign << 12) | to_point5;
        } else {
          result = sign << 12;
        }
      } else {
        uint16_t abs = elem & 0x07;
        uint16_t base = (abs << (to_m_bits - 1)) | (sign << 12);
        result = base + bias_offset;
      }
      dst[2 * i] = result;
    }


  }

}
