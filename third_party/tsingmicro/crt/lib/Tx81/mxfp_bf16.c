//===------------------------ mxfp_bf16.c ---------------------------------===//
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
      // Denormal/subnormal: Flush to zero (keeping sign)
      dst[i] = sign << 8;
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
 * Converts packed FP4 (E2M1) values to BF16 format
 *
 * @param src         Input array of packed FP4 values (2 values per byte)
 * @param dst         Output array for BF16 values (must be pre-allocated)
 * @param elem_count  Number of FP4 elements (not bytes)
 *
 * FP4 (E2M1) format (per element):
 *   [S][EE][M]
 *   1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit
 *
 * Storage format:
 *   Each byte contains two FP4 values:
 *   [S1 E1 E0 M1] [S0 E0 E0 M0] (high nibble first)
 */
void __FP4E2M1_BF16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  for (uint32_t i = 0; i < (elem_count + 1) / 2; i++) {
    uint8_t byte = src[i];

    // Process high nibble (first element)
    {
      uint8_t elem = (byte >> 4) & 0x0F;
      uint8_t sign = elem & 0x08;
      uint8_t exp = (elem >> 1) & 0x03;
      uint8_t mant = elem & 0x01;

      if (exp == 0) {
        // Subnormal: 0.5 * 2^(-1) = 0.25
        dst[2 * i] =
            (sign << 8) | (0x7E << 7) | (mant << 6); // 0x7E = 126 - 127 = -1
      } else if (exp == 0x03) {
        // NaN
        dst[2 * i] = (sign << 8) | (0xFF << 7) | (mant << 6);
      } else {
        // Normal: (1 + M) * 2^(E-1)
        uint8_t bf16_exp = (exp - 1) + 127; // Adjust bias
        dst[2 * i] = (sign << 8) | (bf16_exp << 7) | (mant << 6);
      }
    }

    // Process low nibble (second element) if needed
    if (2 * i + 1 < elem_count) {
      uint8_t elem = byte & 0x0F;
      uint8_t sign = elem & 0x08;
      uint8_t exp = (elem >> 1) & 0x03;
      uint8_t mant = elem & 0x01;

      if (exp == 0) {
        dst[2 * i + 1] = (sign << 8) | (0x7E << 7) | (mant << 6);
      } else if (exp == 0x03) {
        dst[2 * i + 1] = (sign << 8) | (0xFF << 7) | (mant << 6);
      } else {
        uint8_t bf16_exp = (exp - 1) + 127;
        dst[2 * i + 1] = (sign << 8) | (bf16_exp << 7) | (mant << 6);
      }
    }
  }
}
