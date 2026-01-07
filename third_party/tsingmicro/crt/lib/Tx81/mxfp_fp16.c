//===------------------------ mxfp_fp16.c ---------------------------------===//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation
// tx::FP8E5M2ToFP16Op/tx::FP8E4M3ToFP16Op/tx::FP4E2M1ToFP16Op
//
//===----------------------------------------------------------------------===//
#include "tx81.h"
#include <stdio.h>

/**
 * Converts an array of FP8 (E5M2) values to FP16 format
 *
 * @param src         Input array of FP8 values (E5M2 format)
 * @param dst         Output array for FP16 values (must be pre-allocated)
 * @param elem_count  Number of elements to convert
 *
 * FP8 (E5M2) format:
 *   [S][EEEEE][MM]
 *   1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits
 *
 * FP16 output format:
 *   [S][EEEE E][MMMM MMMMMM]
 *   1 sign bit, 5 exponent bits (bias=15), 10 mantissa bits
 */
void __FP8E5M2_FP16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  for (uint32_t i = 0; i < elem_count; i++) {
    uint8_t fp8 = src[i];
    uint8_t sign = fp8 & 0x80;            // Extract sign bit (10000000)
    uint8_t exponent = (fp8 >> 2) & 0x1F; // Extract 5-bit exponent (01111100)
    uint8_t mantissa = fp8 & 0x03;        // Extract 2-bit mantissa (00000011)

    // Handle subnormal values (exponent = 0)
    if (exponent == 0) {
      // According to OCP spec: v = (-1)^S × 2^(1-15) × (0 + 2^(-2) × M)
      // FP16 exponent = (1 - 15) + 15 = 1 (bias adjustment)
      uint16_t fp16_exp = 1;
      dst[i] = (sign << 8) | (fp16_exp << 10) | (mantissa << 8);
      continue;
    }

    // Handle special cases (exponent = 0x1F)
    if (exponent == 0x1F) {
      if (mantissa == 0) {
        // Infinity: set FP16 max exponent (0x1F) with zero mantissa
        dst[i] = (sign << 8) | (0x1F << 10);
      } else {
        // NaN: set FP16 max exponent with non-zero mantissa
        dst[i] = (sign << 8) | (0x1F << 10) | (mantissa << 8);
      }
      continue;
    }

    // Normal case conversion
    // Exponent bias matches (15), extend mantissa to 10 bits
    dst[i] = (sign << 8) |      // Sign bit (bit 15)
             (exponent << 10) | // 5-bit exponent (bits 14-10)
             (mantissa << 8);   // 2-bit mantissa extended to 10 bits (bits 9-8)
  }
}

/**
 * Converts an array of FP8 (E4M3) values to FP16 format
 *
 * @param src         Input array of FP8 values (E4M3 format)
 * @param dst         Output array for FP16 values (must be pre-allocated)
 * @param elem_count  Number of elements to convert
 *
 * FP8 (E4M3) format:
 *   [S][EEEE][MMM]
 *   1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
 *
 * FP16 output format:
 *   [S][EEEE E][MMMM MMMMMM]
 *   1 sign bit, 5 exponent bits (bias=15), 10 mantissa bits
 */
void __FP8E4M3_FP16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  for (uint32_t i = 0; i < elem_count; i++) {
    uint8_t fp8 = src[i];
    uint8_t sign = fp8 & 0x80;            // Extract sign bit (10000000)
    uint8_t exponent = (fp8 >> 3) & 0x0F; // Extract 4-bit exponent (00001111)
    uint8_t mantissa = fp8 & 0x07;        // Extract 3-bit mantissa (00000111)

    // Handle subnormal values (exponent = 0)
    if (exponent == 0) {
      // According to OCP spec: v = (-1)^S × 2^(1-7) × (0 + 2^(-3) × M)
      // FP16 exponent = (1 - 7) + 15 = 9 (bias adjustment)
      static const int denormsAndZeroLut[8] = {0x0000, 0x1800, 0x1C00, 0x1E00,
                                               0x2000, 0x2100, 0x2200, 0x2300};
      dst[i] = (sign << 8) | denormsAndZeroLut[mantissa];
      continue;
    }

    // Handle NaN case (exponent = 0x0F for E4M3)
    if (exponent == 0x0F) {
      // Set FP16 max exponent (0x1F) and extend mantissa
      dst[i] = (sign << 8) | (0x1F << 10) | (mantissa << 7);
      continue;
    }

    // Normal case conversion
    // Convert exponent from FP8 (bias=7) to FP16 (bias=15): E_fp16 = E_fp8 + 8
    uint8_t fp16_exponent = exponent + 8;

    // Construct FP16: extend mantissa to 10 bits
    dst[i] = (sign << 8) |           // Sign bit (bit 15)
             (fp16_exponent << 10) | // 5-bit exponent (bits 14-10)
             (mantissa << 7); // 3-bit mantissa extended to 10 bits (bits 9-7)
  }
}

/**
 * Converts an array of FP8 (E4M3FN) values to FP16 format (supports subnormal
 * numbers)
 *
 * @param src         Input array of FP8 values (E4M3FN: Finite Normal +
 * Subnormal)
 * @param dst         Output array for FP16 values (must be pre-allocated)
 * @param elem_count  Number of elements to convert
 *
 * Based on: OCP Microscaling Formats (MX) Specification v1.0
 * - FP8 E4M3 format: 1 sign bit (S) + 4 exponent bits (E, bias=7) + 3 mantissa
 * bits (M)
 * - Subnormal numbers (E=0): v = (-1)^S × 2^(1-bias) × (M/8) (Section 5.3.1)
 * - NaN (E=0xF): No infinities defined; E=0xF represents NaNs (Table 2)
 * - FP16 format: 1 sign bit + 5 exponent bits (bias=15) + 10 mantissa bits
 * (IEEE 754 compatible)
 */
void __FP8E4M3FN_FP16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  for (uint32_t i = 0; i < elem_count; i++) {
    uint8_t fp8 = src[i];
    uint8_t sign = fp8 & 0x80;            // Extract sign bit (10000000)
    uint8_t exponent = (fp8 >> 3) & 0x0F; // Extract 4-bit exponent (00001111)
    uint8_t mantissa = fp8 & 0x07;        // Extract 3-bit mantissa (00000111)

    // Handle subnormal values (exponent = 0)
    if (exponent == 0) {
      // According to OCP spec: v = (-1)^S × 2^(1-7) × (0 + 2^(-3) × M)
      // FP16 exponent = (1 - 7) + 15 = 9 (bias adjustment)
      static const int denormsAndZeroLut[8] = {0x0000, 0x1800, 0x1C00, 0x1E00,
                                               0x2000, 0x2100, 0x2200, 0x2300};

      dst[i] = (sign << 8) | denormsAndZeroLut[mantissa];
      continue;
    }

    // Handle NaN case (exponent = 0x0F for E4M3)
    if (fp8 == 0x7F || fp8 == 0xFF) {
      // Set FP16 max exponent (0x1F) and extend mantissa
      dst[i] = (sign << 8) | (0x1F << 10) | (mantissa << 7);
      continue;
    }

    // Normal case conversion
    // Convert exponent from FP8 (bias=7) to FP16 (bias=15): E_fp16 = E_fp8 + 8
    uint8_t fp16_exponent = exponent + 8;

    // Construct FP16: extend mantissa to 10 bits
    dst[i] = (sign << 8) |           // Sign bit (bit 15)
             (fp16_exponent << 10) | // 5-bit exponent (bits 14-10)
             (mantissa << 7); // 3-bit mantissa extended to 10 bits (bits 9-7)
  }
}

/**
 * Converts packed FP4 (E2M1) values to FP16 format
 *
 * @param src         Input array of packed FP4 values (2 values per byte)
 * @param dst         Output array for FP16 values (must be pre-allocated)
 * @param elem_count  Number of FP4 elements (not bytes)
 *
 * FP4 (E2M1) format (per element):
 *   [S][EE][M]
 *   1 sign bit (bit3), 2 exponent bits (bit2-1), 1 mantissa bit (bit0)
 *
 * Storage format:
 *   Each byte contains two FP4 values:
 *   [S1 E1 E0 M1] [S0 E1 E0 M0] (high nibble first, corrected symmetry)
 *
 * FP16 output format:
 *   [S][EEEE E][MMMM MMMMMM]
 *   1 sign bit (bit15), 5 exponent bits (bit14-10), 10 mantissa bits (bit9-0)
 */
void __FP4E2M1_FP16(uint8_t *src, uint16_t *dst, uint32_t elem_count) {
  src = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)src);
  dst = (uint16_t *)get_spm_memory_mapping_wrapper((uint64_t)dst);

  // Constants matching Python logic
  const uint16_t to_bias = 15;       // FP16 exponent bias
  const uint16_t to_m_bits = 10;     // FP16 mantissa bits
  const uint16_t to_point5 = 0x3800; // FP16 value for ±0.5 (subnormal mapping)
  const uint16_t bias_offset = (to_bias - 1)
                               << to_m_bits; // (15-1) <<10 = 14<<10

  for (uint32_t i = 0; i < (elem_count + 1) / 2; i++) {
    uint8_t byte = src[i];
    // Process high nibble (first element, bits 7-4)
    if (2 * i + 1 < elem_count) {
      uint8_t elem = (byte >> 4) & 0x0F; // Extract 4-bit FP4 element
      uint16_t sign = elem & 0x08;       // Sign bit (bit3 of element)
      uint16_t exp = (elem >> 1) & 0x03; // Exponent bits (bit2-1 of element)
      uint16_t mant = elem & 0x01;       // Mantissa bit (bit0 of element)
      uint16_t result;
      if (exp == 0) {
        // Subnormal or zero (exp=0)
        if (mant == 1) {
          // Subnormal: map to ±0.5 (to_point5 + sign)
          result =
              (sign << 12) | to_point5; // Sign shifted to FP16 bit15 (3+12=15)
        } else {
          // Zero: only preserve sign
          result = sign << 12;
        }
      } else {
        // Normal number: base mantissa + bias offset
        uint16_t abs = elem & 0x07;
        uint16_t base = (abs << (to_m_bits - 1)) |
                        (sign << 12); // Mantissa shifted to FP16 bit9 (10-1=9)
        result = base + bias_offset;
      }
      dst[2 * i + 1] = result;
    }

    // Process low nibble (second element, bits 3-0) if needed
    {
      uint8_t elem = byte & 0x0F;        // Extract 4-bit FP4 element
      uint16_t sign = elem & 0x08;       // Sign bit (bit3 of element)
      uint16_t exp = (elem >> 1) & 0x03; // Exponent bits (bit2-1 of element)
      uint16_t mant = elem & 0x01;       // Mantissa bit (bit0 of element)

      uint16_t result;
      if (exp == 0) {
        if (mant == 1) {
          result = (sign << 12) | to_point5;
        } else {
          result = sign << 12;
        }
      } else {
        uint16_t abs = elem & 0x07;
        uint16_t base = (abs << (to_m_bits - 1)) |
                        (sign << 12); // Mantissa shifted to FP16 bit9 (10-1=9)
        result = base + bias_offset;
      }
      dst[2 * i] = result;
    }
  }
}
