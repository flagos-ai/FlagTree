//===------------------------- mxfp_scale_fp16.c --------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::MXFPScaleFp16Op see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"
#include <stdio.h>

/**
 * Applies microscaling to FP16 data using E8M0 scale factors.
 *
 * Implements OCP Microscaling Format (MX) specification by scaling blocks
 * of FP16 data with E8M0 scale factors. Special handling for NaN scales
 * as per MX specification requirements.
 *
 * @param value          Source FP16 data pointer
 * @param scale          E8M0 scale factors pointer (1 per block)
 * @param dst            Destination FP16 data pointer
 * @param elem_count     Total elements in source (must be blocks*32)
 */
void __mxfpScaleFP16(uint16_t *value, uint8_t *scale, uint16_t *dst,
                     uint32_t elem_count) {
  const int scaling_block_size = 32; // MXFP4 block size per OCP spec

  // Obtain hardware-specific memory mapping for scale factors
  scale = (uint8_t *)get_spm_memory_mapping_wrapper((uint64_t)scale);

  // Create command buffer.
  TsmArith *cmd = g_intrinsic()->arith_pointer;
  TsmArithInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  for (uint32_t block_idx = 0; block_idx < elem_count / scaling_block_size; block_idx++) {
    uint16_t scale_fp16;
    uint8_t scale_val = scale[block_idx];

    // referrence to test_dot_scaled.py:
    // ```
    //  scale_fp32 = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    //  upcasted_scale = scale_fp32.to(tl.float16)
    // ```
    // Pure bitwise E8M0 to FP16 conversion (matching Python impl)
    if (scale_val == 0xFF) {
      // MX spec: 0xFF = NaN; FP16 NaN (all exp 1s, non-zero mantissa)
      scale_fp16 = 0x7FFF;
    } else {
      // Step 1: Extract float32 exponent (mimic Python's uint32 << 23)
      // E8M0 8-bit value as float32 exponent (float32 bias 127)
      uint32_t f32_exp = scale_val;
      
      // Step 2: Convert to FP16 exponent (FP16 bias 15)
      // FP16 exp = (f32 actual exp) + 15 = (f32_exp - 127) + 15 = f32_exp - 112
      int16_t fp16_exp = (int16_t)f32_exp - 112;
      
      // Step 3: Handle exponent range (FP16 exp: 5-bit, 0~31)
      if (fp16_exp > 31) {
        // Overflow: positive infinity (all exp 1s, mantissa 0s)
        scale_fp16 = 0x7C00;
      } else if (fp16_exp < 0) {
        // Underflow: zero
        scale_fp16 = 0x0000;
      } else {
        // Normal range: construct FP16 (sign 0, 5-bit exp, 10-bit mantissa 0)
        scale_fp16 = (uint16_t)(fp16_exp << 10);  // Mantissa all 0s
      }
    }

    // Calculate block positions
    uint16_t *block_src = value + block_idx * scaling_block_size;
    uint16_t *block_dst = dst + block_idx * scaling_block_size;

    // Apply scaling via vector-scalar multiplication with FP16 format
    cmd->MulVS(&inst, (uint64_t)block_src, (uint32_t)scale_fp16,
               (uint64_t)block_dst, scaling_block_size, RND_NEAREST_EVEN,
               Fmt_FP16);
    TsmExecute(&inst);
  }

}
