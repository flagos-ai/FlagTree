//===------------------------- mxfp_scale_bf16.c --------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::MXFPScaleBF16Op see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "instr_def.h"
#include "tx81.h"
#include <stdio.h>

/**
 * Applies microscaling to BF16 data using E8M0 scale factors.
 *
 * Implements OCP Microscaling Format (MX) specification by scaling blocks
 * of BF16 data with E8M0 scale factors. Special handling for NaN scales
 * as per MX specification requirements.
 *
 * @param value          Source BF16 data pointer (MXFP4-converted BF16 format)
 * @param scale          E8M0 scale factors pointer (1 per block)
 * @param dst            Destination BF16 data pointer
 * @param elem_count     Total elements in source (must be blocks*32)
 */
void __mxfpScaleBf16(uint16_t *value, uint8_t *scale, uint16_t *dst,
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

  // Main processing loop per scaling block
  for (uint32_t block_idx = 0; block_idx < elem_count / scaling_block_size;
       block_idx++) {
    // Convert E8M0 scale to BF16 representation
    uint16_t scale_bf16 = ((uint16_t)scale[block_idx]) << 7;

    // MX spec handling: E8M0 scale value 0xFF indicates NaN
    if (scale[block_idx] == 0xFF) {
      scale_bf16 = 0x7FC0; // BF16 quiet NaN encoding
    }

    // Calculate block positions
    uint16_t *block_src = value + block_idx * scaling_block_size;
    uint16_t *block_dst = dst + block_idx * scaling_block_size;

    // Apply scaling via vector-scalar multiplication
    cmd->MulVS(&inst, (uint64_t)block_src, (uint32_t)scale_bf16,
               (uint64_t)block_dst, scaling_block_size, RND_NEAREST_EVEN,
               Fmt_BF16);
    TsmExecute(&inst);
  }
}
