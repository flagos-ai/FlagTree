//===------------------------ gatherscatter.c -----------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::GatherScatter see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __GatherScatter(uint64_t *src, uint64_t *dst, uint32_t bytes,
                     uint32_t src_strideN, uint32_t src_strideH,
                     uint32_t src_strideW, uint32_t src_iterN,
                     uint32_t src_iterH, uint32_t src_iterW,
                     uint32_t dst_strideN, uint32_t dst_strideH,
                     uint32_t dst_strideW, uint32_t dst_iterN,
                     uint32_t dst_iterH, uint32_t dst_iterW) {
  // Create command buffer.
  TsmDataMove *cmd = g_intrinsic()->datamove_pointer;
  TsmDataMoveInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  St_StrideIteration src_si = {src_strideW, src_iterW,   src_strideH,
                               src_iterH,   src_strideN, src_iterN};
  St_StrideIteration dst_si = {dst_strideW, dst_iterW,   dst_strideH,
                               dst_iterH,   dst_strideN, dst_iterN};

  cmd->GatherScatter(&inst, (uint64_t)src, (uint64_t)dst, bytes, &src_si,
                     &dst_si);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
  // Destroy the command buffer.
}
