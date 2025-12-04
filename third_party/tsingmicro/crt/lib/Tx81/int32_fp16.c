//===------------------------ int32_fp16.cpp
//-------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::INT32_FP16 see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __INT32_FP16(uint64_t *src, uint64_t *dst, uint32_t elem_count,
                  RND_MODE round) {
  // Create command buffer.
  TsmConvert *cmd = g_intrinsic()->convert_pointer;
  TsmConvertInstr inst = {I_CGRA,
                          {
                              0,
                          },
                          {
                              0,
                          }};

  cmd->INT32_FP16(&inst, (uint64_t)src, (uint64_t)dst, elem_count, round);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
