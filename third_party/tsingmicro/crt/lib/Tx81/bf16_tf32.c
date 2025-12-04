//===------------------------ bf16_tf32.c ---------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::BF16_TF32 see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __BF16_TF32(uint64_t *src, uint64_t *dst, uint32_t elem_count) {
  // Create command buffer.
  TsmConvert *cmd = g_intrinsic()->convert_pointer;
  TsmConvertInstr inst = {I_CGRA,
                          {
                              0,
                          },
                          {
                              0,
                          }};

  cmd->BF16_TF32(&inst, (uint64_t)src, (uint64_t)dst, elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
