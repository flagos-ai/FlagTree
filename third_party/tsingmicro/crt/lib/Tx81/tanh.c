//===------------------------ tanh.c --------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Tanh see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Tanh(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmActivation *cmd = g_intrinsic()->activation_pointer;
  TsmActivationInstr inst = {I_CGRA,
                             {
                                 0,
                             },
                             {
                                 0,
                             }};

  cmd->Tanh(&inst, (uint64_t)src, (uint64_t)dst, elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
