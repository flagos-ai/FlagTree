//===------------------------ int8_fp32.c ---------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::INT8_FP32 see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __INT8_FP32(uint64_t *src, uint64_t *dst, uint32_t zp,
                 uint32_t elem_count) {
  // Create command buffer.
  TsmConvert *cmd = g_intrinsic()->convert_pointer;
  TsmConvertInstr inst = {I_CGRA,
                          {
                              0,
                          },
                          {
                              0,
                          }};

  cmd->INT8_FP32(&inst, (uint64_t)src, zp, (uint64_t)dst, elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
  // Destroy the command buffer.
}
