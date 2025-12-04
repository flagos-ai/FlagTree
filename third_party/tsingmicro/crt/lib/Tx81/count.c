//===------------------------ count.c -------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Count see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Count(uint64_t *src, uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmPeripheral *cmd = g_intrinsic()->peripheral_pointer;
  TsmPeripheralInstr inst = {I_CGRA,
                             {
                                 0,
                             },
                             {
                                 0,
                             }};
  ;

  cmd->Count(&inst, (uint64_t)src, elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
