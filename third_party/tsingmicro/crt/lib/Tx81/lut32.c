//===------------------------ lut32.c -------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Lut32 see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Lut32(uint64_t *src, uint64_t *dst, uint64_t *lut32,
             uint32_t src_elem_count, uint32_t lut_elem_count) {
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

  cmd->Lut32(&inst, (uint64_t)src, (uint64_t)dst, (uint64_t)lut32,
             src_elem_count, lut_elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
