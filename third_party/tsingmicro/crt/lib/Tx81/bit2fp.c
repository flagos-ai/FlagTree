//===------------------------ bit2fp.c ------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Bit2Fp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Bit2Fp(uint64_t *src, uint64_t *target, uint32_t elem_count,
              uint16_t fmt) {
  // Create command buffer.
  TsmPeripheral *cmd = g_intrinsic()->peripheral_pointer;
  TsmPeripheralInstr inst = {I_CGRA,
                             {
                                 0,
                             },
                             {
                                 0,
                             }};

  //   assert(elem_count % 8 == 0);

  cmd->Bit2Fp(&inst, (uint64_t)src, (uint64_t)target, elem_count,
              (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
