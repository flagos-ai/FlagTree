//===------------------------ argmax.c ------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::ArgMax see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __ArgMax(uint64_t *src, uint64_t *dst0, uint64_t *dst1,
              uint32_t elem_count, uint16_t fmt) {
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

  cmd->ArgMax(&inst, (uint64_t)src, elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  TsmWaitfinish();

  *(float *)dst0 = *(float *)inst.param.wb_data0;
  *(int32_t *)dst1 = *(int32_t *)inst.param.wb_data1;

  // Destroy the command buffer.
}
