//===------------------------ randgen.c -----------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::RandGen see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __RandGen(uint64_t *src0, uint64_t *src1, uint64_t *dst0, uint64_t *dst1,
               uint64_t *dst2, uint32_t src_elem_num, uint16_t fmt) {
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

  cmd->RandGen(&inst, *src0, *src1, *dst0, *dst1, *dst2, src_elem_num,
               (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
