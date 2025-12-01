//===------------------------ bilinear.c ----------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Bilinear see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Bilinear(uint64_t *src, uint64_t *dst, uint16_t src_n, uint16_t src_h,
                uint16_t src_w, uint16_t src_c, uint16_t dst_n, uint16_t dst_h,
                uint16_t dst_w, uint16_t dst_c, uint16_t fmt) {
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

  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  Data_Shape shape2 = {dst_n, dst_h, dst_w, dst_c};
  cmd->Bilinear(&inst, (uint64_t)src, (uint64_t)dst, shape1, shape2,
                (src_w - 1) / (dst_w - 1), (src_h - 1) / (dst_h - 1),
                (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
