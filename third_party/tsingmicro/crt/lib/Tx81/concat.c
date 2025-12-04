//===------------------------ concat.c ------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Concat see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Concat(uint64_t *src1, uint16_t src1_n, uint16_t src1_h, uint16_t src1_w,
              uint16_t src1_c, uint64_t *src2, uint16_t src2_n, uint16_t src2_h,
              uint16_t src2_w, uint16_t src2_c, uint64_t *dst, uint16_t dst_n,
              uint16_t dst_h, uint16_t dst_w, uint16_t dst_c, uint32_t dim,
              uint16_t fmt) {
  // Create command buffer.
  TsmDataMove *cmd = g_intrinsic()->datamove_pointer;
  TsmMoveInstr inst = {I_CGRA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

  Data_Shape shape1 = {src1_n, src1_h, src1_w, src1_c};
  Data_Shape shape2 = {src2_n, src2_h, src2_w, src2_c};
  Data_Shape shape3 = {dst_n, dst_h, dst_w, dst_c};
  cmd->Concat(&inst, (uint64_t)src1, shape1, (uint64_t)src2, shape2,
              (uint64_t)dst, shape3, dim, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
