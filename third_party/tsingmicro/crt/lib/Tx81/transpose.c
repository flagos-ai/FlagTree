//===------------------------ transpose.c ---------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Transpose see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Transpose(uint64_t *src, uint64_t *dst, int32_t *src_shape,
                 int32_t *dst_shape, uint16_t fmt) {
  // Create command buffer.
  TsmDataMove *cmd = g_intrinsic()->datamove_pointer;
  TsmDataMoveInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  Data_Shape shape1 = {src_shape[0], src_shape[1], src_shape[2], src_shape[3]};
  Data_Shape shape2 = {dst_shape[0], dst_shape[1], dst_shape[2], dst_shape[3]};
  cmd->Transpose(&inst, (uint64_t)src, shape1, (uint64_t)dst, shape2,
                 (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
