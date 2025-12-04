//===------------------------ mask_move.c ---------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::MaskMoveOp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __MaskMove(uint64_t *src, uint64_t *target, uint32_t elem_count,
                uint64_t *mask, int32_t fmt) {
  TsmMaskDataMove *move = g_intrinsic()->maskdatamove_pointer;
  TsmMaskDataMoveInstr inst = {I_CGRA,
                               {
                                   0,
                               },
                               {
                                   0,
                               }};

  move->MaskMove(&inst, (uint64_t)src, (uint64_t)mask, (uint64_t)target,
                 elem_count, (Data_Format)fmt);

  TsmExecute(&inst);
}
