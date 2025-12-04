//===------------------------ memset.c ------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Memset see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Memset(char *dst, int value, int *dst_shape, int *dst_stride, int rank,
              uint16_t fmt) {
  // Create command buffer.
  TsmPeripheral *cmd = g_intrinsic()->peripheral_pointer;
  TsmDataMoveInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  // TODO: Use real stride and iteration, now accumulate all data to elem_count
  int stride0 = 0;
  int stride1 = 0;
  int stride2 = 0;

  int iteration0 = 1;
  int iteration1 = 1;
  int iteration2 = 1;

  int elem_count = 1;
  for (int i = 0; i < rank; i++) {
    elem_count *= dst_shape[i];
  }

  St_StrideIteration si = {stride0,    iteration0, stride1,
                           iteration1, stride1,    iteration2};
  cmd->Memset(&inst, (uint64_t)dst, value, elem_count, &si, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
  // Destroy the command buffer.
}
