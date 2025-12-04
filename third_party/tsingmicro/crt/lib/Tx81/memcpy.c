//===------------------------ memcpy.c ------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::MemCopyOp, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Memcpy(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt) {
  unsigned eleByte;

  switch (fmt) {
  case Fmt_BOOL: {
    // NOTE: Assume bool is 8 byte aligned
    eleByte = 1;
    elem_count = (elem_count + 7) / 8;
    fmt = Fmt_INT8;
    break;
  }
  case Fmt_INT8: {
    eleByte = 1;
    break;
  }
  case Fmt_INT16:
  case Fmt_FP16:
  case Fmt_BF16: {
    eleByte = 2;
    break;
  }
  case Fmt_INT32:
  case Fmt_FP32:
  case Fmt_TF32: {
    eleByte = 4;
    break;
  }
  case Fmt_INT64: {
    eleByte = 8;
    break;
  }
  default:
    // Other formats are not supported.
    assert(false && "Unsupported format\n");
    break;
  }

  if (elem_count == 0) {
    // If elem_count is 0, we don't need to do anything.
    return;
  }

  // Create command buffer.
  TsmDataMove *cmd = g_intrinsic()->datamove_pointer;
  TsmDataMoveInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  St_StrideIteration src_si = {1, 1, 1, 1, 1, 1};
  St_StrideIteration dst_si = {1, 1, 1, 1, 1, 1};

  cmd->GatherScatter(&inst, (uint64_t)src, (uint64_t)dst, eleByte * elem_count,
                     &src_si, &dst_si);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
}
