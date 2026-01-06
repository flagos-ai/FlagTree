//===------------------------ argmin.c ------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::ArgMin see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __ArgMin(uint64_t *src, uint64_t *dst0, uint64_t *dst1,
              uint32_t elem_count, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  volatile void *min_val =
      (volatile void *)get_spm_memory_mapping((uint64_t)dst0);
  volatile void *min_idx =
      (volatile void *)get_spm_memory_mapping((uint64_t)dst1);
  // Create command buffer.
  if (elem_count == 1) {
    volatile void *src_data =
      (volatile void *)get_spm_memory_mapping((uint64_t)src);
    *(uint32_t *)min_idx = 0;
    switch (fmt) {
      case Fmt_FP16:
      case Fmt_BF16:
        *(uint16_t*)min_val = *(uint16_t *)src_data;
        break;
      case Fmt_FP32:
      case Fmt_TF32:
        *(uint32_t*)min_val = *(uint32_t *)src_data;
        break;
      default:
        assert(0 && "ArgMax: Unsupport dtype");
    }
    return;
  }
  TsmPeripheral *cmd = g_intrinsic()->peripheral_pointer;
  TsmPeripheralInstr inst = {I_CGRA, {0,}, {0,}};;

  cmd->ArgMin(&inst, (uint64_t) src, elem_count, (Data_Format) fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  TsmWaitfinish();
  
  switch (fmt) {
    case Fmt_FP16:
    case Fmt_BF16:
      *(uint16_t*)min_val = *(uint16_t *)&inst.param.wb_data0;
      break;
    case Fmt_FP32:
    case Fmt_TF32:
      *(uint32_t*)min_val = *(uint32_t *)&inst.param.wb_data0;
      break;
    default:
      assert(0 && "ArgMin: Unsupport dtype");
  }
  
  *(uint32_t *)min_idx = *(uint32_t *)&inst.param.wb_data1;


  // Destroy the command buffer.
  
}
