//===------------------------ memset.c ------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Memset see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"

void RcsMemset(char *dst, int value, int elem_count, uint16_t fmt) {
  // Create command buffer.
  RcsPeripheral *cmd = g_intrinsic()->peripheral_pointer;
  RcsDataMoveInstr inst = {I_TDMA,
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

  St_StrideIteration si = {stride0,    iteration0, stride1,
                           iteration1, stride2,    iteration2};
  cmd->Memset(&inst, (uint64_t)dst, value, elem_count, &si, (Data_Format)fmt);

  // Dispatch the command to accelerator
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;
}

void SetZero(char *dst, int elem_count, uint16_t fmt) {
  RcsLogic *logic = (RcsLogic *)getRcsOpPointer()->logic_pointer;
  RcsLogicInstr inst = {I_CGRA,{0,},{0,}};
  logic->XorVV(&inst, (uint64_t)dst, (uint64_t)dst, (uint64_t)dst, elem_count,
             (Data_Format)fmt);
  RcsExecute(&inst);
}

void __Memset(char *dst, int value, int *dst_shape, int *dst_stride, int rank,
              uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;

  int elem_count = 1;
  for (int i = 0; i < rank; i++) {
    elem_count *= dst_shape[i];
  }

  if (fmt == Fmt_INT8) {
    if (elem_count % 2 != 0) {
      RcsMemset(dst, value, elem_count, fmt);
      return;
    } else {
       // xor does not support int8, used fp16 replace
      fmt        = Fmt_FP16;
      elem_count = shift_div(elem_count, 2);
    }
  }

  if (value == 0) {
    SetZero(dst, elem_count, fmt);
  }
  else {
    if ((get_dtype_size_new(fmt) == 4 && value == 0xffffffff)
      || (get_dtype_size_new(fmt) == 2 && (uint16_t)value == 0xffff)
      || (get_dtype_size_new(fmt) == 1 && (uint8_t)value == 0xff)) {
      SetZero(dst, elem_count, fmt);
      RcsLogic *logic = (RcsLogic *)getRcsOpPointer()->logic_pointer;
      RcsLogicInstr inst = {I_CGRA,{0,},{0,}};
      // Some specific values, addvs are not supported.
      logic->BoolNotV(&inst, (uint64_t)dst, (uint64_t)dst, elem_count*get_dtype_size_new(fmt)*8);
      RcsExecute(&inst);
    } else {
      RcsMemset(dst, value, elem_count, fmt);
    }
  }

  // Destroy the command buffer.
}
