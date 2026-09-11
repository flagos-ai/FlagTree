//===------------------------ randgen.c -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::RandGen see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"

void __RandGen(uint64_t *src0, uint64_t *src1, uint64_t *dst0, uint64_t *dst1,
               uint64_t *dst2, uint32_t src_elem_num, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  // Create command buffer.
  RcsPeripheral *cmd = g_intrinsic()->peripheral_pointer;
  RcsPeripheralInstr inst = {I_CGRA,
                             {
                                 0,
                             },
                             {
                                 0,
                             }};
  ;

  // Match other CRT peri wrappers: pointer args are SPM addresses, not
  // pointers-to-addresses. Hardware/sim interpret src_elem_num as byte count
  // (must be a multiple of 128); each 128B chunk yields 16 uint64 values.
  cmd->RandGen(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst0,
               (uint64_t)dst1, (uint64_t)dst2, src_elem_num, (Data_Format)fmt);

  // Dispatch the command to accelerator
  RcsExecute(&inst);
  SYNCHRONOUS_INTRINSIC_SWITCH;

  // Destroy the command buffer.
}
