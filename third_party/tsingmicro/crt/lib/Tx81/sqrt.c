//===------------------------ sqrt.c --------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::SqrtVVOp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __SqrtVV(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  // Create command buffer.
  TsmArith *cmd = g_intrinsic()->arith_pointer;
  TsmArithInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->SqrtVV(&inst, (uint64_t)src, (uint64_t)dst, elem_count,
              (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  
}
