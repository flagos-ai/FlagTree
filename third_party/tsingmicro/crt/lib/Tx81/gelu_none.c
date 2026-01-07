//===------------------------ gelu_none.c
//-----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::GeluNone see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "op_gelu.h"

void __GeluNone(uint64_t *src, uint64_t *dst, uint32_t elem_count,
                uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  op_gelu_none(src, dst, elem_count, (Data_Format)fmt);
}
