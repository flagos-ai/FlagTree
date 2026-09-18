//===---------------------- reduce.c --------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::RcsReduce, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "op_reduce_mul_impl.h"
#include "tx81_run.h"
// The arguments list is aligned with RcsConv in Tx81Ops.td
void __ReduceSum(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  // Create reduce command buffer.
  RcsReduce *cmd = g_intrinsic()->reduce_pointer;
  RcsReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};
  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceSum(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;
  // Destroy the command buffer.
}

void __ReduceAvg(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  // Create reduce command buffer.
  RcsReduce *cmd = g_intrinsic()->reduce_pointer;
  RcsReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};
  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceAvg(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;
  // Destroy the command buffer.
}

void __ReduceMax(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  // Create reduce command buffer.
  RcsReduce *cmd = g_intrinsic()->reduce_pointer;
  RcsReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};

  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceMax(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;
  // Destroy the command buffer.
}

void __ReduceMin(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;
  // Create reduce command buffer.
  RcsReduce *cmd = g_intrinsic()->reduce_pointer;
  RcsReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};

  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceMin(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;

  // Destroy the command buffer.
}
void __ReduceMul(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  INTRNISIC_RUN_SWITCH;

  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  op_reduce_mul_impl(src, dst, shape1, dim, fmt);
}
