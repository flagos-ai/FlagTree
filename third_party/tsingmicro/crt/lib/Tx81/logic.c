//===------------------------ logic.c -------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::LogicOp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __AndVV(uint64_t *src0, uint64_t *src1, uint64_t *dst, uint32_t elem_count,
             uint16_t fmt) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->AndVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
             (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}

void __OrVV(uint64_t *src0, uint64_t *src1, uint64_t *dst, uint32_t elem_count,
            uint16_t fmt) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->OrVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
            (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}

void __XorVV(uint64_t *src0, uint64_t *src1, uint64_t *dst, uint32_t elem_count,
             uint16_t fmt) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->XorVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
             (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}

void __BoolNotV(uint64_t *src, uint64_t *dst, uint32_t elem_count) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->BoolNotV(&inst, (uint64_t)src, (uint64_t)dst, elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
}

void __BoolAndV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                uint32_t elem_count) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->BoolAndV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
}

void __BoolOrV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
               uint32_t elem_count) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->BoolOrV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
               elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
}

void __BoolXorV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                uint32_t elem_count) {
  // Create command buffer.
  TsmLogic *cmd = g_intrinsic()->logic_pointer;
  TsmLogicInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->BoolXorV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);
  TsmWaitfinish();
}
