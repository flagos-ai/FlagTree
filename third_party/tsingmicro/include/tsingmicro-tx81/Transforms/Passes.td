//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TX81_TRANSFORMS_PASSES
#define TX81_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def InsertBarrier : Pass<"tx81-insert-barrier", "mlir::ModuleOp"> {
  let summary = "Insert barrier ops based on memory dependency analysis";
  let description = [{
    Analyzes the IR after MK-to-Tx81 conversion and inserts tx.barrier ops
    only where TsmWaitfinish synchronization is actually needed.

    At this IR level there are no memref.load/store; CPU ops are any ops
    whose dialect is NOT tx. DDR↔SPM data movement is exclusively through
    RDMA (DDR→SPM) and WDMA (SPM→DDR). All other NPU compute ops operate
    on SPM and need no barriers between them (hardware handles ordering).

    Barriers are inserted when:
    1. A CPU op (non-tx dialect) uses a Value that may alias a buffer with
       a pending asynchronous write (from RDMA, WDMA, or NPU compute).
    2. An RDMA reads from DDR that a preceding WDMA may have written to
       (RAW dependency through DDR between two async DMA engines).

    Value-based alias analysis is used to determine whether two operations
    access overlapping memory — only if the accessed Value traces back to
    the same base buffer as a pending write target is a barrier inserted.
  }];
  let constructor = "triton::createInsertBarrierPass()";
}

def Tx81ResolveDmaBaseAddr : Pass<"tx81-resolve-dma-base-addr", "mlir::ModuleOp"> {
  let summary = "Resolve DDR base address for tx.rdma and tx.wdma ops";
  let description = [{
    Analyzes each tx.rdma and tx.wdma op to trace its DDR address operand
    back to the originating func.func memref argument (base address).
    Appends the resolved base DDR address as a new last operand to the op.

    This is needed so the CRT __Rdma/__Wdma functions can locate the client
    pointer header (which sits at base_addr - ClientPtrHeaderBytes) for
    DMA bound checking, even when the actual DMA address is base + offset.
  }];
  let constructor = "triton::createTx81ResolveDmaBaseAddrPass()";
}

#endif
