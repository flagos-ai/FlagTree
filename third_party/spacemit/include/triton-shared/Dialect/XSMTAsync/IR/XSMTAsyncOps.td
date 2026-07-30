// ===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: (c) 2024 SpacemiT
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMTASYNC_OPS
#define XSMTASYNC_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.td"

class XSMTAsync_Op<string mnemonic, list<Trait> traits = []> :
        Op<XSMTAsync_Dialect, mnemonic, traits>;

def XSMTAsync_MBarrierAllocOp : XSMTAsync_Op<"mbarrier_alloc", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "mbarrier alloc";
  let arguments = (ins
    Arg<I16>:$parity,
    Arg<I16>:$arr_count,
    Arg<I16>:$tx_count,
    Arg<I16>:$ex_count
  );
  let results = (outs
    Res<I64, "", [MemAlloc<DefaultResource, 0, FullEffect>]>:$mbarrier
  );
  let builders = [
    OpBuilder<(ins "Value":$parity, "Value":$arr_count, "Value":$tx_count,
      "Value":$ex_count)>
  ];
}

def XSMTAsync_MBarrierReleaseOp : XSMTAsync_Op<"mbarrier_release", [
    MemoryEffects<[MemFree]>
  ]> {
  let summary = "mbarrier release";
  let arguments = (ins Arg<I64, "", [MemFree]>:$mbarrier);
}

def XSMTAsync_MBarrierArriveOp : XSMTAsync_Op<"mbarrier_arrive", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "mbarrier arrive";
  let arguments = (ins Arg<I64, "", [MemWrite]>:$mbarrier);
}

def XSMTAsync_MBarrierWaitOp : XSMTAsync_Op<"mbarrier_wait", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "mbarrier wait";
  let arguments = (ins
    Arg<I64, "", [MemWrite]>:$mbarrier,
    Arg<I16>:$parity,
    Arg<I16>:$arr_count
  );
}

#endif // XSMTASYNC_OPS
