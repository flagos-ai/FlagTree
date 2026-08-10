/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_PPUGPU_OPS
#define TRITON_PPUGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "TritonPPUGPUDialect.td"
include "TritonPPUGPUAttrDefs.td"


class TT_PPUGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonPPUGPU_Dialect, mnemonic, !listconcat(traits, [])>;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;


def AsyncAIUCopyGlobalToLocalOp : TT_PPUGPU_Op<"async_aiu_copy_global_to_local", [
  SameVariadicOperandSize
]> {
  let summary = "copy data from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory asynchronously.
    This is analogue to tt.load except the data are copied to local memory pointed
    by by the memory descriptor instread of a distributed tensor. The rest of the
    operands are the same as tt.load.
  }];

  let arguments = (ins
    Arg<TT_PtrType, "", [MemRead<GlobalMemory>]>:$src,
    Variadic<I32>:$coord,
    Variadic<I64>:$shape,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$result,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let results = (outs TTG_AsyncToken:$token);

  // Specify cacheModifier and evictionPolicy explicitly, instead of leaving
  // them in attr-dict, because this way their values get printed as strings,
  // rather than as opaque integers.
  //
  // Note there are no commas between other, cacheModifier, and evictionPolicy,
  // due to limitations in MLIR's asm parser.
  let assemblyFormat = [{
    $src `[` $coord `]` `[` $shape `]` $result
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` type($src) `->` type($result)
  }];
}

#endif
