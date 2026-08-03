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

#ifndef PPUGPU_OPS
#define PPUGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "PPUGPUDialect.td"
include "PPUGPUAttrDefs.td"

def LLVM_PointerGlobal : LLVM_PointerInAddressSpace<1>;
def LLVM_PointerShared : LLVM_PointerInAddressSpace<3>;

def PPUGPU_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def PPUGPU_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def PPUGPU_ScalarLike : AnyTypeOf<[PPUGPU_Float, PPUGPU_Int]>;


def PPUGPU_MemSemanticAttr : I32EnumAttr<
    "MemSemantic", "",
    [
      I32EnumAttrCase<"RELAXED", 1, "relaxed">,
      I32EnumAttrCase<"ACQUIRE", 2, "acquire">,
      I32EnumAttrCase<"RELEASE", 3, "release">,
      I32EnumAttrCase<"ACQUIRE_RELEASE", 4, "acq_rel">,
    ]> {
    let cppNamespace = "::mlir::triton::ppugpu";
}

def PPUGPU_MemSyncScopeAttr : I32EnumAttr<
    "MemSyncScope", "",
    [
      I32EnumAttrCase<"GPU", 1, "gpu">,
      I32EnumAttrCase<"CTA", 2, "cta">,
      I32EnumAttrCase<"SYSTEM", 3, "sys">,
    ]> {
    let cppNamespace = "::mlir::triton::ppugpu";
}

class PPUGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<PPUGPU_Dialect, mnemonic, traits>;

def PPUGPU_PPULoadMatrixOp : PPUGPU_Op<"ppu_ldmatrix", [MemoryEffects<[MemRead]>]> {
  let arguments = (
    ins LLVM_PointerShared:$addr,
    UnitAttr:$trans,
    UnitAttr:$Opb8bLdmatrix
  );
  let results = (outs LLVM_AnyStruct:$result);
  let assemblyFormat = "$addr attr-dict `:` functional-type($addr, $result)";
}

def PPUGPU_ClusterCTAIdOp : PPUGPU_Op<"cluster_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def PPUGPU_LoadAcquireOp : PPUGPU_Op<"ld_acquire", [MemoryEffects<[MemRead]>]> {
  let arguments = (
    ins LLVM_PointerGlobal:$addr,
    Optional<I1>:$mask,
    PPUGPU_MemSemanticAttr:$sem,
    PPUGPU_MemSyncScopeAttr:$scope
  );
  let results = (outs PPUGPU_ScalarLike:$result);
  let assemblyFormat = "$sem `,` $scope `,` $addr (`,` $mask^)? attr-dict `:` functional-type($addr, $result)";
}

def PPUGPU_WarpIdOp : PPUGPU_Op<"warp_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

#endif
