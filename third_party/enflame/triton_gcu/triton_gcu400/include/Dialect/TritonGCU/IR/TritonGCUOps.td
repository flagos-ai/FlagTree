/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRITONGCU_OPS
#define TRITONGCU_OPS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "Dialect/TritonGCU/IR/TritonGCUDialect.td"
include "Dialect/TritonGCU/IR/TritonGCUAttrDefs.td"
include "Dialect/TritonGCU/IR/TritonGCUInterfaces.td"
include "Dialect/TritonGCU/IR/TritonGCUTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/IR/BuiltinTypeInterfaces.td"

//
// Interfaces
//
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

// Constraint-only reference to InputPrecision from triton core.
// Avoids including TritonAttrDefs.td which would regenerate conflicting enum definitions.
def TT_InputPrecisionAttr : Attr<
    CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
    "input precision"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::triton::InputPrecision";
  let convertFromStorage = "static_cast<::mlir::triton::InputPrecision>($_self.getValue().getZExtValue())";
  let constBuilderCall = "$_builder.getIntegerAttr($_builder.getIntegerType(32), static_cast<int32_t>($0))";
}

//
// Op Base
//
class TTGCU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGCU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

//
// Load Op
//
def TTGCU_LoadOp : TTGCU_Op<"load",
                              [MemoryEffects<[MemRead]>, AttrSizedOperandSegments]> {
  let summary = "Load";
  let description = [{
  }];
  let arguments = (ins TTGCU_PtrType:$ptr,
                       Variadic<Index>:$shape,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Optional<AnyType>:$default_value,
                       DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$order_hint);
  let results = (outs AnyStaticShapeTensor:$result);
  let assemblyFormat = [{$ptr `,` `[` $shape `]` `,`
                        `[` $strides `]` `,` `[` $offsets `]`
                        (`,` $default_value^)? `,` `[` $order_hint `]`
                        attr-dict `:` type($ptr)
                        (`,` type($default_value)^)? `->` type($result)}];
  let hasVerifier = 1;
}

//
// Store Op
//
def TTGCU_StoreOp : TTGCU_Op<"store",
                              [MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "store";
  let description = [{
  }];
  let arguments = (ins AnyStaticShapeTensor:$value,
                       TTGCU_PtrType:$ptr,
                       Variadic<Index>:$shape,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$order_hint);
  let assemblyFormat = [{$value `,` $ptr `,` `[` $shape `]` `,` `[` $strides `]`
                        `,` `[` $offsets `]` `,` `[` $order_hint `]`
                        attr-dict `:` type($value) `,` type($ptr)}];
  let hasVerifier = 1;
}

//
// Yield Op
//
def TTGCU_YieldOp : TTGCU_Op<"yield", [
     HasParent<"ElementwiseFusionRegionOp">,
     Pure,
     ReturnLike,
     Terminator]>
{
  let summary = "yield value";

  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{ $operands attr-dict `:` type($operands)}];
}

//
// PtrToInt Op
//
def GCU_PtrToIntOp : TTGCU_Op<"ptr2int", [Elementwise, Pure]>
{
  let summary = "Convert a pointer to integer";

  let description = [{
  }];
  let arguments = (ins TTGCU_PtrType:$ptr);
  let results = (outs I64:$result);
  let assemblyFormat = [{ $ptr attr-dict `:` type($ptr) }];
}

//
// IntToPtr Op
//
def TritonGCU_IntToPtrOp : TTGCU_Op<"int2ptr", [Elementwise, Pure]>
{
  let summary = "Convert an integer to pointer";

  let description = [{
  }];
  let arguments = (ins I64:$value);
  let results = (outs TTGCU_PtrType:$ptr);
  let assemblyFormat = [{ $value attr-dict `:` type($ptr) }];
}

//
// ElementwiseFusionRegionOp
//
def ElementwiseFusionRegionOp : TTGCU_Op<"elementwise_fusion_region", [
    IsolatedFromAbove,
    AllShapesMatch<["results"]>,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Elementwise fusion region";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyTypeOf<[TT_Tensor, AnyInteger, Index, AnyFloat, TT_Ptr]>>:$operands);
  let results = (outs Variadic<TT_Tensor>);
  let regions = (region SizedRegion<1>:$region);
}

//
// Assert Op
//
def TritonGCU_AssertOp : TTGCU_Op<"assert", []>
{
  let summary = "device assert";
  let description = [{
    gcu device assert.
  }];
  let arguments = (ins I1:$condition, StrAttr:$message, StrAttr:$file, StrAttr:$func, I32Attr:$line);
  let assemblyFormat = [{
    $condition attr-dict `:` type($condition)
  }];
}

//
// MemDescToPtr Op
//
def TritonGCU_MemDescToPtrOp : TTGCU_Op<"memdesc_to_ptr", [Pure]>
{
  let summary = "Convert a MemDesc to a Triton pointer";
  let description = [{
    Extract the base pointer from a shared-memory MemDesc and produce
    a `!tt.ptr<elemTy, addrSpace>` that can be consumed by standard
    Triton pointer arithmetic (tt.splat, tt.addptr, tt.load, tt.store).
    The address space is set to 3 (shared memory).
  }];
  let arguments = (ins TTG_MemDescType:$src);
  let results = (outs TT_Ptr:$result);
  let assemblyFormat = [{ $src attr-dict `:` type($src) `->` type($result) }];
}

//
// sync-load-global-to-share
//
def TritonGCU_CopyGlobalToLocalOp : TTGCU_Op<"copy_global_to_local", [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "DTE copy: global -> local (shared) memory";
  let description = [{
    Synchronously copy a contiguous tensor from global memory into local (shared) memory using DTE.
    The global source is described by a scalar base pointer plus shape / strides / offsets.

    Unlike async_load_from_global, this op blocks until the transfer completes
    and does not return a sync token.
  }];
  let arguments = (ins TTGCU_PtrType:$ptr,
                       Variadic<Index>:$shape,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       TTG_MemDescType:$dstMem,
                       Optional<AnyType>:$default_value,
                       DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$order_hint);
  let results = (outs);
  let assemblyFormat = [{$ptr `,` `shape_` `[` $shape `]` `,`
                      `stride_` `[` $strides `]` `,` `offset_` `[` $offsets `]` `,`$dstMem
                      (`,` $default_value^)? `,` `[` $order_hint `]`
                      attr-dict `:` type($ptr)`,`type($dstMem)
                      (`,` type($default_value)^)?}];
}

// gather-global-to-share: scatter-gather load from global ptr tensor into shared memory
def TritonGCU_GatherGlobalToLocalOp : TTGCU_Op<"gather_global_to_local", [
  AttrSizedOperandSegments,
  MemoryEffects<[MemRead, MemWrite]>,
  OptionalTypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)">,
  OptionalTypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)">
]> {
  let summary = "Gather: global pointer tensor -> local (shared) memory";
  let description = [{
    Synchronously gather data from global memory (via a pointer tensor) into
    local (shared) memory. Each element of the src pointer tensor addresses a
    potentially non-contiguous global location. The gathered result is written
    into dstMem. Mask and other (padding) are optional, following the same
    convention as tt.load / async_copy_global_to_local.
  }];

  let arguments = (ins
    TT_PtrTensor:$src,
    TTG_MemDescType:$dstMem,
    Optional<I1Tensor>:$mask,
    Optional<TT_Type>:$other
  );

  let results = (outs);

  let assemblyFormat = [{
    $src `,` $dstMem
    (`,` `mask` `=` $mask^)?
    (`,` `other` `=` $other^)?
    attr-dict `:` type($src) `,` qualified(type($dstMem))
    (`,` type($mask)^)?
    (`,` type($other)^)?
  }];
}


def TritonGCU_MatmulOp : TTGCU_Op<"matmul", [Pure]> {
    let summary = "matmul";

    let description = [{
        $d = matrix_multiply($a, $b)$inputPrecision describes how to exercise the TC
        when the inputs are f32. It can be one of: tf32, tf32x3, ieee.
        tf32: use TC with tf32 ops.
        tf32x3: implement the 3xTF32 trick. For more info see the pass in F32DotTC.cpp
        ieee: don't use TC, implement dot in software.
        If the GPU does not have Tensor cores or the inputs are not f32, this flag is ignored.
    }];

    let arguments = (
      ins
      TT_FpIntTensor:$a,
      TT_FpIntTensor:$b
    );

    let results = (outs TT_FpIntTensor:$d);

    // attr-dict prints enums as integers.  To get inputPrecision printed as a
    // string, we need to specify it explicitly.
    let assemblyFormat = [{
      $a`,` $b attr-dict `:`
      type($a) `*` type($b) `->` type($d)
    }];
}

def TritonGCU_MaskedLoadOp : TTGCU_Op<"maskedload", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments,
  InferTypeOpInterface,
  PredOpTrait<
    "requires the same shape and encoding for ranked tensor operands",
    CPred<[{
      [&](){
        auto resultType = cast<RankedTensorType>(getResult().getType());
        auto encoding = resultType.getEncoding();
        auto shape = resultType.getShape();
        return llvm::all_of(llvm::drop_begin(getOperandTypes(), 1), [&](auto type){
          auto operandType = dyn_cast<RankedTensorType>(type);
          return !operandType || (encoding == operandType.getEncoding() && shape == operandType.getShape());
        });
      }()
    }]>
  >,
  PredOpTrait<
    "value type matches ptr element type",
    CPred<"!getOther() || getPtr().getType().getPointeeType() == getElementTypeOrSelf(getOther())">
  >
]> {
    let summary = "Load from a tensor of pointers";

    let arguments = (ins
      TT_Ptr:$ptr,
      TT_IntTensor:$offset,
      Optional<TT_BoolTensor>:$mask,
      Optional<TT_Type>:$other
    );

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $ptr `,` $offset (`,` $mask^)? (`,` $other^)?
      attr-dict `:` type($ptr) `,` type($offset) (`,` type($mask)^)? (`,` type($other)^)? `->` type($result)
    }];

    let extraClassDeclaration = [{
      static ::llvm::LogicalResult inferReturnTypes(
          ::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location,
          ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
          mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> & inferredReturnTypes) {
        inferredReturnTypes.resize(1);
        auto elementTy = cast<::mlir::triton::PointerType>(operands[0].getType())
                            .getPointeeType();
        auto rankTensorType = cast<::mlir::RankedTensorType>(operands[1].getType());
        auto rank = rankTensorType.getShape();
        auto encoding = rankTensorType.getEncoding();
        inferredReturnTypes[0] = RankedTensorType::get(rank, elementTy, encoding);
        return ::mlir::success();
      }
    }];
}

def TritonGCU_MaskedStoreOp : TTGCU_Op<"maskedstore", [
  MemoryEffects<[MemWrite]>,
  PredOpTrait<
    "requires the same shape and encoding for ranked tensor operands",
    CPred<[{
      [&](){
        auto rankTensorType = cast<RankedTensorType>(getOffset().getType());
        auto encoding = rankTensorType.getEncoding();
        auto shape = rankTensorType.getShape();
        return llvm::all_of(llvm::drop_begin(getOperandTypes(), 2), [&](auto type){
          auto operandType = cast<RankedTensorType>(type);
          return encoding == operandType.getEncoding() && shape == operandType.getShape();
        });
      }()
    }]>
  >,
  PredOpTrait<
    "value type matches ptr type",
    CPred<"getPtr().getType().getPointeeType() == getElementTypeOrSelf(getValue())">
  >,
  OptionalTypesMatchWith<"mask type matches offset type", "offset", "mask",
                 "getI1SameShape($_self)">
]> {
    let summary = "Store by a tensor of pointers";

    let arguments = (ins
      TT_Ptr:$ptr,
      TT_IntTensor:$offset,
      TT_Tensor:$value,
      Optional<TT_BoolTensor>:$mask
    );

    let assemblyFormat = [{
      $ptr `,` $offset `,` $value (`,` $mask^)?
      attr-dict `:` type($ptr) `,` type($offset) `,` type($value) (`,` type($mask)^)?
    }];
}

//
// WarpGroupDot Op
//
def TTGCU_WarpGroupDotOp : TTGCU_Op<"warp_group_dot", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DotOpInterface>,
  TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">
]> {
  let summary = "warp group dot";

  let description = [{
    $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
  }];

  let arguments = (ins
    TTG_TensorOrMemDesc:$a,
    TTG_TensorOrMemDesc:$b,
    TT_FpIntTensor:$c,
    Optional<I1>:$useC,
    DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
    DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
    DefaultValuedAttr<BoolAttr, "false">:$isAsync
  );

  let results = (outs TT_FpIntTensor:$d);

  let assemblyFormat = [{
    $a`,` $b`,` $c (`,` $useC^)? attr-dict
    `:` type($a) `*` qualified(type($b)) `->` type($d)
  }];

  let extraClassDeclaration = [{
    bool needsPartialAccumulator();
  }];

  let hasVerifier = 1;
}

//
// Barrier Op
//
def TTGCU_InitBarrierOp : TTGCU_Op<"init_barrier"> {
  let summary = "Initialize a barrier.";
  let description = [{
      Initializes a barrier.
      `count` is the number of arrives expected by the barrier.
  }];
  let arguments = (ins I32Attr:$count);
  let results = (outs TTGCU_BarrierType:$barrier);
  let assemblyFormat = "`(` `)` attr-dict `:` type($barrier)";
  let hasVerifier = 1;
}

def TTGCU_WaitBarrierOp : TTGCU_Op<"wait_barrier"> {
  let summary = "Wait a barrier";
  let description = [{
    Blocks the program progress until the barrier completes its current phase.
  }];
  let arguments = (ins TTGCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

def TTGCU_ArriveBarrierOp : TTGCU_Op<"arrive_barrier"> {
  let summary = "Arrive a barrier";
  let description = [{
    Perform the "arrive" operation on a barrier.
  }];
  let arguments = (ins TTGCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

#endif // TRITONGCU_OPS
