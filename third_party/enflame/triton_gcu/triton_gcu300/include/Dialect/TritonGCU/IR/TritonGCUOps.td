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
def GCU_PtrToIntOp : TTGCU_Op<"ptr2int", [Pure]>
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
def TritonGCU_IntToPtrOp : TTGCU_Op<"int2ptr", [Pure]>
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
    PredOpTrait<
      "requires the same shape for ranked tensor operands and results",
      CPred<[{
        [&]() {
          ArrayRef<int64_t> shape;
          return llvm::all_of(getOperandTypes(),
                              [&](auto type) {
                                auto operandType = dyn_cast<RankedTensorType>(type);
                                if (operandType) {
                                  if (shape.empty()) {
                                    shape = operandType.getShape();
                                  } else {
                                    return shape == operandType.getShape();
                                  }
                                }
                                return true;
                              }) &&
                ((!shape.empty() && getNumResults() > 0)
                      ? cast<RankedTensorType>(getResultTypes()[0]).getShape() == shape
                      : true);
        }()
      }]>
    >,
    AllShapesMatch<["results"]>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Elementwise fusion region";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyTypeOf<[TT_Tensor, AnyInteger, Index, AnyFloat, TT_Ptr]>>:$operands);
  let results = (outs Variadic<TT_Tensor>);
  let regions = (region AnyRegion:$region);
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
// Allocate shared memory
//
def TritonGCU_ShareAllocOp : TTGCU_Op<"share_alloc", []> {
  let summary = "allocate tensor";
  let description = [{
    This operation allocates buffer in shared memory and return a descriptor
    containing the address and a view of the buffer.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];
  let arguments = (ins Optional<TT_Tensor>:$src);

  let assemblyFormat = [{$src attr-dict `:` functional-type(operands, results)}];

  let results = (outs TTG_MemDescType:$result);
}

//
// Deallocate shared memory
//
def TritonGCU_ShareDeallocOp : TTGCU_Op<"share_dealloc", []> {
  let summary = "dealloc buffer";

  let description = [{
    This operation deallocates a buffer explicitly. Using the buffer after this
    operation is undefined.

    This operation is optional.  If you don't explicitly dealloc a buffer, the
    compiler assumes it's deallocated at the first point that post-dominates all
    uses of the alloc.

    Because we assume a memdesc is dead at the first point that post-dominates
    its uses, ops that wait for an async operation on a memdesc to complete
    (such as triton_nvidia_gpu.dot_wait) should also take the memdesc as an
    operand.
  }];

  let arguments = (ins TTG_MemDescType:$src);

  // Use qualified() otherwise "!tt.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src attr-dict `:` qualified(type($src))}];
}


//
// async-load-global-to-share
//
def TritonGCU_AsyncLoadGlobalToShareOp : TTGCU_Op<"async_load_global_to_share", [MemoryEffects<[MemRead]>, AttrSizedOperandSegments]> {
  let summary = "AsyncLoadGlobalToShare";
  let description = [{
  }];
  let arguments = (ins TTGCU_PtrType:$ptr,
                       Variadic<Index>:$shape,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       TTG_MemDescType:$dstMem,
                       Optional<AnyType>:$default_value,
                       DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$order_hint);
  let results = (outs TTG_AsyncToken:$result);
  let assemblyFormat = [{$ptr `,` `shape_` `[` $shape `]` `,`
                      `stride_` `[` $strides `]` `,` `offset_` `[` $offsets `]` `,`$dstMem
                      (`,` $default_value^)? `,` `[` $order_hint `]`
                      attr-dict `:` type($ptr)`,`type($dstMem)
                      (`,` type($default_value)^)? `->` type($result)}];
}

def TritonGCU_AsyncWaitOp :TTGCU_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins Variadic<TTG_AsyncToken>:$asyncToken);
  let results = (outs TTG_AsyncToken:$retToken);
  let assemblyFormat = "$asyncToken attr-dict";
}

def TritonGCU_LocalLoadOp : TTGCU_Op<"local_load", []> {
  let summary = "Load a buffer from local memory into a distributed tensor";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins TTG_MemDescType:$src, Optional<TTG_AsyncToken> :$token);

  let builders = [
      OpBuilder<(ins "Type":$retType, "Value":$src),
      [{
      build($_builder, $_state, retType, src, /*token=*/static_cast<mlir::Value>(nullptr));
      }]>];

  // Use qualified() otherwise "!tt.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src (`token` $token^)? attr-dict `:` qualified(type($src)) `->` type($result)}];

  let results = (outs TT_Tensor:$result);
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
#endif // TRITONGCU_OPS
