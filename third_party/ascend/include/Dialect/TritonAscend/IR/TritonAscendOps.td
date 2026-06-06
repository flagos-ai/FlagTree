//===-- TritonAscendOps.td - TritonAscend op definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the TritonAscend IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_OPS
#define TRITON_ASCEND_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.td"
include "ascend/include/Dialect/TritonAscend/IR/TritonAscendAttrDefs.td"

//===----------------------------------------------------------------------===//
// TritonAscend op definitions
//===----------------------------------------------------------------------===//

class TT_Ascend_Op<string mnemonic, list<Trait> traits = []> :
  Op<TritonAscend_Dialect, mnemonic, traits>;


//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;


//
// Annotation Op
//
def AnnotationOp : TT_Ascend_Op<"annotation", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Annotate a tensor with key-value attribute pairs";
  let description = [{
    `ascend.annotation` operation can be used to annotate a tensor with
    key-value attribute pairs.

    Example:
    ```mlir
    ascend.annotation %target {key : val}
    ```
  }];
  let arguments = (ins TT_Tensor:$src);
  let assemblyFormat = [{
    $src attr-dict `:` type($src)
  }];
}


//
// Mod Op
//
def ModOp : TT_Ascend_Op<"mod", [Pure]> {
  let summary = "Mod operation (%) of input tensors.";
  let description = [{
    Performs element-wise division with remainder of input tensors.
  }];

  let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) type($rhs) `->` type($result)";
}


//
// IndexPut Op
//
def IndexPutOp : TT_Ascend_Op<"index_put", [
  MemoryEffects<[MemWrite<GlobalMemory>]>,
  SameVariadicOperandSize,
]> {
    let summary = "Scatter store to a tensor pointer with embedding semantics";

    let description = [{
        Index put values from a tensor into a destination tensor.

        The operation takes:
        - ptr: pointer type, the destination tensor pointer (in GM)
        - index: tensor, a index to scatter (in UB)
        - value: tensor, a value to store (in UB)
        - dim: int32, the dimension to scatter along
        - index_boundary: int64, the upper boundary for index values
        - end_offset: tuple of int, the offsets of each dimension for the end of the scatter region
        - start_offset: tuple of int, the offsets of each dimension for the start of the scatter region
        - dst_stride: tuple of int, the stride of each dimension of destination tensor


        Constraints:
        - `ptr` and `value` must have the same rank.
        - `ptr.dtype` only supports `float16`, `bfloat16`, `float32` currently.
        - `index` must be an integer tensor. If `index.rank` != 1, it will be reshaped to 1D.
        - `index.numel` must equal `value.shape[dim]`.
        - `value` support 2~5D tensors.
        - `dim` must be valid (0 <= dim < rank(value) - 1).
    }];

    let arguments = (
      ins TT_Ptr:$ptr,
          TT_Tensor:$index,
          TT_Tensor:$value,
          TT_Int:$dim,
          TT_Int:$indexBoundary,
          Variadic<AnyTypeOf<[I32, I64]>>:$endOffset,
          Variadic<AnyTypeOf<[I32, I64]>>:$startOffset,
          Variadic<AnyTypeOf<[I32, I64]>>:$dstStride
    );

    let assemblyFormat = [{
      $ptr `:` type($ptr) `,` $index `:` type($index) `,`
      $value `:` type($value) `,` $dim `:` type($dim) `,` $indexBoundary `:` type($indexBoundary) `,`
     `[` $endOffset `:` type($endOffset) `]` `,` `[` $startOffset `:` type($startOffset) `]` `,`
     `[` $dstStride `:` type($dstStride) `]`
      attr-dict
    }];
}

//
// GatherOutToUb Op
//
def GatherOutToUbOp : TT_Ascend_Op<"gather_out_to_ub", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  AttrSizedOperandSegments,
]> {
    let summary = "Gather load from a tensor pointer with the embedding semantics";

    let description = [{
        Gather from a source tensor in Global Memory (GM) to Unified Buffer (UB)
        along a specified dimension with out-of-bound handling.

        The operation takes:
        - src: pointer type, the source tensor pointer (in GM)
        - index: tensor, a tensor to gather (in UB)
        - index_boundary: int64, the upper boundary for index values
        - dim: int32, the dimension to gather along
        - src_stride: tuple of int64, the stride of each dimension of src tensor
        - end_offset: tuple of int32, the end offsets of each dimension for index tensor
        - start_offset: tuple of int32, the start offsets of each dimension for index tensor
        - other(Optional): scalar value, the default value when index is out of boundary (in UB)

        Returns:
          a tensor, with the same shape as `index.shape` (in UB)

        Constraints:
        - `src` and `index` must have the same rank.
        - `src.dtype` only supports `float16`, `bfloat16`, `float32` currently.
        - `index` must be an integer tensor, with rank between 1 and 5.
        - `dim` must be valid (0 <= dim < rank(index)).
        - `other` must be a scalar value.
        - For every dimension `i` not equal to `dim`, `index.size[i]` <= `src.size[i]`.
        - The output shape is the same as `index.shape`. If `index` is None, \
            the output tensor will be an empty tensor with the same shape as `index`.
    }];

    let arguments = (
      ins TT_Ptr:$src,
          TT_Tensor:$index,
          TT_Int:$indexBoundary,
          TT_Int:$dim,
          Variadic<AnyTypeOf<[I32, I64]>>:$srcStride,
          Variadic<AnyTypeOf<[I32, I64]>>:$endOffset,
          Variadic<AnyTypeOf<[I32, I64]>>:$startOffset,
          Optional<TT_Type>:$other
    );

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $src `:` type($src) `,` $index `:` type($index) `,`
      $indexBoundary `:` type($indexBoundary) `,` $dim `:` type($dim) `,`
     `[` $srcStride `:` type($srcStride) `]` `,` `[` $endOffset `:` type($endOffset) `]` `,`
     `[` $startOffset `:` type($startOffset) `]` (`,` $other^ `:` type($other))?
      attr-dict `->` type($result)
    }];
}

//
// ScatterUbToOut Op
//
def ScatterUbToOutOp : TT_Ascend_Op<"scatter_ub_to_out", [
  MemoryEffects<[MemWrite<GlobalMemory>]>,
  SameVariadicOperandSize,
]> {
    let summary = "scatter store from a tensor pointer with the embedding semantics";

    let description = [{
        Scatter a tile from Unified Buffer (UB) into a destination tensor in Global Memory (GM)
        along a specified dimension, with index-boundary checking.

        The operation takes:
        - ptr: pointer type, the destination tensor pointer (in GM)
        - value: tensor, a tile value to store (in UB)
        - index: tensor, a tile index to scatter (in UB)
        - index_boundary: int, the upper boundary for index values
        - dim: int, the dimension to scatter along
        - dst_stride: tuple of int, the stride of each dimension of destination tensor
        - end_offset: tuple of int32, the end offsets of each dimension for index tensor
        - start_offset: tuple of int32, the start offsets of each dimension for index tensor

        Constraints:
        - `ptr` and `index` must have the same rank.
        - `ptr.dtype` only supports `float16`, `bfloat16`, `float32` currently.
        - `index` must be an integer tensor, with rank between 1 and 5.
        - `dim` must be valid (0 <= dim < rank(index)).
        - For every dimension `i` not equal to `dim`, `index.size[i]` <= `ptr.size[i]`.
        - The output shape is the same as `index.shape`. If `index` is None, \
            the output tensor will be an empty tensor with the same shape as `index`.
    }];

    let arguments = (
      ins TT_Ptr:$ptr,
          TT_Tensor:$value,
          TT_Tensor:$index,
          TT_Int:$indexBoundary,
          TT_Int:$dim,
          Variadic<AnyTypeOf<[I32, I64]>>:$dstStride,
          Variadic<AnyTypeOf<[I32, I64]>>:$endOffset,
          Variadic<AnyTypeOf<[I32, I64]>>:$startOffset
    );

    let assemblyFormat = [{
      $ptr `:` type($ptr) `,` $value `:` type($value) `,` `,` $index `:` type($index) `,`
      $indexBoundary `:` type($indexBoundary) `,` $dim `:` type($dim) `,`
     `[` $dstStride `:` type($dstStride) `]` `,` `[` $endOffset `:` type($endOffset) `]` `,`
     `[` $startOffset `:` type($startOffset) `]`
      attr-dict
    }];
}


//
// IndexSelectSimd Op
//
def IndexSelectSimdOp : TT_Ascend_Op<"index_select_simd", [
  MemoryEffects<[MemRead<GlobalMemory>]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  AttrSizedOperandSegments
]> {
    let summary = "Index select SIMD operation from global memory";

    let description = [{
        Index select operation (SIMD version) that loads data from multiple indices along a
        specified dimension. The operation selects data from GM and loads them
        as tiles directly to UB with zero-copy semantics.

        The operation takes:
        - src: Source pointer (in GM)
        - index: 1D tensor of indices to select (already in UB)
        - dim: The dimension along which to select
        - src_shape: Complete shape of the source tensor
        - src_offset: Starting offset for reading
        - read_shape: Size to read (tile shape)

        Constraints:
        - read_shape[dim] must be -1
        - src_offset[dim] can be -1 (will be ignored)
    }];

    let arguments = (
      ins
      TT_PtrLike:$src,
      TT_IntTensor:$index,
      I32Attr:$dim,
      Variadic<Index>:$src_shape,
      Variadic<Index>:$src_offset,
      DenseI32ArrayAttr:$read_shape
    );

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $src `,` $index `,` $dim `,`
      `[` $src_shape `]` `,`
      `[` $src_offset `]` `,`
      $read_shape
      attr-dict `:` type($src) `,` type($index) `->` type($result)
    }];
}


//
// Built-in: IndirectLoad Op
//
def IndirectLoadOp : TT_Ascend_Op<"indirect_load", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  AttrSizedOperandSegments
]> {
  let summary = "Built-in: indirect load from global memory using per-element offsets with optional mask/other";

  let description = [{
    Built-in operation emitted by the compiler for unstructured (discrete) memory
    accesses.These are not written directly in the user IR.

    Load values from global memory based on per-element offsets. If `mask`
    is provided, false lanes return `other`.

    The operation takes:
    - src: Source pointer
    - offsets: Tensor of per-element offsets (relative to `src`) for accessing source memory
    - mask (optional):  if mask[idx] is false, do not load the data at address pointer[idx]
    - other (optional): if mask[idx] is false, return other[idx]
  }];

  let arguments = (
    ins TT_Ptr:$src,
        TT_IntTensor:$offsets,
        Optional<TT_BoolLike>:$mask,
        Optional<TT_Type>:$other
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `:` type($src) `,`
    $offsets `:` type($offsets)
    (`,` $mask^ `:` type($mask))?
    (`,` $other^ `:` type($other))?
    attr-dict `->` type($result)
  }];

}


//
// Built-in: IndirectStore Op
//
def IndirectStoreOp : TT_Ascend_Op<"indirect_store", [
  MemoryEffects<[MemWrite<GlobalMemory>]>
]> {
  let summary = "Built-in: indirect store from UB using per-element offsets with optional mask/other";

  let description = [{
    Built-in operation emitted by the compiler for unstructured (discrete) memory
    accesses.These are not written directly in the user IR.

    Store values from UB based to GM on per-element offsets.

    The operation takes:
    - src: Source pointer
    - offsets: Tensor of per-element offsets (relative to `src`) for accessing source memory
    - value: The tensor of elements to be stored
    - mask (optional): If mask[idx] is false, do not store value[idx] at pointer[idx]
  }];

  let arguments = (
    ins TT_Ptr:$src,
        TT_IntTensor:$offsets,
        TT_Type:$value,
        Optional<TT_BoolLike>:$mask
  );

  let assemblyFormat = [{
    $src `:` type($src) `,`
    $offsets `:` type($offsets) `,`
    $value `:` type($value)
    (`,` $mask^ `:` type($mask))?
    attr-dict
  }];

}

//
// Custom Op
//
def CustomOp : TT_Ascend_Op<"custom",  [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "self-defined custom operation";
  let description = [{
    `ascend.custom` triton custom op is designed to pass self-defined custom operation.

    Example:
    ```ascend.custom {str_args = ["sync_block_wait", "cube"]}
    ```
  }];
  let arguments = (ins StrAttr:$op_name, ArrayAttr:$str_args, Variadic<AnyType>:$args);

  let assemblyFormat = "$op_name attr-dict ($args^ `:` type($args))?";
}

def FlipOp : TT_Ascend_Op<"flip", [
  NoMemoryEffect,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Reverse a tensor along a given dimension";
  let description = [{
    Reverses the elements of the input tensor along the specified dimension.
    The output tensor has the same shape and element type as the input.
  }];

  let arguments = (ins
    TT_Tensor:$src,   // Input tensor
    I64Attr:$dim      // Dimension to flip along
  );

  let results = (outs
    TT_Tensor:$flipped  // Flipped values
  );

  let assemblyFormat =
    "$src `,` $dim attr-dict `:` type($src) `->` type($flipped)";
}

def SortOp : TT_Ascend_Op<"sort", [
  NoMemoryEffect,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Sorts a tensor along a given dimension and returns sorted values.";
  let description = [{
    Sorts the elements of the input tensor along the specified dimension.
    Returns one tensor:
    The sorted tensor (same shape and element type as input).
  }];

  let arguments = (ins
    TT_Tensor:$src,       // Input tensor
    I64Attr:$dim,         // Dimension to sort along
    BoolAttr:$descending  // Sort order
  );

  let results = (outs
    TT_Tensor:$sorted    // Sorted values
  );

  let assemblyFormat = "$src `,` $dim `,` $descending attr-dict `:` type($src) `->` type($sorted)";
}

//
// Conv Op
//
def Conv1dOp : TT_Ascend_Op<"conv1d", [
    Pure, 
    DeclareOpInterfaceMethods<InferTypeOpInterface> 
]> {
    let summary = "1D convolution operation";
    let description = [{
        Performs a 1D convolution operation on the input tensor.
    }];

    let arguments = (ins
        TT_Tensor:$input,
        TT_Tensor:$weight,
        Optional<TT_Tensor>:$bias,
        I64Attr:$stride,
        I64Attr:$padding_size,
        I64Attr:$dilation,
        I64Attr:$groups
    );

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
        `(` operands `)` 
        attr-dict `:` functional-type(operands, $result)
    }];

    let hasVerifier = 1;
}

#endif // TRITON_ASCEND_OPS
