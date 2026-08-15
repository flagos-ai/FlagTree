#ifndef TRITON_STRUCTURED_DIALECT
#define TRITON_STRUCTURED_DIALECT

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"

def Triton_Structured_Dialect : Dialect {
  let name = "tts";

  let cppNamespace = "::mlir::tts";

  let summary = "Structured Triton operations";

  let description = [{
    Triton Structured Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect"
  ];

}

//
// Op Base
//
class TTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Structured_Dialect, mnemonic, traits> {
}

def TTS_MakeTensorPtrOp
  : TTS_Op<"make_tptr", [AttrSizedOperandSegments, Pure]> {
  let summary = "create a pointer that points to a tensor in memory";

  // base:    Base pointer used to contruct the tensor of pointers or pointer to tensor.
  // sizes:   Size of the data being loaded or stored.
  // strides: The strides of the parent tensor, which means how much to increase the pointer
  //          by when moving by 1 element in a specific axis.
  // order:   The order of the block, which means how the block is laid out in memory.
  //          It contains the same info as order in tt.make_tensor_ptr.
  // shape:   If order is present, this field signifies the shape of the parent tensor in
  //          memory; if order is not present, it signifies the boundary by which addresses
  //          wraps around (constant zero indicates no wrap-around in the corresponding dimension).
  // offsets: Offset of the block along each dimension from base.
  // originalOffsets: Original offsets before any adjustment (e.g. for padding)
  // result:  If order is present, this op produces a pointer to a tensor; otherwise,
  //          it produces a tensor of pointers.

  let arguments = (ins TT_Ptr:$base,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$originalOffsets,
                       Variadic<Index>:$shape,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_originalOffsets,
                       DenseI64ArrayAttr:$static_shape,
                       DenseI32ArrayAttr:$order);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `originalOffsets` `` `:`
    custom<DynamicIndexList>($originalOffsets, $static_originalOffsets)
    `` `,` `shape` `` `:`
    custom<DynamicIndexList>($shape, $static_shape)
    `` `,` `order` `` `:` $order
    attr-dict `:` type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$originalOffsets,
      "ArrayRef<OpFoldResult>":$shape,
      "ArrayRef<int32_t>":$order)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes; // sizes are always static
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedOriginalOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOriginalOffsets(), getOriginalOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedShape() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticShape(), getShape(), b);
    }
    bool isBlockPtr() {
      return !getOrder().empty();
    }
    bool isStructuredPtr() {
      return !isBlockPtr() &&
             llvm::all_of(getStaticShape(), [](auto shape) { return shape == 0; });
    }
    bool isSplitPtr() {
      return !isBlockPtr() &&
             !isStructuredPtr();
    }
  }];

  // TODO
  //let hasVerifier = 1;
  //let hasCanonicalizer = 1;
}

def TT_IndexTensorLike : AnyTypeOf<[I16Tensor, I32Tensor, I64Tensor]>;

def TTS_MakeGatherScatterTensorPtrOp
  : TTS_Op<"make_gather_scatter_tptr", [AttrSizedOperandSegments, Pure]> {
  // NOTE: Only support cases where the offset for each dimension is defined in a different operation.
  //       Not support case where the offset is a tensor load from other ptr which for multiple dimension.
  //
  //       offset_m = tl.arange(0, M)
  //       offset_n = tl.arange(0, N)
  //       offset_k = tl.arange(0, K)
  //       ld_offsets = tl.load(a_ptr + offset_m[:,None]+offsets_n[None,:])
  //       not_support = tl.load(b_ptr + ld_offsets)
  //       not_support2 = tl.load(b_ptr + ld_offsets * (offset_m[:,None]+offsets_n[None,:]))
  //       not_support3 = tl.load(b_ptr + (ld_offsets * (offset_m[:,None]+offsets_n[None,:]))[:, :, None] + offset_k[None,None,:])
  //
  //       # Support cases where one dimension is structured while the other is not.
  //       # For example, `offset_m[:, None] // K` is not structured, whereas `offset_n[None, :]` is structured in next line.
  //       supported = tl.load(b_ptr + offset_m[:, None] // K + offset_n[None, :])
  //       # `ld_offsets_1d[:, None]` is not structured, whereas `offset_n[None, :]`.
  //       ld_offsets_1d = tl.load(a_ptr + offsets_m)
  //       supported2 = tl.load(b_ptr + ld_offsets_1d[:, None] + offsets_n[None, :])

  let summary = "create an pointer that points to a tensor in memory for gather/scatter";
  let description = [{
    The `tts.make_gather_scatter_tptr` operation is similar to `tts.make_tptr`.
    The key difference is that `make_gather_scatter_tptr` accesses the tensor non-continuously.
    Currently, only one dimension is allowed to be non-continuous.
    This dimension is saved in `gather_scatter_dim`, and the offset for that dimension is saved in `gather_scatter_offset`.
    Each contiguous load will load from this offset.
    Cases with more than one non-continuous dimension are not supported.
  }];

  // base:    Base pointer used to contruct the tensor of pointers or pointer to tensor.
  // gather_scatter_offset: The offset for gather/scatter.
  // gather_scatter_dim: The dimension for gather_scatter_offset.
  // sizes:   Size of the data being loaded or stored.
  // strides: The strides of the parent tensor, which means how much to increase the pointer
  //          by when moving by 1 element in a specific axis.
  // offsets: Offset of the block along each dimension from base.
  // result:  A tensor of pointers.

  let arguments = (ins TT_Ptr:$base,
                       TT_IndexTensorLike:$gather_scatter_offset,
                       I32Attr:$gather_scatter_dim,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `gather_scatter_dim` `` `:` $gather_scatter_dim
    `gather_scatter_offset` `` `:` $gather_scatter_offset
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    attr-dict `:`  type($gather_scatter_offset) type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "Value":$gather_scatter_offset,
      "int":$gather_scatter_dim,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes; // sizes are always static
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
  }];

  // TODO
  //let hasVerifier = 1;
  //let hasCanonicalizer = 1;
}

def TTS_GetStructuredStateOp : TTS_Op<"get_structured_state", [AttrSizedResultSegments, Pure]> {
  let summary = "Placeholder for the structured pointer states computed during PtrAnalysis.";
  let description = "Used to pass the offsets, origiOffsets, and strides to scf.for op to simplify IR rewrites.";

  let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_IndexTensorLike, I1Tensor]>:$input);
  let results = (outs AnyTypeOf<[TT_PtrLike, TT_IndexTensorLike, I1Tensor]>:$structured, Variadic<Index>:$offsets, Variadic<Index>:$origiOffsets, Variadic<Index>:$strides);

  let builders = [
    OpBuilder<(ins "Value":$input)>,
  ];

  let extraClassDeclaration = [{
    static std::optional<std::tuple<SmallVector<Type>, SmallVector<Type>, SmallVector<Type>>>
      getOffsetAndStrideTypes(MLIRContext *context, Type ptrLikeType);

    static std::optional<std::tuple<int32_t, int32_t, int32_t>>
      getOffsetAndStrideSegmentSizes(Type ptrLikeType);
  }];

  let hasFolder = 0;
  let hasVerifier = 1;
}

def TTS_GatherOp : TTS_Op<"gather", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments,
  OptionalTypesMatchWith<"mask type matches ptr type", "offset", "mask", "triton::getI1SameShape($_self)">,
  OptionalTypesMatchWith<"other matches ptr type", "ptr", "other", "triton::getPointeeType($_self)">
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (
    ins
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other
  );

  let results = (outs TT_Type:$result);

  let assemblyFormat = [{
    $ptr `[` $offset `]` (`mask` `=` $mask^)? (`default` `=` $other^)?
    attr-dict `:` `(` type($ptr) `,` type($offset) `)` `->` type($result)
  }];
}

def TTS_ScatterOp : TTS_Op<"scatter", [
  MemoryEffects<[MemWrite]>,
  OptionalTypesMatchWith<"mask type matches offset type", "offset", "mask",
                 "triton::getI1SameShape($_self)">
]> {
  let summary = "optionally store data from in memory to fill a portion of the tensor";

  let arguments = (
    ins
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    TT_Type:$value,
    Optional<TT_BoolLike>:$mask
  );

  let assemblyFormat = [{
    $value `into` $ptr `[` $offset `]` (`mask` `=` $mask^)?
    attr-dict `:` type($value) `into` ` ` `(` type($ptr) `,` type($offset) `)`
  }];
}

def TTS_LoadOp : TTS_Op<"load", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       Variadic<Index>:$mask_dims,
                       DefaultValuedAttr<DenseI64ArrayAttr, "::llvm::ArrayRef<int64_t>{}">:$static_mask_dims,
                       Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other,
                       DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
                       OptionalAttr<TT_PaddingOptionAttr>:$padding
  );


  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "ArrayRef<OpFoldResult>":$mask_dims, "Value":$other, "ArrayRef<int32_t>":$boundaryCheck, "std::optional<triton::PaddingOption>":$padding)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }
  }];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

def TTS_StoreOp : TTS_Op<"store", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "optionally store data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       TT_Tensor:$value,
                       Variadic<Index>:$mask_dims,
                       DefaultValuedAttr<DenseI64ArrayAttr, "::llvm::ArrayRef<int64_t>{}">:$static_mask_dims,
                       DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck
                       );

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<OpFoldResult>":$dims, "ArrayRef<int32_t>":$boundaryCheck)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }
  }];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

#endif // TRITON_STRUCTURED_DIALECT
