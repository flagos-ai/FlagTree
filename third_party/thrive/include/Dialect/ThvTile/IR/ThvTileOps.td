#ifndef THVTILE_OPS
#define THVTILE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ThvTileDialect.td"
include "ThvTileTypes.td"
include "ThvTileAttrDefs.td"

class ThvTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<ThvTile_Dialect, mnemonic, traits>;

def ThvTile_MakeTensorStructureOp : ThvTile_Op<"make_tensor_structure", [Pure, AttrSizedOperandSegments]> {
  let summary = "Construct tensor structure descriptor for structured memory access";
  let description = [{
    Creates a tensor structure object that encapsulates the memory layout of a
    tile region. This structure is consumed by `thvtile.load`
    or `thvtile.store` to perform structured memory transfers.

    Arguments:
    - `base`: scalar pointer to the tensor's base address in memory
    - `shape`: the shape of the tile to be transferred
    - `source_shape`: optional parent/source tensor shape. This is empty when
      the original source extent is unknown, and populated for block pointers
      or other accesses that carry parent shape explicitly.
    - `offsets`: starting offsets from base pointer for this tile (in elements)
    - `strides`: element strides for each dimension (number of elements between
      adjacent indices along that dimension)
    - `valid_seg_sizes`: optional effective segment sizes derived from mask
      analysis. For `boundary`, it is the dense valid region size per rank;
      for `strided`, it is the contiguous size of each valid segment.
    - `valid_num_segs`: optional number of valid segments per rank. Used by
      `strided` masks; empty for `none` and must be empty for `boundary`.
    - `valid_seg_strides`: optional start-to-start interval between adjacent
      valid segments, in logical element units. The interval includes the
      previous valid segment/block; `[T, F, T, F]` has segment size 1 and
      segment stride 2.
    - `order`: dimension ordering (e.g., [1,0] for row-major 2D)
    - `mask_kind`: structured mask pattern classification.

    Note: `shape` represents the tile size (transfer block dimensions), NOT the
    parent tensor shape.

    Example:
      // (1) 2D row-major contiguous tile: 32x64 tile at offset [0, 64],
      //     inner stride == 1 (contiguous)
      %s0 = thvtile.make_tensor_structure %base, [%c32, %c64], [%src_m, %src_n],
              [%o0, %o1], [%c128, %c1], [], [], [] {order = array<i32: 1, 0>}
              : !thvtile.ptr<f16> -> !thvtile.tensor_structure<tensor<32x64xf16>>

      // (2) 2D sampled/strided tile: inner stride == 2 (every 2nd column)
      %s1 = thvtile.make_tensor_structure %base, [%c32, %c32], [], [%o0, %o1],
              [%c128, %c2], [], [], [] {order = array<i32: 1, 0>}
              : !thvtile.ptr<f16> -> !thvtile.tensor_structure<tensor<32x32xf16>>

      // (3) 1D contiguous tile
      %s2 = thvtile.make_tensor_structure %base, [%c256], [], [%o0], [%c1], [], [], []
              {order = array<i32: 0>}
              : !thvtile.ptr<f32> -> !thvtile.tensor_structure<tensor<256xf32>>

      // (4) 1D boundary mask: logical tile has 256 elements, only first
      //     %valid_n elements are valid; masked tail is cropped by metadata.
      %s3 = thvtile.make_tensor_structure %base, [%c256], [], [%o0], [%c1], [%valid_n], [], []
              {order = array<i32: 0>, mask_kind = #thvtile.mask_kind<boundary>}
              : !thvtile.ptr<f32> -> !thvtile.tensor_structure<tensor<256xf32>>
  }];

  let arguments = (ins
    AnyType:$base,
    Variadic<I64>:$shape,
    Variadic<I64>:$source_shape,
    Variadic<I64>:$offsets,
    Variadic<I64>:$strides,
    Variadic<I64>:$valid_seg_sizes,
    Variadic<I64>:$valid_num_segs,
    Variadic<I64>:$valid_seg_strides,
    DenseI32ArrayAttr:$order,
    DefaultValuedAttr<ThvTile_MaskKindAttr,
      "::mlir::thvtile::MaskKind::NONE">:$mask_kind
  );

  let results = (outs ThvTile_TensorStructure:$result);

  let assemblyFormat = [{
    $base `,` `[` $shape `]` `,` `[` $source_shape `]` `,`
    `[` $offsets `]` `,` `[` $strides `]` `,`
    `[` $valid_seg_sizes `]` `,` `[` $valid_num_segs `]` `,`
    `[` $valid_seg_strides `]` attr-dict `:` type($base) `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_LoadOp : ThvTile_Op<"load", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Load a structured tensor tile from memory";
  let description = [{
    Loads a tensor tile using structured access information provided by
    `thvtile.make_tensor_structure`.

    For scalar, per-element pointer tensor, or arbitrary-mask access patterns
    that cannot be represented by tensor_structure metadata, use
    `thvtile.iterative_load` instead.

    Boundary/strided mask information is carried by the tensor structure itself.
    If an `other` value is provided, lowering must initialize invalid positions
    with `other` before copying valid regions.

    Example:
      // (1) plain 2D load, no mask metadata
      %t0 = thvtile.load %s0
              : !thvtile.tensor_structure<tensor<64x64xf16>> -> tensor<64x64xf16>

      // (2) boundary/strided masked load; mask metadata lives in %s_boundary,
      //     masked-off positions are filled with %other
      %t1 = thvtile.load %s_boundary, %other
              : !thvtile.tensor_structure<tensor<64x64xf16>>, f16
              -> tensor<64x64xf16>
  }];

  let arguments = (ins
    ThvTile_TensorStructure:$structure,
    Optional<ThvTile_Scalar>:$other
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $structure (`,` $other^)? attr-dict `:`
    type($structure) (`,` type($other)^)? `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_StoreOp : ThvTile_Op<"store", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Store a structured tensor tile to memory";
  let description = [{
    Stores a tensor tile to memory using structured access information
    provided by `thvtile.make_tensor_structure`.

    For scalar, per-element pointer tensor, or arbitrary-mask access patterns
    that cannot be represented by tensor_structure metadata, use
    `thvtile.iterative_store` instead.

    Example:
      // (1) plain 2D store, no mask metadata
      thvtile.store %tile, %s0 : tensor<64x64xf16>,
          !thvtile.tensor_structure<tensor<64x64xf16>>

      // (2) boundary/strided store; mask metadata lives in %s_boundary
      thvtile.store %tile, %s_boundary : tensor<64x64xf16>,
          !thvtile.tensor_structure<tensor<64x64xf16>>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$value,
    ThvTile_TensorStructure:$structure
  );

  let assemblyFormat = [{
    $value `,` $structure attr-dict `:` type($value) `,` type($structure)
  }];

  let hasVerifier = 1;
}

def ThvTile_IterativeLoadOp : ThvTile_Op<"iterative_load", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Iterative load from memory (scalar pointer or per-element pointer tensor)";
  let description = [{
    Fallback load op for non-structured memory access patterns.
    When pointer analysis cannot extract affine offset/stride information,
    this op performs iterative (per-element) load from DRAM to SRAM.

    The `ptrs` operand can be either:
    - a single `!thvtile.ptr<T>` scalar pointer → loads a single scalar element, or
    - a tensor of pointers `tensor<...x!thvtile.ptr<T>>` → per-element load,
      where each element specifies its own address.

    The optional `mask` controls which elements are actually loaded; masked-off
    positions are filled with `other` if provided.

    Example:
      // (1) scalar load
      %v0 = thvtile.iterative_load %ptr : !thvtile.ptr<f32> -> f32

      // (2) per-element load from a pointer tensor, no mask
      %t0 = thvtile.iterative_load %ptrs
          : tensor<64x64x!thvtile.ptr<f16>> -> tensor<64x64xf16>

      // (3) per-element load with mask + fill value
      %t1 = thvtile.iterative_load %ptrs, %mask, %other
          : tensor<64x64x!thvtile.ptr<f16>>, tensor<64x64xi1>, f16
          -> tensor<64x64xf16>
  }];

  let arguments = (ins
    AnyType:$ptrs,
    Optional<ThvTile_BoolLike>:$mask,
    Optional<ThvTile_ScalarOrTensor>:$other
  );
  let results = (outs ThvTile_ScalarOrTensor:$result);

  let assemblyFormat = [{
    $ptrs (`,` $mask^)? (`,` $other^)? attr-dict `:`
    type($ptrs) (`,` type($mask)^)? (`,` type($other)^)? `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_IterativeStoreOp : ThvTile_Op<"iterative_store", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Iterative store to memory (scalar pointer or per-element pointer tensor)";
  let description = [{
    Fallback store op for non-structured memory access patterns.
    When pointer analysis cannot extract affine offset/stride information,
    this op performs iterative (per-element) store to memory.

    The `ptrs` operand can be either:
    - a single `!thvtile.ptr<T>` scalar pointer → stores a single scalar element, or
    - a tensor of pointers `tensor<...x!thvtile.ptr<T>>` → per-element store.

    Example:
      // (1) scalar store
      thvtile.iterative_store %val, %ptr : f32, !thvtile.ptr<f32>

      // (2) per-element store to a pointer tensor, no mask
      thvtile.iterative_store %value, %ptrs
          : tensor<64x64xf16>, tensor<64x64x!thvtile.ptr<f16>>

      // (3) per-element store with mask
      thvtile.iterative_store %value, %ptrs, %mask
          : tensor<64x64xf16>, tensor<64x64x!thvtile.ptr<f16>>, tensor<64x64xi1>
  }];

  let arguments = (ins
    ThvTile_ScalarOrTensor:$value,
    AnyType:$ptrs,
    Optional<ThvTile_BoolLike>:$mask
  );

  let assemblyFormat = [{
    $value `,` $ptrs (`,` $mask^)? attr-dict `:`
    type($value) `,` type($ptrs) (`,` type($mask)^)?
  }];

  let hasVerifier = 1;
}

def ThvTile_EmptyOp : ThvTile_Op<"empty", [Pure]> {
  let summary = "Create an uninitialized tensor";
  let description = [{
    Creates a new tensor with uninitialized (undefined) contents. The result
    shape is determined by the result type; any dynamic dimensions are supplied
    via the `dynamicSizes` operands, in the order they appear in the type.

    Example:
      // (1) fully static 2D shape
      %e = thvtile.empty() : tensor<64x64xf16>

      // (2) fully dynamic shape (two dynamic dims)
      %d = thvtile.empty(%d0, %d1) : tensor<?x?xf32>

      // (3) mixed static/dynamic (only the leading dim is dynamic)
      %m = thvtile.empty(%d0) : tensor<?x128xf16>
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = "`(` $dynamicSizes `)` attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def ThvTile_FillOp : ThvTile_Op<"fill", [Pure]> {
  let summary = "Create a tensor filled with a constant value";
  let description = [{
    Creates a new tensor where every element is set to the given scalar value.
    The result shape is determined by the result type.

    Example:
      // (1) fill a 2D f16 tensor with a constant
      %zeros = thvtile.fill %cst : f16 -> tensor<64x64xf16>
  }];

  let arguments = (ins ThvTile_Scalar:$value);
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def ThvTile_PadOp : ThvTile_Op<"pad", [Pure,
    SameOperandsAndResultElementType]> {
  let summary = "Pad a tensor with a scalar value";
  let description = [{
    Pads the input tensor according to `low` and `high`.
    `low` and `high` specify padding before and after each dimension.
    `value` is a scalar SSA value used to fill the padded region.

    Example:
      // (1) pad only the trailing dim on the right with zeros (56 → 64)
      %p0 = thvtile.pad %x, %zero low = [0, 0] high = [0, 8]
           : tensor<64x56xf16>, f16 -> tensor<64x64xf16>

      // (2) symmetric padding on both sides of each dim with a scalar value
      %p1 = thvtile.pad %x, %pad low = [2, 2] high = [2, 2]
           : tensor<60x60xf16>, f16 -> tensor<64x64xf16>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$src,
    ThvTile_Scalar:$value,
    DenseI64ArrayAttr:$low,
    DenseI64ArrayAttr:$high
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $src `,` $value `low` `=` $low `high` `=` $high attr-dict
    `:` type($src) `,` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_PermuteOp : ThvTile_Op<"permute", [Pure,
    SameOperandsAndResultElementType]> {
  let summary = "Permute tensor dimensions";
  let description = [{
    Rearranges the dimensions of a tensor according to the given permutation.

    The `permutation` attribute specifies the new dimension ordering:
    permutation[i] = j means the i-th dimension of the result comes from the j-th
    dimension of the input.

    Example:
      // (1) 2D transpose (swap the two dims)
      %t = thvtile.permute %x {permutation = array<i32: 1, 0>}
           : tensor<64x128xf16> -> tensor<128x64xf16>

      // (2) 3D permute: result dim i comes from input dim permutation[i]
      %p = thvtile.permute %y {permutation = array<i32: 2, 0, 1>}
           : tensor<2x64x128xf32> -> tensor<128x2x64xf32>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$src,
    DenseI32ArrayAttr:$permutation
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def ThvTile_FlipOp : ThvTile_Op<"flip", [Pure,
    SameOperandsAndResultType]> {
  let summary = "Reverse elements along a specified dimension";
  let description = [{
    Reverses the order of elements along the given axis.

    Example:
      // (1) reverse along the last axis (columns)
      %f0 = thvtile.flip %x {axis = 1 : i32}
           : tensor<64x128xf16> -> tensor<64x128xf16>

      // (2) reverse along axis 0 (rows)
      %f1 = thvtile.flip %x {axis = 0 : i32}
           : tensor<64x128xf16> -> tensor<64x128xf16>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$src,
    I32Attr:$axis
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def ThvTile_CastOp : ThvTile_Op<"cast", [Pure,
    SameOperandsAndResultShape]> {
  let summary = "Type conversion between floating-point and integer types";
  let description = [{
    Converts tensor elements between different numeric types:
    - Float to float (e.g., f32 -> f16, f16 -> bf16)
    - Float to int (e.g., f32 -> i32)
    - Int to float (e.g., i16 -> f16)
    - Int to int (e.g., i32 -> i8, with truncation/extension)

    The optional `rounding` attribute specifies how values are rounded for
    conversions that require rounding (e.g., floating-point narrowing or
    floating-point to integer conversion). If absent, the target lowering may
    use its default rounding mode.

    Both source and result must be tensors.

    Example:
      // (1) float → float (narrowing)
      %r0 = thvtile.cast %x : tensor<64x64xf32> -> tensor<64x64xf16>

      // (2) float → int with explicit round-toward-zero
      %r2 = thvtile.cast %x, rounding = #thvtile.rounding_mode<rtz>
          : tensor<64x64xf32> -> tensor<64x64xi32>

      // (3) int → float
      %r3 = thvtile.cast %y : tensor<64x64xi16> -> tensor<64x64xf16>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$src,
    OptionalAttr<ThvTile_RoundingModeAttr>:$rounding
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $src attr-dict (`,` `rounding` `=` $rounding^)? `:` type($src) `->` type($result)
  }];
  let hasVerifier = 1;
}

def ThvTile_BitcastOp : ThvTile_Op<"bitcast", [Pure,
    SameOperandsAndResultShape]> {
  let summary = "Bitwise reinterpretation between types of equal bit-width";
  let description = [{
    Reinterprets the bits of a tensor element-by-element without any value
    conversion. Source and result must have the same bit-width per element.

    Example:
      // (1) f16 → i16 (same 16-bit width)
      %r0 = thvtile.bitcast %x : tensor<64x64xf16> -> tensor<64x64xi16>

      // (2) f32 → i32 (32-bit reinterpretation)
      %r1 = thvtile.bitcast %y : tensor<64x64xf32> -> tensor<64x64xi32>
  }];

  let arguments = (ins ThvTile_FpIntTensor:$src);
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def ThvTile_ExtractSliceOp : ThvTile_Op<"extract_slice", [
  DestinationStyleOpInterface,
  Pure,
  TypesMatchWith<"expected result type to match dest type",
                 "dest", "result", "$_self">
]> {
  let summary = "Extract a tile or gather-like sub-tensor";
  let description = [{
    Extracts tile/data from source tensor into an explicit destination tensor.

    This op models two access patterns:
    - Full-tile access: `static_sizes` covers the whole source tile and all
      `static_strides` are 1.
    - Gather-like access: the source is read as a regular strided pattern.
      `static_sizes` gives the number of selected elements per dimension, and
      `static_strides` gives the period between selected elements per dimension.
      All non-unit periods must be the same, so the access has a uniform gap.
      This can represent repeated equal-size blocks by using one dimension as
      the block-count dimension and another as the contiguous block dimension.

    When all static sizes are 1, a scalar-like extraction can be represented with
    a rank-0 tensor destination/result. During lowering, compatible cases may be
    optimized into a subview-like representation instead of materializing a data
    copy, depending on aliasing and target support.

    - `source`: tensor to read from
    - `dest`: destination tensor whose type matches the result type
    - `offsets`: starting position for each source dimension
    - `sizes`: number of elements to extract per source dimension; it must either
      cover the full source tile for full-tile access or describe the selected
      element/block layout for gather-like access
    - `strides`: period between selected elements per source dimension; it must
      be all 1 for full-tile/scalar-like access, or all non-unit values must be
      the same to represent one uniform gap

    Example:
      // (1) full 2D tile, unit stride
      %s0 = thvtile.extract_slice %src into %dst [%c0, %c0] [64, 64] [1, 1]
             : tensor<64x64xf16> into tensor<64x64xf16>

      // (2) gather-like 2D tile: sizes [32, 32], uniform non-unit period 2
      %s1 = thvtile.extract_slice %src into %dst [%c0, %c0] [32, 32] [2, 1]
             : tensor<64x64xf16> into tensor<32x32xf16>

      // (3) scalar-like extract at dynamic-loop-lowered position i
      %v0 = thvtile.extract_slice %vec into %dst0 [%i] [1] [1]
             : tensor<8xf32> into tensor<f32>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$source,
    ThvTile_FpIntTensor:$dest,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $source `into` $dest `[` $offsets `]` $static_sizes $static_strides
    attr-dict `:` type($source) `into` type($dest)
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];

  let hasVerifier = 1;
}

def ThvTile_InsertSliceOp : ThvTile_Op<"insert_slice", [
  DestinationStyleOpInterface,
  Pure,
  TypesMatchWith<"expected result type to match dest type",
                 "dest", "result", "$_self">
]> {
  let summary = "Insert a tile or scatter-like sub-tensor";
  let description = [{
    Inserts data into a destination tensor and returns the updated destination.
    This is the inverse of extract_slice.

    This op models two access patterns:
    - Full-tile access: `static_sizes` covers the whole destination tile and all
      `static_strides` are 1.
    - Scatter-like access: the source is written as a regular strided pattern.
      `static_sizes` gives the number of written elements per dimension, and
      `static_strides` gives the period between written elements per dimension.
      All non-unit periods must be the same, so the access has a uniform gap.
      This can represent repeated equal-size blocks by using one dimension as
      the block-count dimension and another as the contiguous block dimension.

    - `source`: scalar or tensor to insert
    - `dest`: destination tensor whose type matches the result type
    - `offsets`: starting position for each dimension in the result
    - `sizes`: number of elements inserted per dimension; it must either cover
      the full destination tile for full-tile access, describe the selected
      element/block layout for scatter-like access, or be all 1 for
      scalar/rank-0 source
    - `strides`: period between written elements per dimension; it must be all 1
      for full-tile/scalar-like access, or all non-unit values must be the same
      to represent one uniform gap

    Values outside the inserted region are preserved from `dest`.

    Example:
      // (1) full 2D tile, unit stride
      %u0 = thvtile.insert_slice %src into %dst [%c0, %c0] [64, 64] [1, 1]
                 : tensor<64x64xf16> into tensor<64x64xf16>

      // (2) scatter-like 2D tile: sizes [32, 32], uniform non-unit period 2
      %u1 = thvtile.insert_slice %src into %dst [%c0, %c0] [32, 32] [2, 1]
                 : tensor<32x32xf16> into tensor<64x64xf16>

      // (3) scalar insert at dynamic-loop-lowered position i
      %u3 = thvtile.insert_slice %value into %dst_vec [%i] [1] [1]
                 : f32 into tensor<8xf32>
  }];

  let arguments = (ins
    ThvTile_ScalarOrTensor:$source,
    ThvTile_FpIntTensor:$dest,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $source `into` $dest `[` $offsets `]` $static_sizes $static_strides
    attr-dict `:` type($source) `into` type($dest)
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];

  let hasVerifier = 1;
}

def ThvTile_ExpandShapeOp : ThvTile_Op<"expand_shape", [Pure,
    SameOperandsAndResultElementType]> {
  let summary = "Expand tensor dimensions by splitting existing ones";
  let description = [{
    Expands a tensor by splitting one or more dimensions into multiple.
    The `reassociation` attribute maps result dimensions to source dimensions.

    Example:
      // (1) split dim 0 (128 = 2 x 64), keep dim 1
      %e = thvtile.expand_shape %x [[0, 1], [2]]
           : tensor<128x64xf16> -> tensor<2x64x64xf16>

      // (2) split the trailing dim (256 = 16 x 16)
      %f = thvtile.expand_shape %y [[0], [1, 2]]
           : tensor<4x256xf32> -> tensor<4x16x16xf32>

      // (3) 1D → 3D, split into three factors (64 = 4 x 4 x 4)
      %g = thvtile.expand_shape %z [[0, 1, 2]]
           : tensor<64xf16> -> tensor<4x4x4xf16>

      // (4) insert a trailing size-1 dim (32 → 32 x 1)
      %h = thvtile.expand_shape %v [[0, 1]]
           : tensor<32xf16> -> tensor<32x1xf16>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$src,
    IndexListArrayAttr:$reassociation
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $src $reassociation attr-dict `:` type($src) `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_CollapseShapeOp : ThvTile_Op<"collapse_shape", [Pure,
    SameOperandsAndResultElementType]> {
  let summary = "Collapse tensor dimensions by merging adjacent ones";
  let description = [{
    Collapses a tensor by merging adjacent dimensions into one.
    The `reassociation` attribute maps result dimensions to source dimensions.

    Example:
      // (1) merge dims 0 and 1 (2 x 64 → 128), keep dim 2
      %c = thvtile.collapse_shape %x [[0, 1], [2]]
           : tensor<2x64x64xf16> -> tensor<128x64xf16>

      // (2) merge trailing dims (16 x 16 → 256)
      %d = thvtile.collapse_shape %y [[0], [1, 2]]
           : tensor<4x16x16xf32> -> tensor<4x256xf32>

      // (3) fully flatten 3D → 1D
      %e = thvtile.collapse_shape %z [[0, 1, 2]]
           : tensor<4x4x4xf16> -> tensor<64xf16>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$src,
    IndexListArrayAttr:$reassociation
  );
  let results = (outs ThvTile_FpIntTensor:$result);

  let assemblyFormat = [{
    $src $reassociation attr-dict `:` type($src) `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_MatmulOp : ThvTile_Op<"matmul", [Pure, AttrSizedOperandSegments]> {
  let summary = "Tile-level matrix multiply-accumulate";
  let description = [{
    Computes D = A * B + C (when C is provided) or D = A * B.
    C may be either a 2D accumulator with shape [M, N] or a 1D bias with
    shape [N], broadcast across the M dimension.

    Supports multiple precision modes:
    - Standard: FP16/BF16/FP32 x FP16/BF16/FP32 -> FP16/BF16/FP32
    - Integer: INT8 x INT8 -> INT32

    C is optional; when absent, accumulation starts from zero. When present,
    C may be a 2D accumulator [M, N] or a 1D bias [N].

    Example:
      // (1) standard, with 2D accumulator C
      %d = thvtile.matmul %a : tensor<64x32xf16>, %b : tensor<32x64xf16>
           acc %c : tensor<64x64xf32> -> tensor<64x64xf32>

      // (2) standard, with 1D bias C broadcast across M
      %d = thvtile.matmul %a : tensor<64x32xf16>, %b : tensor<32x64xf16>
           acc %bias : tensor<64xf32> -> tensor<64x64xf32>
  }];

  let arguments = (ins
    ThvTile_FpIntTensor:$a,
    ThvTile_FpIntTensor:$b,
    Optional<ThvTile_FpIntTensor>:$c,
    Optional<ThvTile_FpIntTensor>:$a_scale,
    Optional<ThvTile_FpIntTensor>:$b_scale,
    DefaultValuedAttr<ThvTile_QuantFormatAttr,
      "::mlir::thvtile::QuantFormat::NONE">:$a_format,
    DefaultValuedAttr<ThvTile_QuantFormatAttr,
      "::mlir::thvtile::QuantFormat::NONE">:$b_format
  );

  let results = (outs ThvTile_FpIntTensor:$d);

  let assemblyFormat = [{
    $a `:` type($a) `,` $b `:` type($b)
    (`acc` $c^ `:` type($c))?
    (`a_scale` $a_scale^ `:` type($a_scale))?
    (`b_scale` $b_scale^ `:` type($b_scale))?
    attr-dict `->` type($d)
  }];

  let hasVerifier = 1;
}

def ThvTile_AtomicRMWOp : ThvTile_Op<"atomic_rmw", []> {
  let summary = "Atomic read-modify-write";
  let description = [{
    Atomically loads the value at the pointer(s), applies `atomic_op` with
    `value`, stores the result back, and returns the old value(s).

    Supports both scalar and tensor operands with integer element types only:
    - Scalar: `ptr` is a single pointer to an integer, `value` and result are
      integer scalars.
    - Tensor: `ptr` is a tensor of integer pointers, `value` and result are
      integer tensors. An optional `mask` controls which elements participate.

    The `scope` attribute currently only accepts `block` for this op.

    Example:
      // (1) scalar integer add
      %old = thvtile.atomic_rmw "add", "relaxed", "block", %ptr, %val
             : !thvtile.ptr<i32>, i32 -> i32

      // (2) scalar integer exchange, acquire-release
      %old = thvtile.atomic_rmw "xchg", "acq_rel", "block", %ptr, %val
             : !thvtile.ptr<i64>, i64 -> i64

      // (3) tensor integer add with mask
      %old = thvtile.atomic_rmw "add", "relaxed", "block", %ptrs, %vals, %mask
             : tensor<128x!thvtile.ptr<i32>>, tensor<128xi32>, tensor<128xi1>
             -> tensor<128xi32>
  }];

  let arguments = (ins
    ThvTile_AtomicRMWKindAttr:$atomic_op,
    ThvTile_MemSemanticAttr:$sem,
    ThvTile_MemScopeAttr:$scope,
    AnyType:$ptr,
    ThvTile_IntLike:$value,
    Optional<ThvTile_BoolTensor>:$mask
  );
  let results = (outs ThvTile_IntLike:$result);

  let assemblyFormat = [{
    $atomic_op `,` $sem `,` $scope `,` $ptr `,` $value (`,` $mask^ `:` type($mask))?
    attr-dict `:` type($ptr) `,` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_AtomicCASOp : ThvTile_Op<"atomic_cas", []> {
  let summary = "Atomic compare-and-swap";
  let description = [{
    Atomically compares value at the pointer(s) with `cmp`. If equal, stores
    `val` and returns the original value. Otherwise returns the current value.

    Supports both scalar and tensor operands with integer element types only:
    - Scalar: `ptr` is a single pointer to an integer, `cmp`/`val`/result are
      integer scalars.
    - Tensor: `ptr` is a tensor of integer pointers, `cmp`/`val`/result are
      integer tensors.

    The `scope` attribute currently only accepts `block` for this op.

    Example:
      // (1) scalar compare-and-swap
      %old = thvtile.atomic_cas "relaxed", "block", %ptr, %cmp, %val
             : !thvtile.ptr<i32>, i32, i32 -> i32

      // (2) scalar compare-and-swap with stronger ordering
      %old = thvtile.atomic_cas "acq_rel", "block", %ptr, %cmp, %val
             : !thvtile.ptr<i64>, i64, i64 -> i64

      // (3) tensor compare-and-swap
      %old = thvtile.atomic_cas "relaxed", "block", %ptrs, %cmp, %val
             : tensor<64x!thvtile.ptr<i32>>, tensor<64xi32>, tensor<64xi32>
             -> tensor<64xi32>
  }];

  let arguments = (ins
    ThvTile_MemSemanticAttr:$sem,
    ThvTile_MemScopeAttr:$scope,
    AnyType:$ptr,
    ThvTile_IntLike:$cmp,
    ThvTile_IntLike:$val
  );
  let results = (outs ThvTile_IntLike:$result);

  let assemblyFormat = [{
    $sem `,` $scope `,` $ptr `,` $cmp `,` $val
    attr-dict `:` type($ptr) `,` type($cmp) `,` type($val) `->` type($result)
  }];

  let hasVerifier = 1;
}

def ThvTile_LibdeviceCallOp : ThvTile_Op<"libdevice_call", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Call a libdevice/external function by name";
  let description = [{
    Calls an external libdevice-style function identified by a string symbol
    name.

    The callee name is carried as an attribute, and the op accepts an arbitrary
    number of operands with arbitrary types. It returns an arbitrary number of
    values with arbitrary types, specified by the result types.

    The `pure` attribute controls side-effect modeling: when `pure = true`
    (default) the call has no memory effects; when `pure = false` it is modeled
    with a read+write on the default resource.

    Example:
      %r3:2 = thvtile.libdevice_call "custom_divmod"(%x, %y) : (i32, i32) -> (i32, i32)
  }];

  let arguments = (ins
    StrAttr:$callee,
    Variadic<AnyType>:$operands,
    BoolAttr:$pure
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
}

def ThvTile_GetBlockIdOp : ThvTile_Op<"get_block_id", [Pure]> {
  let summary = "Get the current block ID along a grid dimension";
  let description = [{
    Returns the block index along the specified grid dimension.

    Example:
      %bid_x = thvtile.get_block_id {dim = #thvtile.block_dim<x>} : i32
      %bid_y = thvtile.get_block_id {dim = #thvtile.block_dim<y>} : i32
      %bid_z = thvtile.get_block_id {dim = #thvtile.block_dim<z>} : i32
  }];

  let arguments = (ins ThvTile_BlockDimAttr:$dim);
  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ThvTile_GetNumBlocksOp : ThvTile_Op<"get_num_blocks", [Pure]> {
  let summary = "Get the total number of blocks along a grid dimension";
  let description = [{
    Returns the grid size (number of blocks) along the specified dimension.

    Example:
      %grid_x = thvtile.get_num_blocks {dim = #thvtile.block_dim<x>} : i32
      %grid_y = thvtile.get_num_blocks {dim = #thvtile.block_dim<y>} : i32
      %grid_z = thvtile.get_num_blocks {dim = #thvtile.block_dim<z>} : i32
  }];

  let arguments = (ins ThvTile_BlockDimAttr:$dim);
  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ThvTile_PrintOp : ThvTile_Op<"print", [
    MemoryEffects<[MemWrite]>
]> {
  let summary = "Device-side print for debugging";
  let description = [{
    Prints a message prefix followed by the values of the given arguments.
    Useful for debugging kernel behavior on-device.

    Example:
      // (1) print a tensor with a label
      thvtile.print "tile value:" %x : tensor<4x4xf32>

      // (2) print multiple values
      thvtile.print "a, b:" %a, %b : f32, tensor<8xi32>

      // (3) print a plain message with no operands
      thvtile.print "reached checkpoint"
  }];

  let arguments = (ins
    StrAttr:$prefix,
    Variadic<AnyType>:$args
  );

  let assemblyFormat = [{
    $prefix attr-dict ($args^ `:` type($args))?
  }];
}

def ThvTile_AssertOp : ThvTile_Op<"assert", [
    MemoryEffects<[MemWrite]>
]> {
  let summary = "Device-side assertion for correctness checking";
  let description = [{
    Checks a boolean condition. If false, prints the message and aborts.
    The condition can be a scalar i1 or a tensor of i1.

    Example:
      // (1) scalar i1 condition
      thvtile.assert %cond, "matrix dimensions must match" : i1

      // (2) tensor of i1 (aborts if any element is false)
      thvtile.assert %conds, "all values must be positive" : tensor<128xi1>
  }];

  let arguments = (ins
    ThvTile_BoolLike:$condition,
    StrAttr:$message
  );
  let assemblyFormat = "$condition `,` $message attr-dict `:` type($condition)";
}

def ThvTile_AddPtrOp : ThvTile_Op<"addptr", [Pure]> {
  let summary = "Add an integer offset to a pointer (scalar or tensor)";
  let description = [{
    Performs elementwise pointer arithmetic: adds an integer offset (in
    elements) to each pointer in the `ptr` operand. Both scalar pointers
    and tensors of pointers are supported.

    Example:
      // (1) scalar
      %p1 = thvtile.addptr %p, %off : !thvtile.ptr<f16>, i64

      // (2) tensor-of-pointers
      %p2 = thvtile.addptr %ps, %offs
            : tensor<4x8x!thvtile.ptr<f16>>, tensor<4x8xi64>
  }];

  let arguments = (ins AnyType:$ptr, ThvTile_IntLike:$offset);
  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$ptr `,` $offset attr-dict `:` functional-type($ptr, $result) `,` type($offset)";

  let hasVerifier = 1;

  let hasFolder = 1;
}

def ThvTile_IntToPtrOp : ThvTile_Op<"int_to_ptr", [
    Elementwise, SameOperandsAndResultShape, Pure
]> {
    let summary = "Cast an i64 (scalar or tensor) to a pointer (scalar or tensor)";
    let description = [{
      Reinterprets a 64-bit integer address as a `!thvtile.ptr<T>`. Operates elementwise,
      so a `tensor<...xi64>` input yields a `tensor<...x!thvtile.ptr<T>>` output;
      a scalar `i64` yields a scalar `!thvtile.ptr<T>`.

      Example:
        // (1) scalar
        %p = thvtile.int_to_ptr %addr : i64 -> !thvtile.ptr<f32>

        // (2) tensor-of-pointers (gather/scatter addresses)
        %ps = thvtile.int_to_ptr %addrs : tensor<4x8xi64>
              -> tensor<4x8x!thvtile.ptr<f32>>
    }];

    let arguments = (ins ThvTile_I64Like:$src);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def ThvTile_PtrToIntOp : ThvTile_Op<"ptr_to_int", [
    Elementwise, SameOperandsAndResultShape, Pure
]> {
  let summary = "Cast a pointer (scalar or tensor) to an i64 (scalar or tensor)";
  let description = [{
    Reinterprets a `!thvtile.ptr<T>` as its 64-bit integer address. Operates
    elementwise: a `tensor<...x!thvtile.ptr<T>>` input yields a `tensor<...xi64>`
    output with matching shape; a scalar pointer yields a scalar `i64`.

    Example:
      // (1) scalar
      %addr = thvtile.ptr_to_int %p : !thvtile.ptr<f32> -> i64

      // (2) tensor-of-pointers
      %addrs = thvtile.ptr_to_int %ps : tensor<4x8x!thvtile.ptr<f32>>
               -> tensor<4x8xi64>
  }];

  let arguments = (ins AnyType:$src);
  let results = (outs ThvTile_I64Like:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif
