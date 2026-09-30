#ifndef THVTILE_TYPES
#define THVTILE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "ThvTileDialect.td"

//===----------------------------------------------------------------------===//
// Type base class
//===----------------------------------------------------------------------===//

class ThvTile_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<ThvTile_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// Pointer Type
//===----------------------------------------------------------------------===//

def ThvTile_PtrType : ThvTile_TypeDef<"Ptr", "ptr"> {
  let summary = "Pointer type in ThvTile IR";
  let description = [{
    A pointer to a scalar element in memory, can be used as the base address.

    Example:
      !thvtile.ptr<f32>    // pointer to f32 scalar
  }];

  let parameters = (ins "Type":$pointeeType);
  let assemblyFormat = "`<` $pointeeType `>`";
}

//===----------------------------------------------------------------------===//
// Tensor Structure Type
//===----------------------------------------------------------------------===//

def ThvTile_TensorStructureType : ThvTile_TypeDef<"TensorStructure", "tensor_structure"> {
  let summary = "Tensor structure descriptor type";
  let description = [{
    Encapsulates memory layout information for a tile region:
    base pointer, tile shape, offsets, strides, and dimension order.
    This is the result type of `thvtile.make_tensor_structure` and is
    consumed by `thvtile.load` or `thvtile.store` and so on to
    perform structured memory transfers.

    The `blockType` parameter specifies the tile shape and element type
    that will be loaded.

    Example:
      !thvtile.tensor_structure<tensor<64x64xf16>>
  }];

  let parameters = (ins "RankedTensorType":$blockType);
  let assemblyFormat = "`<` $blockType `>`";
}

//===----------------------------------------------------------------------===//
// Type Constraint Predicates
//===----------------------------------------------------------------------===//

// Floating-point scalar types supported by ThvTile
def ThvTile_Float : AnyTypeOf<[
  F16, BF16, F32, F64
], "thvtile floating-point">;

// Integer scalar types
// NOTE: MLIR only predefines I1/I8/I16/I32/I64/I128.
def ThvTile_I4 : I<4>;
def ThvTile_Int : AnyTypeOf<[I1, ThvTile_I4, I8, I16, I32, I64], "thvtile integer">;

// Numeric scalar
def ThvTile_Scalar : AnyTypeOf<[ThvTile_Float, ThvTile_Int]>;

// Tensor of FP or Int elements
def ThvTile_FpIntTensor : RankedTensorOf<[ThvTile_Float, ThvTile_Int]>;

// Integer scalar or tensor value
def ThvTile_IntTensor : RankedTensorOf<[ThvTile_Int]>;
def ThvTile_IntLike : AnyTypeOf<[ThvTile_Int, ThvTile_IntTensor],
  "thvtile integer scalar or tensor">;

// Boolean predicates
def ThvTile_BoolTensor : RankedTensorOf<[I1]>;
def ThvTile_BoolLike : AnyTypeOf<[I1, ThvTile_BoolTensor]>;

// I64 scalar or tensor of I64 predicate
def ThvTile_I64Tensor : RankedTensorOf<[I64]>;
def ThvTile_I64Like : AnyTypeOf<[I64, ThvTile_I64Tensor],
                                "thvtile i64-like">;

// Pointer type predicate
def ThvTile_Ptr : Type<
  CPred<"::mlir::isa<::mlir::thvtile::PtrType>($_self)">,
  "thvtile pointer type">;

// Pointer OR tensor-of-pointers
def ThvTile_AnyPtr : AnyTypeOf<[ThvTile_Ptr, RankedTensorOf<[ThvTile_PtrType]>],
  "thvtile pointer or tensor of pointers">;

// Scalar OR tensor value
def ThvTile_ScalarOrTensor : AnyTypeOf<[ThvTile_Scalar, ThvTile_FpIntTensor],
  "thvtile scalar or tensor value">;

// TensorStructure type predicate
def ThvTile_TensorStructure : Type<
  CPred<"::mlir::isa<::mlir::thvtile::TensorStructureType>($_self)">,
  "thvtile tensor structure type">;

#endif
