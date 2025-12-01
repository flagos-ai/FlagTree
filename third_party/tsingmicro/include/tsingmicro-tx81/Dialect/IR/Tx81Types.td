//===-------------------------- Tx81Types.td ------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// TODO: Update this file to define the customized type used by Tx81 dialect,
// it is now copy-and-pasted from MagicKernelTypes.td.
//
//===----------------------------------------------------------------------===//

#ifndef TSINGMICRO_TX81_TYPES_TD
#define TSINGMICRO_TX81_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.td"

//
// Types
//
class MKTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<Tx81Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

// Floating-point Type
def MKFloat : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def MKFloatTensor : RankedTensorOf<[MKFloat]>;
def MKFloatLike : AnyTypeOf<[MKFloat, MKFloatTensor]>;

// Boolean Type
// TT_Bool -> I1
def MKBoolTensor : RankedTensorOf<[I1]>;
def MKBoolLike : AnyTypeOf<[I1, MKBoolTensor]>;

// Integer Type
def I4 : I<4>;
def MKInt : AnyTypeOf<[I1, I4, I8, I16, I32, I64], "integer">;
def MKIntTensor : RankedTensorOf<[MKInt]>;
def MKIntLike : AnyTypeOf<[MKInt, MKIntTensor]>;

// I32 Type
// MKI32 -> I32
// MKI32Tensor -> I32Tensor
def MKI32Like : AnyTypeOf<[I32, I32Tensor]>;

// I64 Type
// MKI64 -> I64
// MKI64Tensor -> I64Tensor
def MKI64Like : AnyTypeOf<[I64, I64Tensor]>;

// Pointer Type in TableGen
class MKPtrOf<list<Type> pointeeTypes> :
    DialectType<Tx81Dialect,
                And<[CPred<"::mlir::isa<::mlir::triton::PointerType>($_self)">,
                     Concat<"[](::mlir::Type pointeeType) { return ",
                            SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                                        "; }(::mlir::cast<::mlir::triton::PointerType>($_self).getPointeeType())">]>,
                "ptr", "::mlir::triton::PointerType">;

// Pointer Type in C++ (corresponding to `MKPtrOf`)
def MKPtrType : MKTypeDef<"Pointer", "ptr"> {
  let summary = "Pointer type (`::mlir::triton::PointerType`) in Triton IR type system";

  let description = [{
      Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
  }];

  let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

  let builders = [
      TypeBuilderWithInferredContext<(ins
          "Type":$pointeeType,
          "int":$addressSpace
      ), [{
          return $_get(pointeeType.getContext(), pointeeType, addressSpace);
      }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
}

// Scalar Pointer Type: `ptr<>`
def MKPtr : MKPtrOf<[AnyType]>;

// Tensor of Pointer Type: `tensor<ptr<>>`
def MKPtrTensor : RankedTensorOf<[MKPtr]>;

// Tensor of Pointer Type or Pointer type: `tensor<ptr<>>` or `ptr<>`
def MKPtrLike : AnyTypeOf<[MKPtr, MKPtrTensor]>;

// Tensor Type
def MKFpIntTensor : RankedTensorOf<[MKFloat, MKInt]>;
def MKTensor : RankedTensorOf<[MKFloat, MKInt, MKPtr]>;

// Pointer Type to Tensor Type: `ptr<tensor<>>`
def MKTensorPtr : MKPtrOf<[MKTensor]>;

// Any Type in Magic Kernel IR
def MKType : AnyTypeOf<[MKFloatLike, MKIntLike, MKPtrLike, MKTensorPtr]>;

#endif // TSINGMICRO_TX81_TYPES_TD
