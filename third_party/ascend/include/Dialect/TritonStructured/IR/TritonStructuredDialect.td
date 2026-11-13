#ifndef TRITON_STRUCTURED_DIALECT
#define TRITON_STRUCTURED_DIALECT

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
//include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



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

  let usePropertiesForAttributes = 1;
}

//
// Op Base
//
class TTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Structured_Dialect, mnemonic, traits> {
}


// SameVariadicResultSize
// AttrSizedResultSegments
def TTS_GetStructuredStateOp : TTS_Op<"get_structured_state", [AttrSizedResultSegments, Pure]> {
  let summary = "Placeholder for the structured pointer states computed during PtrAnalysis.";
  let description = "Used to pass the offsets and strides to scf.for op to simplify IR rewrites.";

  let arguments = (ins AnyTypeOf<[TT_PtrLike, I32Tensor, I64Tensor,I16Tensor,I8Tensor,I1Tensor]>:$input);
  let results = (outs AnyTypeOf<[TT_PtrLike, I32Tensor, I64Tensor,I16Tensor,I8Tensor,I1Tensor]>:$structured, Variadic<Index>:$offsets, Variadic<Index>:$strides);

  let builders = [
    OpBuilder<(ins "Value":$input)>,
  ];

  let extraClassDeclaration = [{
    static std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
      getOffsetAndStrideTypes(MLIRContext *context, Type ptrLikeType);

    static std::optional<std::pair<int32_t, int32_t>>
      getOffsetAndStrideSegmentSizes(Type ptrLikeType);
  }];

  let hasFolder = 0;
  let hasVerifier = 1;
}



#endif // TRITON_STRUCTURED_DIALECT
