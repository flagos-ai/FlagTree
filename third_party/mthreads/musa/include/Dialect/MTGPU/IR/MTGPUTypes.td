#ifndef MTGPU_TYPES
#define MTGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MTGPUDialect.td"

class MTGPU_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<MTGPUDialect, name, traits> {
  let mnemonic = _mnemonic;
}

def MTGPU_SqmmaAccumulatorType
    : MTGPU_TypeDef<"SqmmaAccumulator", "sqmma_accumulator"> {
  let summary = "backend-private SQMMA accumulator carrier";
  let description = [{
    A backend-private carrier type that preserves SQMMA loop-carried
    accumulators in compact fragment form while keeping the wrapped tensor
    type as the logical accumulator semantics.
  }];

  let parameters = (ins "Type":$tensorType);
  let assemblyFormat = "`<` $tensorType `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    RankedTensorType getAccumulatorType() const {
      return llvm::cast<RankedTensorType>(getTensorType());
    }
  }];
}

#endif
