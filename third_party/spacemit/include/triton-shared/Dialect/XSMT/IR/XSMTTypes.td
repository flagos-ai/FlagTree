include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "triton-shared/Dialect/XSMT/IR/XSMTDialect.td"

class XSMT_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<XSMT_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def XSMT_BufferType : XSMT_TypeDef<"Buffer", "buffer", [ShapedTypeInterface]> {
  let summary = "buffer type for SMT tensors";
  let description = [{
    A buffer type that represents multi-buffered tensors with different scope kinds.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int":$copies,
    "StringAttr":$scopeKind
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      "int":$copies,
      "StringRef":$scopeKind
    ), [{
      auto *ctx = elementType.getContext();
      return $_get(ctx, shape, elementType, copies,
                   StringAttr::get(ctx, scopeKind));
    }]>
  ];

  let extraClassDeclaration = [{
    mlir::ShapedType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                               mlir::Type elementType) const;
    bool hasRank() const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def XSMT_MBarrierType : XSMT_TypeDef<"MBarrier", "mbarrier"> {
  let summary = "Multi-copy mbarrier type";
  let description = [{
    Represents a handle to multiple mbarrier copies.
    Used for async copy synchronization with double/triple buffering.
  }];

  let parameters = (ins
    "int":$numBarriers
  );

  let assemblyFormat = "`<` $numBarriers `>`";
}
