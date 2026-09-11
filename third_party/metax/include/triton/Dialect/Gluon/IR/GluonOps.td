#ifndef GLUON_OPS
#define GLUON_OPS

include "triton/Dialect/Gluon/IR/GluonDialect.td"
include "triton/Dialect/Gluon/IR/GluonAttrDefs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class Gluon_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gluon_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def Gluon_SetAutoLayoutOp : Gluon_Op<"set_auto_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType]> {
  let summary = "set auto encoding to a concrete encoding type";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$encoding, "Value":$value)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Gluon_RequireLayoutOp : Gluon_Op<"require_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType, Pure]> {
  let summary = "explicit internal tensor or memdesc layout requirement";
  let description = [{
    Internal layout contract used by backend-specific Gluon layout pipelines.
    The result type carries the required concrete encoding. Layout propagation
    may retag the source producer/carrier to satisfy the contract. Tensor
    cleanup may materialize an explicit convert_layout when the source is
    already concrete; memdesc cleanup requires the contract to fold to an
    identity after retagging.
  }];

  let arguments = (ins AnyTypeOf<[TT_Tensor, TTG_MemDescType]>:$src);
  let results = (outs AnyTypeOf<[TT_Tensor, TTG_MemDescType]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Gluon_ReleaseLayoutOp : Gluon_Op<"release_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType, Pure]> {
  let summary = "explicit internal tensor layout release boundary";
  let description = [{
    Internal layout contract boundary used when a value leaves a specialized
    layout region. The result type carries the released concrete encoding.
    Tensor cleanup may erase the op or materialize convert_layout; memdesc
    cleanup requires an identity after retagging.
  }];

  let arguments = (ins AnyTypeOf<[TT_Tensor, TTG_MemDescType]>:$src);
  let results = (outs AnyTypeOf<[TT_Tensor, TTG_MemDescType]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Gluon_ExtractSliceOp
    : Gluon_Op<"extract_slice", [Pure, SameOperandsAndResultEncoding]> {
  let summary = "layout-parametric register tensor slice glue op";
  let description = [{
    Gluon dataflow-level register slice. This op carries element-offset slice
    intent while tensor layout is still AutoEncodingAttr. Backend lowering
    verifies that the slice is representable for the concrete layout and may
    rewrite it to triton_gpu.extract_tensor after TTG layouts have been
    inferred.
  }];

  let arguments = (ins
    AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$source,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$result);
}

def Gluon_InsertSliceOp
    : Gluon_Op<"insert_slice", [Pure, SameOperandsAndResultEncoding]> {
  let summary = "layout-parametric register tensor slice update glue op";
  let description = [{
    Gluon dataflow-level register slice update. This op carries element-offset
    update intent while tensor layout is still AutoEncodingAttr. Backend
    lowering verifies that the update is representable for the concrete layout
    and may rewrite it to triton_gpu.insert_tensor after TTG layouts have been
    inferred.
  }];

  let arguments = (ins
    AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$base,
    AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$update,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$result);
}

#endif // GLUON_OPS
