#ifndef GLUON_OPS
#define GLUON_OPS

include "triton/Dialect/Gluon/IR/GluonDialect.td"
include "triton/Dialect/Gluon/IR/GluonAttrDefs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

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
