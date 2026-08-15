#ifndef TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES
#define TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToUnstructured : Pass<"triton-to-unstructured", "mlir::ModuleOp"> {
  let summary = "Transforms tt.addptr ops into offset accumulation ops";
  let constructor = "triton::createTritonToUnstructuredPass()";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", /*default*/"32",
             "Bitwidth used for the starting offset of each pointer">
  ];
}

#endif
