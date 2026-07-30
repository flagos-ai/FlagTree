#ifndef TRITON_ADD_TARGET_DESCRIPTION_CONVERSION_PASSES
#define TRITON_ADD_TARGET_DESCRIPTION_CONVERSION_PASSES


include "mlir/Pass/PassBase.td"

def AddTargetDescription : Pass<"add-target-description", "mlir::ModuleOp"> {
  let summary = "Add the target description from ModuleOp DLTI attribute";
  let constructor = "triton::createAddTargetDescriptionPass()";
  let dependentDialects = ["DLTIDialect"];
}

#endif
