#ifndef FLAGTREE_PASSES
#define FLAGTREE_PASSES

include "mlir/Pass/PassBase.td"

def FlagTreeConvertArgToMemDesc : Pass<"flagtree-convert-arg-to-memdesc", "mlir::ModuleOp"> {}
def FlagTreeDSLRegionInline : Pass<"flagtree-dslregion-inline", "mlir::ModuleOp"> {}

#endif
