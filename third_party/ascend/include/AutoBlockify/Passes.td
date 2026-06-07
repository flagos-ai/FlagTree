#ifndef AUTO_BLOCKIFY_PASSES
#define AUTO_BLOCKIFY_PASSES

include "mlir/Pass/PassBase.td"

def AutoBlockify : Pass<"auto-blockify", "mlir::ModuleOp"> {
    let summary = "Apply auto blockify v2";
    let constructor = "triton::createAutoBlockifyPass()";
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
        "mlir::triton::TritonDialect"
    ];
    let options = [
        Option<"autoBlockifySize", "auto-blockify-size", "int", "1",
           "Apply auto blockify v2 when TRITON_ALL_BLOCKS_PARALLEL is 1."
           "Expand highest dimension with blockify size">
    ];
}

#endif // AUTO_BLOCKIFY_PASSES
