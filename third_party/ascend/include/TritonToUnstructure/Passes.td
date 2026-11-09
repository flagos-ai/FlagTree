#ifndef TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
#define TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToUnstructure : Pass<"triton-to-unstructure", "mlir::ModuleOp"> {
    let summary = "Convert Triton for unstructure case";
    let constructor = "triton::createTritonToUnstructurePass()";
    let options = [
        Option<"forceScalarizeMode", "force-scalarize-mode", "bool", "false",
           "Scalarize unstructured memory access even if structured dimensions are mixed.">,
        Option<"compileOn91095", "compile-on-910-95",
            "bool", /*default*/"false",
            "compile on 910_95">,
        Option<"compileMode", "compile-mode",
            "std::string", /*default*/"\"simd_simt_template\"",
            "Compilation mode: simd, simd_simt_template (or unstructured_in_simt), simt_only">
    ];
}

def BubbleUpOperation : Pass<"bubble-up-operation", "mlir::ModuleOp"> {
    let summary = "Apply bubble up operation optimization";
    let constructor = "triton::createBubbleUpOperationPass()";
    let options = [
        Option<"enableAggressiveMode", "enable-aggressive-mode", "bool", "true",
           "Enable aggressive bubble up operation.">,
    ];
}

#endif // TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
