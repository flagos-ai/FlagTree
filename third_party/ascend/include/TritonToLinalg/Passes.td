#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalg : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
    let summary = "Convert Triton to Linalg dialect";
    let constructor = "triton::createTritonToLinalgPass()";
    let options = [
        Option<"globalKernel", "global-kernel",
            "bool", /*default*/"true",
            "generate a global kernel">,
        Option<"namedOps", "named-ops",
            "bool", /*default*/"true",
            "use linalg named ops instead of linalg.generic">,
        Option<"enableNd2nzOnVector", "enable-nd2nz-on-vector",
            "bool", /*default*/"false",
            "enable nd2nz on vector">,
        Option<"enableSelectAnalysis", "enable-select-analysis",
            "bool", /*default*/"true",
            "enable select analysis">,
        Option<"compileOn91095", "compile-on-910-95",
            "bool", /*default*/"false",
            "compile on 910_95">,
        Option<"compileMode", "compile-mode",
            "std::string", /*default*/"\"simd_simt_template\"",
            "Compilation mode: simd, simd_simt_template (or unstructured_in_simt), simt_only">
    ];
}

def MarkTensorKind : Pass<"mark-tensor-kind", "mlir::ModuleOp"> {
    let summary = "Mark tensor kind (INPUT/OUTPUT/INPUT_OUTPUT) on Triton function arguments";
    let constructor = "triton::createMarkTensorKindPass()";
}

def DeduplicateDebugNops : Pass<"deduplicate-debug-nops", "mlir::ModuleOp"> {
    let summary = "Deduplicate llvm.inline_asm 'nop' debug-anchor ops by source line.";
    let description = [{
        Removes redundant `llvm.inline_asm "nop"` ops that share a
        `(filename, line)` source location within a function. Keeps the
        first occurrence in IR walk order; erases the rest.

        Opt-in via env var `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1`. With the
        env var unset the pass is a no-op.
    }];
    let constructor = "triton::createDeduplicateDebugNopsPass()";
}

#endif // TRITON_TO_LINALG_CONVERSION_PASSES
