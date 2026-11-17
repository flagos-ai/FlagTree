/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_DYNAMIC_CV_PIPELINE_PASSES
#define TRITON_DYNAMIC_CV_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

def AddDynamicCVPipeline : Pass<"add_dynamic_cv_pipeline", "mlir::ModuleOp"> {
    let summary = "Add dynamic cv pipeline based on ssbufer";
    let constructor = "triton::createAddDynamicCVPipelinePass()";
    let options = [
        Option<"compileOn91095", "compile-on-910-95",
        "bool", /*default*/"false",
        "compile on 910_95">
    ];
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect", "LLVM::LLVMDialect"];
}

def PreCheckAvailablePass : Pass<"pre-check-dynamic-cv-pipeline-available", "mlir::ModuleOp"> {
  let summary = "Check whether dynamic cv pipeline can process the input module";
  let constructor = "triton::createPreCheckAvailablePass()";
  let dependentDialects = ["scope::ScopeDialect"];
}

def OpClassifierPass : Pass<"op-classifier", "mlir::ModuleOp"> {
  let summary = "Classify operations as CUBE or VECTOR";
  let constructor = "triton::createOpClassifierPass()";
}

def PlanComputeBlockPass : Pass<"plan-compute-block", "mlir::ModuleOp"> {
  let summary = "Plan compute block";
  let constructor = "triton::createPlanComputeBlockPass()";
}

def AddBlockIdForControlOpsPass : Pass<"add-block-id-for-control-ops", "mlir::ModuleOp"> {
    let summary = "Add block_id attribute to control flow operations";
    let constructor = "triton::createAddBlockIdForControlOpsPass()";
}

def DataDependencyAnalysisPass : Pass<"data-dependency-analysis", "mlir::ModuleOp"> {
    let summary = "Analyze data dependencies between Vector and Cube blocks";
    let constructor = "triton::createDataDependencyAnalysisPass()";
}

def InterCoreTransferAndSyncPass : Pass<"inter-core-transfer-and-sync", "mlir::ModuleOp"> {
    let summary = "Insert inter-core transfer operations and PIPE_S synchronization between Vector and Cube cores";
    let constructor = "triton::createInterCoreTransferAndSyncPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def MarkMainLoopPass : Pass<"mark-main-loop", "mlir::ModuleOp"> {
    let summary = "Mark main computation loops containing FixpipeOp or CopyOp with main_loop attribute";
    let constructor = "triton::createMarkMainLoopPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def SeparateCVScopePass : Pass<"separate-cv-scope", "mlir::ModuleOp"> {
    let summary = "separate ops into cv scope";
    let constructor = "triton::createSeparateCVScopePass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def PreserveControlAttrsCanonicalizePass : Pass<"preserve-control-attrs-canonicalize", "mlir::ModuleOp"> {
    let summary = "Canonicalize while preserving selected control-flow attrs";
    let constructor = "triton::createPreserveControlAttrsCanonicalizePass()";
}

def ComputeBlockOptPass : Pass<"compute-block-id-opt", "mlir::ModuleOp"> {
  let summary = "Block ID Optimization Pass";
  let constructor = "triton::createComputeBlockOptPass()";
}

def RemoveSsbufAttrPass : Pass<"remove-ssbuf-attr", "mlir::ModuleOp"> {
  let summary = "Remove ssbuf attributes";
  let constructor = "triton::createRemoveSsbufAttrPass()";
}

def CheckUnsupportedScenarioPass : Pass<"ssbuf-check-unsupported-scenario", "mlir::ModuleOp"> {
  let summary = "Check unsupported scenario pass for ssbuf";
  let constructor = "triton::createCheckUnsupportedScenarioPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

#endif // TRITON_DYNAMIC_CV_PIPELINE_PASSES
