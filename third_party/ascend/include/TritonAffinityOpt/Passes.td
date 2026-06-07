/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_AFFINITY_OPTIMIZATION_PASSES
#define TRITON_AFFINITY_OPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def DAGSSBuffer : Pass<"dag-ssbuf", "mlir::ModuleOp"> {
  let summary = "Convert vector operations to shared storage buffer operations";
  let constructor = "triton::createDAGSSBufferPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def DAGScope : Pass<"dag-scope", "mlir::ModuleOp"> {
  let summary = "Convert native triton code to NPU-affine code";
  let constructor = "triton::createDAGScopePass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def DAGSync : Pass<"dag-sync", "mlir::ModuleOp"> {
  let summary = "DAG sync";
  let constructor = "triton::createDAGSyncPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "annotation::AnnotationDialect"];
}

#endif // TRITON_AFFINITY_OPTIMIZATION_PASSES
