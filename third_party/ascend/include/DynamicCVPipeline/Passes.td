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
}

#endif // TRITON_DYNAMIC_CV_PIPELINE_PASSES