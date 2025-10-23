/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_ANNOTATION_CONVERSION_PASSES
#define TRITON_TO_ANNOTATION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToAnnotation : Pass<"triton-to-annotation", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Annotation dialect";
  let constructor = "triton::createTritonToAnnotationPass()";
  let dependentDialects = ["annotation::AnnotationDialect"];
}

#endif // TRITON_TO_ANNOTATION_CONVERSION_PASSES
