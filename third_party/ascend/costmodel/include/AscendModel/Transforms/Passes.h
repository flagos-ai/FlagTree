//===- Passes.h - AscendModel Transformation Passes -------------*- C++ -*-===//
//
// This file declares transformation passes for the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_TRANSFORMS_PASSES_H
#define ASCEND_MODEL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// Pass Declarations (TableGen generated)
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "AscendModel/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass Registration (TableGen generated)
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "AscendModel/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

/// Register the performance modeling pipeline.
void registerAscendModelPipeline();

/// Register all AscendModel passes and pipelines.
inline void registerAllAscendModelPasses() {
  registerAscendModelPasses();   // TableGen generated
  registerAscendModelPipeline(); // Custom pipeline
}

} // namespace ascend
} // namespace mlir

#endif // ASCEND_MODEL_TRANSFORMS_PASSES_H
