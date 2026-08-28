//===----------------------------------------------------------------------===//
// MKPipeline: software-pipeline scf.for loops with mk.dot / SPM buffers
//===----------------------------------------------------------------------===//
#ifndef MK_PIPELINE_PASSES_H
#define MK_PIPELINE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#include "magic-kernel/Conversion/MKPipeline/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "magic-kernel/Conversion/MKPipeline/Passes.h.inc"

} // namespace mlir::triton

#endif // MK_PIPELINE_PASSES_H