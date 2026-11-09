//===- triton-opt.cpp - Triton-Ascend Pass Pipeline Driver ------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Reproduces the backend's ttir->ttadapter pass pipeline outside of Python.
// `third_party/ascend/backend/compiler.py` prints the equivalent command line
// under `if opt.debug:`, and this driver is what runs it.
//
// Deliberately does NOT reuse the core `bin/RegisterTritonDialects.h`:
//   - it registers test-only passes needing TritonTestAnalysis /
//     TritonTestDialect / TritonTestProton, which an ascend configuration
//     does not build (only amd's TritonAMDGPUTestAnalysis is available),
//   - its CommonIR dialect registration sits behind __FLAGTREE_COMMON_IR__,
//     a macro the ascend build never defines, so `tile.*` would not parse,
//   - it carries no HIVM/HFusion/Scope/HACC/Annotation registration.
// Keeping a self-contained registry here also means this file does not have
// to be kept in sync with a header shared by every other backend.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"

#include "ascend/include/AutoBlockify/Passes.h"
#include "ascend/include/CommonIRToHIVM/Passes.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"
#include "ascend/include/TritonControlFlowOpt/Passes.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonToGraph/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "ascend/include/TritonToStructured/Passes.h"
#include "ascend/include/TritonToUnstructure/Passes.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/RefineArgsBlockId.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

static void registerAscendPasses() {
  // Groups behind GEN_PASS_REGISTRATION. Between them these cover every pass
  // the backend names in the printed pipeline: triton-control-flow-opt,
  // triton-to-structured, discrete-mask-access-conversion,
  // triton-to-annotation, triton-to-unstructure (+ bubble-up-operation),
  // triton-to-hivm, triton-to-hfusion, triton-to-llvm, commonir-to-hivm
  // (+ reduce-sum-strength), triton-to-linalg, merge-concat-load-buffer
  // (via TritonToGraph) and add_dynamic_cv_pipeline.
  mlir::triton::registerTritonControlFlowOptPasses();
  mlir::triton::registerTritonToStructuredPasses();
  mlir::triton::registerDiscreteMaskAccessConversionPasses();
  mlir::triton::registerTritonToAnnotationPasses();
  mlir::triton::registerTritonToUnstructurePasses();
  mlir::triton::registerTritonToHIVMPasses();
  mlir::triton::registerTritonToHFusionPasses();
  mlir::triton::registerTritonToLLVMPasses();
  mlir::triton::registerCommonIRToHIVMPasses();
  mlir::triton::registerTritonToLinalgPasses();
  mlir::triton::registerAutoBlockifyPasses();
  mlir::triton::cfg::registerTritonToGraphPasses();
  mlir::triton::registerAddDynamicCVPipelinePasses();

  // DynamicCVPipeline sub-passes declared by hand rather than in Passes.td.
  // add_dynamic_cv_pipeline drives these through a nested pass manager, so
  // they are not required to replay the printed pipeline, but registering
  // them lets a single stage be run in isolation. Passes also covered by
  // registerAddDynamicCVPipelinePasses above resolve to the same allocator,
  // which the pass registry accepts.
  mlir::triton::registerStandardizeOpPasses();
  mlir::triton::registerPreCheckAvailablePasses();
  mlir::triton::registerAddControlFlowConditionPasses();
  mlir::triton::registerAddMultiBufferOuterScopePasses();
  mlir::triton::registerAddMultiBufferInnerScopePasses();
  mlir::triton::registerUBOverflowCheckerPasses();
  mlir::triton::registerSeparateMemoryFromComputePasses();
  mlir::triton::registerAnalyzeDataFlowPasses();
  mlir::triton::registerComputeBlockOptPasses();
  mlir::triton::registerPlanComputeBlockPasses();
  mlir::triton::registerAddOpClassifierPasses();
  mlir::triton::registerRefineArgsBlockIdPasses();
  // Not registered: registerAsyncLoadHoistingPasses. It is declared in
  // DynamicCVPipeline/DecoupleComputeAndMemory/AsyncLoadHoistingPass.h and
  // defined in the matching .cpp, but lib/DynamicCVPipeline/CMakeLists.txt
  // does not glob DecoupleComputeAndMemory/*.cpp, so that directory is not
  // compiled into libDynamicCVPipeline and the symbol does not exist.
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Upstream MLIR: supplies the dialects the lowering targets (linalg, memref,
  // tensor, bufferization, func, scf, ...) plus canonicalize / cse / inline /
  // symbol-dce, which the printed pipeline interleaves with ascend passes.
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  // mlir::registerAllExtensions does not pull in the func dialect's inliner
  // interface. The pipeline runs `inline` right after commonir-to-hivm, and
  // the Inliner aborts on func.func without it. This mirrors the in-process
  // path, which registers the same extension in triton_ascend.cc load_dialects.
  mlir::func::registerInlinerExtension(registry);

  // BiShengIR: hivm / hfusion / scope / hacc / annotation dialects and their
  // bufferization and tiling external models.
  bishengir::registerAllDialects(registry);

  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::ascend::TritonAscendDialect,
                  mlir::triton::tile::CommonIRDialect, mlir::tts::TritonStructuredDialect>();

  registerAscendPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton-Ascend pass pipeline driver\n", registry));
}
