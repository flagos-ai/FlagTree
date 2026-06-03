#include "evas/Conversion/TritonToEvas/TritonToEvasPipeline.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "evas/Transform/Linalg/Passes.h"

using namespace mlir;

namespace mlir::triton::evas {

void buildTritonToEvasPipeline(OpPassManager &pm) {
  pm.addPass(triton::createTritonToStructuredPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(triton::createTritonToUnstructuredPass());
  pm.addPass(createEvasTritonArithToLinalgPass(/*tensorPtrToLinalg=*/true));
  pm.addPass(triton::createStructuredToMemrefPass());
  pm.addPass(triton::createUnstructuredToMemrefPass());
  pm.addPass(triton::createTritonPtrToMemrefPass());
  pm.addPass(triton::createTritonToPtrPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(triton::createReconcilePtrCastsPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // EVAS does not support every scalar/index width that upstream Triton emits.
  pm.addPass(ev::createRewriteDataTypePass());

  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.allowReturnAllocsFromLoops = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  pm.addPass(ev::createRemoveLoopIterArgsWithMemrefTypePass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(ev::createSetMemRefScopePass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  pm.addPass(ev::createRemoveRedundencyCopyPass());

  pm.addPass(ev::createInsertDeallocOpPass());
  pm.addPass(ev::createMemoryAllocPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

void registerTritonToEvasPipeline() {
  PassPipelineRegistration<>(
      "triton-to-evas",
      "Lower Triton IR through triton-shared and EVAS backend passes",
      [](OpPassManager &pm) { buildTritonToEvasPipeline(pm); });
}

} // namespace mlir::triton::evas
