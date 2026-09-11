#ifndef TRITON_METAX_GLUON_PASSES_H
#define TRITON_METAX_GLUON_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DECL
#include "Gluon/Passes.h.inc"

std::unique_ptr<Pass> createTritonMETAXGPUGluonInsertRequireLayoutPass(
    int computeCapability = 80);

std::unique_ptr<Pass> createTritonMETAXGPUGluonPropagateLayoutPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonLegalizeRegisterSlicesPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonResolvePlaceholderLayoutsPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonVerifySynchronizationPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonVerifyLayoutContractsPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonLegalizeLocalDotStagingPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonSelectC500LayoutTransfersPass();

std::unique_ptr<Pass>
createTritonMETAXGPUGluonReorderInstructionsPass();

LogicalResult
legalizeTritonMETAXGPUGluonC500AsyncCopyLayout(ModuleOp module);

LogicalResult
insertTritonMETAXGPUGluonGvmArriveBarrierShared(ModuleOp module);

LogicalResult verifyTritonMETAXGPUGluonSynchronization(ModuleOp module);

LogicalResult verifyTritonMETAXGPUGluonLayoutContracts(ModuleOp module);

#define GEN_PASS_REGISTRATION
#include "Gluon/Passes.h.inc"

} // namespace mlir

#endif // TRITON_METAX_GLUON_PASSES_H
