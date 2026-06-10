#ifndef EV_LINALG_TRANSFORMS_PASSES_H_
#define EV_LINALG_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace ev {

std::unique_ptr<Pass> createRemoveLoopIterArgsWithMemrefTypePass();
std::unique_ptr<Pass> createRemoveScalarPass();
std::unique_ptr<Pass> createRewriteFuncOpArgsTypePass();
std::unique_ptr<Pass> createSplitComputationalOpPass();
std::unique_ptr<Pass> createInsertDeallocOpPass();
std::unique_ptr<Pass> createSetDeviceInfoPass();
std::unique_ptr<Pass> createMaterializeAnnotationPass();
std::unique_ptr<Pass> createDoubleBufferPass();
std::unique_ptr<Pass> createBufferizePass();
std::unique_ptr<Pass> createMemoryAllocPass();
std::unique_ptr<Pass> createMemoryAllocPass(size_t memScope, size_t alignment,
                                            bool preview, bool bankopt);
std::unique_ptr<Pass> createEncapsulateLinalgOpPass();
std::unique_ptr<Pass> createSetMemRefScopePass();
std::unique_ptr<Pass> createMemoryPromotionPass();
std::unique_ptr<Pass> createRemoveRedundencyCopyPass();
std::unique_ptr<Pass> createRewriteDataTypePass();
} // namespace ev
#define GEN_PASS_REGISTRATION
#include "evas/Transform/Linalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

#endif
