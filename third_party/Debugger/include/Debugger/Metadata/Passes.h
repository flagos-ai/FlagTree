#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
class Operation;
class Pass;
} // namespace mlir

namespace mlir {
namespace flagtree {
namespace debugger {

// CTT-1 / B-stub: validate begin/end pairing and assign `scope_id` on marker
// ops. Does not erase markers (production pass erases after this succeeds).
mlir::LogicalResult
assignDebugCollectScopeIdsWithoutErase(mlir::ModuleOp module);

// Direct helpers for compile paths where running an MLIR pass manager would
// trigger verifier assertions on Triton block/tensor pointers.  These keep the
// debugger metadata path available without inserting dynamic record IR.
mlir::LogicalResult
assignDebugOpIdsAndMetadataWithoutPassManager(mlir::ModuleOp module);
void eraseDebugCollectMarkers(mlir::ModuleOp module);
bool hasTritonTensorPointerTypes(mlir::ModuleOp module);

// Module B entry points.
// - createResolveDebugScopePass: validate collect begin/end pairs and freeze
//   the set of tracked scopes.
// - createAssignOpIdPass: assign stable op ids and export static metadata.
std::unique_ptr<mlir::Pass> createResolveDebugScopePass();
std::unique_ptr<mlir::Pass> createAssignOpIdPass();

bool hasDebugCollectMarkers(mlir::Operation *op);
mlir::LogicalResult insertDefaultDebugCollectMarkers(mlir::ModuleOp module,
                                                     int32_t level,
                                                     int32_t addrLevel);

std::string getDebugTrackedOpTableJson(mlir::ModuleOp module);
std::string getDebugKernelMetadataJson(mlir::ModuleOp module);
uint32_t getDebugKernelId(mlir::ModuleOp module);
void setDebugKernelIdSeed(mlir::ModuleOp module, llvm::StringRef seed);

void registerFlagTreeDebuggerMetadataPasses();

} // namespace debugger
} // namespace flagtree
} // namespace mlir
