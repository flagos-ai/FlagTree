#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
class Pass;
}

namespace mlir {
namespace flagtree {
namespace debugger {

void setDebugHiddenArgAbiEnabled(mlir::ModuleOp module, bool enabled);
void setDebugAddrLevel(mlir::ModuleOp module, int32_t addrLevel);
void setDebugTimelineEnabled(mlir::ModuleOp module, bool enabled);
void setDebugTimelineOnly(mlir::ModuleOp module, bool enabled);
uint32_t getDebugRecordsPerInstance(mlir::ModuleOp module);
uint32_t getDebugRecordSize(mlir::ModuleOp module);
std::string getDebugRecordLayout(mlir::ModuleOp module);
std::string getDebugRecordPlanJson(mlir::ModuleOp module);
uint64_t getDebugFullDumpPayloadBytesPerInstance(mlir::ModuleOp module);
std::string getDebugFullDumpPlanJson(mlir::ModuleOp module);

// Module C entry point.
// This pass will eventually:
// - consume B's op-id/static metadata annotations
// - inject summary / memory-event instrumentation
// - write protocol records to the control block passed by A/F
std::unique_ptr<mlir::Pass> createInsertInstrumentationPass();
std::unique_ptr<mlir::Pass> createSimplifyRecordMemrefWritesPass();
void registerFlagTreeDebuggerInstrumentationPasses();

} // namespace debugger
} // namespace flagtree
} // namespace mlir
