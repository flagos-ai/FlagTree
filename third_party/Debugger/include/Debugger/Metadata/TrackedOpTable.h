#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {

// Module B's stable static metadata view.
// C only needs op_id-oriented compile-time facts for instrumentation.
// D uses the same table to map decoded runtime records back to source-level
// semantics.
struct StaticValueInfo {
  std::string valueKind;
  std::string dtype;
  std::string elementDtype;
  std::string shape;
  std::string stride;
  std::string layout;
  std::string encoding;
  std::string addrSpace;
  uint32_t rank = 0;
  uint32_t elementBits = 0;
  uint32_t vecWidth = 0;
};

struct OperandStaticInfo {
  uint32_t operandIndex = 0;
  std::string operandRole;
  uint32_t producerOpId = 0;
  bool isConstant = false;
  bool isPredicate = false;
  bool isKernelArgument = false;
  std::string constantValueRepr;
  StaticValueInfo value;
};

struct StatementValueInfo {
  std::string sourceName;
  std::string sourceRole;
  bool hasOperandIndex = false;
  uint32_t operandIndex = 0;
  uint32_t producerOpId = 0;
  uint32_t captureOpId = 0;
  std::string capturePolicy;
  bool isConstant = false;
  std::string constantValueRepr;
  StaticValueInfo value;
};

struct TrackedOpEntry {
  uint32_t opId = 0;
  uint32_t scopeId = 0;
  uint32_t resultIndex = 0;
  uint32_t statementId = 0;
  bool isSyntheticStatementCapture = false;
  bool isMemoryOp = false;
  std::string opCategory;
  std::string role;
  std::string mlirOpName;
  std::string sourceLoc;
  std::string tritonStatement;
  std::string statementResultName;
  std::string inlineCallPath;
  StaticValueInfo result;
  std::vector<OperandStaticInfo> operands;
  std::vector<StatementValueInfo> statementValues;
  std::string addrSpace;
  std::string accessType;
  uint32_t accessBytes = 0;
  uint32_t alignmentRequired = 0;
  bool hasMask = false;
  std::string maskDtype;
  std::string cacheModifier;
  std::string evictionPolicy;
  bool isVolatile = false;
  std::string boundaryCheckPolicy;
  std::string paddingSemantics;
};

using TrackedOpTable = std::vector<TrackedOpEntry>;

struct KernelDebugMetadata {
  uint32_t debugKernelId = 0;
  std::string kernelName;
  std::string backendName;
  std::string targetName;
  uint32_t scopeCount = 0;
  uint32_t trackedOpCount = 0;
  TrackedOpTable trackedOps;
};

// Contract: `debugKernelId` must match `BufferMeta.kernelId` at runtime.

const TrackedOpEntry *findTrackedOp(const TrackedOpTable &table, uint32_t opId);
TrackedOpEntry *findTrackedOp(TrackedOpTable &table, uint32_t opId);

std::string serializeTrackedOpTableToJson(const TrackedOpTable &table);
bool parseTrackedOpTableFromJson(std::string_view text, TrackedOpTable &table,
                                 std::string *errorMessage = nullptr);

std::string
serializeKernelDebugMetadataToJson(const KernelDebugMetadata &metadata);
bool parseKernelDebugMetadataFromJson(std::string_view text,
                                      KernelDebugMetadata &metadata,
                                      std::string *errorMessage = nullptr);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
