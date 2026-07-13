#include "Debugger/Decode/Reporter.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace mlir::flagtree::debugger;

namespace {

StaticValueInfo scalarF32() {
  StaticValueInfo value;
  value.valueKind = "scalar";
  value.dtype = "f32";
  value.elementDtype = "f32";
  value.shape = "[]";
  value.stride = "unknown";
  value.layout = "unknown";
  value.elementBits = 32;
  value.vecWidth = 1;
  return value;
}

TrackedOpEntry makeOp(uint32_t opId, std::string name) {
  TrackedOpEntry entry;
  entry.opId = opId;
  entry.scopeId = 1;
  entry.resultIndex = 0;
  entry.statementId = 1;
  entry.mlirOpName =
      opId == 1 ? "arith.addf" : "flagtree.debug.operand_capture";
  entry.sourceLoc = "loc(\"third_party/Debugger/test/lit/"
                    "statement-operand-capture-example.mlir\":4:10)";
  entry.tritonStatement = "y = a + b";
  entry.statementResultName = std::move(name);
  entry.result = scalarF32();
  if (opId != 1) {
    entry.isSyntheticStatementCapture = true;
    entry.opCategory = "statement_operand_capture";
    entry.role = "operand";
  }
  return entry;
}

StatementValueInfo makeStatementValue(std::string role, std::string name,
                                      uint32_t captureOpId, std::string policy,
                                      uint32_t operandIndex = 0,
                                      bool hasOperandIndex = false) {
  StatementValueInfo value;
  value.sourceRole = std::move(role);
  value.sourceName = std::move(name);
  value.captureOpId = captureOpId;
  value.capturePolicy = std::move(policy);
  value.hasOperandIndex = hasOperandIndex;
  value.operandIndex = operandIndex;
  value.value = scalarF32();
  return value;
}

DecodedSummaryCountBundleRecord countRecord(uint32_t opId) {
  DecodedSummaryCountBundleRecord record;
  record.raw.header.recordKind = RecordKind::SUMMARY_COUNT_BUNDLE_U64;
  record.raw.header.opId = opId;
  record.raw.header.logicalInstanceId = 0;
  record.raw.nanCount = 0;
  record.raw.infCount = 0;
  record.raw.zeroCount = 0;
  record.raw.elementCount = 1;
  return record;
}

DecodedSummaryValueBundleRecord valueRecord(uint32_t opId, float value) {
  DecodedSummaryValueBundleRecord record;
  record.raw.header.recordKind = RecordKind::SUMMARY_VALUE_BUNDLE_F32;
  record.raw.header.opId = opId;
  record.raw.header.logicalInstanceId = 0;
  record.raw.meanFinite = value;
  record.raw.minFinite = value;
  record.raw.maxFinite = value;
  record.raw.l2Norm = value;
  return record;
}

} // namespace

int main(int argc, char **argv) {
  KernelDebugMetadata metadata;
  metadata.debugKernelId = 2534481613u;
  metadata.kernelName = "statement_operand_capture";
  metadata.scopeCount = 1;

  TrackedOpEntry anchor = makeOp(1, "y");
  anchor.statementValues.push_back(
      makeStatementValue("result", "y", 1, "captured_current_op"));
  anchor.statementValues.push_back(makeStatementValue(
      "operand", "a", 2, "captured_at_current_statement", 0, true));
  anchor.statementValues.push_back(makeStatementValue(
      "operand", "b", 3, "captured_at_current_statement", 1, true));

  metadata.trackedOps.push_back(anchor);
  metadata.trackedOps.push_back(makeOp(2, "a"));
  metadata.trackedOps.push_back(makeOp(3, "b"));
  metadata.trackedOpCount = metadata.trackedOps.size();

  DecodedDebugRun run;
  run.meta.runId = 1;
  run.meta.kernelId = metadata.debugKernelId;
  run.meta.protocolVer = kProtocolVersion;
  run.meta.recordLevel = RecordLevel::LEVEL_SUMMARY;
  run.meta.exportMode = ExportMode::POST_KERNEL_EXPORT;
  run.meta.backendKind = BackendKind::CANN;
  run.header.writeIdx = 6;
  run.header.capacity = 64;
  run.header.recordSize = kBundleRecordSize;

  run.records.push_back(countRecord(2));
  run.records.push_back(valueRecord(2, 1.0f));
  run.records.push_back(countRecord(3));
  run.records.push_back(valueRecord(3, 2.0f));
  run.records.push_back(countRecord(1));
  run.records.push_back(valueRecord(1, 3.0f));

  std::string report = renderTextReport(run, metadata);
  if (argc > 1) {
    std::ofstream out(argv[1]);
    out << report;
  }
  std::cout << report;
  return 0;
}
