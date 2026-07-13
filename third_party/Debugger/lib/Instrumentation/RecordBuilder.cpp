#include "Debugger/Instrumentation/RecordBuilder.h"

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

RecordHeader buildHeader(RecordKind recordKind, uint32_t opId,
                         uint64_t logicalInstanceId) {
  RecordHeader header{};
  header.recordKind = recordKind;
  header.reserved0 = 0;
  header.opId = opId;
  header.logicalInstanceId = logicalInstanceId;
  return header;
}

SummaryRecord buildSummaryRecord(uint32_t opId, uint64_t logicalInstanceId,
                                 CollectorKind kind, ResultType resultType) {
  SummaryRecord record{};
  record.header = buildHeader(RecordKind::SUMMARY, opId, logicalInstanceId);
  record.collectorKind = kind;
  record.resultType = resultType;
  record.reserved1 = 0;
  return record;
}

} // namespace

SummaryRecord buildSummaryU64Record(uint32_t opId, uint64_t logicalInstanceId,
                                    CollectorKind kind, uint64_t value) {
  SummaryRecord record =
      buildSummaryRecord(opId, logicalInstanceId, kind, ResultType::U64);
  record.resultData.u64Val = value;
  return record;
}

SummaryRecord buildSummaryF32Record(uint32_t opId, uint64_t logicalInstanceId,
                                    CollectorKind kind, float value) {
  SummaryRecord record =
      buildSummaryRecord(opId, logicalInstanceId, kind, ResultType::F32);
  record.resultData.f32Val = value;
  return record;
}

SummaryRecord buildSummaryF64Record(uint32_t opId, uint64_t logicalInstanceId,
                                    CollectorKind kind, double value) {
  SummaryRecord record =
      buildSummaryRecord(opId, logicalInstanceId, kind, ResultType::F64);
  record.resultData.f64Val = value;
  return record;
}

MemoryEventRecord buildMemoryEventRecord(uint32_t opId,
                                         uint64_t logicalInstanceId,
                                         uint64_t addr, MemoryEventKind kind,
                                         uint32_t ext0) {
  MemoryEventRecord record{};
  record.header =
      buildHeader(RecordKind::MEMORY_EVENT, opId, logicalInstanceId);
  record.addr = addr;
  record.eventKind = kind;
  record.reserved1 = 0;
  record.ext0 = ext0;
  return record;
}

FullValueRefRecord buildFullValueRefRecord(uint32_t opId,
                                           uint64_t logicalInstanceId,
                                           uint32_t payloadOffset,
                                           uint32_t payloadLength) {
  FullValueRefRecord record{};
  record.header = buildHeader(RecordKind::FULL_VALUE, opId, logicalInstanceId);
  record.payloadOffset = payloadOffset;
  record.payloadLength = payloadLength;
  record.reserved1 = 0;
  return record;
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
