#pragma once

#include "Debugger/Common/Protocol.h"

namespace mlir {
namespace flagtree {
namespace debugger {

// Module C record-construction helpers.
// The structs in Protocol.h are the stable ABI. These helpers exist so C can
// centralize how a protocol record is populated before writing it into F's
// ring buffer.
SummaryRecord buildSummaryU64Record(uint32_t opId, uint64_t logicalInstanceId,
                                    CollectorKind kind, uint64_t value);
SummaryRecord buildSummaryF32Record(uint32_t opId, uint64_t logicalInstanceId,
                                    CollectorKind kind, float value);
SummaryRecord buildSummaryF64Record(uint32_t opId, uint64_t logicalInstanceId,
                                    CollectorKind kind, double value);

MemoryEventRecord buildMemoryEventRecord(uint32_t opId,
                                         uint64_t logicalInstanceId,
                                         uint64_t addr, MemoryEventKind kind,
                                         uint32_t ext0 = 0);

FullValueRefRecord buildFullValueRefRecord(uint32_t opId,
                                           uint64_t logicalInstanceId,
                                           uint32_t payloadOffset,
                                           uint32_t payloadLength);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
