#pragma once

#include "Debugger/Common/Protocol.h"
#include "Debugger/Runtime/TransferEngine.h"

#include <string>
#include <variant>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {

// Module D runtime decode view.
// The decoder consumes raw bytes exported by F and converts them into strongly
// typed protocol records. Reporter logic can then join those records with B's
// TrackedOpTable.
struct DecodedSummaryRecord {
  SummaryRecord raw{};
};

struct DecodedSummaryCountBundleRecord {
  SummaryCountBundleRecord raw{};
};

struct DecodedSummaryValueBundleRecord {
  SummaryValueBundleRecord raw{};
};

struct DecodedMemoryEventRecord {
  MemoryEventRecord raw{};
};

struct DecodedFullValueRefRecord {
  FullValueRefRecord raw{};
};

struct DecodedTimelineRecord {
  TimelineRecord raw{};
};

using DecodedRecord =
    std::variant<DecodedSummaryRecord, DecodedSummaryCountBundleRecord,
                 DecodedSummaryValueBundleRecord, DecodedMemoryEventRecord,
                 DecodedFullValueRefRecord, DecodedTimelineRecord>;

struct DecodedDebugRun {
  BufferMeta meta{};
  RingBufferHeader header{};
  DebugRuntimeMetadata runtimeMetadata{};
  std::vector<DecodedRecord> records;
};

bool decodeExportedRun(const DebugExportedRun &run, DecodedDebugRun &decoded,
                       std::string *errorMessage = nullptr);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
