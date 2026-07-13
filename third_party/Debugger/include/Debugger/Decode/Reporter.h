#pragma once

#include "Debugger/Decode/Decoder.h"
#include "Debugger/Metadata/TrackedOpTable.h"

#include <string>

namespace mlir {
namespace flagtree {
namespace debugger {

struct ReportOptions {
  bool includeDynamicRecords = true;
  bool includeStaticMetadata = true;
  bool includeStaticOpCatalog = false;
  bool includeAggregates = false;
  bool includeStatementRecords = true;
  bool includeOpLog = true;
  bool includeReportHeader = true;
};

// Module D report entry point.
// This layer owns the final human-readable presentation after runtime records
// have been decoded and joined with B's static metadata.
std::string renderTextReport(const DecodedDebugRun &run,
                             const KernelDebugMetadata &metadata,
                             const ReportOptions &options = {});

// Machine-readable report with the same op/instance grouping as the text
// report. Metric arrays are aligned with the per-op `instances` array.
std::string renderJsonReport(const DecodedDebugRun &run,
                             const KernelDebugMetadata &metadata,
                             const ReportOptions &options = {});

} // namespace debugger
} // namespace flagtree
} // namespace mlir
