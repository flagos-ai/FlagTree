#pragma once

#include "Debugger/Common/Protocol.h"
#include "Debugger/Instrumentation/Writer.h"

#include <cstddef>
#include <string_view>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {

// ─── Phase-1 summary collector spec ─────────────────────────────────────────
// Static descriptor table for all summary collectors C is responsible for.
struct SummaryCollectorSpec {
  CollectorKind kind;
  std::string_view name;
  bool enabledByDefault;
  bool phase1Core;
};

const std::vector<SummaryCollectorSpec> &getSummaryCollectorSpecs();
std::string_view getCollectorName(CollectorKind kind);
std::vector<CollectorKind> getEnabledCollectors(RecordLevel level);
bool isCollectorEnabledAtLevel(CollectorKind kind, RecordLevel level);
bool isKnownCollector(CollectorKind kind);

// ─── Host-side summary computation ──────────────────────────────────────────
// Used by C-module unit tests to verify summary record fields without running
// on GPU.  The device-side MLIR lowering is expected to produce the same
// numeric results for well-defined (finite) inputs.
//
// Metrics are computed over *finite* values where specified:
//   - nanCount      : number of NaN elements
//   - infCount      : number of ±Inf elements
//   - zeroCount     : number of zero finite elements, including -0.0
//   - mean          : arithmetic mean of finite elements (0.0 if none)
//   - min           : minimum of finite elements (0.0 if none)
//   - max           : maximum of finite elements (0.0 if none)
//   - l2Norm        : sqrt(sum(x*x)) over finite elements
//   - elementCount  : total number of elements (including NaN and Inf)
struct SummaryStats {
  uint64_t nanCount = 0;
  uint64_t infCount = 0;
  uint64_t zeroCount = 0;
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double l2Norm = 0.0;
  uint64_t elementCount = 0;
};

SummaryStats computeSummaryStatsF32(const float *data, size_t count);
SummaryStats computeSummaryStatsF64(const double *data, size_t count);

// Write one SummaryRecord per enabled collector (determined by `level`) into
// `sink`.  This mirrors what the instrumented GPU kernel does per tracked-op
// instance: one call per op per CTA-level execution.
//
// Standalone usage:
//   SummaryStats stats = computeSummaryStatsF32(data, n);
//   writeSummaryRecordsToSink(opId, instanceId, stats,
//                             RecordLevel::LEVEL_SUMMARY, *mySink);
void writeSummaryRecordsToSink(uint32_t opId, uint64_t logicalInstanceId,
                               const SummaryStats &stats, RecordLevel level,
                               RecordSink &sink);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
