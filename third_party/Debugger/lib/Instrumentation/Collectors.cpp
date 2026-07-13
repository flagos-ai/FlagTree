#include "Debugger/Instrumentation/Collectors.h"

#include "Debugger/Instrumentation/RecordBuilder.h"

#include <cmath>
#include <limits>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

const std::vector<SummaryCollectorSpec> kSummaryCollectorSpecs = {
    {CollectorKind::NAN_COUNT, "nan_count", true, true},
    {CollectorKind::INF_COUNT, "inf_count", true, true},
    {CollectorKind::ZERO_COUNT, "zero_count", true, true},
    {CollectorKind::MEAN_FINITE, "mean_finite", true, true},
    {CollectorKind::MIN_FINITE, "min_finite", true, true},
    {CollectorKind::MAX_FINITE, "max_finite", true, true},
    {CollectorKind::L2_NORM, "l2_norm", true, true},
    {CollectorKind::ELEMENT_COUNT, "element_count", true, true},
};

} // namespace

const std::vector<SummaryCollectorSpec> &getSummaryCollectorSpecs() {
  return kSummaryCollectorSpecs;
}

std::string_view getCollectorName(CollectorKind kind) {
  for (const auto &spec : getSummaryCollectorSpecs()) {
    if (spec.kind == kind) {
      return spec.name;
    }
  }
  return {};
}

std::vector<CollectorKind> getEnabledCollectors(RecordLevel level) {
  std::vector<CollectorKind> collectors;
  for (const auto &spec : getSummaryCollectorSpecs()) {
    if (isCollectorEnabledAtLevel(spec.kind, level)) {
      collectors.push_back(spec.kind);
    }
  }
  return collectors;
}

bool isCollectorEnabledAtLevel(CollectorKind kind, RecordLevel level) {
  switch (level) {
  case RecordLevel::LEVEL_SUMMARY:
  case RecordLevel::LEVEL_TENSOR_FULL:
    return isKnownCollector(kind);
  }
  return false;
}

bool isKnownCollector(CollectorKind kind) {
  for (const auto &spec : getSummaryCollectorSpecs()) {
    if (spec.kind == kind) {
      return true;
    }
  }
  return false;
}

// ─── Host-side summary computation ──────────────────────────────────────────

SummaryStats computeSummaryStatsF32(const float *data, size_t count) {
  SummaryStats stats;
  stats.elementCount = static_cast<uint64_t>(count);
  if (count == 0)
    return stats;

  double sum = 0.0;
  double squareSum = 0.0;
  double minVal = std::numeric_limits<double>::infinity();
  double maxVal = -std::numeric_limits<double>::infinity();
  uint64_t finiteCount = 0;

  for (size_t i = 0; i < count; ++i) {
    const float v = data[i];
    if (std::isnan(v)) {
      ++stats.nanCount;
    } else if (std::isinf(v)) {
      ++stats.infCount;
    } else {
      const double dv = static_cast<double>(v);
      if (v == 0.0f)
        ++stats.zeroCount;
      sum += dv;
      squareSum += dv * dv;
      if (dv < minVal)
        minVal = dv;
      if (dv > maxVal)
        maxVal = dv;
      ++finiteCount;
    }
  }

  if (finiteCount > 0) {
    stats.mean = sum / static_cast<double>(finiteCount);
    stats.min = minVal;
    stats.max = maxVal;
    stats.l2Norm = std::sqrt(squareSum);
  }
  return stats;
}

SummaryStats computeSummaryStatsF64(const double *data, size_t count) {
  SummaryStats stats;
  stats.elementCount = static_cast<uint64_t>(count);
  if (count == 0)
    return stats;

  double sum = 0.0;
  double squareSum = 0.0;
  double minVal = std::numeric_limits<double>::infinity();
  double maxVal = -std::numeric_limits<double>::infinity();
  uint64_t finiteCount = 0;

  for (size_t i = 0; i < count; ++i) {
    const double v = data[i];
    if (std::isnan(v)) {
      ++stats.nanCount;
    } else if (std::isinf(v)) {
      ++stats.infCount;
    } else {
      if (v == 0.0)
        ++stats.zeroCount;
      sum += v;
      squareSum += v * v;
      if (v < minVal)
        minVal = v;
      if (v > maxVal)
        maxVal = v;
      ++finiteCount;
    }
  }

  if (finiteCount > 0) {
    stats.mean = sum / static_cast<double>(finiteCount);
    stats.min = minVal;
    stats.max = maxVal;
    stats.l2Norm = std::sqrt(squareSum);
  }
  return stats;
}

void writeSummaryRecordsToSink(uint32_t opId, uint64_t logicalInstanceId,
                               const SummaryStats &stats, RecordLevel level,
                               RecordSink &sink) {
  for (const CollectorKind kind : getEnabledCollectors(level)) {
    SummaryRecord record{};
    switch (kind) {
    case CollectorKind::NAN_COUNT:
      record =
          buildSummaryU64Record(opId, logicalInstanceId, kind, stats.nanCount);
      break;
    case CollectorKind::INF_COUNT:
      record =
          buildSummaryU64Record(opId, logicalInstanceId, kind, stats.infCount);
      break;
    case CollectorKind::ZERO_COUNT:
      record =
          buildSummaryU64Record(opId, logicalInstanceId, kind, stats.zeroCount);
      break;
    case CollectorKind::MEAN_FINITE:
      record = buildSummaryF64Record(opId, logicalInstanceId, kind, stats.mean);
      break;
    case CollectorKind::MIN_FINITE:
      record = buildSummaryF64Record(opId, logicalInstanceId, kind, stats.min);
      break;
    case CollectorKind::MAX_FINITE:
      record = buildSummaryF64Record(opId, logicalInstanceId, kind, stats.max);
      break;
    case CollectorKind::L2_NORM:
      record =
          buildSummaryF64Record(opId, logicalInstanceId, kind, stats.l2Norm);
      break;
    case CollectorKind::ELEMENT_COUNT:
      record = buildSummaryU64Record(opId, logicalInstanceId, kind,
                                     stats.elementCount);
      break;
    }
    sink.writeSummary(record);
  }
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
