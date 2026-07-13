#include "Debugger/Decode/Reporter.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

template <typename EnumT> uint64_t enumNumber(EnumT value) {
  using UnderlyingT = typename std::underlying_type<EnumT>::type;
  return static_cast<uint64_t>(static_cast<UnderlyingT>(value));
}

template <typename EnumT>
std::string unknownEnum(const char *name, EnumT value) {
  std::ostringstream os;
  os << name << "(" << enumNumber(value) << ")";
  return os.str();
}

std::string toString(RecordLevel value) {
  switch (value) {
  case RecordLevel::LEVEL_SUMMARY:
    return "LEVEL_SUMMARY";
  case RecordLevel::LEVEL_TENSOR_FULL:
    return "LEVEL_TENSOR_FULL";
  }
  return unknownEnum("RecordLevel", value);
}

std::string toString(ExportMode value) {
  switch (value) {
  case ExportMode::POST_KERNEL_EXPORT:
    return "POST_KERNEL_EXPORT";
  case ExportMode::STREAMING_EXPORT:
    return "STREAMING_EXPORT";
  }
  return unknownEnum("ExportMode", value);
}

std::string toString(BackendKind value) {
  switch (value) {
  case BackendKind::UNKNOWN:
    return "UNKNOWN";
  case BackendKind::CUDA:
    return "CUDA";
  case BackendKind::HIP:
    return "HIP";
  case BackendKind::MUSA:
    return "MUSA";
  case BackendKind::CANN:
    return "CANN";
  }
  return unknownEnum("BackendKind", value);
}

std::string toMetricName(CollectorKind value) {
  switch (value) {
  case CollectorKind::NAN_COUNT:
    return "nan_count";
  case CollectorKind::INF_COUNT:
    return "inf_count";
  case CollectorKind::ZERO_COUNT:
    return "zero_count";
  case CollectorKind::MEAN_FINITE:
    return "mean";
  case CollectorKind::MIN_FINITE:
    return "min";
  case CollectorKind::MAX_FINITE:
    return "max";
  case CollectorKind::L2_NORM:
    return "l2_norm";
  case CollectorKind::ELEMENT_COUNT:
    return "element_count";
  }
  return unknownEnum("CollectorKind", value);
}

std::string toString(ResultType value) {
  switch (value) {
  case ResultType::U64:
    return "U64";
  case ResultType::F32:
    return "F32";
  case ResultType::F64:
    return "F64";
  }
  return unknownEnum("ResultType", value);
}

std::string toString(MemoryEventKind value) {
  switch (value) {
  case MemoryEventKind::LAST_ALIGNED_ADDR:
    return "LAST_ALIGNED_ADDR";
  case MemoryEventKind::BASE_ALIGNED_ADDR:
    return "BASE_ALIGNED_ADDR";
  case MemoryEventKind::FIRST_ADDR:
    return "FIRST_ADDR";
  case MemoryEventKind::LAST_ADDR:
    return "LAST_ADDR";
  case MemoryEventKind::MIN_ADDR:
    return "MIN_ADDR";
  case MemoryEventKind::MAX_ADDR:
    return "MAX_ADDR";
  case MemoryEventKind::ACTIVE_LANE_COUNT:
    return "ACTIVE_LANE_COUNT";
  case MemoryEventKind::ADDRESS_SPAN_BYTES:
    return "ADDRESS_SPAN_BYTES";
  }
  return unknownEnum("MemoryEventKind", value);
}

bool isAddressSummaryMemoryEventKind(MemoryEventKind value) {
  switch (value) {
  case MemoryEventKind::FIRST_ADDR:
  case MemoryEventKind::LAST_ADDR:
  case MemoryEventKind::MIN_ADDR:
  case MemoryEventKind::MAX_ADDR:
  case MemoryEventKind::ACTIVE_LANE_COUNT:
  case MemoryEventKind::ADDRESS_SPAN_BYTES:
    return true;
  default:
    return false;
  }
}

const char *addressSummaryFieldName(MemoryEventKind value) {
  switch (value) {
  case MemoryEventKind::FIRST_ADDR:
    return "first_addr";
  case MemoryEventKind::LAST_ADDR:
    return "last_addr";
  case MemoryEventKind::MIN_ADDR:
    return "min_addr";
  case MemoryEventKind::MAX_ADDR:
    return "max_addr";
  case MemoryEventKind::ACTIVE_LANE_COUNT:
    return "active_lane_count";
  case MemoryEventKind::ADDRESS_SPAN_BYTES:
    return "address_span_bytes";
  default:
    return "unknown";
  }
}

std::string boolString(bool value) { return value ? "true" : "false"; }

std::string hexValue(uint64_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << std::nouppercase << value;
  return os.str();
}

std::string stringOrNA(const std::string &value) {
  return value.empty() ? "<none>" : value;
}

std::string joinInt64Vector(const std::vector<int64_t> &values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

std::string formatSummaryValue(const SummaryRecord &record) {
  std::ostringstream os;
  switch (record.resultType) {
  case ResultType::U64:
    os << record.resultData.u64Val;
    break;
  case ResultType::F32:
    os << record.resultData.f32Val;
    break;
  case ResultType::F64:
    os << record.resultData.f64Val;
    break;
  default:
    os << "<unknown-result-type>";
    break;
  }
  return os.str();
}

uint32_t getRecordOpId(const DecodedRecord &record) {
  return std::visit([](const auto &decoded) { return decoded.raw.header.opId; },
                    record);
}

uint64_t getRecordLogicalInstanceId(const DecodedRecord &record) {
  return std::visit(
      [](const auto &decoded) { return decoded.raw.header.logicalInstanceId; },
      record);
}

const BufferRegistrationInfo *
findContainingBuffer(const DebugRuntimeMetadata &metadata, uint64_t addr) {
  for (const auto &buffer : metadata.buffers) {
    if (addr >= buffer.baseAddress &&
        addr - buffer.baseAddress < buffer.sizeBytes) {
      return &buffer;
    }
  }
  return nullptr;
}

std::string formatDtypeInBlock(const TrackedOpEntry &entry) {
  if (entry.operands.empty()) {
    return "<none>";
  }

  std::ostringstream os;
  for (size_t i = 0; i < entry.operands.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << "arg" << entry.operands[i].operandIndex << "="
       << stringOrNA(entry.operands[i].value.dtype);
  }
  return os.str();
}

void renderTrackedOpBlock(std::ostringstream &os,
                          const TrackedOpEntry *trackedOp) {
  if (!trackedOp) {
    os << "  static: <missing>\n";
    return;
  }

  os << "  static:\n";
  os << "    mlir_op: " << stringOrNA(trackedOp->mlirOpName) << "\n";
  if (trackedOp->isMemoryOp) {
    os << "    role: " << stringOrNA(trackedOp->role) << "\n";
    os << "    category: " << stringOrNA(trackedOp->opCategory) << "\n";
  }
  os << "    source_loc: " << stringOrNA(trackedOp->sourceLoc) << "\n";
  os << "    triton_statement: " << stringOrNA(trackedOp->tritonStatement)
     << "\n";
  os << "    dtype_in: " << formatDtypeInBlock(*trackedOp) << "\n";
  os << "    dtype_out: " << stringOrNA(trackedOp->result.dtype) << "\n";
  os << "    shape: " << stringOrNA(trackedOp->result.shape) << "\n";
  os << "    stride: " << stringOrNA(trackedOp->result.stride) << "\n";
  os << "    layout: " << stringOrNA(trackedOp->result.layout) << "\n";
  if (trackedOp->isMemoryOp) {
    os << "    memory_semantics: addr_space="
       << stringOrNA(trackedOp->addrSpace)
       << " access_type=" << stringOrNA(trackedOp->accessType)
       << " access_bytes=" << trackedOp->accessBytes
       << " alignment_required=" << trackedOp->alignmentRequired
       << " has_mask=" << boolString(trackedOp->hasMask)
       << " boundary_check_policy="
       << stringOrNA(trackedOp->boundaryCheckPolicy) << "\n";
  }
}

void renderRuntimeAddressContext(std::ostringstream &os,
                                 const DecodedDebugRun &run,
                                 const TrackedOpEntry *trackedOp, uint64_t addr,
                                 const char *indent = "  ") {
  const auto *buffer = findContainingBuffer(run.runtimeMetadata, addr);
  if (buffer) {
    os << indent << "runtime_address: bufferId=" << buffer->bufferId
       << " name=" << stringOrNA(buffer->bufferName) << " range=["
       << hexValue(buffer->baseAddress) << ","
       << hexValue(buffer->baseAddress + buffer->sizeBytes) << ")"
       << " sizeBytes=" << buffer->sizeBytes
       << " offset=" << (addr - buffer->baseAddress) << "\n";
  } else {
    os << indent
       << "runtime_address: bufferId=not captured offset=not captured "
          "range=not captured\n";
  }

  if (trackedOp && trackedOp->alignmentRequired != 0) {
    os << indent << "alignment_ok="
       << boolString(addr % trackedOp->alignmentRequired == 0) << "\n";
  } else {
    os << indent << "alignment_ok=not captured\n";
  }
  os << indent << "local_address_snapshot=not captured\n";
}

struct SummaryMetricValue {
  ResultType resultType = ResultType::U64;
  std::string resultTypeName;
  std::string value;
  uint64_t u64Value = 0;
  float f32Value = 0.0f;
  double f64Value = 0.0;
};

struct MemoryEventView {
  MemoryEventKind kind = MemoryEventKind::LAST_ALIGNED_ADDR;
  uint64_t addr = 0;
  uint32_t ext0 = 0;
};

struct FullValueRefView {
  uint32_t payloadOffset = 0;
  uint32_t payloadLength = 0;
  std::string kind;
  std::string path;
};

struct InstanceReportGroup {
  uint64_t logicalInstanceId = 0;
  std::map<std::string, SummaryMetricValue> summaryValues;
  std::vector<MemoryEventView> memoryEvents;
  std::vector<FullValueRefView> fullValueRefs;
};

struct OpReportGroup {
  uint32_t opId = 0;
  uint32_t scopeId = kInvalidScopeId;
  const TrackedOpEntry *trackedOp = nullptr;
  std::map<uint64_t, InstanceReportGroup> instances;
};

struct MemoryUseView {
  std::string addressRole;
  uint32_t addressExt0 = 0;
  bool hasAddressExt0 = false;
  const TrackedOpEntry *accessOp = nullptr;
  const OpReportGroup *runtime = nullptr;
};

struct StatementRenderIndex {
  std::map<uint32_t, std::string> resultLabelsByCaptureOp;
  std::map<uint32_t, std::vector<MemoryUseView>> memoryUsesByStatementOp;
};

const char *const kSummaryMetricOrder[] = {
    "element_count", "nan_count", "inf_count", "zero_count",
    "mean",          "min",       "max",       "l2_norm",
};

bool isOrderedSummaryMetric(const std::string &metric) {
  for (const char *ordered : kSummaryMetricOrder) {
    if (metric == ordered) {
      return true;
    }
  }
  return false;
}

SummaryMetricValue makeSummaryMetricValue(const SummaryRecord &record) {
  SummaryMetricValue value;
  value.resultType = record.resultType;
  value.resultTypeName = toString(record.resultType);
  value.value = formatSummaryValue(record);
  value.u64Value = record.resultData.u64Val;
  value.f32Value = record.resultData.f32Val;
  value.f64Value = record.resultData.f64Val;
  return value;
}

SummaryMetricValue makeU64SummaryMetricValue(uint64_t rawValue) {
  SummaryMetricValue value;
  value.resultType = ResultType::U64;
  value.resultTypeName = toString(ResultType::U64);
  value.value = std::to_string(rawValue);
  value.u64Value = rawValue;
  return value;
}

SummaryMetricValue makeF32SummaryMetricValue(float rawValue) {
  SummaryMetricValue value;
  value.resultType = ResultType::F32;
  value.resultTypeName = toString(ResultType::F32);
  std::ostringstream os;
  os << rawValue;
  value.value = os.str();
  value.f32Value = rawValue;
  return value;
}

void addCountBundleSummaryValues(
    std::map<std::string, SummaryMetricValue> &summaryValues,
    const SummaryCountBundleRecord &record) {
  summaryValues["nan_count"] = makeU64SummaryMetricValue(record.nanCount);
  summaryValues["inf_count"] = makeU64SummaryMetricValue(record.infCount);
  summaryValues["zero_count"] = makeU64SummaryMetricValue(record.zeroCount);
  summaryValues["element_count"] =
      makeU64SummaryMetricValue(record.elementCount);
}

void addValueBundleSummaryValues(
    std::map<std::string, SummaryMetricValue> &summaryValues,
    const SummaryValueBundleRecord &record) {
  summaryValues["mean"] = makeF32SummaryMetricValue(record.meanFinite);
  summaryValues["min"] = makeF32SummaryMetricValue(record.minFinite);
  summaryValues["max"] = makeF32SummaryMetricValue(record.maxFinite);
  summaryValues["l2_norm"] = makeF32SummaryMetricValue(record.l2Norm);
}

std::string formatSummaryCell(const SummaryMetricValue &value) {
  std::ostringstream os;
  os << value.value << " (" << value.resultTypeName << ")";
  return os.str();
}

std::string padRight(std::string value, size_t width) {
  if (value.size() < width) {
    value.append(width - value.size(), ' ');
  }
  return value;
}

std::string renderAlignedArray(const std::vector<std::string> &cells,
                               const std::vector<size_t> &widths) {
  std::ostringstream os;
  os << "[";
  const bool applyPadding = cells.size() > 1;
  for (size_t i = 0; i < cells.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    size_t width =
        applyPadding && i < widths.size() ? widths[i] : cells[i].size();
    os << padRight(cells[i], width);
  }
  os << "]";
  return os.str();
}

void updateColumnWidths(std::vector<size_t> &widths,
                        const std::vector<std::string> &cells) {
  if (widths.size() < cells.size()) {
    widths.resize(cells.size(), 0);
  }
  for (size_t i = 0; i < cells.size(); ++i) {
    widths[i] = std::max(widths[i], cells[i].size());
  }
}

std::vector<std::string> collectSummaryMetrics(
    const std::map<uint64_t, InstanceReportGroup> &instances) {
  std::vector<std::string> metrics;
  for (const char *metric : kSummaryMetricOrder) {
    for (const auto &instanceEntry : instances) {
      if (instanceEntry.second.summaryValues.count(metric) != 0) {
        metrics.push_back(metric);
        break;
      }
    }
  }

  for (const auto &instanceEntry : instances) {
    for (const auto &summaryEntry : instanceEntry.second.summaryValues) {
      if (isOrderedSummaryMetric(summaryEntry.first)) {
        continue;
      }
      if (std::find(metrics.begin(), metrics.end(), summaryEntry.first) ==
          metrics.end()) {
        metrics.push_back(summaryEntry.first);
      }
    }
  }
  return metrics;
}

const MemoryEventKind *addressSummaryKindsBegin() {
  static const MemoryEventKind orderedKinds[] = {
      MemoryEventKind::FIRST_ADDR,        MemoryEventKind::LAST_ADDR,
      MemoryEventKind::MIN_ADDR,          MemoryEventKind::MAX_ADDR,
      MemoryEventKind::ACTIVE_LANE_COUNT, MemoryEventKind::ADDRESS_SPAN_BYTES,
  };
  return orderedKinds;
}

const MemoryEventKind *addressSummaryKindsEnd() {
  return addressSummaryKindsBegin() + 6;
}

std::map<MemoryEventKind, const MemoryEventView *>
collectAddressSummary(const std::vector<MemoryEventView> &events,
                      std::optional<uint32_t> ext0Filter = std::nullopt) {
  std::map<MemoryEventKind, const MemoryEventView *> summary;
  for (const MemoryEventView &event : events) {
    if (ext0Filter && event.ext0 != *ext0Filter)
      continue;
    if (isAddressSummaryMemoryEventKind(event.kind)) {
      summary[event.kind] = &event;
    }
  }
  return summary;
}

bool isAddressSummaryComplete(
    const std::map<MemoryEventKind, const MemoryEventView *> &summary) {
  for (const MemoryEventKind *kind = addressSummaryKindsBegin();
       kind != addressSummaryKindsEnd(); ++kind) {
    if (summary.count(*kind) == 0) {
      return false;
    }
  }
  return true;
}

std::string addressSummaryStatus(
    const std::map<MemoryEventKind, const MemoryEventView *> &summary) {
  if (summary.empty()) {
    return "not captured";
  }
  return isAddressSummaryComplete(summary) ? "complete" : "partial";
}

std::string formatMemorySummaryCell(MemoryEventKind kind,
                                    const MemoryEventView *event) {
  if (!event) {
    return "not captured";
  }
  if (kind == MemoryEventKind::ACTIVE_LANE_COUNT ||
      kind == MemoryEventKind::ADDRESS_SPAN_BYTES) {
    return std::to_string(event->addr);
  }
  return hexValue(event->addr);
}

struct TextMetricRow {
  std::string label;
  std::vector<std::string> cells;
};

std::vector<TextMetricRow>
buildSummaryRows(const std::map<uint64_t, InstanceReportGroup> &instances) {
  std::vector<TextMetricRow> rows;
  for (const std::string &metric : collectSummaryMetrics(instances)) {
    TextMetricRow row;
    row.label = metric;
    for (const auto &instanceEntry : instances) {
      auto it = instanceEntry.second.summaryValues.find(metric);
      if (it == instanceEntry.second.summaryValues.end()) {
        row.cells.push_back("not captured");
      } else {
        row.cells.push_back(formatSummaryCell(it->second));
      }
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<TextMetricRow> buildAddressSummaryRows(
    const std::map<uint64_t, InstanceReportGroup> &instances,
    std::optional<uint32_t> ext0Filter = std::nullopt) {
  std::vector<std::map<MemoryEventKind, const MemoryEventView *>> summaries;
  summaries.reserve(instances.size());
  bool hasAnyAddressSummary = false;
  for (const auto &instanceEntry : instances) {
    summaries.push_back(
        collectAddressSummary(instanceEntry.second.memoryEvents, ext0Filter));
    hasAnyAddressSummary = hasAnyAddressSummary || !summaries.back().empty();
  }
  if (!hasAnyAddressSummary) {
    return {};
  }

  std::vector<TextMetricRow> rows;
  TextMetricRow statusRow;
  statusRow.label = "status";
  for (const auto &summary : summaries) {
    statusRow.cells.push_back(addressSummaryStatus(summary));
  }
  rows.push_back(std::move(statusRow));

  for (const MemoryEventKind *kind = addressSummaryKindsBegin();
       kind != addressSummaryKindsEnd(); ++kind) {
    TextMetricRow row;
    row.label = addressSummaryFieldName(*kind);
    for (const auto &summary : summaries) {
      auto it = summary.find(*kind);
      row.cells.push_back(formatMemorySummaryCell(
          *kind, it == summary.end() ? nullptr : it->second));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

bool memoryEventMatchesExt0(const MemoryEventView &event,
                            std::optional<uint32_t> ext0Filter) {
  return !ext0Filter || event.ext0 == *ext0Filter;
}

bool hasMemoryEvents(const OpReportGroup *opGroup,
                     std::optional<uint32_t> ext0Filter = std::nullopt) {
  if (!opGroup)
    return false;
  for (const auto &instanceEntry : opGroup->instances) {
    for (const MemoryEventView &event : instanceEntry.second.memoryEvents) {
      if (memoryEventMatchesExt0(event, ext0Filter))
        return true;
    }
  }
  return false;
}

size_t maxLabelWidth(const std::vector<TextMetricRow> &rows) {
  size_t width = 0;
  for (const auto &row : rows) {
    width = std::max(width, row.label.size());
  }
  return width;
}

bool hasFallbackMemoryEvents(
    const InstanceReportGroup &instance,
    std::optional<uint32_t> ext0Filter = std::nullopt) {
  for (const MemoryEventView &event : instance.memoryEvents) {
    if (!memoryEventMatchesExt0(event, ext0Filter))
      continue;
    if (!isAddressSummaryMemoryEventKind(event.kind)) {
      return true;
    }
  }
  return false;
}

void renderMetricRows(std::ostringstream &os, const char *sectionName,
                      const std::vector<TextMetricRow> &rows,
                      const std::vector<size_t> &columnWidths,
                      const char *sectionIndent, const char *rowIndent) {
  if (rows.empty()) {
    return;
  }
  os << sectionIndent << sectionName << ":\n";
  const size_t labelWidth = maxLabelWidth(rows);
  for (const auto &row : rows) {
    os << rowIndent << padRight(row.label, labelWidth) << ": "
       << renderAlignedArray(row.cells, columnWidths) << "\n";
  }
}

void renderMetricRows(std::ostringstream &os, const char *sectionName,
                      const std::vector<TextMetricRow> &rows,
                      const std::vector<size_t> &columnWidths) {
  renderMetricRows(os, sectionName, rows, columnWidths, "  ", "    ");
}

void renderFallbackMemoryEventsByInstance(
    std::ostringstream &os, const DecodedDebugRun &run,
    const TrackedOpEntry *trackedOp,
    const std::map<uint64_t, InstanceReportGroup> &instances,
    std::optional<uint32_t> ext0Filter = std::nullopt,
    const char *sectionIndent = "  ", const char *rowIndent = "    ") {
  bool hasAny = false;
  for (const auto &instanceEntry : instances) {
    hasAny =
        hasAny || hasFallbackMemoryEvents(instanceEntry.second, ext0Filter);
  }
  if (!hasAny) {
    return;
  }

  os << sectionIndent << "memory_events_by_instance:\n";
  for (const auto &instanceEntry : instances) {
    const auto &instance = instanceEntry.second;
    if (!hasFallbackMemoryEvents(instance, ext0Filter)) {
      continue;
    }
    os << rowIndent << "logical_instance_id=" << instance.logicalInstanceId
       << "\n";
    size_t eventIndex = 0;
    for (const MemoryEventView &event : instance.memoryEvents) {
      if (!memoryEventMatchesExt0(event, ext0Filter))
        continue;
      if (isAddressSummaryMemoryEventKind(event.kind)) {
        continue;
      }
      os << rowIndent << "  [" << eventIndex++
         << "] kind=" << toString(event.kind)
         << " addr=" << hexValue(event.addr) << " ext0=" << event.ext0 << "\n";
      std::string addressIndent = std::string(rowIndent) + "    ";
      renderRuntimeAddressContext(os, run, trackedOp, event.addr,
                                  addressIndent.c_str());
    }
  }
}

bool hasFullValueRefs(
    const std::map<uint64_t, InstanceReportGroup> &instances) {
  for (const auto &instanceEntry : instances) {
    if (!instanceEntry.second.fullValueRefs.empty())
      return true;
  }
  return false;
}

void renderFullValueRefs(
    std::ostringstream &os,
    const std::map<uint64_t, InstanceReportGroup> &instances,
    const char *indent, const char *heading) {
  if (!hasFullValueRefs(instances))
    return;

  const std::string instanceIndent = std::string(indent) + "  ";
  const std::string refIndent = std::string(indent) + "    ";
  os << indent << heading << ":\n";
  for (const auto &instanceEntry : instances) {
    const auto &instance = instanceEntry.second;
    if (instance.fullValueRefs.empty()) {
      continue;
    }
    os << instanceIndent << "logical_instance_id=" << instance.logicalInstanceId
       << "\n";
    for (size_t i = 0; i < instance.fullValueRefs.size(); ++i) {
      const FullValueRefView &ref = instance.fullValueRefs[i];
      os << refIndent << "[" << i << "]";
      if (!ref.kind.empty())
        os << " kind=" << ref.kind;
      if (!ref.path.empty()) {
        os << " path=" << ref.path << "\n";
      } else {
        os << " payload_offset=" << ref.payloadOffset
           << " payload_length=" << ref.payloadLength << " path=not exported\n";
      }
    }
  }
}

void renderFullValueRefsByInstance(
    std::ostringstream &os,
    const std::map<uint64_t, InstanceReportGroup> &instances) {
  renderFullValueRefs(os, instances, "  ", "full_value_refs_by_instance");
}

std::string artifactFileName(const FullValueRefView &ref) {
  if (ref.path.empty())
    return "";
  size_t pos = ref.path.find_last_of("/\\");
  return pos == std::string::npos ? ref.path : ref.path.substr(pos + 1);
}

void renderStatementArtifactFiles(
    std::ostringstream &os,
    const std::map<uint64_t, InstanceReportGroup> &instances,
    llvm::StringRef kind, llvm::StringRef label, const char *indent) {
  std::vector<std::string> files;
  for (const auto &instanceEntry : instances) {
    for (const FullValueRefView &ref : instanceEntry.second.fullValueRefs) {
      if (llvm::StringRef(ref.kind) != kind)
        continue;
      std::string file = artifactFileName(ref);
      if (!file.empty())
        files.push_back(std::move(file));
    }
  }
  if (files.empty())
    return;

  if (files.size() == 1) {
    os << indent << label.str() << ": " << files.front() << "\n";
    return;
  }

  os << indent << label.str() << "s:\n";
  const std::string fileIndent = std::string(indent) + "  ";
  for (size_t i = 0; i < files.size(); ++i)
    os << fileIndent << "[" << i << "] " << files[i] << "\n";
}

const FullDumpArtifactInfo *
findFullDumpArtifact(const DebugRuntimeMetadata &metadata, uint32_t opId,
                     uint64_t logicalInstanceId, uint32_t payloadOffset,
                     uint32_t payloadLength) {
  for (const FullDumpArtifactInfo &artifact : metadata.fullDumpArtifacts) {
    if (artifact.opId == opId &&
        artifact.logicalInstanceId == logicalInstanceId &&
        artifact.payloadOffset == payloadOffset &&
        artifact.payloadLength == payloadLength) {
      return &artifact;
    }
  }
  return nullptr;
}

std::map<uint32_t, OpReportGroup>
buildOpReportGroups(const DecodedDebugRun &run,
                    const KernelDebugMetadata &metadata) {
  std::map<uint32_t, OpReportGroup> byOp;

  for (const auto &record : run.records) {
    const uint32_t opId = getRecordOpId(record);
    const uint64_t instanceId = getRecordLogicalInstanceId(record);
    const auto *trackedOp = findTrackedOp(metadata.trackedOps, opId);
    auto &opGroup = byOp[opId];
    opGroup.opId = opId;
    opGroup.scopeId = trackedOp ? trackedOp->scopeId : kInvalidScopeId;
    opGroup.trackedOp = trackedOp;

    auto &instance = opGroup.instances[instanceId];
    instance.logicalInstanceId = instanceId;

    if (const auto *summary = std::get_if<DecodedSummaryRecord>(&record)) {
      instance.summaryValues[toMetricName(summary->raw.collectorKind)] =
          makeSummaryMetricValue(summary->raw);
    } else if (const auto *bundle =
                   std::get_if<DecodedSummaryCountBundleRecord>(&record)) {
      addCountBundleSummaryValues(instance.summaryValues, bundle->raw);
    } else if (const auto *bundle =
                   std::get_if<DecodedSummaryValueBundleRecord>(&record)) {
      addValueBundleSummaryValues(instance.summaryValues, bundle->raw);
    } else if (const auto *memory =
                   std::get_if<DecodedMemoryEventRecord>(&record)) {
      instance.memoryEvents.push_back({
          memory->raw.eventKind,
          memory->raw.addr,
          memory->raw.ext0,
      });
    } else if (const auto *full =
                   std::get_if<DecodedFullValueRefRecord>(&record)) {
      const FullDumpArtifactInfo *artifact = findFullDumpArtifact(
          run.runtimeMetadata, opId, instanceId, full->raw.payloadOffset,
          full->raw.payloadLength);
      instance.fullValueRefs.push_back({
          full->raw.payloadOffset,
          full->raw.payloadLength,
          artifact ? artifact->kind : std::string(),
          artifact ? artifact->path : std::string(),
      });
    }
  }
  return byOp;
}

void renderInstanceAlignedData(std::ostringstream &os,
                               const DecodedDebugRun &run,
                               const OpReportGroup &opGroup) {
  std::vector<std::string> instanceCells;
  for (const auto &instanceEntry : opGroup.instances) {
    instanceCells.push_back(
        std::to_string(instanceEntry.second.logicalInstanceId));
  }

  std::vector<TextMetricRow> summaryRows = buildSummaryRows(opGroup.instances);
  std::vector<TextMetricRow> addressRows =
      buildAddressSummaryRows(opGroup.instances);

  std::vector<size_t> columnWidths;
  updateColumnWidths(columnWidths, instanceCells);
  for (const auto &row : summaryRows) {
    updateColumnWidths(columnWidths, row.cells);
  }
  for (const auto &row : addressRows) {
    updateColumnWidths(columnWidths, row.cells);
  }

  os << "  instances: " << renderAlignedArray(instanceCells, columnWidths)
     << "\n";
  renderMetricRows(os, "summary", summaryRows, columnWidths);
  renderMetricRows(os, "address_summary", addressRows, columnWidths);
  renderFallbackMemoryEventsByInstance(os, run, opGroup.trackedOp,
                                       opGroup.instances);
  renderFullValueRefsByInstance(os, opGroup.instances);
}

bool isLoadAccessName(llvm::StringRef kind) {
  return kind == "load" || kind == "descriptor_load" ||
         kind == "experimental_descriptor_load" || kind == "tptr_load" ||
         kind == "tts_load";
}

bool isStoreAccessName(llvm::StringRef kind) {
  return kind == "store" || kind == "descriptor_store" ||
         kind == "experimental_descriptor_store" || kind == "tptr_store" ||
         kind == "tts_store";
}

std::string statementAddressSummaryLabel(const TrackedOpEntry *trackedOp) {
  if (!trackedOp)
    return "address_summary";
  if (isLoadAccessName(trackedOp->accessType))
    return "address_summary(load from)";
  if (isStoreAccessName(trackedOp->accessType))
    return "address_summary(store to)";
  return "address_summary";
}

bool hasStatementRuntimeDetails(const OpReportGroup *opGroup) {
  if (!opGroup || opGroup->instances.empty())
    return false;
  if (!collectSummaryMetrics(opGroup->instances).empty())
    return true;
  if (!buildAddressSummaryRows(opGroup->instances).empty())
    return true;
  for (const auto &instanceEntry : opGroup->instances) {
    if (hasFallbackMemoryEvents(instanceEntry.second))
      return true;
  }
  return false;
}

void renderStatementRuntime(std::ostringstream &os,
                            const OpReportGroup *opGroup, const char *indent,
                            bool includeArtifactFiles) {
  if (!opGroup || opGroup->instances.empty()) {
    os << indent << "runtime: not captured\n";
    return;
  }

  std::vector<std::string> instanceCells;
  for (const auto &instanceEntry : opGroup->instances) {
    instanceCells.push_back(
        std::to_string(instanceEntry.second.logicalInstanceId));
  }

  std::vector<TextMetricRow> summaryRows = buildSummaryRows(opGroup->instances);
  std::vector<TextMetricRow> addressRows =
      buildAddressSummaryRows(opGroup->instances);

  std::vector<size_t> columnWidths;
  updateColumnWidths(columnWidths, instanceCells);
  for (const auto &row : summaryRows)
    updateColumnWidths(columnWidths, row.cells);
  for (const auto &row : addressRows)
    updateColumnWidths(columnWidths, row.cells);

  os << indent
     << "instances: " << renderAlignedArray(instanceCells, columnWidths)
     << "\n";
  const std::string sectionIndent = std::string(indent);
  const std::string rowIndent = sectionIndent + "  ";
  renderMetricRows(os, "summary", summaryRows, columnWidths,
                   sectionIndent.c_str(), rowIndent.c_str());
  if (includeArtifactFiles && !summaryRows.empty())
    renderStatementArtifactFiles(os, opGroup->instances, "value",
                                 "full_value_file", sectionIndent.c_str());
  const std::string addressSummaryLabel =
      statementAddressSummaryLabel(opGroup->trackedOp);
  renderMetricRows(os, addressSummaryLabel.c_str(), addressRows, columnWidths,
                   sectionIndent.c_str(), rowIndent.c_str());
  if (includeArtifactFiles && !addressRows.empty())
    renderStatementArtifactFiles(os, opGroup->instances, "memory_address",
                                 "memory_address_file", sectionIndent.c_str());
}

bool hasStatementValues(const KernelDebugMetadata &metadata) {
  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    if (!entry.isSyntheticStatementCapture && !entry.statementValues.empty())
      return true;
  }
  return false;
}

bool hasTextContent(std::ostringstream &os) {
  return os.tellp() > std::ostringstream::pos_type(0);
}

void renderSectionHeading(std::ostringstream &os, const char *heading) {
  if (hasTextContent(os))
    os << "\n";
  os << heading << "\n";
}

bool isStatementResult(const StatementValueInfo &value) {
  return value.sourceRole == "result";
}

bool isStatementOperand(const StatementValueInfo &value) {
  return value.sourceRole == "operand";
}

std::map<std::string, uint32_t>
countStatementResultsByName(const KernelDebugMetadata &metadata) {
  std::map<std::string, uint32_t> resultCounts;
  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    if (entry.isSyntheticStatementCapture || entry.statementValues.empty())
      continue;
    for (const StatementValueInfo &value : entry.statementValues) {
      if (isStatementResult(value))
        ++resultCounts[stringOrNA(value.sourceName)];
    }
  }
  return resultCounts;
}

std::string formatResultOrdinal(uint32_t ordinal) {
  std::ostringstream os;
  os << std::setw(3) << std::setfill('0') << ordinal;
  return os.str();
}

std::string makeResultLabel(const StatementValueInfo &value,
                            const std::map<std::string, uint32_t> &counts,
                            std::map<std::string, uint32_t> &ordinals) {
  const std::string name = stringOrNA(value.sourceName);
  const uint32_t count = counts.count(name) == 0 ? 0 : counts.at(name);
  if (count > 1) {
    const uint32_t ordinal = ++ordinals[name];
    return "[result " + name + ":" + formatResultOrdinal(ordinal) + "]";
  }
  return "[result " + name + "]";
}

std::string makeOperandLabel(const StatementValueInfo &value) {
  return "<operand " + stringOrNA(value.sourceName) + ">";
}

std::string findResultReference(
    const StatementValueInfo &value,
    const std::map<uint32_t, std::string> &resultLabelsByCaptureOp) {
  if (value.producerOpId != 0) {
    auto producerIt = resultLabelsByCaptureOp.find(value.producerOpId);
    if (producerIt != resultLabelsByCaptureOp.end())
      return producerIt->second;
  }
  if (value.captureOpId != 0) {
    auto captureIt = resultLabelsByCaptureOp.find(value.captureOpId);
    if (captureIt != resultLabelsByCaptureOp.end())
      return captureIt->second;
  }
  return "";
}

const OpReportGroup *
findCaptureGroup(const StatementValueInfo &value,
                 const std::map<uint32_t, OpReportGroup> &byOp) {
  auto captureIt = byOp.find(value.captureOpId);
  return captureIt == byOp.end() ? nullptr : &captureIt->second;
}

bool shouldRenderStatementValue(const StatementValueInfo &value,
                                const std::map<uint32_t, OpReportGroup> &byOp,
                                const std::string &referenceLabel) {
  if (isStatementResult(value) || !referenceLabel.empty() || value.isConstant)
    return true;
  return hasStatementRuntimeDetails(findCaptureGroup(value, byOp));
}

std::string memoryAccessKind(const TrackedOpEntry &entry) {
  return entry.accessType.empty() ? "memory" : entry.accessType;
}

bool isLoadAccessKind(const std::string &kind) {
  return isLoadAccessName(kind);
}

bool isStatementLevelMemoryAccessKind(const std::string &kind) {
  return kind == "memref_copy" || kind == "tensormap_create" ||
         kind == "tensormap_fenceproxy_acquire" || kind == "async_copy" ||
         kind == "async_tma_copy";
}

struct MemoryUseTarget {
  uint32_t ext0 = 0;
  std::string role;
};

bool isMemoryAddressOperandRole(llvm::StringRef role) {
  return role == "ptr" || role == "descriptor" || role == "desc_ptr" ||
         role == "descriptor_base" || role == "global_address" ||
         role == "src" || role == "dst";
}

std::vector<MemoryUseTarget>
memoryUseTargetsForEntry(const TrackedOpEntry &entry) {
  std::vector<MemoryUseTarget> targets;
  const std::string accessKind = memoryAccessKind(entry);
  for (const OperandStaticInfo &operand : entry.operands) {
    if (isMemoryAddressOperandRole(operand.operandRole))
      targets.push_back(
          MemoryUseTarget{operand.operandIndex, operand.operandRole});
  }
  if (targets.empty())
    targets.push_back(MemoryUseTarget{0, ""});
  if (!isStatementLevelMemoryAccessKind(accessKind) && targets.size() > 1)
    targets.resize(1);
  return targets;
}

const std::vector<MemoryUseView> *
findStatementMemoryUses(uint32_t opId,
                        const StatementRenderIndex &renderIndex) {
  auto it = renderIndex.memoryUsesByStatementOp.find(opId);
  if (it == renderIndex.memoryUsesByStatementOp.end() || it->second.empty())
    return nullptr;
  return &it->second;
}

StatementRenderIndex
buildStatementRenderIndex(const KernelDebugMetadata &metadata,
                          const std::map<uint32_t, OpReportGroup> &byOp) {
  StatementRenderIndex index;
  std::map<std::string, uint32_t> resultCounts =
      countStatementResultsByName(metadata);
  std::map<std::string, uint32_t> resultOrdinals;

  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    if (entry.isSyntheticStatementCapture || entry.statementValues.empty())
      continue;
    for (const StatementValueInfo &value : entry.statementValues) {
      if (!isStatementResult(value) || value.captureOpId == 0)
        continue;
      index.resultLabelsByCaptureOp[value.captureOpId] =
          makeResultLabel(value, resultCounts, resultOrdinals);
    }
  }

  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    if (entry.isSyntheticStatementCapture || entry.statementValues.empty() ||
        !entry.isMemoryOp)
      continue;

    auto runtimeIt = byOp.find(entry.opId);
    const OpReportGroup *runtime =
        runtimeIt == byOp.end() ? nullptr : &runtimeIt->second;

    const std::string accessKind = memoryAccessKind(entry);
    bool addressAlreadyShownOnResult = false;
    if (isLoadAccessKind(accessKind)) {
      for (const StatementValueInfo &value : entry.statementValues) {
        if (isStatementResult(value) && value.captureOpId != 0 &&
            index.resultLabelsByCaptureOp.count(value.captureOpId) != 0) {
          addressAlreadyShownOnResult = true;
          break;
        }
      }
    }
    if (addressAlreadyShownOnResult)
      continue;

    for (const MemoryUseTarget &target : memoryUseTargetsForEntry(entry)) {
      const std::optional<uint32_t> ext0Filter = target.ext0;
      if (!hasMemoryEvents(runtime, ext0Filter))
        continue;

      MemoryUseView use;
      use.addressRole = target.role;
      use.addressExt0 = target.ext0;
      use.hasAddressExt0 = true;
      use.accessOp = &entry;
      use.runtime = runtime;

      index.memoryUsesByStatementOp[entry.opId].push_back(use);
    }
  }

  return index;
}

void renderMemoryAccess(std::ostringstream &os, const DecodedDebugRun &run,
                        const MemoryUseView &use, const char *indent) {
  const std::string rowIndent = std::string(indent) + "  ";

  if (!use.runtime || use.runtime->instances.empty()) {
    os << indent << "runtime: not captured\n";
    return;
  }

  const std::optional<uint32_t> ext0Filter =
      use.hasAddressExt0 ? std::optional<uint32_t>(use.addressExt0)
                         : std::nullopt;
  std::vector<TextMetricRow> addressRows =
      buildAddressSummaryRows(use.runtime->instances, ext0Filter);
  const std::string addressSummaryLabel =
      statementAddressSummaryLabel(use.accessOp);
  if (addressRows.empty()) {
    os << indent << addressSummaryLabel << ": not captured\n";
    renderFallbackMemoryEventsByInstance(os, run, use.accessOp,
                                         use.runtime->instances, ext0Filter,
                                         indent, rowIndent.c_str());
    return;
  }

  std::vector<std::string> instanceCells;
  for (const auto &instanceEntry : use.runtime->instances)
    instanceCells.push_back(
        std::to_string(instanceEntry.second.logicalInstanceId));

  std::vector<size_t> columnWidths;
  updateColumnWidths(columnWidths, instanceCells);
  for (const auto &row : addressRows)
    updateColumnWidths(columnWidths, row.cells);

  os << indent
     << "instances: " << renderAlignedArray(instanceCells, columnWidths)
     << "\n";
  renderMetricRows(os, addressSummaryLabel.c_str(), addressRows, columnWidths,
                   indent, rowIndent.c_str());
  renderFallbackMemoryEventsByInstance(os, run, use.accessOp,
                                       use.runtime->instances, ext0Filter,
                                       indent, rowIndent.c_str());
}

std::string memoryAccessLabel(const MemoryUseView &use) {
  if (!use.addressRole.empty())
    return use.addressRole;
  return "access";
}

void renderMemoryAccesses(std::ostringstream &os, const DecodedDebugRun &run,
                          const std::vector<MemoryUseView> &memoryAccesses,
                          const char *indent) {
  if (memoryAccesses.empty())
    return;
  const std::string useIndent = std::string(indent) + "  ";
  if (memoryAccesses.size() == 1) {
    os << indent << "memory_access:\n";
    renderMemoryAccess(os, run, memoryAccesses.front(), useIndent.c_str());
    return;
  }

  os << indent << "memory_accesses:\n";
  const std::string nestedIndent = useIndent + "  ";
  for (const MemoryUseView &use : memoryAccesses) {
    os << useIndent << memoryAccessLabel(use) << ":\n";
    renderMemoryAccess(os, run, use, nestedIndent.c_str());
  }
}

void renderStatementValue(std::ostringstream &os, const DecodedDebugRun &run,
                          const StatementValueInfo &value,
                          const std::map<uint32_t, OpReportGroup> &byOp,
                          const std::string &label,
                          const std::string &referenceLabel = "",
                          bool includeArtifactFiles = false) {
  os << "  " << label;
  if (!referenceLabel.empty()) {
    os << ": " << referenceLabel << "\n";
    return;
  }
  os << ":\n";
  if (value.isConstant)
    os << "    constant_value: " << stringOrNA(value.constantValueRepr) << "\n";

  const OpReportGroup *captureGroup = findCaptureGroup(value, byOp);
  if (value.isConstant && !hasStatementRuntimeDetails(captureGroup))
    return;
  renderStatementRuntime(os, captureGroup, "    ", includeArtifactFiles);
}

void renderStatementRecords(std::ostringstream &os, const DecodedDebugRun &run,
                            const KernelDebugMetadata &metadata) {
  if (!hasStatementValues(metadata))
    return;

  std::map<uint32_t, OpReportGroup> byOp = buildOpReportGroups(run, metadata);
  StatementRenderIndex renderIndex = buildStatementRenderIndex(metadata, byOp);

  renderSectionHeading(os, "Triton Statement Records");
  bool firstStatement = true;
  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    if (entry.isSyntheticStatementCapture || entry.statementValues.empty())
      continue;
    if (!firstStatement)
      os << "\n";
    firstStatement = false;
    os << "source_loc: " << stringOrNA(entry.sourceLoc) << "\n";
    os << "statement_id: " << entry.statementId << "\n";
    os << "statement: " << stringOrNA(entry.tritonStatement) << "\n";
    if (const std::vector<MemoryUseView> *statementMemoryUses =
            findStatementMemoryUses(entry.opId, renderIndex))
      renderMemoryAccesses(os, run, *statementMemoryUses, "  ");

    for (const StatementValueInfo &value : entry.statementValues) {
      if (isStatementResult(value)) {
        std::string resultLabel =
            "[result " + stringOrNA(value.sourceName) + "]";
        auto labelIt =
            renderIndex.resultLabelsByCaptureOp.find(value.captureOpId);
        if (labelIt != renderIndex.resultLabelsByCaptureOp.end())
          resultLabel = labelIt->second;
        renderStatementValue(os, run, value, byOp, resultLabel,
                             /*referenceLabel=*/"",
                             /*includeArtifactFiles=*/true);
        continue;
      }

      std::string operandLabel = isStatementOperand(value)
                                     ? makeOperandLabel(value)
                                     : stringOrNA(value.sourceRole) + " " +
                                           stringOrNA(value.sourceName);
      std::string referenceLabel =
          isStatementOperand(value)
              ? findResultReference(value, renderIndex.resultLabelsByCaptureOp)
              : "";
      if (!shouldRenderStatementValue(value, byOp, referenceLabel))
        continue;
      renderStatementValue(os, run, value, byOp, operandLabel, referenceLabel,
                           /*includeArtifactFiles=*/referenceLabel.empty());
    }
  }
}

void renderRecordsByOp(
    std::ostringstream &os, const DecodedDebugRun &run,
    const KernelDebugMetadata &metadata,
    const char *sectionName = "IR Op Log Records",
    const char *staticOnlySectionName = "IR Op Log Static Only Ops") {
  std::map<uint32_t, OpReportGroup> byOp = buildOpReportGroups(run, metadata);

  renderSectionHeading(os, sectionName);
  bool firstOp = true;
  for (const auto &entry : byOp) {
    if (!firstOp) {
      os << "\n";
    }
    firstOp = false;
    const auto &opGroup = entry.second;
    os << "op_id=" << opGroup.opId << " scope_id=" << opGroup.scopeId << "\n";
    renderTrackedOpBlock(os, opGroup.trackedOp);
    os << "\n";
    renderInstanceAlignedData(os, run, opGroup);
  }

  bool printedStaticOnlyHeader = false;
  for (const TrackedOpEntry &trackedOp : metadata.trackedOps) {
    if (byOp.count(trackedOp.opId) != 0)
      continue;
    if (!printedStaticOnlyHeader) {
      os << "\n" << staticOnlySectionName << "\n";
      printedStaticOnlyHeader = true;
    } else {
      os << "\n";
    }
    os << "op_id=" << trackedOp.opId << " scope_id=" << trackedOp.scopeId
       << "\n";
    os << "  dynamic_record_status: static_only\n";
    os << "  dynamic_record_note: no runtime record was emitted for this op; "
          "static metadata is kept for producer/context analysis\n";
    renderTrackedOpBlock(os, &trackedOp);
  }
}

void renderStaticMetadata(std::ostringstream &os,
                          const KernelDebugMetadata &metadata) {
  os << "\nStatic Op Catalog\n";
  os << "kernel_debug_id: " << metadata.debugKernelId << "\n";
  os << "kernel_name: " << stringOrNA(metadata.kernelName) << "\n";
  os << "backend_name: " << stringOrNA(metadata.backendName) << "\n";
  os << "target_name: " << stringOrNA(metadata.targetName) << "\n";
  os << "scope_count: " << metadata.scopeCount << "\n";
  os << "tracked_ops: " << metadata.trackedOps.size() << "\n";

  for (const auto &entry : metadata.trackedOps) {
    os << "  op_id=" << entry.opId << " scope_id=" << entry.scopeId
       << " mlir_op=" << stringOrNA(entry.mlirOpName)
       << " source_loc=" << stringOrNA(entry.sourceLoc)
       << " dtype_out=" << stringOrNA(entry.result.dtype)
       << " shape=" << stringOrNA(entry.result.shape)
       << " stride=" << stringOrNA(entry.result.stride)
       << " layout=" << stringOrNA(entry.result.layout);
    if (entry.isMemoryOp) {
      os << " addr_space=" << stringOrNA(entry.addrSpace)
         << " access_type=" << stringOrNA(entry.accessType)
         << " access_bytes=" << entry.accessBytes
         << " alignment_required=" << entry.alignmentRequired;
    }
    os << "\n";
  }
}

void renderRuntimeMetadata(std::ostringstream &os,
                           const DebugRuntimeMetadata &metadata) {
  os << "\nRuntime Inventory\n";
  os << "buffers: " << metadata.buffers.size() << "\n";
  for (const auto &buffer : metadata.buffers) {
    os << "  bufferId=" << buffer.bufferId
       << " name=" << stringOrNA(buffer.bufferName)
       << " base=" << hexValue(buffer.baseAddress)
       << " sizeBytes=" << buffer.sizeBytes << " alignment=" << buffer.alignment
       << " range=[" << hexValue(buffer.baseAddress) << ","
       << hexValue(buffer.baseAddress + buffer.sizeBytes) << ")\n";
  }

  os << "tensors: " << metadata.tensors.size() << "\n";
  for (const auto &tensor : metadata.tensors) {
    os << "  arg=" << tensor.argumentIndex
       << " name=" << stringOrNA(tensor.logicalName)
       << " dtype=" << stringOrNA(tensor.dtype)
       << " shape=" << joinInt64Vector(tensor.shape)
       << " stride=" << joinInt64Vector(tensor.stride)
       << " layout=" << stringOrNA(tensor.layout)
       << " bufferId=" << tensor.bufferId
       << " base=" << hexValue(tensor.baseAddress)
       << " sizeBytes=" << tensor.sizeBytes << "\n";
  }
  if (metadata.buffers.empty() && metadata.tensors.empty()) {
    os << "runtime metadata not captured\n";
  }
}

llvm::json::Value jsonUnsigned(uint64_t value) {
  if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return llvm::json::Value(static_cast<int64_t>(value));
  }
  return llvm::json::Value(std::to_string(value));
}

llvm::json::Array jsonInt64Vector(const std::vector<int64_t> &values) {
  llvm::json::Array array;
  for (int64_t value : values) {
    array.push_back(value);
  }
  return array;
}

llvm::json::Object jsonStaticValueInfo(const StaticValueInfo &info) {
  return llvm::json::Object{
      {"value_kind", info.valueKind},
      {"dtype", info.dtype},
      {"element_dtype", info.elementDtype},
      {"shape", info.shape},
      {"stride", info.stride},
      {"layout", info.layout},
      {"encoding", info.encoding},
      {"addr_space", info.addrSpace},
      {"rank", static_cast<int64_t>(info.rank)},
      {"element_bits", static_cast<int64_t>(info.elementBits)},
      {"vec_width", static_cast<int64_t>(info.vecWidth)}};
}

llvm::json::Object jsonOperandStaticInfo(const OperandStaticInfo &info) {
  return llvm::json::Object{
      {"operand_index", static_cast<int64_t>(info.operandIndex)},
      {"operand_role", info.operandRole},
      {"producer_op_id", static_cast<int64_t>(info.producerOpId)},
      {"is_constant", info.isConstant},
      {"is_predicate", info.isPredicate},
      {"is_kernel_argument", info.isKernelArgument},
      {"constant_value_repr", info.constantValueRepr},
      {"value", jsonStaticValueInfo(info.value)}};
}

llvm::json::Object
jsonStatementValueStaticInfo(const StatementValueInfo &info) {
  llvm::json::Object object{
      {"source_name", info.sourceName},
      {"source_role", info.sourceRole},
      {"has_operand_index", info.hasOperandIndex},
      {"operand_index", static_cast<int64_t>(info.operandIndex)},
      {"producer_op_id", static_cast<int64_t>(info.producerOpId)},
      {"capture_op_id", static_cast<int64_t>(info.captureOpId)},
      {"capture_policy", info.capturePolicy},
      {"is_constant", info.isConstant},
      {"constant_value_repr", info.constantValueRepr},
      {"value", jsonStaticValueInfo(info.value)},
  };
  return object;
}

llvm::json::Object jsonTrackedOp(const TrackedOpEntry *trackedOp) {
  if (!trackedOp) {
    return llvm::json::Object{{"status", "missing"}};
  }

  llvm::json::Array operands;
  for (const OperandStaticInfo &operand : trackedOp->operands) {
    operands.push_back(jsonOperandStaticInfo(operand));
  }

  llvm::json::Array statementValues;
  for (const StatementValueInfo &value : trackedOp->statementValues) {
    statementValues.push_back(jsonStatementValueStaticInfo(value));
  }

  llvm::json::Object object{
      {"status", "captured"},
      {"mlir_op", trackedOp->mlirOpName},
      {"source_loc", trackedOp->sourceLoc},
      {"triton_statement", trackedOp->tritonStatement},
      {"statement_id", static_cast<int64_t>(trackedOp->statementId)},
      {"statement_result_name", trackedOp->statementResultName},
      {"is_synthetic_statement_capture",
       trackedOp->isSyntheticStatementCapture},
      {"inline_call_path", trackedOp->inlineCallPath},
      {"result_index", static_cast<int64_t>(trackedOp->resultIndex)},
      {"is_memory_op", trackedOp->isMemoryOp},
      {"op_category", trackedOp->opCategory},
      {"role", trackedOp->role},
      {"result", jsonStaticValueInfo(trackedOp->result)},
      {"operands", std::move(operands)},
      {"statement_values", std::move(statementValues)},
  };

  if (trackedOp->isMemoryOp) {
    object["memory_semantics"] = llvm::json::Object{
        {"addr_space", trackedOp->addrSpace},
        {"access_type", trackedOp->accessType},
        {"access_bytes", static_cast<int64_t>(trackedOp->accessBytes)},
        {"alignment_required",
         static_cast<int64_t>(trackedOp->alignmentRequired)},
        {"has_mask", trackedOp->hasMask},
        {"mask_dtype", trackedOp->maskDtype},
        {"cache_modifier", trackedOp->cacheModifier},
        {"eviction_policy", trackedOp->evictionPolicy},
        {"is_volatile", trackedOp->isVolatile},
        {"boundary_check_policy", trackedOp->boundaryCheckPolicy},
        {"padding_semantics", trackedOp->paddingSemantics},
    };
  }
  return object;
}

llvm::json::Object jsonRuntimeAddressContext(const DecodedDebugRun &run,
                                             const TrackedOpEntry *trackedOp,
                                             uint64_t addr) {
  llvm::json::Object object;
  const auto *buffer = findContainingBuffer(run.runtimeMetadata, addr);
  if (buffer) {
    object["buffer_id"] = static_cast<int64_t>(buffer->bufferId);
    object["buffer_name"] = buffer->bufferName;
    object["range_start"] = hexValue(buffer->baseAddress);
    object["range_end"] = hexValue(buffer->baseAddress + buffer->sizeBytes);
    object["size_bytes"] = jsonUnsigned(buffer->sizeBytes);
    object["offset"] = jsonUnsigned(addr - buffer->baseAddress);
  } else {
    object["buffer_id"] = nullptr;
    object["buffer_name"] = nullptr;
    object["range_start"] = nullptr;
    object["range_end"] = nullptr;
    object["size_bytes"] = nullptr;
    object["offset"] = nullptr;
  }

  if (trackedOp && trackedOp->alignmentRequired != 0) {
    object["alignment_ok"] = (addr % trackedOp->alignmentRequired == 0);
  } else {
    object["alignment_ok"] = nullptr;
  }
  object["local_address_snapshot"] = nullptr;
  return object;
}

llvm::json::Object jsonSummaryCell(const SummaryMetricValue *value) {
  if (!value) {
    return llvm::json::Object{{"status", "not_captured"}};
  }

  llvm::json::Object object{{"status", "captured"},
                            {"result_type", value->resultTypeName},
                            {"display", formatSummaryCell(*value)}};
  switch (value->resultType) {
  case ResultType::U64:
    object["value"] = jsonUnsigned(value->u64Value);
    break;
  case ResultType::F32:
    object["value"] = static_cast<double>(value->f32Value);
    break;
  case ResultType::F64:
    object["value"] = value->f64Value;
    break;
  }
  return object;
}

llvm::json::Object jsonAddressSummaryCell(MemoryEventKind kind,
                                          const MemoryEventView *event) {
  if (!event) {
    return llvm::json::Object{{"status", "not_captured"}};
  }
  return llvm::json::Object{{"status", "captured"},
                            {"value", jsonUnsigned(event->addr)},
                            {"display", formatMemorySummaryCell(kind, event)}};
}

llvm::json::Object jsonMemoryEvent(const DecodedDebugRun &run,
                                   const TrackedOpEntry *trackedOp,
                                   const MemoryEventView &event) {
  return llvm::json::Object{
      {"kind", toString(event.kind)},
      {"addr", jsonUnsigned(event.addr)},
      {"addr_hex", hexValue(event.addr)},
      {"ext0", static_cast<int64_t>(event.ext0)},
      {"runtime_address",
       jsonRuntimeAddressContext(run, trackedOp, event.addr)},
  };
}

llvm::json::Object jsonFullValueRef(const FullValueRefView &ref) {
  llvm::json::Object object;
  object["kind"] = ref.kind;
  if (!ref.path.empty()) {
    object["path"] = ref.path;
  } else {
    object["payload_offset"] = static_cast<int64_t>(ref.payloadOffset);
    object["payload_length"] = static_cast<int64_t>(ref.payloadLength);
    object["path"] = nullptr;
  }
  return object;
}

llvm::json::Array jsonFullValueRefsByInstance(
    const std::map<uint64_t, InstanceReportGroup> &instances) {
  llvm::json::Array fullValueRefsByInstance;
  for (const auto &instanceEntry : instances) {
    const auto &instance = instanceEntry.second;
    if (instance.fullValueRefs.empty())
      continue;
    llvm::json::Array refs;
    for (const FullValueRefView &ref : instance.fullValueRefs)
      refs.push_back(jsonFullValueRef(ref));
    fullValueRefsByInstance.push_back(llvm::json::Object{
        {"logical_instance_id", jsonUnsigned(instance.logicalInstanceId)},
        {"refs", std::move(refs)},
    });
  }
  return fullValueRefsByInstance;
}

llvm::json::Array
jsonInstanceIds(const std::map<uint64_t, InstanceReportGroup> &instances) {
  llvm::json::Array ids;
  for (const auto &instanceEntry : instances)
    ids.push_back(jsonUnsigned(instanceEntry.second.logicalInstanceId));
  return ids;
}

bool addJsonAddressSummary(
    llvm::json::Object &object,
    const std::map<uint64_t, InstanceReportGroup> &instances,
    std::optional<uint32_t> ext0Filter = std::nullopt) {
  std::vector<std::map<MemoryEventKind, const MemoryEventView *>>
      addressSummaries;
  bool hasAnyAddressSummary = false;
  for (const auto &instanceEntry : instances) {
    addressSummaries.push_back(
        collectAddressSummary(instanceEntry.second.memoryEvents, ext0Filter));
    hasAnyAddressSummary =
        hasAnyAddressSummary || !addressSummaries.back().empty();
  }
  if (!hasAnyAddressSummary)
    return false;

  llvm::json::Object addressSummary;
  llvm::json::Array status;
  for (const auto &summaryForInstance : addressSummaries)
    status.push_back(addressSummaryStatus(summaryForInstance));
  addressSummary["status"] = std::move(status);

  for (const MemoryEventKind *kind = addressSummaryKindsBegin();
       kind != addressSummaryKindsEnd(); ++kind) {
    llvm::json::Array cells;
    for (const auto &summaryForInstance : addressSummaries) {
      auto it = summaryForInstance.find(*kind);
      cells.push_back(jsonAddressSummaryCell(
          *kind, it == summaryForInstance.end() ? nullptr : it->second));
    }
    addressSummary[addressSummaryFieldName(*kind)] = std::move(cells);
  }
  object["address_summary"] = std::move(addressSummary);
  return true;
}

llvm::json::Array jsonMemoryEventsByInstance(
    const DecodedDebugRun &run, const TrackedOpEntry *trackedOp,
    const std::map<uint64_t, InstanceReportGroup> &instances,
    std::optional<uint32_t> ext0Filter = std::nullopt) {
  llvm::json::Array memoryEventsByInstance;
  for (const auto &instanceEntry : instances) {
    const auto &instance = instanceEntry.second;
    if (!hasFallbackMemoryEvents(instance, ext0Filter))
      continue;
    llvm::json::Array events;
    for (const MemoryEventView &event : instance.memoryEvents) {
      if (!memoryEventMatchesExt0(event, ext0Filter))
        continue;
      if (!isAddressSummaryMemoryEventKind(event.kind))
        events.push_back(jsonMemoryEvent(run, trackedOp, event));
    }
    memoryEventsByInstance.push_back(llvm::json::Object{
        {"logical_instance_id", jsonUnsigned(instance.logicalInstanceId)},
        {"events", std::move(events)},
    });
  }
  return memoryEventsByInstance;
}

llvm::json::Object jsonMemoryAccessReport(const DecodedDebugRun &run,
                                          const MemoryUseView &use,
                                          bool includeLabel) {
  llvm::json::Object object;
  if (includeLabel)
    object["label"] = memoryAccessLabel(use);

  llvm::json::Object runtime;
  if (!use.runtime || use.runtime->instances.empty()) {
    runtime["status"] = "not_captured";
  } else {
    const std::optional<uint32_t> ext0Filter =
        use.hasAddressExt0 ? std::optional<uint32_t>(use.addressExt0)
                           : std::nullopt;
    runtime["instances"] = jsonInstanceIds(use.runtime->instances);
    if (!addJsonAddressSummary(runtime, use.runtime->instances, ext0Filter))
      runtime["address_summary"] =
          llvm::json::Object{{"status", "not_captured"}};
    llvm::json::Array memoryEvents = jsonMemoryEventsByInstance(
        run, use.accessOp, use.runtime->instances, ext0Filter);
    if (!memoryEvents.empty())
      runtime["memory_events_by_instance"] = std::move(memoryEvents);
  }
  object["runtime"] = std::move(runtime);
  return object;
}

llvm::json::Object jsonStatementRuntimeReport(const OpReportGroup *opGroup) {
  if (!opGroup || opGroup->instances.empty())
    return llvm::json::Object{{"status", "not_captured"}};

  llvm::json::Object object;
  object["instances"] = jsonInstanceIds(opGroup->instances);

  llvm::json::Object summary;
  for (const std::string &metric : collectSummaryMetrics(opGroup->instances)) {
    llvm::json::Array cells;
    for (const auto &instanceEntry : opGroup->instances) {
      auto it = instanceEntry.second.summaryValues.find(metric);
      cells.push_back(jsonSummaryCell(
          it == instanceEntry.second.summaryValues.end() ? nullptr
                                                         : &it->second));
    }
    summary[metric] = std::move(cells);
  }
  if (!summary.empty())
    object["summary"] = std::move(summary);

  addJsonAddressSummary(object, opGroup->instances);

  llvm::json::Array fullValueRefsByInstance =
      jsonFullValueRefsByInstance(opGroup->instances);
  if (!fullValueRefsByInstance.empty())
    object["full_value_refs_by_instance"] = std::move(fullValueRefsByInstance);

  return object;
}

llvm::json::Object jsonOpReportGroup(const DecodedDebugRun &run,
                                     const OpReportGroup &opGroup) {
  llvm::json::Object object{
      {"op_id", static_cast<int64_t>(opGroup.opId)},
      {"scope_id", static_cast<int64_t>(opGroup.scopeId)},
      {"static", jsonTrackedOp(opGroup.trackedOp)},
  };

  object["instances"] = jsonInstanceIds(opGroup.instances);

  llvm::json::Object summary;
  for (const std::string &metric : collectSummaryMetrics(opGroup.instances)) {
    llvm::json::Array cells;
    for (const auto &instanceEntry : opGroup.instances) {
      auto it = instanceEntry.second.summaryValues.find(metric);
      cells.push_back(jsonSummaryCell(
          it == instanceEntry.second.summaryValues.end() ? nullptr
                                                         : &it->second));
    }
    summary[metric] = std::move(cells);
  }
  if (!summary.empty()) {
    object["summary"] = std::move(summary);
  }

  addJsonAddressSummary(object, opGroup.instances);

  llvm::json::Array memoryEventsByInstance =
      jsonMemoryEventsByInstance(run, opGroup.trackedOp, opGroup.instances);
  if (!memoryEventsByInstance.empty()) {
    object["memory_events_by_instance"] = std::move(memoryEventsByInstance);
  }

  llvm::json::Array fullValueRefsByInstance =
      jsonFullValueRefsByInstance(opGroup.instances);
  if (!fullValueRefsByInstance.empty()) {
    object["full_value_refs_by_instance"] = std::move(fullValueRefsByInstance);
  }

  return object;
}

llvm::json::Object
jsonStatementValueReport(const DecodedDebugRun &run,
                         const StatementValueInfo &value,
                         const std::map<uint32_t, OpReportGroup> &byOp) {
  (void)run;
  llvm::json::Object object{
      {"source_name", value.sourceName},
      {"source_role", value.sourceRole},
      {"is_constant", value.isConstant},
      {"constant_value_repr", value.constantValueRepr},
      {"value", jsonStaticValueInfo(value.value)},
  };
  const auto captureIt = byOp.find(value.captureOpId);
  if (captureIt != byOp.end()) {
    object["runtime"] = jsonStatementRuntimeReport(&captureIt->second);
  } else {
    object["runtime"] = llvm::json::Object{
        {"status", value.isConstant ? "constant" : "not_captured"}};
  }
  return object;
}

llvm::json::Array
jsonStatementRecords(const DecodedDebugRun &run,
                     const KernelDebugMetadata &metadata,
                     const std::map<uint32_t, OpReportGroup> &byOp) {
  llvm::json::Array statements;
  StatementRenderIndex renderIndex = buildStatementRenderIndex(metadata, byOp);
  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    if (entry.isSyntheticStatementCapture || entry.statementValues.empty())
      continue;

    llvm::json::Array values;
    for (const StatementValueInfo &value : entry.statementValues) {
      values.push_back(jsonStatementValueReport(run, value, byOp));
    }

    llvm::json::Object statement{
        {"source_loc", entry.sourceLoc},
        {"statement_id", static_cast<int64_t>(entry.statementId)},
        {"statement", entry.tritonStatement},
        {"statement_result_name", entry.statementResultName},
        {"values", std::move(values)},
    };
    if (const std::vector<MemoryUseView> *statementMemoryUses =
            findStatementMemoryUses(entry.opId, renderIndex)) {
      llvm::json::Array uses;
      const bool includeLabels = statementMemoryUses->size() > 1;
      for (const MemoryUseView &use : *statementMemoryUses)
        uses.push_back(jsonMemoryAccessReport(run, use, includeLabels));
      statement["memory_accesses"] = std::move(uses);
    }
    statements.push_back(std::move(statement));
  }
  return statements;
}

llvm::json::Object jsonStaticOnlyOp(const TrackedOpEntry &trackedOp) {
  return llvm::json::Object{
      {"op_id", static_cast<int64_t>(trackedOp.opId)},
      {"scope_id", static_cast<int64_t>(trackedOp.scopeId)},
      {"dynamic_record_status", "static_only"},
      {"dynamic_record_note",
       "no runtime record was emitted for this op; static metadata is kept "
       "for producer/context analysis"},
      {"static", jsonTrackedOp(&trackedOp)},
  };
}

llvm::json::Object jsonOpLog(const DecodedDebugRun &run,
                             const KernelDebugMetadata &metadata,
                             const std::map<uint32_t, OpReportGroup> &byOp) {
  llvm::json::Array recordsByOp;
  for (const auto &entry : byOp) {
    recordsByOp.push_back(jsonOpReportGroup(run, entry.second));
  }

  llvm::json::Array staticOnlyOps;
  for (const TrackedOpEntry &trackedOp : metadata.trackedOps) {
    if (byOp.count(trackedOp.opId) == 0) {
      staticOnlyOps.push_back(jsonStaticOnlyOp(trackedOp));
    }
  }

  return llvm::json::Object{{"records_by_op", std::move(recordsByOp)},
                            {"static_only_ops", std::move(staticOnlyOps)}};
}

llvm::json::Object jsonStaticOpCatalog(const KernelDebugMetadata &metadata) {
  llvm::json::Array trackedOps;
  for (const TrackedOpEntry &entry : metadata.trackedOps) {
    trackedOps.push_back(jsonTrackedOp(&entry));
  }

  return llvm::json::Object{
      {"kernel_debug_id", static_cast<int64_t>(metadata.debugKernelId)},
      {"kernel_name", metadata.kernelName},
      {"backend_name", metadata.backendName},
      {"target_name", metadata.targetName},
      {"scope_count", static_cast<int64_t>(metadata.scopeCount)},
      {"tracked_ops", std::move(trackedOps)},
  };
}

llvm::json::Object jsonRuntimeInventory(const DebugRuntimeMetadata &metadata) {
  llvm::json::Array buffers;
  for (const auto &buffer : metadata.buffers) {
    buffers.push_back(llvm::json::Object{
        {"buffer_id", static_cast<int64_t>(buffer.bufferId)},
        {"name", buffer.bufferName},
        {"base", hexValue(buffer.baseAddress)},
        {"base_value", jsonUnsigned(buffer.baseAddress)},
        {"size_bytes", jsonUnsigned(buffer.sizeBytes)},
        {"alignment", static_cast<int64_t>(buffer.alignment)},
        {"range_start", hexValue(buffer.baseAddress)},
        {"range_end", hexValue(buffer.baseAddress + buffer.sizeBytes)},
    });
  }

  llvm::json::Array tensors;
  for (const auto &tensor : metadata.tensors) {
    tensors.push_back(llvm::json::Object{
        {"argument_index", static_cast<int64_t>(tensor.argumentIndex)},
        {"name", tensor.logicalName},
        {"dtype", tensor.dtype},
        {"shape", jsonInt64Vector(tensor.shape)},
        {"stride", jsonInt64Vector(tensor.stride)},
        {"layout", tensor.layout},
        {"buffer_id", static_cast<int64_t>(tensor.bufferId)},
        {"base", hexValue(tensor.baseAddress)},
        {"base_value", jsonUnsigned(tensor.baseAddress)},
        {"size_bytes", jsonUnsigned(tensor.sizeBytes)},
    });
  }

  return llvm::json::Object{{"buffers", std::move(buffers)},
                            {"tensors", std::move(tensors)}};
}

struct AggregateStats {
  uint32_t opId = 0;
  uint32_t scopeId = kInvalidScopeId;
  uint64_t totalRecords = 0;
  uint64_t summaryRecords = 0;
  uint64_t memoryEventRecords = 0;
  uint64_t fullValueRefRecords = 0;
  std::map<std::string, std::string> latestSummaryValues;
};

void renderAggregates(std::ostringstream &os, const DecodedDebugRun &run,
                      const KernelDebugMetadata &metadata) {
  std::map<uint32_t, AggregateStats> byOp;
  std::map<uint32_t, uint64_t> byScope;
  uint64_t summaryCount = 0;
  uint64_t memoryCount = 0;
  uint64_t fullValueCount = 0;

  for (const auto &record : run.records) {
    const uint32_t opId = getRecordOpId(record);
    const auto *trackedOp = findTrackedOp(metadata.trackedOps, opId);
    const uint32_t scopeId = trackedOp ? trackedOp->scopeId : kInvalidScopeId;
    auto &stats = byOp[opId];
    stats.opId = opId;
    stats.scopeId = scopeId;
    ++stats.totalRecords;
    ++byScope[scopeId];

    if (const auto *summary = std::get_if<DecodedSummaryRecord>(&record)) {
      ++stats.summaryRecords;
      ++summaryCount;
      stats.latestSummaryValues[toMetricName(summary->raw.collectorKind)] =
          formatSummaryValue(summary->raw);
    } else if (const auto *bundle =
                   std::get_if<DecodedSummaryCountBundleRecord>(&record)) {
      ++stats.summaryRecords;
      ++summaryCount;
      stats.latestSummaryValues["nan_count"] =
          std::to_string(bundle->raw.nanCount);
      stats.latestSummaryValues["inf_count"] =
          std::to_string(bundle->raw.infCount);
      stats.latestSummaryValues["zero_count"] =
          std::to_string(bundle->raw.zeroCount);
      stats.latestSummaryValues["element_count"] =
          std::to_string(bundle->raw.elementCount);
    } else if (const auto *bundle =
                   std::get_if<DecodedSummaryValueBundleRecord>(&record)) {
      ++stats.summaryRecords;
      ++summaryCount;
      stats.latestSummaryValues["mean"] =
          makeF32SummaryMetricValue(bundle->raw.meanFinite).value;
      stats.latestSummaryValues["min"] =
          makeF32SummaryMetricValue(bundle->raw.minFinite).value;
      stats.latestSummaryValues["max"] =
          makeF32SummaryMetricValue(bundle->raw.maxFinite).value;
      stats.latestSummaryValues["l2_norm"] =
          makeF32SummaryMetricValue(bundle->raw.l2Norm).value;
    } else if (std::holds_alternative<DecodedMemoryEventRecord>(record)) {
      ++stats.memoryEventRecords;
      ++memoryCount;
    } else if (std::holds_alternative<DecodedFullValueRefRecord>(record)) {
      ++stats.fullValueRefRecords;
      ++fullValueCount;
    }
  }

  os << "\nAggregates\n";
  os << "kernel: total_records=" << run.records.size()
     << " summary_records=" << summaryCount
     << " memory_event_records=" << memoryCount
     << " full_value_ref_records=" << fullValueCount << "\n";

  os << "by_scope:\n";
  for (const auto &entry : byScope) {
    os << "  scope_id=" << entry.first << " records=" << entry.second << "\n";
  }

  os << "by_op:\n";
  for (const auto &entry : byOp) {
    const auto &stats = entry.second;
    os << "  op_id=" << stats.opId << " scope_id=" << stats.scopeId
       << " total_records=" << stats.totalRecords
       << " summary_records=" << stats.summaryRecords
       << " memory_event_records=" << stats.memoryEventRecords
       << " full_value_ref_records=" << stats.fullValueRefRecords << "\n";
    for (const auto &metric : stats.latestSummaryValues) {
      os << "    latest." << metric.first << "=" << metric.second << "\n";
    }
  }
}

llvm::json::Object jsonAggregates(const DecodedDebugRun &run,
                                  const KernelDebugMetadata &metadata) {
  std::map<uint32_t, AggregateStats> byOp;
  std::map<uint32_t, uint64_t> byScope;
  uint64_t summaryCount = 0;
  uint64_t memoryCount = 0;
  uint64_t fullValueCount = 0;

  for (const auto &record : run.records) {
    const uint32_t opId = getRecordOpId(record);
    const auto *trackedOp = findTrackedOp(metadata.trackedOps, opId);
    const uint32_t scopeId = trackedOp ? trackedOp->scopeId : kInvalidScopeId;
    auto &stats = byOp[opId];
    stats.opId = opId;
    stats.scopeId = scopeId;
    ++stats.totalRecords;
    ++byScope[scopeId];

    if (const auto *summary = std::get_if<DecodedSummaryRecord>(&record)) {
      ++stats.summaryRecords;
      ++summaryCount;
      stats.latestSummaryValues[toMetricName(summary->raw.collectorKind)] =
          formatSummaryValue(summary->raw);
    } else if (const auto *bundle =
                   std::get_if<DecodedSummaryCountBundleRecord>(&record)) {
      ++stats.summaryRecords;
      ++summaryCount;
      stats.latestSummaryValues["nan_count"] =
          std::to_string(bundle->raw.nanCount);
      stats.latestSummaryValues["inf_count"] =
          std::to_string(bundle->raw.infCount);
      stats.latestSummaryValues["zero_count"] =
          std::to_string(bundle->raw.zeroCount);
      stats.latestSummaryValues["element_count"] =
          std::to_string(bundle->raw.elementCount);
    } else if (const auto *bundle =
                   std::get_if<DecodedSummaryValueBundleRecord>(&record)) {
      ++stats.summaryRecords;
      ++summaryCount;
      stats.latestSummaryValues["mean"] =
          makeF32SummaryMetricValue(bundle->raw.meanFinite).value;
      stats.latestSummaryValues["min"] =
          makeF32SummaryMetricValue(bundle->raw.minFinite).value;
      stats.latestSummaryValues["max"] =
          makeF32SummaryMetricValue(bundle->raw.maxFinite).value;
      stats.latestSummaryValues["l2_norm"] =
          makeF32SummaryMetricValue(bundle->raw.l2Norm).value;
    } else if (std::holds_alternative<DecodedMemoryEventRecord>(record)) {
      ++stats.memoryEventRecords;
      ++memoryCount;
    } else if (std::holds_alternative<DecodedFullValueRefRecord>(record)) {
      ++stats.fullValueRefRecords;
      ++fullValueCount;
    }
  }

  llvm::json::Array scopes;
  for (const auto &entry : byScope) {
    scopes.push_back(llvm::json::Object{
        {"scope_id", static_cast<int64_t>(entry.first)},
        {"records", jsonUnsigned(entry.second)},
    });
  }

  llvm::json::Array ops;
  for (const auto &entry : byOp) {
    const auto &stats = entry.second;
    llvm::json::Object latest;
    for (const auto &metric : stats.latestSummaryValues) {
      latest[metric.first] = metric.second;
    }
    ops.push_back(llvm::json::Object{
        {"op_id", static_cast<int64_t>(stats.opId)},
        {"scope_id", static_cast<int64_t>(stats.scopeId)},
        {"total_records", jsonUnsigned(stats.totalRecords)},
        {"summary_records", jsonUnsigned(stats.summaryRecords)},
        {"memory_event_records", jsonUnsigned(stats.memoryEventRecords)},
        {"full_value_ref_records", jsonUnsigned(stats.fullValueRefRecords)},
        {"latest_summary_values", std::move(latest)},
    });
  }

  return llvm::json::Object{
      {"kernel",
       llvm::json::Object{
           {"total_records", jsonUnsigned(run.records.size())},
           {"summary_records", jsonUnsigned(summaryCount)},
           {"memory_event_records", jsonUnsigned(memoryCount)},
           {"full_value_ref_records", jsonUnsigned(fullValueCount)},
       }},
      {"by_scope", std::move(scopes)},
      {"by_op", std::move(ops)},
  };
}

std::string renderJsonValue(llvm::json::Value value) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << value;
  return text;
}

} // namespace

std::string renderTextReport(const DecodedDebugRun &run,
                             const KernelDebugMetadata &metadata,
                             const ReportOptions &options) {
  std::ostringstream os;

  if (options.includeReportHeader) {
    os << "FlagTree Debug Report\n";
    os << "protocol_version: " << run.meta.protocolVer << "\n";
    os << "run_id: " << run.meta.runId << "\n";
    os << "kernel_id: " << run.meta.kernelId << "\n";
    os << "kernel_name: " << stringOrNA(metadata.kernelName) << "\n";
    os << "device: " << run.meta.deviceId << "\n";
    os << "backend: " << toString(run.meta.backendKind) << "\n";
    os << "record_level: " << toString(run.meta.recordLevel) << "\n";
    os << "export_mode: " << toString(run.meta.exportMode) << "\n";
    os << "record_count: " << run.records.size() << "\n";
    os << "record_count_note: number of debug record slots written; not tensor "
          "element count\n";
    os << "overflow_count: " << run.header.overflowCount << "\n";
    os << "flags: " << run.header.flags << "\n";
    if (metadata.debugKernelId != 0 &&
        metadata.debugKernelId != run.meta.kernelId) {
      os << "metadata_warning: debugKernelId " << metadata.debugKernelId
         << " does not match runtime kernel_id " << run.meta.kernelId << "\n";
    }
    if (run.header.overflowCount != 0 ||
        (run.header.flags & RB_FLAG_OVERFLOW) != 0) {
      os << "overflow_warning: device debug buffer overflowed; report may be "
            "truncated\n";
    }
  }

  if (options.includeDynamicRecords) {
    if (options.includeStatementRecords) {
      renderStatementRecords(os, run, metadata);
    }
    if (options.includeOpLog) {
      renderRecordsByOp(os, run, metadata);
    }
  }

  if (options.includeStaticMetadata) {
    renderRuntimeMetadata(os, run.runtimeMetadata);
  }

  if (options.includeStaticOpCatalog) {
    renderStaticMetadata(os, metadata);
  }

  if (options.includeAggregates) {
    renderAggregates(os, run, metadata);
  }

  return os.str();
}

std::string renderJsonReport(const DecodedDebugRun &run,
                             const KernelDebugMetadata &metadata,
                             const ReportOptions &options) {
  llvm::json::Array warnings;
  if (metadata.debugKernelId != 0 &&
      metadata.debugKernelId != run.meta.kernelId) {
    warnings.push_back(
        "metadata debugKernelId does not match runtime kernel_id");
  }
  if (run.header.overflowCount != 0 ||
      (run.header.flags & RB_FLAG_OVERFLOW) != 0) {
    warnings.push_back(
        "device debug buffer overflowed; report may be truncated");
  }

  llvm::json::Object root{
      {"protocol_version", static_cast<int64_t>(run.meta.protocolVer)},
      {"run_id", jsonUnsigned(run.meta.runId)},
      {"kernel_id", static_cast<int64_t>(run.meta.kernelId)},
      {"kernel_name", metadata.kernelName},
      {"device", static_cast<int64_t>(run.meta.deviceId)},
      {"backend", toString(run.meta.backendKind)},
      {"record_level", toString(run.meta.recordLevel)},
      {"export_mode", toString(run.meta.exportMode)},
      {"record_count", jsonUnsigned(run.records.size())},
      {"record_count_note",
       "number of debug record slots written; not tensor element count"},
      {"overflow_count", static_cast<int64_t>(run.header.overflowCount)},
      {"flags", static_cast<int64_t>(run.header.flags)},
      {"warnings", std::move(warnings)},
  };

  std::map<uint32_t, OpReportGroup> byOp = buildOpReportGroups(run, metadata);
  if (options.includeDynamicRecords) {
    if (options.includeStatementRecords) {
      root["records_by_op"] = jsonStatementRecords(run, metadata, byOp);
    }
    if (options.includeOpLog) {
      root["op_log"] = jsonOpLog(run, metadata, byOp);
    }
  }

  if (options.includeStaticMetadata) {
    root["static_op_catalog"] = jsonStaticOpCatalog(metadata);
    root["runtime_inventory"] = jsonRuntimeInventory(run.runtimeMetadata);
  }

  if (options.includeAggregates) {
    root["aggregates"] = jsonAggregates(run, metadata);
  }

  return renderJsonValue(std::move(root));
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
