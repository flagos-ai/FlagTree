#include "Debugger/Decode/Decoder.h"
#include "Debugger/Decode/Reporter.h"
#include "Debugger/Frontend/Bridge.h"
#include "Debugger/Metadata/TrackedOpTable.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace {

using namespace mlir::flagtree::debugger;

std::atomic<uint64_t> nextRunId{1};

py::handle lookup(const py::dict &dict, const char *key) {
  py::str pyKey(key);
  if (!dict.contains(pyKey)) {
    return py::handle();
  }
  py::object value = dict[pyKey];
  return value.is_none() ? py::handle() : value;
}

bool getBoolOr(const py::dict &dict, const char *key, bool fallback) {
  py::handle value = lookup(dict, key);
  return value ? py::cast<bool>(value) : fallback;
}

uint32_t getUInt32Or(const py::dict &dict, const char *key, uint32_t fallback) {
  py::handle value = lookup(dict, key);
  return value ? py::cast<uint32_t>(value) : fallback;
}

uint64_t getUInt64Or(const py::dict &dict, const char *key, uint64_t fallback) {
  py::handle value = lookup(dict, key);
  return value ? py::cast<uint64_t>(value) : fallback;
}

std::string getStringOr(const py::dict &dict, const char *key,
                        std::string_view fallback) {
  py::handle value = lookup(dict, key);
  return value ? py::cast<std::string>(value) : std::string(fallback);
}

std::vector<int64_t> castInt64Vector(py::handle value) {
  if (!value || value.is_none()) {
    return {};
  }
  return py::cast<std::vector<int64_t>>(value);
}

uint32_t getSequenceUInt32(py::handle value, size_t index, uint32_t fallback) {
  if (!value || value.is_none() || !py::isinstance<py::sequence>(value)) {
    return fallback;
  }
  py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
  if (index >= static_cast<size_t>(sequence.size())) {
    return fallback;
  }
  int64_t dim = py::cast<int64_t>(sequence[index]);
  if (dim <= 0 ||
      static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("launch grid dimension must fit uint32_t");
  }
  return static_cast<uint32_t>(dim);
}

std::string toLower(std::string_view input) {
  std::string result;
  result.reserve(input.size());
  for (char ch : input) {
    if (ch >= 'A' && ch <= 'Z') {
      result.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

RecordLevel parseRecordLevel(py::handle value) {
  if (!value || value.is_none()) {
    return RecordLevel::LEVEL_SUMMARY;
  }
  if (py::isinstance<py::int_>(value)) {
    return py::cast<uint32_t>(value) == 2 ? RecordLevel::LEVEL_TENSOR_FULL
                                          : RecordLevel::LEVEL_SUMMARY;
  }
  std::string lowered = toLower(py::cast<std::string>(py::str(value)));
  if (lowered == "level_tensor_full" || lowered == "tensor_full" ||
      lowered == "full") {
    return RecordLevel::LEVEL_TENSOR_FULL;
  }
  return RecordLevel::LEVEL_SUMMARY;
}

ExportMode parseExportMode(py::handle value) {
  if (!value || value.is_none()) {
    return ExportMode::POST_KERNEL_EXPORT;
  }
  if (py::isinstance<py::int_>(value)) {
    return py::cast<uint32_t>(value) == 2 ? ExportMode::STREAMING_EXPORT
                                          : ExportMode::POST_KERNEL_EXPORT;
  }
  std::string lowered = toLower(py::cast<std::string>(py::str(value)));
  if (lowered == "streaming_export" || lowered == "streaming") {
    return ExportMode::STREAMING_EXPORT;
  }
  return ExportMode::POST_KERNEL_EXPORT;
}

BackendKind parseBackendKind(std::string_view backendName) {
  std::string lowered = toLower(backendName);
  if (lowered == "cuda" || lowered == "nvidia") {
    return BackendKind::CUDA;
  }
  if (lowered == "hip" || lowered == "rocm" || lowered == "amd") {
    return BackendKind::HIP;
  }
  if (lowered == "musa") {
    return BackendKind::MUSA;
  }
  if (lowered == "cann" || lowered == "ascend" || lowered == "npu") {
    return BackendKind::CANN;
  }
  return BackendKind::UNKNOWN;
}

uint64_t checkedMulU64(uint64_t lhs, uint64_t rhs, const char *what) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw std::overflow_error(std::string(what) + " overflows uint64_t");
  }
  return lhs * rhs;
}

uint64_t launchGridProduct(const DebugRuntimeMetadata &metadata) {
  if (!metadata.hasLaunchGrid)
    throw std::invalid_argument(
        "level-2 debugger full dump requires launch grid metadata");
  uint64_t product =
      checkedMulU64(metadata.gridX, metadata.gridY, "debug launch grid");
  return checkedMulU64(product, metadata.gridZ, "debug launch grid");
}

BufferRegistrationInfo parseBufferRegistrationInfo(const py::dict &dict) {
  BufferRegistrationInfo info;
  info.bufferId = getUInt32Or(dict, "buffer_id", 0);
  info.bufferName = getStringOr(dict, "buffer_name", "");
  info.baseAddress = getUInt64Or(dict, "base_address", 0);
  info.sizeBytes = getUInt64Or(dict, "size_bytes", 0);
  info.alignment = getUInt32Or(dict, "alignment", 0);
  return info;
}

LaunchTensorInfo parseLaunchTensorInfo(const py::dict &dict) {
  LaunchTensorInfo info;
  info.argumentIndex = getUInt32Or(dict, "argument_index", 0);
  info.logicalName = getStringOr(dict, "logical_name", "");
  info.dtype = getStringOr(dict, "dtype", "");
  info.shape = castInt64Vector(lookup(dict, "shape"));
  info.stride = castInt64Vector(lookup(dict, "stride"));
  info.layout = getStringOr(dict, "layout", "");
  info.bufferId = getUInt32Or(dict, "buffer_id", 0);
  info.baseAddress = getUInt64Or(dict, "base_address", 0);
  info.sizeBytes = getUInt64Or(dict, "size_bytes", 0);
  return info;
}

DebugRecordPlanEntry parseDebugRecordPlanEntry(const py::dict &dict) {
  DebugRecordPlanEntry entry;
  entry.recordIndex = getUInt32Or(dict, "record_index", 0);
  entry.opId = getUInt32Or(dict, "op_id", 0);
  entry.scopeId = getUInt32Or(dict, "scope_id", kInvalidScopeId);
  entry.recordKind =
      static_cast<RecordKind>(getUInt32Or(dict, "record_kind", 0));
  entry.collectorKind =
      static_cast<CollectorKind>(getUInt32Or(dict, "collector_kind", 0));
  entry.resultType =
      static_cast<ResultType>(getUInt32Or(dict, "result_type", 0));
  entry.eventKind =
      static_cast<MemoryEventKind>(getUInt32Or(dict, "event_kind", 0));
  return entry;
}

FullDumpArtifactInfo parseFullDumpArtifactInfo(const py::dict &dict) {
  FullDumpArtifactInfo info;
  info.opId = getUInt32Or(dict, "op_id", 0);
  info.logicalInstanceId = getUInt64Or(dict, "logical_instance_id", 0);
  info.payloadOffset = getUInt32Or(dict, "payload_offset", 0);
  info.payloadLength = getUInt32Or(dict, "payload_length", 0);
  info.kind = getStringOr(dict, "kind", "");
  info.path = getStringOr(dict, "path", "");
  return info;
}

DebugRuntimeMetadata parseRuntimeMetadata(py::handle value) {
  DebugRuntimeMetadata runtimeMetadata;
  if (!value || value.is_none()) {
    return runtimeMetadata;
  }

  py::dict dict = py::cast<py::dict>(value);

  py::handle buffers = lookup(dict, "buffers");
  if (buffers && !buffers.is_none()) {
    for (py::handle item : py::cast<py::list>(buffers)) {
      runtimeMetadata.buffers.push_back(
          parseBufferRegistrationInfo(py::cast<py::dict>(item)));
    }
  }

  py::handle tensors = lookup(dict, "tensors");
  if (tensors && !tensors.is_none()) {
    for (py::handle item : py::cast<py::list>(tensors)) {
      runtimeMetadata.tensors.push_back(
          parseLaunchTensorInfo(py::cast<py::dict>(item)));
    }
  }

  py::handle grid = lookup(dict, "grid");
  if (grid && !grid.is_none()) {
    runtimeMetadata.hasLaunchGrid = true;
    runtimeMetadata.gridX = getSequenceUInt32(grid, 0, 1);
    runtimeMetadata.gridY = getSequenceUInt32(grid, 1, 1);
    runtimeMetadata.gridZ = getSequenceUInt32(grid, 2, 1);
  }
  runtimeMetadata.recordsPerInstance =
      getUInt32Or(dict, "records_per_instance", 0);
  runtimeMetadata.recordLayout = getStringOr(dict, "record_layout", "");
  py::handle recordPlan = lookup(dict, "record_plan");
  if (recordPlan && !recordPlan.is_none()) {
    for (py::handle item : py::cast<py::list>(recordPlan)) {
      runtimeMetadata.recordPlan.push_back(
          parseDebugRecordPlanEntry(py::cast<py::dict>(item)));
    }
  }
  py::handle fullDumpArtifacts = lookup(dict, "full_dump_artifacts");
  if (fullDumpArtifacts && !fullDumpArtifacts.is_none()) {
    for (py::handle item : py::cast<py::list>(fullDumpArtifacts)) {
      runtimeMetadata.fullDumpArtifacts.push_back(
          parseFullDumpArtifactInfo(py::cast<py::dict>(item)));
    }
  }

  return runtimeMetadata;
}

py::dict toPyBufferMeta(const BufferMeta &meta) {
  py::dict dict;
  dict["run_id"] = meta.runId;
  dict["device_id"] = meta.deviceId;
  dict["kernel_id"] = meta.kernelId;
  dict["protocol_version"] = meta.protocolVer;
  dict["record_level"] = static_cast<uint32_t>(meta.recordLevel);
  dict["export_mode"] = static_cast<uint32_t>(meta.exportMode);
  dict["backend_kind"] = static_cast<uint32_t>(meta.backendKind);
  return dict;
}

BufferMeta parseBufferMeta(const py::dict &dict) {
  BufferMeta meta{};
  meta.runId = getUInt64Or(dict, "run_id", 0);
  meta.deviceId = getUInt32Or(dict, "device_id", 0);
  meta.kernelId = getUInt32Or(dict, "kernel_id", 0);
  meta.protocolVer = static_cast<uint16_t>(
      getUInt32Or(dict, "protocol_version", kProtocolVersion));
  meta.recordLevel = parseRecordLevel(lookup(dict, "record_level"));
  meta.exportMode = parseExportMode(lookup(dict, "export_mode"));
  meta.backendKind =
      static_cast<BackendKind>(getUInt32Or(dict, "backend_kind", 0));
  return meta;
}

py::dict toPyRuntimeMetadata(const DebugRuntimeMetadata &runtimeMetadata) {
  py::dict dict;

  py::list buffers;
  for (const auto &buffer : runtimeMetadata.buffers) {
    py::dict entry;
    entry["buffer_id"] = buffer.bufferId;
    entry["buffer_name"] = buffer.bufferName;
    entry["base_address"] = buffer.baseAddress;
    entry["size_bytes"] = buffer.sizeBytes;
    entry["alignment"] = buffer.alignment;
    buffers.append(entry);
  }
  dict["buffers"] = std::move(buffers);

  py::list tensors;
  for (const auto &tensor : runtimeMetadata.tensors) {
    py::dict entry;
    entry["argument_index"] = tensor.argumentIndex;
    entry["logical_name"] = tensor.logicalName;
    entry["dtype"] = tensor.dtype;
    entry["shape"] = tensor.shape;
    entry["stride"] = tensor.stride;
    entry["layout"] = tensor.layout;
    entry["buffer_id"] = tensor.bufferId;
    entry["base_address"] = tensor.baseAddress;
    entry["size_bytes"] = tensor.sizeBytes;
    tensors.append(entry);
  }
  dict["tensors"] = std::move(tensors);
  if (runtimeMetadata.hasLaunchGrid) {
    dict["grid"] = py::make_tuple(runtimeMetadata.gridX, runtimeMetadata.gridY,
                                  runtimeMetadata.gridZ);
  }
  dict["records_per_instance"] = runtimeMetadata.recordsPerInstance;
  dict["record_layout"] = runtimeMetadata.recordLayout;
  py::list recordPlan;
  for (const auto &entry : runtimeMetadata.recordPlan) {
    py::dict planEntry;
    planEntry["record_index"] = entry.recordIndex;
    planEntry["op_id"] = entry.opId;
    planEntry["scope_id"] = entry.scopeId;
    planEntry["record_kind"] = static_cast<uint32_t>(entry.recordKind);
    planEntry["collector_kind"] = static_cast<uint32_t>(entry.collectorKind);
    planEntry["result_type"] = static_cast<uint32_t>(entry.resultType);
    planEntry["event_kind"] = static_cast<uint32_t>(entry.eventKind);
    recordPlan.append(planEntry);
  }
  dict["record_plan"] = std::move(recordPlan);

  py::list fullDumpArtifacts;
  for (const auto &artifact : runtimeMetadata.fullDumpArtifacts) {
    py::dict entry;
    entry["op_id"] = artifact.opId;
    entry["logical_instance_id"] = artifact.logicalInstanceId;
    entry["payload_offset"] = artifact.payloadOffset;
    entry["payload_length"] = artifact.payloadLength;
    entry["kind"] = artifact.kind;
    entry["path"] = artifact.path;
    fullDumpArtifacts.append(entry);
  }
  dict["full_dump_artifacts"] = std::move(fullDumpArtifacts);

  return dict;
}

py::dict toPyRingBufferHeader(const RingBufferHeader &header) {
  py::dict dict;
  dict["write_idx"] = header.writeIdx;
  dict["capacity"] = header.capacity;
  dict["overflow_count"] = header.overflowCount;
  dict["flags"] = header.flags;
  dict["record_size"] = header.recordSize;
  dict["payload_offset"] = header.payloadOffset;
  return dict;
}

py::dict toPyExportedRun(const DebugExportedRun &run) {
  py::dict dict;
  dict["meta"] = toPyBufferMeta(run.meta);
  dict["runtime_metadata"] = toPyRuntimeMetadata(run.runtimeMetadata);
  dict["raw_buffer"] =
      py::bytes(reinterpret_cast<const char *>(run.rawBuffer.data()),
                static_cast<py::ssize_t>(run.rawBuffer.size()));
  return dict;
}

DebugExportedRun parseExportedRun(const py::dict &dict) {
  DebugExportedRun run;
  py::handle meta = lookup(dict, "meta");
  if (meta && !meta.is_none()) {
    run.meta = parseBufferMeta(py::cast<py::dict>(meta));
  }

  py::handle runtimeMetadata = lookup(dict, "runtime_metadata");
  run.runtimeMetadata = parseRuntimeMetadata(runtimeMetadata);

  py::handle rawBuffer = lookup(dict, "raw_buffer");
  if (rawBuffer && !rawBuffer.is_none()) {
    py::bytes bytes = py::cast<py::bytes>(rawBuffer);
    std::string data = bytes;
    run.rawBuffer.assign(data.begin(), data.end());
  }
  return run;
}

void parseReportInputs(const py::dict &exportedRun,
                       const std::string &metadataJson,
                       DecodedDebugRun &decoded,
                       KernelDebugMetadata &metadata) {
  DebugExportedRun run = parseExportedRun(exportedRun);
  std::string errorMessage;
  if (!decodeExportedRun(run, decoded, &errorMessage)) {
    throw py::value_error(errorMessage);
  }
  if (!parseKernelDebugMetadataFromJson(metadataJson, metadata,
                                        &errorMessage)) {
    throw py::value_error(errorMessage);
  }
}

ReportOptions makeStatementReportOptions() {
  ReportOptions options;
  options.includeDynamicRecords = true;
  options.includeStaticMetadata = false;
  options.includeAggregates = false;
  options.includeStatementRecords = true;
  options.includeOpLog = false;
  options.includeReportHeader = false;
  return options;
}

ReportOptions makeOpLogReportOptions() {
  ReportOptions options;
  options.includeDynamicRecords = true;
  options.includeStaticMetadata = true;
  options.includeAggregates = false;
  options.includeStatementRecords = false;
  options.includeOpLog = true;
  options.includeReportHeader = false;
  return options;
}

py::dict toPyDecodedRecord(const DecodedRecord &record) {
  py::dict dict;
  if (const auto *summary = std::get_if<DecodedSummaryRecord>(&record)) {
    const auto &raw = summary->raw;
    dict["record_kind"] = "SUMMARY";
    dict["op_id"] = raw.header.opId;
    dict["logical_instance_id"] = raw.header.logicalInstanceId;
    dict["collector_kind"] = static_cast<uint32_t>(raw.collectorKind);
    dict["result_type"] = static_cast<uint32_t>(raw.resultType);
    dict["u64_value"] = raw.resultData.u64Val;
    dict["f32_value"] = raw.resultData.f32Val;
    dict["f64_value"] = raw.resultData.f64Val;
    return dict;
  }
  if (const auto *bundle =
          std::get_if<DecodedSummaryCountBundleRecord>(&record)) {
    const auto &raw = bundle->raw;
    dict["record_kind"] = "SUMMARY_COUNT_BUNDLE_U64";
    dict["op_id"] = raw.header.opId;
    dict["logical_instance_id"] = raw.header.logicalInstanceId;
    dict["nan_count"] = raw.nanCount;
    dict["inf_count"] = raw.infCount;
    dict["zero_count"] = raw.zeroCount;
    dict["element_count"] = raw.elementCount;
    return dict;
  }
  if (const auto *bundle =
          std::get_if<DecodedSummaryValueBundleRecord>(&record)) {
    const auto &raw = bundle->raw;
    dict["record_kind"] = "SUMMARY_VALUE_BUNDLE_F32";
    dict["op_id"] = raw.header.opId;
    dict["logical_instance_id"] = raw.header.logicalInstanceId;
    dict["mean"] = raw.meanFinite;
    dict["min"] = raw.minFinite;
    dict["max"] = raw.maxFinite;
    dict["l2_norm"] = raw.l2Norm;
    return dict;
  }
  if (const auto *memory = std::get_if<DecodedMemoryEventRecord>(&record)) {
    const auto &raw = memory->raw;
    dict["record_kind"] = "MEMORY_EVENT";
    dict["op_id"] = raw.header.opId;
    dict["logical_instance_id"] = raw.header.logicalInstanceId;
    dict["addr"] = raw.addr;
    dict["event_kind"] = static_cast<uint32_t>(raw.eventKind);
    dict["ext0"] = raw.ext0;
    return dict;
  }
  if (const auto *timeline = std::get_if<DecodedTimelineRecord>(&record)) {
    const auto &raw = timeline->raw;
    dict["record_kind"] = "TIMELINE";
    dict["op_id"] = raw.header.opId;
    dict["logical_instance_id"] = raw.header.logicalInstanceId;
    dict["start_cycle"] = raw.startCycle;
    dict["end_cycle"] = raw.endCycle;
    dict["duration_cycle"] = raw.durationCycle;
    return dict;
  }
  const auto &raw = std::get<DecodedFullValueRefRecord>(record).raw;
  dict["record_kind"] = "FULL_VALUE";
  dict["op_id"] = raw.header.opId;
  dict["logical_instance_id"] = raw.header.logicalInstanceId;
  dict["payload_offset"] = raw.payloadOffset;
  dict["payload_length"] = raw.payloadLength;
  return dict;
}

py::dict toPyDecodedRun(const DecodedDebugRun &run) {
  py::dict dict;
  dict["meta"] = toPyBufferMeta(run.meta);
  dict["header"] = toPyRingBufferHeader(run.header);
  dict["runtime_metadata"] = toPyRuntimeMetadata(run.runtimeMetadata);

  py::list records;
  for (const auto &record : run.records)
    records.append(toPyDecodedRecord(record));
  dict["records"] = std::move(records);
  return dict;
}

class PreparedLaunchHandle {
public:
  PreparedLaunchHandle(const py::dict &metadata, uint64_t streamHandle,
                       py::handle runtimeMetadataValue) {
    auto bridge = createFrontendBridge();

    DebugCompileRequest compileRequest;
    compileRequest.kernelName = getStringOr(metadata, "debug_kernel_name",
                                            getStringOr(metadata, "name", ""));
    compileRequest.backendName =
        getStringOr(metadata, "debug_backend_name",
                    getStringOr(metadata, "backend_name", ""));
    compileRequest.targetName = getStringOr(metadata, "debug_target_name",
                                            getStringOr(metadata, "arch", ""));
    compileRequest.options.enabled = getBoolOr(metadata, "debug_enabled", true);
    compileRequest.options.recordLevel =
        parseRecordLevel(lookup(metadata, "debug_record_level"));
    compileRequest.options.exportMode =
        parseExportMode(lookup(metadata, "debug_export_mode"));
    compileRequest.options.recordCapacity =
        getUInt32Or(metadata, "debug_record_capacity", 1024);
    compileRequest = bridge->normalizeCompileRequest(compileRequest);

    KernelDebugMetadata kernelMetadata;
    kernelMetadata.debugKernelId = getUInt32Or(metadata, "debug_kernel_id", 1);
    kernelMetadata.kernelName = compileRequest.kernelName;
    kernelMetadata.backendName = compileRequest.backendName;
    kernelMetadata.targetName = compileRequest.targetName;

    DebugRuntimeMetadata runtimeMetadata =
        parseRuntimeMetadata(runtimeMetadataValue);

    DebugKernelArtifacts artifacts;
    bridge->attachKernelMetadata(compileRequest, artifacts, kernelMetadata);
    artifacts.metadataJson = getStringOr(metadata, "debug_metadata_json", "");
    artifacts.bufferPlan.recordSize = getUInt32Or(
        metadata, "debug_record_size", artifacts.bufferPlan.recordSize);
    uint64_t fullDumpPayloadBytesPerInstance =
        getUInt64Or(metadata, "debug_full_dump_payload_bytes_per_instance", 0);
    if (compileRequest.options.recordLevel == RecordLevel::LEVEL_TENSOR_FULL &&
        fullDumpPayloadBytesPerInstance != 0) {
      uint64_t gridProduct = launchGridProduct(runtimeMetadata);
      uint64_t requiredRecords =
          checkedMulU64(gridProduct, runtimeMetadata.recordsPerInstance,
                        "level-2 debug record count");
      if (requiredRecords > artifacts.bufferPlan.recordCapacity) {
        throw std::invalid_argument(
            "level-2 debugger full dump requires debug_record_capacity >= "
            "grid_product * records_per_instance");
      }
      uint64_t payloadBytes =
          checkedMulU64(gridProduct, fullDumpPayloadBytesPerInstance,
                        "level-2 debug payload bytes");
      uint64_t totalBytes = static_cast<uint64_t>(sizeof(RingBufferHeader)) +
                            checkedMulU64(artifacts.bufferPlan.recordCapacity,
                                          artifacts.bufferPlan.recordSize,
                                          "debug record area bytes") +
                            payloadBytes;
      if (totalBytes > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("level-2 debugger full dump buffer must "
                                    "fit 32-bit payload offsets");
      }
      artifacts.bufferPlan.payloadBytes = static_cast<size_t>(payloadBytes);
    }

    BufferMeta bufferMeta{};
    bufferMeta.runId =
        getUInt64Or(metadata, "debug_run_id",
                    nextRunId.fetch_add(1, std::memory_order_relaxed));
    bufferMeta.deviceId = getUInt32Or(metadata, "debug_device_id", 0);
    bufferMeta.kernelId = kernelMetadata.debugKernelId;
    bufferMeta.protocolVer = static_cast<uint16_t>(
        getUInt32Or(metadata, "debug_protocol_version", kProtocolVersion));
    bufferMeta.recordLevel =
        parseRecordLevel(lookup(metadata, "debug_record_level"));
    bufferMeta.exportMode =
        parseExportMode(lookup(metadata, "debug_export_mode"));
    bufferMeta.backendKind = parseBackendKind(compileRequest.backendName);

    prepared_.emplace(bridge->prepareOwnedLaunch(
        artifacts, bufferMeta, runtimeMetadata, streamHandle));
  }

  ~PreparedLaunchHandle() { releaseInternal(); }

  uint64_t hiddenArgValue() const {
    ensurePrepared();
    return prepared_->request.hiddenArgValue;
  }

  py::dict bufferMeta() const {
    ensurePrepared();
    return toPyBufferMeta(prepared_->request.bufferMeta);
  }

  py::dict runtimeMetadata() const {
    ensurePrepared();
    return toPyRuntimeMetadata(prepared_->request.runtimeMetadata);
  }

  py::dict finish() {
    ensurePrepared();
    DebugExportedRun run =
        prepared_->transferEngine->syncExport(prepared_->request.launchContext);
    py::dict exportedRun = toPyExportedRun(run);
    releaseInternal();
    return exportedRun;
  }

  void release() { releaseInternal(); }

private:
  void ensurePrepared() const {
    if (!prepared_.has_value()) {
      throw py::value_error("debugger launch handle is no longer active");
    }
  }

  void releaseInternal() {
    if (!prepared_.has_value()) {
      return;
    }
    if (prepared_->transferEngine) {
      prepared_->transferEngine->release(prepared_->request.launchContext);
      prepared_->transferEngine.reset();
    }
    prepared_.reset();
  }

  std::optional<PreparedDebugLaunch> prepared_;
};

} // namespace

void init_triton_debugger(py::module &&m) {
  py::class_<PreparedLaunchHandle>(m, "PreparedLaunchHandle")
      .def_property_readonly("hidden_arg_value",
                             &PreparedLaunchHandle::hiddenArgValue)
      .def_property_readonly("buffer_meta", &PreparedLaunchHandle::bufferMeta)
      .def_property_readonly("runtime_metadata",
                             &PreparedLaunchHandle::runtimeMetadata)
      .def("finish", &PreparedLaunchHandle::finish)
      .def("release", &PreparedLaunchHandle::release);

  m.def(
      "prepare_launch",
      [](const py::dict &metadata, uint64_t streamHandle,
         py::object runtimeMetadata) {
        return std::make_unique<PreparedLaunchHandle>(metadata, streamHandle,
                                                      runtimeMetadata);
      },
      py::arg("metadata"), py::arg("stream_handle"),
      py::arg("runtime_metadata") = py::none());

  m.def(
      "decode_exported_run",
      [](const py::dict &exportedRun) {
        DebugExportedRun run = parseExportedRun(exportedRun);
        DecodedDebugRun decoded;
        std::string errorMessage;
        if (!decodeExportedRun(run, decoded, &errorMessage)) {
          throw py::value_error(errorMessage);
        }
        return toPyDecodedRun(decoded);
      },
      py::arg("exported_run"));

  m.def(
      "render_text_report",
      [](const py::dict &exportedRun, const std::string &metadataJson) {
        DecodedDebugRun decoded;
        KernelDebugMetadata metadata;
        parseReportInputs(exportedRun, metadataJson, decoded, metadata);
        return renderTextReport(decoded, metadata);
      },
      py::arg("exported_run"), py::arg("metadata_json"));

  m.def(
      "render_json_report",
      [](const py::dict &exportedRun, const std::string &metadataJson) {
        DecodedDebugRun decoded;
        KernelDebugMetadata metadata;
        parseReportInputs(exportedRun, metadataJson, decoded, metadata);
        return renderJsonReport(decoded, metadata);
      },
      py::arg("exported_run"), py::arg("metadata_json"));

  m.def(
      "render_text_statement_report",
      [](const py::dict &exportedRun, const std::string &metadataJson) {
        DecodedDebugRun decoded;
        KernelDebugMetadata metadata;
        parseReportInputs(exportedRun, metadataJson, decoded, metadata);
        return renderTextReport(decoded, metadata,
                                makeStatementReportOptions());
      },
      py::arg("exported_run"), py::arg("metadata_json"));

  m.def(
      "render_json_statement_report",
      [](const py::dict &exportedRun, const std::string &metadataJson) {
        DecodedDebugRun decoded;
        KernelDebugMetadata metadata;
        parseReportInputs(exportedRun, metadataJson, decoded, metadata);
        return renderJsonReport(decoded, metadata,
                                makeStatementReportOptions());
      },
      py::arg("exported_run"), py::arg("metadata_json"));

  m.def(
      "render_text_op_log_report",
      [](const py::dict &exportedRun, const std::string &metadataJson) {
        DecodedDebugRun decoded;
        KernelDebugMetadata metadata;
        parseReportInputs(exportedRun, metadataJson, decoded, metadata);
        return renderTextReport(decoded, metadata, makeOpLogReportOptions());
      },
      py::arg("exported_run"), py::arg("metadata_json"));

  m.def(
      "render_json_op_log_report",
      [](const py::dict &exportedRun, const std::string &metadataJson) {
        DecodedDebugRun decoded;
        KernelDebugMetadata metadata;
        parseReportInputs(exportedRun, metadataJson, decoded, metadata);
        return renderJsonReport(decoded, metadata, makeOpLogReportOptions());
      },
      py::arg("exported_run"), py::arg("metadata_json"));
}
