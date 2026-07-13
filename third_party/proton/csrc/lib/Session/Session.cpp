#include "Session/Session.h"
#include "Context/Python.h"
#include "Context/Shadow.h"
#include "Data/TraceData.h"
#include "Data/TreeData.h"
#include "Device.h"
#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Profiler/Vendor/Adapter.h"
#include "Utility/String.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <variant>

namespace proton {

using json = nlohmann::json;

namespace {
Profiler *getProfiler(const std::string &profilerName,
                      const std::string &profilerPath,
                      const std::string &mode) {
  if (proton::toLower(profilerName) == "cupti") {
    auto *profiler = &CuptiProfiler::instance();
    profiler->setLibPath(profilerPath);
    if (proton::toLower(mode).find("pcsampling") == 0)
      profiler->enablePCSampling();
    return profiler;
  }
  if (proton::toLower(profilerName) == "cupti_pcsampling") {
    return &CuptiProfiler::instance().enablePCSampling();
  }
  if (proton::toLower(profilerName) == "roctracer") {
    return &RoctracerProfiler::instance();
  }
  if (proton::toLower(profilerName) == "instrumentation") {
    return InstrumentationProfiler::instance().setMode(
        proton::split(mode, ":"));
  }
  throw std::runtime_error("Unknown profiler: " + profilerName);
}

std::vector<std::string> splitMode(const std::string &mode) {
  std::vector<std::string> parts;
  if (mode.empty())
    return parts;
  size_t start = 0;
  while (start <= mode.size()) {
    size_t next = mode.find(':', start);
    parts.push_back(mode.substr(
        start, next == std::string::npos ? std::string::npos : next - start));
    if (next == std::string::npos)
      break;
    start = next + 1;
  }
  return parts;
}

Profiler *validateAndSetProfilerMode(
    Profiler *profiler, const std::string &mode,
    const std::map<size_t, std::unique_ptr<Session>> &sessions) {
  auto modeAndOptions = splitMode(mode);
  for (const auto &[id, session] : sessions) {
    if (session->getProfiler() == profiler &&
        session->getProfiler()->getMode() != modeAndOptions) {
      throw std::runtime_error("Cannot reuse a profiler with a different mode "
                               "across active sessions");
    }
  }
  return profiler->setMode(modeAndOptions);
}

std::string makeArtifactPath(const std::string &basePath,
                             const std::string &suffix) {
  if (basePath.empty() || basePath == "-") {
    return basePath;
  }
  return basePath + suffix;
}

json metricValueToJson(const MetricValueType &value) {
  return std::visit([](auto &&v) { return json(v); }, value);
}

json toJson(const RuntimeTraceEventKey &key) {
  return {{"scope_id", key.scopeId},
          {"op_name", key.opName},
          {"task_id", key.taskId},
          {"correlation_id", key.correlationId},
          {"device_id", key.deviceId},
          {"stream_id", key.streamId},
          {"start_time_ns", key.startTimeNs},
          {"end_time_ns", key.endTimeNs}};
}

std::string vendorMetricStateToString(VendorMetricState state) {
  switch (state) {
  case VendorMetricState::Requested:
    return "requested";
  case VendorMetricState::Enabled:
    return "enabled";
  case VendorMetricState::Collected:
    return "collected";
  case VendorMetricState::Unsupported:
    return "unsupported";
  case VendorMetricState::Unavailable:
    return "unavailable";
  case VendorMetricState::Unmatched:
    return "unmatched";
  case VendorMetricState::Count:
    break;
  }
  return "unknown";
}

json toJson(const VendorMetricAssociation &association) {
  json metrics = json::object();
  for (const auto &[name, value] : association.metrics) {
    metrics[name] = metricValueToJson(value);
  }
  return {{"runtime_event", toJson(association.runtimeEvent)},
          {"state", vendorMetricStateToString(association.state)},
          {"source", association.source},
          {"note", association.note},
          {"metrics", std::move(metrics)}};
}

json summarizeVendorArtifact(const VendorProfileArtifact &artifact) {
  std::map<std::string, size_t> countsBySource;
  std::map<std::string, size_t> countsByState;
  std::map<std::string, size_t> topOpTypes;
  size_t bandwidthAssociationCount = 0;
  size_t mstxRangeCount = 0;
  size_t timedAssociationCount = 0;

  for (const auto &association : artifact.associations) {
    countsBySource[association.source]++;
    countsByState[vendorMetricStateToString(association.state)]++;
    if (association.runtimeEvent.endTimeNs >
        association.runtimeEvent.startTimeNs) {
      timedAssociationCount++;
    }
    if (association.source == "msprof_mstx") {
      mstxRangeCount++;
    }
    auto bandwidthIt = association.metrics.find("bandwidth_gb_s");
    if (bandwidthIt != association.metrics.end()) {
      bandwidthAssociationCount++;
    }
    auto opTypeIt = association.metrics.find("op_type");
    if (opTypeIt != association.metrics.end()) {
      if (auto opType = std::get_if<std::string>(&opTypeIt->second)) {
        if (!opType->empty()) {
          topOpTypes[*opType]++;
        }
      }
    }
  }

  json topOpTypesJson = json::array();
  std::vector<std::pair<std::string, size_t>> sortedOpTypes(topOpTypes.begin(),
                                                            topOpTypes.end());
  std::sort(sortedOpTypes.begin(), sortedOpTypes.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });
  for (const auto &[name, count] : sortedOpTypes) {
    topOpTypesJson.push_back({{"op_type", name}, {"count", count}});
    if (topOpTypesJson.size() >= 20) {
      break;
    }
  }

  return {{"raw_input_count", artifact.rawInputs.size()},
          {"association_count", artifact.associations.size()},
          {"timed_association_count", timedAssociationCount},
          {"counts_by_source", countsBySource},
          {"counts_by_state", countsByState},
          {"bandwidth_association_count", bandwidthAssociationCount},
          {"mstx_range_count", mstxRangeCount},
          {"top_op_types", std::move(topOpTypesJson)}};
}

std::string metricValueToString(const MetricValueType &value) {
  return std::visit(
      [](auto &&v) -> std::string {
        using ValueType = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<ValueType, std::string>) {
          return v;
        } else {
          return std::to_string(v);
        }
      },
      value);
}

std::string
metricValueOrEmpty(const std::map<std::string, MetricValueType> &metrics,
                   const std::string &name) {
  auto it = metrics.find(name);
  if (it == metrics.end()) {
    return "";
  }
  return metricValueToString(it->second);
}

std::string
makeSupplementalVendorOpName(const VendorMetricAssociation &association,
                             uint64_t syntheticIndex) {
  std::string name = "CANN " + association.source;
  auto rowIndex = metricValueOrEmpty(association.metrics, "summary_row_index");
  name += " #" + std::to_string(syntheticIndex);
  if (!rowIndex.empty() && rowIndex != std::to_string(syntheticIndex)) {
    name += " row " + rowIndex;
  }

  auto opType = metricValueOrEmpty(association.metrics, "op_type");
  auto apiName = metricValueOrEmpty(association.metrics, "api_name");
  auto metricName = metricValueOrEmpty(association.metrics, "metric");
  if (!opType.empty()) {
    name += " " + opType;
  } else if (!apiName.empty()) {
    name += " " + apiName;
  } else if (!metricName.empty()) {
    name += " " + metricName;
  }
  return name;
}

std::string makeUniqueRuntimeOpName(const RuntimeTraceEventKey &event) {
  auto name = event.opName.empty() ? std::string("CANN task") : event.opName;
  if (event.taskId != 0) {
    name += " [task " + std::to_string(event.taskId) + "]";
  } else if (event.correlationId != 0) {
    name += " [corr " + std::to_string(event.correlationId) + "]";
  } else {
    name += " [" + std::to_string(event.startTimeNs) + "]";
  }
  return name;
}

std::map<std::string, MetricValueType>
makeVendorMetrics(const VendorMetricAssociation &association,
                  const RuntimeTraceEventKey &event,
                  bool syntheticTimelineEvent) {
  std::map<std::string, MetricValueType> vendorMetrics;
  vendorMetrics["vendor.source"] = association.source;
  vendorMetrics["vendor.state"] = vendorMetricStateToString(association.state);
  vendorMetrics["vendor.synthetic_timeline_event"] =
      syntheticTimelineEvent ? std::string("true") : std::string("false");
  if (!association.note.empty()) {
    vendorMetrics["vendor.note"] = association.note;
  }
  vendorMetrics["runtime.op_name"] = event.opName;
  vendorMetrics["runtime.task_id"] = event.taskId;
  vendorMetrics["runtime.correlation_id"] = event.correlationId;
  vendorMetrics["runtime.device_id"] = event.deviceId;
  vendorMetrics["runtime.stream_id"] = event.streamId;
  vendorMetrics["runtime.start_time_ns"] = event.startTimeNs;
  vendorMetrics["runtime.end_time_ns"] = event.endTimeNs;
  vendorMetrics["runtime.duration_us"] =
      static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
  if (association.source == "aclprof_op_summary" &&
      event.endTimeNs > event.startTimeNs) {
    vendorMetrics["cann.op_summary_begin_time_us"] =
        static_cast<double>(event.startTimeNs) / 1000.0;
    vendorMetrics["cann.op_summary_finish_time_us"] =
        static_cast<double>(event.endTimeNs) / 1000.0;
  }
  for (const auto &[name, value] : association.metrics) {
    vendorMetrics["cann." + name] = value;
  }
  return vendorMetrics;
}

void preserveLaunchTimingForMergedOpSummary(
    const VendorMetricAssociation &association,
    std::map<std::string, MetricValueType> &vendorMetrics) {
  if (association.source != "aclprof_op_summary" ||
      (association.runtimeEvent.scopeId == 0 &&
       association.metrics.find("merged_with_launch_range") ==
           association.metrics.end())) {
    return;
  }
  auto taskDurationIt = association.metrics.find("task_duration_us");
  if (taskDurationIt != association.metrics.end()) {
    vendorMetrics["cann.op_summary_task_duration_us"] = taskDurationIt->second;
  }
  vendorMetrics.erase("cann.task_duration_us");
  vendorMetrics.erase("runtime.duration_us");
  vendorMetrics.erase("runtime.start_time_ns");
  vendorMetrics.erase("runtime.end_time_ns");
}

json toJson(const VendorMetricRequest &request) {
  return {{"name", request.name}, {"required", request.required}};
}

json toJson(const VendorProfileArtifact &artifact) {
  json requested = json::array();
  for (const auto &request : artifact.requestedMetrics) {
    requested.push_back(toJson(request));
  }

  json associations = json::array();
  for (const auto &association : artifact.associations) {
    associations.push_back(toJson(association));
  }

  return {{"schema_version", artifact.schemaVersion},
          {"backend", artifact.backend},
          {"importer", artifact.importer},
          {"requested_metrics", std::move(requested)},
          {"enabled_metrics", artifact.enabledMetrics},
          {"raw_inputs", artifact.rawInputs},
          {"summary", summarizeVendorArtifact(artifact)},
          {"degrade_reasons", artifact.degradeReasons},
          {"associations", std::move(associations)}};
}

void writeJsonArtifact(const std::string &path, const json &object) {
  if (path.empty() || path == "-") {
    std::cout << object.dump(2) << std::endl;
    return;
  }
  std::ofstream os(path);
  os << object.dump(2) << std::endl;
}

std::string makeRunId(size_t sessionId) {
  auto now = std::chrono::system_clock::now().time_since_epoch().count();
  return std::to_string(now) + "-" + std::to_string(sessionId);
}

void appendUnique(std::vector<std::string> &target,
                  const std::vector<std::string> &source) {
  for (const auto &item : source) {
    if (std::find(target.begin(), target.end(), item) == target.end()) {
      target.push_back(item);
    }
  }
}

bool parseBoolOption(const std::string &value) {
  auto normalized = toLower(trim(value));
  return normalized == "1" || normalized == "true" || normalized == "on" ||
         normalized == "yes";
}

std::string makeTemporaryCannOutputRoot(size_t sessionId) {
  auto now = std::chrono::system_clock::now().time_since_epoch().count();
  auto name = "proton_cann_profile_" + std::to_string(now) + "_" +
              std::to_string(sessionId);
  std::error_code ec;
  auto root = std::filesystem::temp_directory_path(ec);
  if (ec) {
    root = std::filesystem::path("/tmp");
  }
  return (root / name).string();
}

bool adapterOptionEnabled(const std::map<std::string, std::string> &options,
                          const std::vector<std::string> &keys,
                          bool defaultValue) {
  for (const auto &key : keys) {
    auto it = options.find(key);
    if (it != options.end()) {
      return parseBoolOption(it->second);
    }
  }
  return defaultValue;
}

void isolateCannRuntimeOutputPath(VendorProfilePlan &plan, size_t sessionId) {
  if (!adapterOptionEnabled(plan.requested.adapterOptions,
                            {"aclprof_runtime_enabled", "cann_aclprof_runtime"},
                            false)) {
    return;
  }

  auto &options = plan.requested.adapterOptions;
  std::string pathKey = "aclprof_output_path";
  if (options.find(pathKey) == options.end() &&
      options.find("output_path") != options.end()) {
    pathKey = "output_path";
  }

  bool useTemporaryRoot =
      adapterOptionEnabled(options, {"aclprof_output_path_temporary"}, false);
  auto rootIt = options.find(pathKey);
  std::string root = rootIt != options.end() ? rootIt->second : "";
  if (trim(root).empty() && useTemporaryRoot) {
    root = makeTemporaryCannOutputRoot(sessionId);
  }
  if (trim(root).empty()) {
    root = "./proton_cann_profile";
  }

  auto isolated = std::filesystem::path(root) /
                  ("proton_session_" + std::to_string(sessionId));
  options["aclprof_output_root"] = root;
  options[pathKey] = isolated.string();
  if (useTemporaryRoot) {
    options["aclprof_output_retained"] = "false";
  }
  if (pathKey == "aclprof_output_path") {
    options.erase("output_path");
  } else {
    options.erase("aclprof_output_path");
  }
}

void cleanupTemporaryCannOutput(const VendorProfilePlan &plan,
                                std::map<std::string, std::string> &config,
                                std::vector<std::string> &degradeReasons) {
  if (!adapterOptionEnabled(plan.requested.adapterOptions,
                            {"aclprof_output_path_temporary"}, false)) {
    return;
  }
  auto rootIt = plan.requested.adapterOptions.find("aclprof_output_root");
  if (rootIt == plan.requested.adapterOptions.end() ||
      trim(rootIt->second).empty()) {
    return;
  }
  std::error_code ec;
  auto removed = std::filesystem::remove_all(rootIt->second, ec);
  if (ec) {
    appendUnique(degradeReasons,
                 {"Failed to remove temporary CANN profiler output '" +
                  rootIt->second + "': " + ec.message()});
    config["aclprof_output_cleanup"] = "failed";
  } else {
    config["aclprof_output_cleanup"] = "removed";
    config["aclprof_output_cleanup_removed_entries"] = std::to_string(removed);
  }
}

size_t overlayVendorRuntimeMetrics(Data *treeData, Data *timelineData,
                                   const VendorProfileArtifact &artifact,
                                   DeviceType deviceType) {
  if (!treeData) {
    return 0;
  }

  uint64_t firstTimedStartNs = std::numeric_limits<uint64_t>::max();
  for (const auto &association : artifact.associations) {
    const auto &event = association.runtimeEvent;
    if (event.endTimeNs > event.startTimeNs) {
      firstTimedStartNs = std::min(firstTimedStartNs, event.startTimeNs);
    }
  }
  if (firstTimedStartNs == std::numeric_limits<uint64_t>::max()) {
    firstTimedStartNs = 0;
  }

  constexpr uint64_t kSupplementalStreamBase = 100000;
  std::map<std::string, uint64_t> supplementalStreams;
  uint64_t nextSupplementalStream = kSupplementalStreamBase;
  uint64_t nextSupplementalOffsetNs = 0;
  uint64_t nextSupplementalIndex = 0;

  size_t overlayCount = 0;
  for (const auto &association : artifact.associations) {
    if (association.state != VendorMetricState::Collected &&
        association.state != VendorMetricState::Unmatched) {
      continue;
    }
    auto event = association.runtimeEvent;
    bool syntheticTimelineEvent = false;

    auto scopeId = event.scopeId;
    if (event.endTimeNs <= event.startTimeNs) {
      syntheticTimelineEvent = true;
      event.startTimeNs = firstTimedStartNs + nextSupplementalOffsetNs;
      event.endTimeNs = event.startTimeNs + 1000;
      nextSupplementalOffsetNs += 1000;
      auto streamIt = supplementalStreams.find(association.source);
      if (streamIt == supplementalStreams.end()) {
        streamIt = supplementalStreams
                       .emplace(association.source, nextSupplementalStream++)
                       .first;
      }
      event.streamId = streamIt->second;
      event.opName =
          makeSupplementalVendorOpName(association, nextSupplementalIndex++);
      scopeId = 0;
    }

    if (scopeId == 0) {
      if (event.opName.empty()) {
        continue;
      }
      scopeId = Scope::getNewScopeId();
      auto opName = syntheticTimelineEvent ? event.opName
                                           : makeUniqueRuntimeOpName(event);
      if (!syntheticTimelineEvent) {
        treeData->addOp(scopeId, opName);
      }
      if (timelineData) {
        timelineData->addOp(scopeId, opName);
      }
    }

    auto metric = std::make_shared<KernelMetric>(
        event.startTimeNs, event.endTimeNs, 1, event.deviceId,
        static_cast<uint64_t>(deviceType), event.streamId);
    auto vendorMetrics =
        makeVendorMetrics(association, event, syntheticTimelineEvent);
    auto mergeOpSummaryIntoExistingScope =
        association.source == "aclprof_op_summary" &&
        (association.runtimeEvent.scopeId != 0 ||
         association.metrics.find("merged_with_launch_range") !=
             association.metrics.end());
    if (mergeOpSummaryIntoExistingScope) {
      preserveLaunchTimingForMergedOpSummary(association, vendorMetrics);
    }
    if (!syntheticTimelineEvent) {
      if (!mergeOpSummaryIntoExistingScope) {
        treeData->addMetric(scopeId, metric);
      }
      treeData->addMetrics(scopeId, vendorMetrics);
    }
    if (timelineData) {
      if (!mergeOpSummaryIntoExistingScope) {
        timelineData->addMetric(scopeId, metric);
      }
      timelineData->addMetrics(scopeId, vendorMetrics);
    }
    overlayCount++;
  }
  return overlayCount;
}

size_t countAssociationsByState(const VendorProfileArtifact &artifact,
                                VendorMetricState state) {
  size_t count = 0;
  for (const auto &association : artifact.associations) {
    if (association.state == state) {
      count++;
    }
  }
  return count;
}

size_t countAssociationsBySource(const VendorProfileArtifact &artifact,
                                 const std::string &source) {
  size_t count = 0;
  for (const auto &association : artifact.associations) {
    if (association.source == source) {
      count++;
    }
  }
  return count;
}

size_t countAssociationsBySources(const VendorProfileArtifact &artifact,
                                  const std::set<std::string> &sources) {
  size_t count = 0;
  for (const auto &association : artifact.associations) {
    if (sources.count(association.source) > 0) {
      count++;
    }
  }
  return count;
}

std::unique_ptr<Data> makeData(const std::string &dataName,
                               const std::string &path,
                               ContextSource *contextSource) {
  if (toLower(dataName) == "tree") {
    return std::make_unique<TreeData>(path, contextSource);
  }
  if (toLower(dataName) == "trace") {
    return std::make_unique<TraceData>(path, contextSource);
  }
  throw std::runtime_error("Unknown data: " + dataName);
}

std::unique_ptr<ContextSource>
makeContextSource(const std::string &contextSourceName) {
  if (toLower(contextSourceName) == "shadow") {
    return std::make_unique<ShadowContextSource>();
  } else if (toLower(contextSourceName) == "python") {
    return std::make_unique<PythonContextSource>();
  }
  throw std::runtime_error("Unknown context source: " + contextSourceName);
}

void throwIfSessionNotInitialized(
    const std::map<size_t, std::unique_ptr<Session>> &sessions,
    size_t sessionId) {
  if (!sessions.count(sessionId)) {
    throw std::runtime_error("Session has not been initialized: " +
                             std::to_string(sessionId));
  }
}

} // namespace

void Session::activate() {
  profiler->start();
  profiler->flush();
  profiler->registerData(data.get());
  if (timelineData) {
    profiler->registerData(timelineData.get());
  }
}

void Session::deactivate() {
  profiler->flush();
  if (timelineData) {
    profiler->unregisterData(timelineData.get());
    if (!vendorAdapter) {
      timelineData->clear();
    }
  }
  profiler->unregisterData(data.get());
  if (!vendorAdapter) {
    data->clear();
  }
}

void Session::finalize(const std::string &outputFormat) {
  profiler->stop();
  if (vendorAdapter) {
    auto importer = vendorAdapter->createImporter();
    SessionProfileMetadata metadata;
    metadata.runId = makeRunId(id);
    metadata.sessionName = path;
    metadata.backend = vendorAdapter->getName();
    metadata.profilerName = profilerName;
    metadata.context = contextSourceName;
    metadata.data = dataName;
    metadata.hook = hookName;
    metadata.mode = mode;
    metadata.runtimeBaseEnabled = vendorPlan.runtimeBaseEnabled;
    metadata.degradeReasons = vendorPlan.degradeReasons;
    metadata.device.type = vendorAdapter->getDeviceType();
    metadata.device.typeName = getDeviceTypeString(metadata.device.type);
    uint64_t deviceId = 0;
    auto deviceIdIt = vendorPlan.requested.adapterOptions.find("device_id");
    if (deviceIdIt != vendorPlan.requested.adapterOptions.end()) {
      try {
        deviceId = std::stoull(deviceIdIt->second);
      } catch (...) {
        deviceId = 0;
      }
    }
    auto deviceSnapshot = getDevice(metadata.device.type, deviceId);
    metadata.device.id = deviceSnapshot.id;
    metadata.device.arch = deviceSnapshot.arch;
    metadata.device.name = deviceSnapshot.arch;
    metadata.device.clockRate = deviceSnapshot.clockRate;
    metadata.device.memoryClockRate = deviceSnapshot.memoryClockRate;
    metadata.device.busWidth = deviceSnapshot.busWidth;
    metadata.device.numSms = deviceSnapshot.numSms;

    for (const auto &request : vendorPlan.requested.vendorMetrics) {
      metadata.vendorMetricsRequested.push_back(request.name);
    }
    metadata.vendorMetricsEnabled = vendorPlan.enabledVendorMetrics;
    metadata.config["artifact_layout"] = "<base>.hatchet,<base>.timeline.json,<"
                                         "base>.meta.json,<base>.vendor.json";
    for (const auto &[key, value] : vendorPlan.requested.adapterOptions) {
      metadata.config[key] = value;
    }

    auto vendorArtifact = importer->import(metadata, vendorPlan);
    appendUnique(metadata.degradeReasons, vendorArtifact.degradeReasons);
    auto overlayCount = overlayVendorRuntimeMetrics(
        data.get(), timelineData.get(), vendorArtifact, metadata.device.type);
    auto collectedCount =
        countAssociationsByState(vendorArtifact, VendorMetricState::Collected);
    auto unmatchedCount =
        countAssociationsByState(vendorArtifact, VendorMetricState::Unmatched);
    auto hostFallbackCount =
        countAssociationsBySource(vendorArtifact, "runtime_base_fallback");
    auto nativeBaseCount = countAssociationsBySources(
        vendorArtifact, {"aclprof_op_summary", "aclprof_task_time"});
    metadata.config["vendor_runtime_metric_overlays"] =
        std::to_string(overlayCount);
    metadata.config["vendor_association_collected"] =
        std::to_string(collectedCount);
    metadata.config["vendor_association_unmatched"] =
        std::to_string(unmatchedCount);
    metadata.config["runtime_base_native_associations"] =
        std::to_string(nativeBaseCount);
    metadata.config["runtime_base_host_fallback_associations"] =
        std::to_string(hostFallbackCount);

    cleanupTemporaryCannOutput(vendorPlan, metadata.config,
                               metadata.degradeReasons);

    data->dumpToPath(makeArtifactPath(path, ".hatchet"), "hatchet");
    if (timelineData) {
      timelineData->dumpToPath(makeArtifactPath(path, ".timeline.json"),
                               "chrome_trace");
    }

    json config = json::object();
    for (const auto &[key, value] : metadata.config) {
      config[key] = value;
    }
    json device = {{"type", metadata.device.typeName},
                   {"id", metadata.device.id},
                   {"name", metadata.device.name},
                   {"arch", metadata.device.arch},
                   {"clock_rate", metadata.device.clockRate},
                   {"memory_clock_rate", metadata.device.memoryClockRate},
                   {"bus_width", metadata.device.busWidth},
                   {"num_sms", metadata.device.numSms}};
    json meta = {{"schema_version", metadata.schemaVersion},
                 {"run_id", metadata.runId},
                 {"session_name", metadata.sessionName},
                 {"backend", metadata.backend},
                 {"profiler_name", metadata.profilerName},
                 {"context", metadata.context},
                 {"data", metadata.data},
                 {"hook", metadata.hook},
                 {"mode", metadata.mode},
                 {"runtime_base_enabled", metadata.runtimeBaseEnabled},
                 {"vendor_metrics_requested", metadata.vendorMetricsRequested},
                 {"vendor_metrics_enabled", metadata.vendorMetricsEnabled},
                 {"degrade_reasons", metadata.degradeReasons},
                 {"config", std::move(config)},
                 {"versions",
                  {{"driver", metadata.versions.driver},
                   {"runtime", metadata.versions.runtime}}},
                 {"device", std::move(device)}};

    writeJsonArtifact(makeArtifactPath(path, ".meta.json"), meta);
    writeJsonArtifact(makeArtifactPath(path, ".vendor.json"),
                      toJson(vendorArtifact));
    return;
  }
  data->dump(outputFormat);
}

size_t Session::getContextDepth() { return contextSource->getDepth(); }

std::unique_ptr<Session> SessionManager::makeSession(
    size_t id, const std::string &path, const std::string &profilerName,
    const std::string &profilerPath, const std::string &contextSourceName,
    const std::string &dataName, const std::string &mode,
    const std::string &hookName) {
  if (const auto *vendorAdapter = VendorAdapterRegistry::find(profilerName)) {
    for (const auto &[existingId, existingSession] : sessions) {
      (void)existingId;
      if (existingSession->vendorAdapter == vendorAdapter) {
        throw std::runtime_error(
            "Vendor backend '" + vendorAdapter->getName() +
            "' does not support overlapping sessions. Finalize the active "
            "session before starting another one.");
      }
    }

    auto vendorOptions = parseVendorProfileMode(mode);
    auto vendorPlan = vendorAdapter->makePlan(vendorOptions);
    if (vendorAdapter->getName() == "cann") {
      isolateCannRuntimeOutputPath(vendorPlan, id);
    }
    if (toLower(dataName) != "tree") {
      vendorPlan.degradeReasons.push_back(
          "backend=cann currently emits tree base data; requested data=" +
          dataName + " was ignored.");
    }
    auto *profiler = vendorAdapter->getRuntimeProfiler();
    if (!profiler) {
      throw std::runtime_error("Vendor backend has no runtime profiler: " +
                               profilerName);
    }
    std::string profilerMode = mode;
    for (const auto &[key, value] : vendorPlan.requested.adapterOptions) {
      if (!profilerMode.empty()) {
        profilerMode += ":";
      }
      profilerMode += key + "=" + value;
    }
    profiler = validateAndSetProfilerMode(profiler, profilerMode, sessions);
    auto contextSource = makeContextSource(contextSourceName);
    auto data = makeData("tree", path, contextSource.get());
    auto timelineData = std::make_unique<TraceData>(path, contextSource.get());
    auto *session = new Session(
        id, path, profiler, std::move(contextSource), std::move(data),
        profilerName, contextSourceName, dataName, mode, hookName,
        vendorAdapter, std::move(vendorPlan), std::move(timelineData));
    return std::unique_ptr<Session>(session);
  }

  auto *profiler = getProfiler(profilerName, profilerPath, mode);
  profiler = validateAndSetProfilerMode(profiler, mode, sessions);
  auto contextSource = makeContextSource(contextSourceName);
  auto data = makeData(dataName, path, contextSource.get());
  auto *session =
      new Session(id, path, profiler, std::move(contextSource), std::move(data),
                  profilerName, contextSourceName, dataName, mode, hookName);
  return std::unique_ptr<Session>(session);
}

void SessionManager::activateSession(size_t sessionId) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  activateSessionImpl(sessionId);
}

void SessionManager::activateAllSessions() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  for (const auto &[sessionId, _] : activeSessions) {
    activateSessionImpl(sessionId);
  }
}

void SessionManager::deactivateSession(size_t sessionId) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  deActivateSessionImpl(sessionId);
}

void SessionManager::deactivateAllSessions() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  for (const auto &[sessionId, _] : activeSessions) {
    deActivateSessionImpl(sessionId);
  }
}

void SessionManager::activateSessionImpl(size_t sessionId) {
  throwIfSessionNotInitialized(sessions, sessionId);
  if (activeSessions[sessionId])
    return;
  activeSessions[sessionId] = true;
  sessions[sessionId]->activate();
  registerInterface<ScopeInterface>(sessionId, scopeInterfaceCounts);
  registerInterface<OpInterface>(sessionId, opInterfaceCounts);
  registerInterface<InstrumentationInterface>(sessionId,
                                              instrumentationInterfaceCounts);
  registerInterface<ContextSource>(sessionId, contextSourceCounts);
}

void SessionManager::deActivateSessionImpl(size_t sessionId) {
  throwIfSessionNotInitialized(sessions, sessionId);
  if (!activeSessions[sessionId]) {
    return;
  }
  activeSessions[sessionId] = false;
  sessions[sessionId]->deactivate();
  unregisterInterface<ScopeInterface>(sessionId, scopeInterfaceCounts);
  unregisterInterface<OpInterface>(sessionId, opInterfaceCounts);
  unregisterInterface<InstrumentationInterface>(sessionId,
                                                instrumentationInterfaceCounts);
  unregisterInterface<ContextSource>(sessionId, contextSourceCounts);
}

void SessionManager::removeSession(size_t sessionId) {
  if (!hasSession(sessionId)) {
    return;
  }
  auto path = sessions[sessionId]->path;
  sessionPaths.erase(path);
  sessions.erase(sessionId);
}

size_t SessionManager::addSession(const std::string &path,
                                  const std::string &profilerName,
                                  const std::string &profilerPath,
                                  const std::string &contextSourceName,
                                  const std::string &dataName,
                                  const std::string &mode,
                                  const std::string &hookName) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  if (hasSession(path)) {
    auto sessionId = getSessionId(path);
    activateSessionImpl(sessionId);
    return sessionId;
  }
  auto sessionId = nextSessionId++;
  sessionPaths[path] = sessionId;
  sessions[sessionId] =
      makeSession(sessionId, path, profilerName, profilerPath,
                  contextSourceName, dataName, mode, hookName);
  return sessionId;
}

void SessionManager::finalizeSession(size_t sessionId,
                                     const std::string &outputFormat) {
  std::unique_ptr<Session> session;
  {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (!hasSession(sessionId)) {
      return;
    }
    deActivateSessionImpl(sessionId);
    auto sessionIt = sessions.find(sessionId);
    auto path = sessionIt->second->path;
    sessionPaths.erase(path);
    activeSessions.erase(sessionId);
    session = std::move(sessionIt->second);
    sessions.erase(sessionIt);
  }
  session->finalize(outputFormat);
}

void SessionManager::finalizeAllSessions(const std::string &outputFormat) {
  std::vector<std::unique_ptr<Session>> sessionsToFinalize;
  {
    std::unique_lock<std::shared_mutex> lock(mutex);
    auto sessionIds = std::vector<size_t>{};
    for (const auto &[sessionId, session] : sessions) {
      (void)session;
      sessionIds.push_back(sessionId);
    }
    for (auto sessionId : sessionIds) {
      deActivateSessionImpl(sessionId);
    }
    for (auto &[sessionId, session] : sessions) {
      (void)sessionId;
      sessionsToFinalize.push_back(std::move(session));
    }
    sessions.clear();
    sessionPaths.clear();
    activeSessions.clear();
  }
  for (auto &session : sessionsToFinalize) {
    session->finalize(outputFormat);
  }
}

void SessionManager::enterScope(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : scopeInterfaceCounts) {
    auto [scopeInterface, count] = iter;
    if (count > 0) {
      scopeInterface->enterScope(scope);
    }
  }
}

void SessionManager::exitScope(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : scopeInterfaceCounts) {
    auto [scopeInterface, count] = iter;
    if (count > 0) {
      scopeInterface->exitScope(scope);
    }
  }
}

void SessionManager::enterOp(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto [sessionId, active] : activeSessions) {
    if (!active) {
      continue;
    }
    sessions[sessionId]->data->addOp(scope.scopeId, scope.name);
    if (sessions[sessionId]->timelineData) {
      sessions[sessionId]->timelineData->addOp(scope.scopeId, scope.name);
    }
  }
  for (auto iter : opInterfaceCounts) {
    auto [opInterface, count] = iter;
    if (count > 0) {
      opInterface->enterOp(scope);
    }
  }
}

void SessionManager::exitOp(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : opInterfaceCounts) {
    auto [opInterface, count] = iter;
    if (count > 0) {
      opInterface->exitOp(scope);
    }
  }
}

void SessionManager::initFunctionMetadata(
    uint64_t functionId, const std::string &functionName,
    const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
    const std::vector<std::pair<size_t, size_t>> &scopeIdParents,
    const std::string &metadataPath) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (const auto &[interface, count] : instrumentationInterfaceCounts) {
    if (count > 0) {
      interface->initFunctionMetadata(functionId, functionName, scopeIdNames,
                                      scopeIdParents, metadataPath);
    }
  }
}

void SessionManager::enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                                         uint8_t *buffer, size_t size) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (const auto &[interface, count] : instrumentationInterfaceCounts) {
    if (count > 0) {
      interface->enterInstrumentedOp(streamId, functionId, buffer, size);
    }
  }
}

void SessionManager::exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                                        uint8_t *buffer, size_t size) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (const auto &[interface, count] : instrumentationInterfaceCounts) {
    if (count > 0) {
      interface->exitInstrumentedOp(streamId, functionId, buffer, size);
    }
  }
}

void SessionManager::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto [sessionId, active] : activeSessions) {
    if (active) {
      sessions[sessionId]->data->addMetrics(scopeId, metrics);
      if (sessions[sessionId]->timelineData) {
        sessions[sessionId]->timelineData->addMetrics(scopeId, metrics);
      }
    }
  }
}

void SessionManager::setState(std::optional<Context> context) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (const auto &[contextSource, count] : contextSourceCounts) {
    if (count > 0) {
      contextSource->setState(context);
    }
  }
}

size_t SessionManager::getContextDepth(size_t sessionId) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  throwIfSessionNotInitialized(sessions, sessionId);
  return sessions[sessionId]->getContextDepth();
}

} // namespace proton
