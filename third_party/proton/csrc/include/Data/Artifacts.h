#ifndef PROTON_DATA_ARTIFACTS_H_
#define PROTON_DATA_ARTIFACTS_H_

#include "Data/Metric.h"
#include "Device.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace proton {

constexpr uint32_t kVendorProfileSchemaVersion = 1;

enum class ArtifactKind {
  Hatchet,
  Timeline,
  Meta,
  Vendor,
  Count,
};

enum class VendorMetricState {
  Requested,
  Enabled,
  Collected,
  Unsupported,
  Unavailable,
  Unmatched,
  Count,
};

struct ArtifactPathSpec {
  ArtifactKind kind{};
  std::string suffix{};
  bool required{false};
};

struct SessionArtifactLayout {
  std::string basePath{};
  ArtifactPathSpec hatchet{
      ArtifactKind::Hatchet, ".hatchet",
      true}; // used for aggregated analysis and regressions
  ArtifactPathSpec timeline{ArtifactKind::Timeline, ".timeline.json",
                            true}; // used for chrome/perfetto timelines
  ArtifactPathSpec meta{ArtifactKind::Meta, ".meta.json",
                        true}; // used for run metadata and schema tracking
  ArtifactPathSpec vendor{
      ArtifactKind::Vendor, ".vendor.json",
      false}; // omitted or empty when vendor mode is disabled
};

inline SessionArtifactLayout
makeDefaultSessionArtifactLayout(const std::string &basePath) {
  SessionArtifactLayout layout;
  layout.basePath = basePath;
  return layout;
}

struct RuntimeVersionInfo {
  std::string driver{};
  std::string runtime{};
};

struct DeviceSnapshot {
  DeviceType type{DeviceType::COUNT};
  std::string typeName{};
  std::string name{};
  std::string arch{};
  uint64_t id{0};
  uint64_t clockRate{0};
  uint64_t memoryClockRate{0};
  uint64_t busWidth{0};
  uint64_t numSms{0};
};

struct RuntimeTraceEventKey {
  size_t scopeId{0};
  std::string opName{};
  uint64_t taskId{0};
  uint64_t correlationId{0};
  uint64_t deviceId{0};
  uint64_t streamId{0};
  uint64_t startTimeNs{0};
  uint64_t endTimeNs{0};
};

struct VendorMetricRequest {
  std::string name{};
  bool required{false};
};

struct VendorMetricAssociation {
  RuntimeTraceEventKey runtimeEvent{};
  VendorMetricState state{VendorMetricState::Requested};
  std::string source{};
  std::string note{};
  std::map<std::string, MetricValueType> metrics{};
};

struct SessionProfileMetadata {
  uint32_t schemaVersion{kVendorProfileSchemaVersion};
  std::string runId{};
  std::string sessionName{};
  std::string backend{};
  std::string profilerName{};
  std::string context{};
  std::string data{};
  std::string hook{};
  std::string mode{};
  bool runtimeBaseEnabled{true};
  std::vector<std::string> vendorMetricsRequested{};
  std::vector<std::string> vendorMetricsEnabled{};
  std::vector<std::string> degradeReasons{};
  std::map<std::string, std::string> config{};
  RuntimeVersionInfo versions{};
  DeviceSnapshot device{};
};

struct VendorProfileArtifact {
  uint32_t schemaVersion{kVendorProfileSchemaVersion};
  std::string backend{};
  std::string importer{};
  std::vector<VendorMetricRequest> requestedMetrics{};
  std::vector<std::string> enabledMetrics{};
  std::vector<std::string> rawInputs{};
  std::vector<std::string> degradeReasons{};
  std::vector<VendorMetricAssociation> associations{};
};

} // namespace proton

#endif // PROTON_DATA_ARTIFACTS_H_
