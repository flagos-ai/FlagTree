#include "Profiler/Vendor/CannAdapter.h"
#include "Profiler/Vendor/CannProfiler.h"
#include "Utility/String.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace proton {

namespace {

// Keep enum values aligned with aclprofAicoreMetrics in AscendCL docs.
constexpr uint64_t kAclAicoreArithmeticUtilization = 0;
constexpr uint64_t kAclAicoreMemoryAccess = 8;

std::string join(const std::vector<std::string> &items) {
  std::ostringstream os;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << items[i];
  }
  return os.str();
}

} // namespace

std::string CannMetricsImporter::getName() const { return "msprof_importer"; }

VendorProfileArtifact
CannMetricsImporter::import(const SessionProfileMetadata &metadata,
                            const VendorProfilePlan &plan) const {
  auto artifact = CannProfiler::importMsprofOutput(metadata, plan);
  artifact.backend = metadata.backend;
  artifact.importer = getName();
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;
  artifact.degradeReasons.insert(artifact.degradeReasons.begin(),
                                 plan.degradeReasons.begin(),
                                 plan.degradeReasons.end());

  if (artifact.associations.empty() && artifact.degradeReasons.empty()) {
    artifact.degradeReasons.push_back(
        "No CANN profiling associations could be imported.");
  }
  return artifact;
}

const CannAdapter &CannAdapter::instance() {
  static const CannAdapter adapter;
  return adapter;
}

std::string CannAdapter::getName() const { return "cann"; }

DeviceType CannAdapter::getDeviceType() const { return DeviceType::ASCEND; }

std::vector<std::string> CannAdapter::getSupportedVendorMetrics() const {
  return {"aicore", "bandwidth"};
}

VendorProfilePlan
CannAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = true;

  if (!options.runtimeBaseEnabled) {
    plan.degradeReasons.push_back(
        "runtime_base=false is not supported; forcing runtime_base=true");
  }

  const auto supported = getSupportedVendorMetrics();
  for (const auto &request : options.vendorMetrics) {
    auto metricName = toLower(trim(request.name));
    if (std::find(supported.begin(), supported.end(), metricName) !=
        supported.end()) {
      if (std::find(plan.enabledVendorMetrics.begin(),
                    plan.enabledVendorMetrics.end(),
                    metricName) == plan.enabledVendorMetrics.end()) {
        plan.enabledVendorMetrics.push_back(metricName);
      }
    } else {
      plan.disabledVendorMetrics.push_back(request.name);
    }
  }

  if (!plan.disabledVendorMetrics.empty()) {
    plan.degradeReasons.push_back("Unsupported CANN vendor metrics: " +
                                  join(plan.disabledVendorMetrics));
  }

  // Prefer the in-process aclprof runtime path so proton.start()/finalize()
  // can produce vendor artifacts without requiring users to wrap their program
  // in an external msprof command. Users can still set
  // aclprof_runtime_enabled=false for file-only import or external-msprof
  // flows.
  if (plan.requested.adapterOptions.count("mstx_domain") == 0 &&
      plan.requested.adapterOptions.count("mstx_domain_name") == 0) {
    plan.requested.adapterOptions["mstx_domain"] = "proton";
  }
  if (plan.requested.adapterOptions.count("aclprof_runtime_enabled") == 0 &&
      plan.requested.adapterOptions.count("cann_aclprof_runtime") == 0) {
    plan.requested.adapterOptions["aclprof_runtime_enabled"] = "true";
  }
  if (plan.requested.adapterOptions.count("aclprof_auto_export") == 0 &&
      plan.requested.adapterOptions.count("cann_aclprof_auto_export") == 0 &&
      plan.requested.adapterOptions.count("msprof_auto_export") == 0) {
    plan.requested.adapterOptions["aclprof_auto_export"] = "true";
  }

  // Keep the legacy aclprof data-type config available for installations that
  // still enable the optional in-process aclprof runtime path.
  std::vector<std::string> aclProfFlags = {
      "ACL_PROF_ACL_API", "ACL_PROF_TASK_TIME", "ACL_PROF_RUNTIME_API"};
  if (!plan.enabledVendorMetrics.empty()) {
    aclProfFlags.push_back("ACL_PROF_AICORE_METRICS");
  }
  if (plan.requested.adapterOptions.count("aclprof_data_type_flags") == 0) {
    plan.requested.adapterOptions["aclprof_data_type_flags"] =
        join(aclProfFlags);
  }
  if (plan.requested.adapterOptions.count("aclprof_set_config_mem_freq_hz") ==
      0) {
    plan.requested.adapterOptions["aclprof_set_config_mem_freq_hz"] = "15";
  }
  if (plan.requested.adapterOptions.count("output_path") == 0 &&
      plan.requested.adapterOptions.count("aclprof_output_path") == 0) {
    const char *envOutputPath = std::getenv("PROTON_CANN_PROFILE_OUTPUT");
    if (envOutputPath && *envOutputPath) {
      plan.requested.adapterOptions["aclprof_output_path"] =
          std::string(envOutputPath);
    } else {
      plan.requested.adapterOptions["aclprof_output_path_temporary"] = "true";
    }
  }
  if (plan.requested.adapterOptions.count("runtime_host_timing_fallback") ==
          0 &&
      plan.requested.adapterOptions.count("runtime_base_host_fallback") == 0) {
    const char *envHostFallback =
        std::getenv("PROTON_CANN_RUNTIME_HOST_FALLBACK");
    plan.requested.adapterOptions["runtime_host_timing_fallback"] =
        envHostFallback && *envHostFallback ? std::string(envHostFallback)
                                            : "true";
  }
  auto useMsprofTx = std::find(plan.enabledVendorMetrics.begin(),
                               plan.enabledVendorMetrics.end(),
                               "aicore") != plan.enabledVendorMetrics.end() ||
                     std::find(plan.enabledVendorMetrics.begin(),
                               plan.enabledVendorMetrics.end(),
                               "bandwidth") != plan.enabledVendorMetrics.end();
  if (plan.requested.adapterOptions.count("mstx_enabled") == 0 &&
      plan.requested.adapterOptions.count("msproftx_enabled") == 0) {
    plan.requested.adapterOptions["mstx_enabled"] =
        useMsprofTx ? "true" : "false";
  }
  if (plan.requested.adapterOptions.count("aclprof_msproftx_enabled") == 0) {
    plan.requested.adapterOptions["aclprof_msproftx_enabled"] =
        useMsprofTx ? "true" : "false";
  }

  // aclprofCreateConfig accepts one aicore metric selector.
  // Prefer memory-access metrics if bandwidth enhancement is requested.
  uint64_t aclAicoreMetric = kAclAicoreArithmeticUtilization;
  if (std::find(plan.enabledVendorMetrics.begin(),
                plan.enabledVendorMetrics.end(),
                "bandwidth") != plan.enabledVendorMetrics.end()) {
    aclAicoreMetric = kAclAicoreMemoryAccess;
  } else if (std::find(plan.enabledVendorMetrics.begin(),
                       plan.enabledVendorMetrics.end(),
                       "aicore") != plan.enabledVendorMetrics.end()) {
    aclAicoreMetric = kAclAicoreArithmeticUtilization;
  }
  if (plan.requested.adapterOptions.count("aclprof_aicore_metric_id") == 0) {
    plan.requested.adapterOptions["aclprof_aicore_metric_id"] =
        std::to_string(aclAicoreMetric);
  }

  return plan;
}

Profiler *CannAdapter::getRuntimeProfiler() const {
  return &CannProfiler::instance();
}

std::unique_ptr<VendorMetricsImporter> CannAdapter::createImporter() const {
  return std::make_unique<CannMetricsImporter>();
}

} // namespace proton
