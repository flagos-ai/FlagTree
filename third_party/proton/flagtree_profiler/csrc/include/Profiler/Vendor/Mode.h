#ifndef PROTON_PROFILER_VENDOR_MODE_H_
#define PROTON_PROFILER_VENDOR_MODE_H_

#include "Data/Artifacts.h"

#include <map>
#include <string>
#include <vector>

namespace proton {

// Parsed user request, before any adapter-specific validation.
struct VendorProfileOptions {
  std::string rawMode{};
  bool runtimeBaseEnabled{true};
  std::vector<VendorMetricRequest> vendorMetrics{};
  std::map<std::string, std::string> adapterOptions{};
};

// Normalized execution plan after adapter-specific capability checks.
struct VendorProfilePlan {
  VendorProfileOptions requested{};
  bool runtimeBaseEnabled{true};
  std::vector<std::string> enabledVendorMetrics{};
  std::vector<std::string> disabledVendorMetrics{};
  std::vector<std::string> degradeReasons{};
};

VendorProfileOptions parseVendorProfileMode(const std::string &mode);

} // namespace proton

#endif // PROTON_PROFILER_VENDOR_MODE_H_
