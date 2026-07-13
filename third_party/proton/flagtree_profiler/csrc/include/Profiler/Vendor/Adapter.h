#ifndef PROTON_PROFILER_VENDOR_ADAPTER_H_
#define PROTON_PROFILER_VENDOR_ADAPTER_H_

#include "Data/Artifacts.h"
#include "Profiler/Vendor/Mode.h"

#include <memory>
#include <string>
#include <vector>

namespace proton {

class Profiler;

// Imports adapter-specific profiler output, such as aclprof/msprof exports.
class VendorMetricsImporter {
public:
  virtual ~VendorMetricsImporter() = default;

  virtual std::string getName() const = 0;

  virtual VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                                       const VendorProfilePlan &plan) const = 0;
};

// Adapter contract for vendor runtime backends such as CANN.
class VendorAdapter {
public:
  virtual ~VendorAdapter() = default;

  virtual std::string getName() const = 0;

  virtual DeviceType getDeviceType() const = 0;

  virtual std::vector<std::string> getSupportedVendorMetrics() const = 0;

  virtual VendorProfilePlan
  makePlan(const VendorProfileOptions &options) const = 0;

  // The concrete runtime profiler is still expected to inherit
  // proton::Profiler. Returning nullptr is acceptable for an unfinished adapter
  // skeleton.
  virtual Profiler *getRuntimeProfiler() const = 0;

  virtual std::unique_ptr<VendorMetricsImporter> createImporter() const = 0;
};

class VendorAdapterRegistry {
public:
  static const VendorAdapter *find(const std::string &name);

  static std::vector<std::string> names();
};

} // namespace proton

#endif // PROTON_PROFILER_VENDOR_ADAPTER_H_
