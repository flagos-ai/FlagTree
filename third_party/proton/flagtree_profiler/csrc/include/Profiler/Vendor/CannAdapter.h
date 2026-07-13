#ifndef PROTON_PROFILER_VENDOR_CANN_ADAPTER_H_
#define PROTON_PROFILER_VENDOR_CANN_ADAPTER_H_

#include "Profiler/Vendor/Adapter.h"

namespace proton {

class CannMetricsImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override;

  VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                               const VendorProfilePlan &plan) const override;
};

class CannAdapter final : public VendorAdapter {
public:
  static const CannAdapter &instance();

  std::string getName() const override;

  DeviceType getDeviceType() const override;

  std::vector<std::string> getSupportedVendorMetrics() const override;

  VendorProfilePlan
  makePlan(const VendorProfileOptions &options) const override;

  Profiler *getRuntimeProfiler() const override;

  std::unique_ptr<VendorMetricsImporter> createImporter() const override;

private:
  CannAdapter() = default;
};

} // namespace proton

#endif // PROTON_PROFILER_VENDOR_CANN_ADAPTER_H_
