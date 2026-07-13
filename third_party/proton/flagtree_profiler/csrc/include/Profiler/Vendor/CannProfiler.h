#ifndef PROTON_PROFILER_CANN_PROFILER_H_
#define PROTON_PROFILER_CANN_PROFILER_H_

#include "Context/Context.h"
#include "Data/Artifacts.h"
#include "Profiler/Profiler.h"
#include "Profiler/Vendor/Mode.h"
#include "Utility/Singleton.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace proton {

class CannProfiler : public Profiler,
                     public OpInterface,
                     public Singleton<CannProfiler> {
public:
  CannProfiler();
  virtual ~CannProfiler();

  static VendorProfileArtifact
  importMsprofOutput(const SessionProfileMetadata &metadata,
                     const VendorProfilePlan &plan);

private:
  struct MsProfTxRange {
    uint64_t rangeId{0};
    bool active{false};
    bool mstxDomainRange{false};
  };

  std::vector<std::string> drainRuntimeDegradeReasons();
  std::vector<RuntimeTraceEventKey> drainRuntimeEvents();

  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;

  void doStart() override;
  void doFlush() override;
  void doStop() override;
  void doSetMode(const std::vector<std::string> &modeAndOptions) override;

  std::vector<std::string> modeAndOptions;
  std::mutex mutex;
  std::unordered_map<size_t, uint64_t> opStartTimesNs;
  std::unordered_map<size_t, MsProfTxRange> opRanges;
  std::vector<RuntimeTraceEventKey> runtimeEvents;
  std::vector<std::string> runtimeDegradeReasons;
  uint64_t aclprofDataTypeConfig{0};
  uint64_t aclprofAicoreMetricId{0};
  uint32_t deviceId{0};
  std::string outputPath{};
  std::string memFreqHz{"15"};
  std::string mstxDomainName{"proton"};
  bool msproftxEnabled{false};
  bool hostTimingFallbackEnabled{false};
  bool mstxActive{false};
  void *mstxDomain{nullptr};
  void *mstxStream{nullptr};
  bool mstxStreamOwned{false};
  void *aclprofConfig{nullptr};
  bool aclprofActive{false};
  bool aclprofRuntimeEnabled{false};
  bool aclprofAutoExportEnabled{true};
};

} // namespace proton

#endif // PROTON_PROFILER_CANN_PROFILER_H_
