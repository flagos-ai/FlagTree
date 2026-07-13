#include "Profiler/Vendor/CannProfiler.h"

#include "Data/Data.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Utility/String.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace proton {

namespace {

constexpr uint64_t kAclProfAclApi = 0x00000001;
constexpr uint64_t kAclProfTaskTime = 0x00000002;
constexpr uint64_t kAclProfAicoreMetrics = 0x00000004;
constexpr uint64_t kAclProfMsprofTx = 0x00000080;
constexpr uint64_t kAclProfRuntimeApi = 0x00000100;
constexpr int kAclProfSysHardwareMemFreq = 3;

constexpr uint64_t kAclAicoreArithmeticUtilization = 0;
constexpr uint64_t kAclAicoreMemoryAccess = 8;

uint64_t nowNs() {
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::string getEnvOrDefault(const char *key, const std::string &fallback) {
  const char *value = std::getenv(key);
  if (!value) {
    return fallback;
  }
  return std::string(value);
}

uint64_t parseAclProfFlags(const std::string &flags) {
  static const std::map<std::string, uint64_t> kFlagMap = {
      {"ACL_PROF_ACL_API", kAclProfAclApi},
      {"ACL_PROF_TASK_TIME", kAclProfTaskTime},
      {"ACL_PROF_AICORE_METRICS", kAclProfAicoreMetrics},
      {"ACL_PROF_MSPROFTX", kAclProfMsprofTx},
      {"ACL_PROF_RUNTIME_API", kAclProfRuntimeApi},
  };
  uint64_t mask = 0;
  for (const auto &token : split(flags, ",")) {
    auto key = trim(token);
    auto it = kFlagMap.find(key);
    if (it != kFlagMap.end()) {
      mask |= it->second;
    }
  }
  return mask;
}

bool parseBool(const std::string &value) {
  auto normalized = toLower(trim(value));
  return normalized == "true" || normalized == "1" || normalized == "yes" ||
         normalized == "on";
}

bool adapterOptionEnabled(const std::map<std::string, std::string> &options,
                          const std::vector<std::string> &keys) {
  for (const auto &key : keys) {
    auto it = options.find(key);
    if (it != options.end() && parseBool(it->second)) {
      return true;
    }
  }
  return false;
}

#if defined(__linux__)
using AclError = int;
using AclProfConfig = void;

struct AclProfApi {
  using FnInit = AclError (*)(const char *, size_t);
  using FnFinalize = AclError (*)();
  using FnCreateConfig = AclProfConfig *(*)(uint32_t *, uint32_t, uint32_t,
                                            const void *, uint64_t);
  using FnDestroyConfig = AclError (*)(const AclProfConfig *);
  using FnStart = AclError (*)(const AclProfConfig *);
  using FnStop = AclError (*)(const AclProfConfig *);
  using FnSetConfig = AclError (*)(int, const char *, size_t);
  using FnCreateStamp = void *(*)();
  using FnDestroyStamp = AclError (*)(void *);
  using FnSetStampTraceMessage = AclError (*)(void *, const char *, size_t);
  using FnRangeStart = AclError (*)(void *, uint32_t *);
  using FnRangeStop = AclError (*)(uint32_t);

  void *aclLib{nullptr};
  void *profLib{nullptr};
  FnInit init{nullptr};
  FnFinalize finalize{nullptr};
  FnCreateConfig createConfig{nullptr};
  FnDestroyConfig destroyConfig{nullptr};
  FnStart start{nullptr};
  FnStop stop{nullptr};
  FnSetConfig setConfig{nullptr};
  FnCreateStamp createStamp{nullptr};
  FnDestroyStamp destroyStamp{nullptr};
  FnSetStampTraceMessage setStampTraceMessage{nullptr};
  FnRangeStart rangeStart{nullptr};
  FnRangeStop rangeStop{nullptr};
  bool loaded{false};
  std::string loadError{};

  ~AclProfApi() {
    if (profLib) {
      dlclose(profLib);
    }
    if (aclLib) {
      dlclose(aclLib);
    }
  }

  bool load() {
    if (loaded) {
      return true;
    }

    const std::vector<std::string> searchRoots = {
        getEnvOrDefault("ASCEND_TOOLKIT_PATH", ""),
        getEnvOrDefault("ASCEND_TOOLKIT_HOME", ""),
        "/usr/local/Ascend/cann-8.5.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit/6.0.1",
    };
    std::vector<std::string> aclCandidates = {"libacl_rt.so", "libacl.so",
                                              "libascendcl.so"};
    std::vector<std::string> profCandidates = {"libmsprofiler.so",
                                               "libacl_prof.so"};
    for (const auto &root : searchRoots) {
      if (root.empty()) {
        continue;
      }
      aclCandidates.push_back(root + "/aarch64-linux/lib64/libacl_rt.so");
      aclCandidates.push_back(root + "/lib64/libacl.so");
      aclCandidates.push_back(root + "/lib64/libascendcl.so");
      aclCandidates.push_back(root + "/aarch64-linux/lib64/libacl.so");
      aclCandidates.push_back(root + "/aarch64-linux/lib64/libascendcl.so");
      profCandidates.push_back(root + "/aarch64-linux/lib64/libmsprofiler.so");
      profCandidates.push_back(root + "/lib64/libmsprofiler.so");
      profCandidates.push_back(root + "/tools/profiler/lib64/libmsprofiler.so");
      profCandidates.push_back(
          root + "/toolkit/tools/profiler/lib64/libmsprofiler.so");

      aclCandidates.push_back(root + "/aarch64-linux/devlib/libacl_rt.so");
      aclCandidates.push_back(root + "/aarch64-linux/devlib/libacl.so");
      aclCandidates.push_back(root + "/aarch64-linux/devlib/libascendcl.so");
      aclCandidates.push_back(
          root + "/aarch64-linux/devlib/linux/aarch64/libacl_rt.so");
      aclCandidates.push_back(root +
                              "/aarch64-linux/devlib/linux/aarch64/libacl.so");
      aclCandidates.push_back(
          root + "/aarch64-linux/devlib/linux/aarch64/libascendcl.so");
      profCandidates.push_back(root + "/lib64/libacl_prof.so");
      profCandidates.push_back(root + "/aarch64-linux/lib64/libacl_prof.so");
      profCandidates.push_back(root + "/aarch64-linux/devlib/libacl_prof.so");
      profCandidates.push_back(root + "/aarch64-linux/devlib/linux/aarch64/"
                                      "libacl_prof.so");
    }

    for (const auto &cand : aclCandidates) {
      aclLib = dlopen(cand.c_str(), RTLD_LAZY | RTLD_GLOBAL);
      if (aclLib) {
        break;
      }
    }
    for (const auto &cand : profCandidates) {
      profLib = dlopen(cand.c_str(), RTLD_LAZY | RTLD_GLOBAL);
      if (profLib) {
        break;
      }
    }
    if (!aclLib || !profLib) {
      loadError = "Failed to load libacl.so/libacl_prof.so";
      return false;
    }

    init = reinterpret_cast<FnInit>(dlsym(profLib, "aclprofInit"));
    finalize = reinterpret_cast<FnFinalize>(dlsym(profLib, "aclprofFinalize"));
    createConfig =
        reinterpret_cast<FnCreateConfig>(dlsym(profLib, "aclprofCreateConfig"));
    destroyConfig = reinterpret_cast<FnDestroyConfig>(
        dlsym(profLib, "aclprofDestroyConfig"));
    start = reinterpret_cast<FnStart>(dlsym(profLib, "aclprofStart"));
    stop = reinterpret_cast<FnStop>(dlsym(profLib, "aclprofStop"));
    setConfig =
        reinterpret_cast<FnSetConfig>(dlsym(profLib, "aclprofSetConfig"));
    createStamp =
        reinterpret_cast<FnCreateStamp>(dlsym(profLib, "aclprofCreateStamp"));
    destroyStamp =
        reinterpret_cast<FnDestroyStamp>(dlsym(profLib, "aclprofDestroyStamp"));
    setStampTraceMessage = reinterpret_cast<FnSetStampTraceMessage>(
        dlsym(profLib, "aclprofSetStampTraceMessage"));
    rangeStart =
        reinterpret_cast<FnRangeStart>(dlsym(profLib, "aclprofRangeStart"));
    rangeStop =
        reinterpret_cast<FnRangeStop>(dlsym(profLib, "aclprofRangeStop"));

    if (!init || !finalize || !createConfig || !destroyConfig || !start ||
        !stop) {
      loadError = "Missing aclprof symbols required for runtime profiling";
      return false;
    }

    loaded = true;
    return true;
  }
};

AclProfApi &aclProfApi() {
  static AclProfApi api;
  return api;
}

struct AclRuntimeApi {
  using FnInit = AclError (*)(const char *);
  using FnSetDevice = AclError (*)(int32_t);
  using FnCreateStream = AclError (*)(void **);
  using FnSynchronizeStream = AclError (*)(void *);
  using FnDestroyStream = AclError (*)(void *);

  void *lib{nullptr};
  FnInit init{nullptr};
  FnSetDevice setDevice{nullptr};
  FnCreateStream createStream{nullptr};
  FnSynchronizeStream synchronizeStream{nullptr};
  FnDestroyStream destroyStream{nullptr};
  bool loaded{false};
  std::string loadError{};
  std::string loadedFrom{};

  ~AclRuntimeApi() {
    if (lib) {
      dlclose(lib);
    }
  }

  bool load() {
    if (loaded) {
      return true;
    }

    const std::vector<std::string> searchRoots = {
        getEnvOrDefault("ASCEND_TOOLKIT_PATH", ""),
        getEnvOrDefault("ASCEND_TOOLKIT_HOME", ""),
        "/usr/local/Ascend/cann-8.5.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit/6.0.1",
    };
    std::vector<std::string> candidates = {"libascendcl.so", "libacl.so"};
    for (const auto &root : searchRoots) {
      if (root.empty()) {
        continue;
      }
      candidates.push_back(root + "/lib64/libascendcl.so");
      candidates.push_back(root + "/lib64/libacl.so");
      candidates.push_back(root + "/aarch64-linux/lib64/libascendcl.so");
      candidates.push_back(root + "/aarch64-linux/lib64/libacl.so");
      candidates.push_back(root + "/aarch64-linux/devlib/libascendcl.so");
      candidates.push_back(root + "/aarch64-linux/devlib/libacl.so");
      candidates.push_back(
          root + "/aarch64-linux/devlib/linux/aarch64/libascendcl.so");
      candidates.push_back(root +
                           "/aarch64-linux/devlib/linux/aarch64/libacl.so");
    }

    std::set<std::string> tried;
    for (const auto &candidate : candidates) {
      if (!tried.insert(candidate).second) {
        continue;
      }
      lib = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_GLOBAL);
      if (!lib) {
        continue;
      }
      setDevice = reinterpret_cast<FnSetDevice>(dlsym(lib, "aclrtSetDevice"));
      createStream =
          reinterpret_cast<FnCreateStream>(dlsym(lib, "aclrtCreateStream"));
      destroyStream =
          reinterpret_cast<FnDestroyStream>(dlsym(lib, "aclrtDestroyStream"));
      synchronizeStream = reinterpret_cast<FnSynchronizeStream>(
          dlsym(lib, "aclrtSynchronizeStream"));
      init = reinterpret_cast<FnInit>(dlsym(lib, "aclInit"));
      if (setDevice && createStream && destroyStream) {
        loaded = true;
        loadedFrom = candidate;
        return true;
      }
      dlclose(lib);
      lib = nullptr;
    }

    loadError =
        "Failed to load AscendCL runtime stream APIs required for mstx ranges.";
    return false;
  }
};

AclRuntimeApi &aclRuntimeApi() {
  static AclRuntimeApi api;
  return api;
}

struct MstxApi {
  using RangeId = uint64_t;
  using DomainHandle = void *;
  using FnRangeStartA = RangeId (*)(const char *, void *);
  using FnRangeEnd = void (*)(RangeId);
  using FnDomainCreateA = DomainHandle (*)(const char *);
  using FnDomainDestroy = void (*)(DomainHandle);
  using FnDomainRangeStartA = RangeId (*)(DomainHandle, const char *, void *);
  using FnDomainRangeEnd = void (*)(DomainHandle, RangeId);

  std::vector<void *> libs;
  FnRangeStartA rangeStartA{nullptr};
  FnRangeEnd rangeEnd{nullptr};
  FnDomainCreateA domainCreateA{nullptr};
  FnDomainDestroy domainDestroy{nullptr};
  FnDomainRangeStartA domainRangeStartA{nullptr};
  FnDomainRangeEnd domainRangeEnd{nullptr};
  bool loaded{false};
  std::string loadError{};
  std::string loadedFrom{};

  ~MstxApi() {
    for (auto *lib : libs) {
      if (lib) {
        dlclose(lib);
      }
    }
  }

  bool resolveFrom(void *lib, const std::string &source) {
    auto candidateRangeStartA =
        reinterpret_cast<FnRangeStartA>(dlsym(lib, "mstxRangeStartA"));
    auto candidateRangeEnd =
        reinterpret_cast<FnRangeEnd>(dlsym(lib, "mstxRangeEnd"));
    if (!candidateRangeStartA || !candidateRangeEnd) {
      return false;
    }

    rangeStartA = candidateRangeStartA;
    rangeEnd = candidateRangeEnd;
    domainCreateA =
        reinterpret_cast<FnDomainCreateA>(dlsym(lib, "mstxDomainCreateA"));
    domainDestroy =
        reinterpret_cast<FnDomainDestroy>(dlsym(lib, "mstxDomainDestroy"));
    domainRangeStartA = reinterpret_cast<FnDomainRangeStartA>(
        dlsym(lib, "mstxDomainRangeStartA"));
    domainRangeEnd =
        reinterpret_cast<FnDomainRangeEnd>(dlsym(lib, "mstxDomainRangeEnd"));
    loaded = true;
    loadedFrom = source;
    return true;
  }

  bool load() {
    if (loaded) {
      return true;
    }

    const std::vector<std::string> searchRoots = {
        getEnvOrDefault("ASCEND_TOOLKIT_PATH", ""),
        getEnvOrDefault("ASCEND_TOOLKIT_HOME", ""),
        "/usr/local/Ascend/cann-8.5.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit/6.0.1",
    };
    const std::vector<std::string> libraryNames = {
        "libmstx.so", "libms_tools_ext.so", "libmsprofiler.so",
        "libprofapi.so"};

    std::vector<std::string> candidates;
    for (const auto &root : searchRoots) {
      if (root.empty()) {
        continue;
      }
      for (const auto &name : libraryNames) {
        candidates.push_back(root + "/tools/mstx/lib64/" + name);
        candidates.push_back(root + "/toolkit/tools/mstx/lib64/" + name);
        candidates.push_back(root + "/lib64/" + name);
        candidates.push_back(root + "/aarch64-linux/lib64/" + name);
        candidates.push_back(root + "/tools/profiler/lib64/" + name);
        candidates.push_back(root + "/tools/profiler/profiler_tool/lib64/" +
                             name);
        candidates.push_back(root + "/tools/mspti/lib64/" + name);
        candidates.push_back(root + "/toolkit/tools/mspti/lib64/" + name);
      }
    }
    candidates.insert(candidates.end(), libraryNames.begin(),
                      libraryNames.end());

    std::set<std::string> tried;
    for (const auto &candidate : candidates) {
      if (!tried.insert(candidate).second) {
        continue;
      }
      void *lib = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_GLOBAL);
      if (!lib) {
        continue;
      }
      if (resolveFrom(lib, candidate)) {
        libs.push_back(lib);
        return true;
      }
      dlclose(lib);
    }

#if defined(RTLD_DEFAULT)
    if (resolveFrom(RTLD_DEFAULT, "global symbol table")) {
      return true;
    }
#endif

    loadError = "Failed to load CANN mstx APIs. Run the program under "
                "`msprof --msproftx=on` and ensure the mstx runtime library is "
                "visible.";
    return false;
  }
};

MstxApi &mstxApi() {
  static MstxApi api;
  return api;
}
#endif

std::string normalizeColumnName(const std::string &name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9')) {
      out.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> values;
  std::string current;
  bool inQuotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (c == '"') {
      if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (c == ',' && !inQuotes) {
      values.push_back(trim(current));
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  values.push_back(trim(current));
  return values;
}

std::optional<double> parseDouble(const std::string &value) {
  auto cleaned = trim(value);
  if (cleaned.empty()) {
    return std::nullopt;
  }
  try {
    size_t parsedChars = 0;
    auto parsed = std::stod(cleaned, &parsedChars);
    if (parsedChars != cleaned.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<uint64_t> parseU64(const std::string &value) {
  auto cleaned = trim(value);
  if (cleaned.empty()) {
    return std::nullopt;
  }
  try {
    size_t parsedChars = 0;
    auto parsed = std::stoull(cleaned, &parsedChars);
    if (parsedChars != cleaned.size()) {
      return std::nullopt;
    }
    return static_cast<uint64_t>(parsed);
  } catch (...) {
    return std::nullopt;
  }
}

std::string normalizeMetricKey(const std::string &name) {
  std::string out;
  out.reserve(name.size());
  bool lastUnderscore = false;
  for (char c : name) {
    auto uc = static_cast<unsigned char>(c);
    if ((uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z') ||
        (uc >= '0' && uc <= '9')) {
      out.push_back(static_cast<char>(std::tolower(uc)));
      lastUnderscore = false;
      continue;
    }
    if (!lastUnderscore) {
      out.push_back('_');
      lastUnderscore = true;
    }
  }
  while (!out.empty() && out.front() == '_') {
    out.erase(out.begin());
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  return out;
}

std::optional<uint64_t>
parseMetricU64(const VendorMetricAssociation &association,
               const std::string &metricName) {
  auto it = association.metrics.find(metricName);
  if (it == association.metrics.end()) {
    return std::nullopt;
  }
  if (std::holds_alternative<uint64_t>(it->second)) {
    return std::get<uint64_t>(it->second);
  }
  if (std::holds_alternative<int64_t>(it->second)) {
    auto value = std::get<int64_t>(it->second);
    if (value >= 0) {
      return static_cast<uint64_t>(value);
    }
  }
  if (std::holds_alternative<double>(it->second)) {
    auto value = std::get<double>(it->second);
    if (value >= 0.0) {
      return static_cast<uint64_t>(value);
    }
  }
  return std::nullopt;
}

std::optional<double>
parseMetricDouble(const VendorMetricAssociation &association,
                  const std::string &metricName) {
  auto it = association.metrics.find(metricName);
  if (it == association.metrics.end()) {
    return std::nullopt;
  }
  if (std::holds_alternative<double>(it->second)) {
    return std::get<double>(it->second);
  }
  if (std::holds_alternative<uint64_t>(it->second)) {
    return static_cast<double>(std::get<uint64_t>(it->second));
  }
  if (std::holds_alternative<int64_t>(it->second)) {
    auto value = std::get<int64_t>(it->second);
    if (value >= 0) {
      return static_cast<double>(value);
    }
  }
  return std::nullopt;
}

bool metricEnabled(const std::vector<std::string> &enabledVendorMetrics,
                   const std::string &metricName) {
  return std::find(enabledVendorMetrics.begin(), enabledVendorMetrics.end(),
                   metricName) != enabledVendorMetrics.end();
}

bool looksLikeMemoryByteMetric(const std::string &key) {
  auto lower = toLower(key);
  bool memoryLike = lower.find("memory") != std::string::npos ||
                    lower.find("mem") != std::string::npos ||
                    lower.find("ddr") != std::string::npos ||
                    lower.find("hbm") != std::string::npos ||
                    lower.find("mte") != std::string::npos ||
                    lower.find("bytes") != std::string::npos ||
                    lower.find("byte") != std::string::npos;
  bool byteLike = lower.find("bytes") != std::string::npos ||
                  lower.find("byte") != std::string::npos ||
                  lower.find("_b") != std::string::npos ||
                  lower.find("_kb") != std::string::npos ||
                  lower.find("_mb") != std::string::npos ||
                  lower.find("_gb") != std::string::npos;
  bool accessLike = lower.find("access") != std::string::npos ||
                    lower.find("read") != std::string::npos ||
                    lower.find("write") != std::string::npos ||
                    lower.find("load") != std::string::npos ||
                    lower.find("store") != std::string::npos ||
                    lower.find("traffic") != std::string::npos;
  return memoryLike && byteLike && accessLike;
}

bool looksLikeBandwidthMetric(const std::string &key) {
  auto lower = toLower(key);
  return lower.find("bandwidth") != std::string::npos ||
         lower.find("throughput") != std::string::npos ||
         ((lower.find("read") != std::string::npos ||
           lower.find("write") != std::string::npos) &&
          (lower.find("_gb_s") != std::string::npos ||
           lower.find("_mb_s") != std::string::npos ||
           lower.find("_kb_s") != std::string::npos ||
           lower.find("_b_s") != std::string::npos));
}

double scaleMemoryMetricToBytes(const std::string &key, double value) {
  auto lower = toLower(key);
  if (lower.find("_gbytes") != std::string::npos ||
      lower.find("_gbyte") != std::string::npos ||
      lower.find("_gb") != std::string::npos) {
    return value * 1'000'000'000.0;
  }
  if (lower.find("_mbytes") != std::string::npos ||
      lower.find("_mbyte") != std::string::npos ||
      lower.find("_mb") != std::string::npos) {
    return value * 1'000'000.0;
  }
  if (lower.find("_kbytes") != std::string::npos ||
      lower.find("_kbyte") != std::string::npos ||
      lower.find("_kb") != std::string::npos) {
    return value * 1'000.0;
  }
  return value;
}

std::optional<double> scaleBandwidthMetricToGbps(const std::string &key,
                                                 double value) {
  auto lower = toLower(key);
  if (lower.find("_gb_s") != std::string::npos ||
      lower.find("_gbyte_s") != std::string::npos ||
      lower.find("_gbytes_s") != std::string::npos ||
      lower.find("_gbps") != std::string::npos) {
    return value;
  }
  if (lower.find("_mb_s") != std::string::npos ||
      lower.find("_mbyte_s") != std::string::npos ||
      lower.find("_mbytes_s") != std::string::npos ||
      lower.find("_mbps") != std::string::npos) {
    return value / 1'000.0;
  }
  if (lower.find("_kb_s") != std::string::npos ||
      lower.find("_kbyte_s") != std::string::npos ||
      lower.find("_kbytes_s") != std::string::npos ||
      lower.find("_kbps") != std::string::npos) {
    return value / 1'000'000.0;
  }
  if (lower.find("_b_s") != std::string::npos ||
      lower.find("_byte_s") != std::string::npos ||
      lower.find("_bytes_s") != std::string::npos ||
      lower.find("_bps") != std::string::npos) {
    return value / 1'000'000'000.0;
  }
  return std::nullopt;
}

void enrichBandwidthMetrics(
    VendorMetricAssociation &association,
    const std::vector<std::string> &enabledVendorMetrics) {
  if (!metricEnabled(enabledVendorMetrics, "bandwidth")) {
    return;
  }

  std::optional<double> directBandwidthGbps;
  std::optional<double> directReadBandwidthGbps;
  std::optional<double> directWriteBandwidthGbps;
  std::string directBandwidthKey;
  std::optional<double> readBytes;
  std::optional<double> writeBytes;
  std::optional<double> totalBytes;
  std::string totalBytesKey;

  for (const auto &[key, value] : association.metrics) {
    auto numeric = parseMetricDouble(association, key);
    if (!numeric.has_value()) {
      continue;
    }

    if (looksLikeBandwidthMetric(key)) {
      auto scaled = scaleBandwidthMetricToGbps(key, numeric.value());
      if (scaled.has_value()) {
        auto lower = toLower(key);
        if (lower.find("read") != std::string::npos) {
          directReadBandwidthGbps =
              directReadBandwidthGbps.value_or(0.0) + scaled.value();
        } else if (lower.find("write") != std::string::npos) {
          directWriteBandwidthGbps =
              directWriteBandwidthGbps.value_or(0.0) + scaled.value();
        } else if (!directBandwidthGbps.has_value()) {
          directBandwidthGbps = scaled.value();
          directBandwidthKey = key;
        }
      }
    }

    if (looksLikeMemoryByteMetric(key)) {
      auto bytes = scaleMemoryMetricToBytes(key, numeric.value());
      auto lower = toLower(key);
      if (lower.find("read") != std::string::npos ||
          lower.find("load") != std::string::npos) {
        readBytes = readBytes.value_or(0.0) + bytes;
      } else if (lower.find("write") != std::string::npos ||
                 lower.find("store") != std::string::npos) {
        writeBytes = writeBytes.value_or(0.0) + bytes;
      } else if (!totalBytes.has_value()) {
        totalBytes = bytes;
        totalBytesKey = key;
      }
    }
  }

  if (!totalBytes.has_value() &&
      (readBytes.has_value() || writeBytes.has_value())) {
    totalBytes = readBytes.value_or(0.0) + writeBytes.value_or(0.0);
    totalBytesKey = "memory_read_write_bytes";
  }

  if (readBytes.has_value() && association.metrics.find("memory_read_bytes") ==
                                   association.metrics.end()) {
    association.metrics["memory_read_bytes"] = readBytes.value();
  }
  if (writeBytes.has_value() &&
      association.metrics.find("memory_write_bytes") ==
          association.metrics.end()) {
    association.metrics["memory_write_bytes"] = writeBytes.value();
  }
  if (totalBytes.has_value() &&
      association.metrics.find("memory_access_bytes") ==
          association.metrics.end()) {
    association.metrics["memory_access_bytes"] = totalBytes.value();
  }

  if (!directBandwidthGbps.has_value() &&
      (directReadBandwidthGbps.has_value() ||
       directWriteBandwidthGbps.has_value())) {
    directBandwidthGbps = directReadBandwidthGbps.value_or(0.0) +
                          directWriteBandwidthGbps.value_or(0.0);
    directBandwidthKey = "memory_read_write_bandwidth";
  }

  if (directReadBandwidthGbps.has_value()) {
    association.metrics["memory_read_bandwidth_gb_s"] =
        directReadBandwidthGbps.value();
  }
  if (directWriteBandwidthGbps.has_value()) {
    association.metrics["memory_write_bandwidth_gb_s"] =
        directWriteBandwidthGbps.value();
  }
  if (directBandwidthGbps.has_value()) {
    association.metrics["bandwidth_gb_s"] = directBandwidthGbps.value();
    association.metrics["bandwidth_source"] =
        std::string("direct_csv_column:") + directBandwidthKey;
    return;
  }

  auto durationUs = parseMetricDouble(association, "task_duration_us");
  if (totalBytes.has_value() && durationUs.has_value() &&
      durationUs.value() > 0.0) {
    association.metrics["bandwidth_gb_s"] =
        totalBytes.value() / (durationUs.value() * 1000.0);
    association.metrics["bandwidth_source"] =
        std::string("derived_from_bytes_and_task_duration:") + totalBytesKey;
  }
}

std::optional<size_t>
findColumn(const std::vector<std::string> &headers,
           const std::unordered_set<std::string> &candidates) {
  for (size_t i = 0; i < headers.size(); ++i) {
    if (candidates.count(normalizeColumnName(headers[i])) > 0) {
      return i;
    }
  }
  return std::nullopt;
}

bool endsWithLower(const std::string &value, const std::string &suffix) {
  if (value.size() < suffix.size()) {
    return false;
  }
  return toLower(value.substr(value.size() - suffix.size())) == suffix;
}

bool isCandidateVendorCsvFileName(const std::string &fileName) {
  auto lowerName = toLower(fileName);
  return endsWithLower(lowerName, ".csv") &&
         (lowerName.find("op_summary") != std::string::npos ||
          lowerName.find("task_time") != std::string::npos ||
          lowerName.find("op_statistic") != std::string::npos ||
          lowerName.find("api_statistic") != std::string::npos ||
          lowerName.find("step_trace") != std::string::npos ||
          lowerName.find("kernel_details") != std::string::npos ||
          lowerName.find("mstx") != std::string::npos ||
          lowerName.find("msproftx") != std::string::npos ||
          lowerName.find("msprof_tx") != std::string::npos ||
          lowerName.find("aicpu") != std::string::npos ||
          lowerName.find("aicore") != std::string::npos ||
          lowerName.find("hbm") != std::string::npos ||
          lowerName.find("llc") != std::string::npos ||
          lowerName.find("bandwidth") != std::string::npos ||
          lowerName.find("memory") != std::string::npos ||
          lowerName.find("mem") != std::string::npos ||
          lowerName.find("summary") != std::string::npos);
}

uint64_t parseTimeToNs(const std::string &rawValue,
                       const std::string &rawHeader) {
  auto parsed = parseDouble(rawValue);
  if (!parsed.has_value()) {
    return 0;
  }
  auto header = toLower(rawHeader);
  if (header.find("ns") != std::string::npos) {
    return static_cast<uint64_t>(parsed.value());
  }
  if (header.find("ms") != std::string::npos) {
    return static_cast<uint64_t>(parsed.value() * 1'000'000.0);
  }
  // Default to microseconds for msprof/aclprof summary fields.
  return static_cast<uint64_t>(parsed.value() * 1'000.0);
}

std::vector<std::filesystem::path>
collectCandidateCsvFiles(const SessionProfileMetadata &metadata) {
  std::vector<std::filesystem::path> roots;
  std::vector<std::filesystem::path> importRoots;
  std::vector<std::filesystem::path> outputRoots;
  auto addImportRoot = [&importRoots](const std::string &value) {
    auto cleaned = trim(value);
    if (!cleaned.empty()) {
      importRoots.push_back(std::filesystem::path(cleaned));
    }
  };
  auto addOutputRoot = [&outputRoots](const std::string &value) {
    auto cleaned = trim(value);
    if (!cleaned.empty()) {
      outputRoots.push_back(std::filesystem::path(cleaned));
    }
  };

  const std::vector<std::string> importRootKeys = {
      "msprof_import_path", "cann_import_path", "vendor_import_path"};
  for (const auto &key : importRootKeys) {
    auto it = metadata.config.find(key);
    if (it != metadata.config.end()) {
      addImportRoot(it->second);
    }
  }
  addImportRoot(getEnvOrDefault("PROTON_CANN_IMPORT_PATH", ""));

  if (!importRoots.empty()) {
    roots = std::move(importRoots);
  } else {
    const std::vector<std::string> outputRootKeys = {
        "aclprof_output_path", "output_path", "msprof_output_path"};
    for (const auto &key : outputRootKeys) {
      auto it = metadata.config.find(key);
      if (it != metadata.config.end()) {
        addOutputRoot(it->second);
      }
    }

    if (!outputRoots.empty()) {
      roots = std::move(outputRoots);
    }
  }

  if (roots.empty()) {
    auto sessionPath = std::filesystem::path(metadata.sessionName);
    if (!sessionPath.empty()) {
      if (sessionPath.has_parent_path()) {
        roots.push_back(sessionPath.parent_path());
      }
      roots.push_back(sessionPath);
    }
  }

  std::set<std::filesystem::path> uniqueRoots(roots.begin(), roots.end());
  std::set<std::string> uniqueCsvFiles;
  std::vector<std::filesystem::path> csvFiles;
  auto addCsv = [&](const std::filesystem::path &path) {
    auto key = path.lexically_normal().string();
    if (uniqueCsvFiles.insert(key).second) {
      csvFiles.push_back(path);
    }
  };

  for (const auto &root : uniqueRoots) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
      continue;
    }
    if (std::filesystem::is_regular_file(root, ec)) {
      auto fileName = toLower(root.filename().string());
      if (endsWithLower(fileName, ".csv")) {
        addCsv(root);
      }
      continue;
    }
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(root, ec)) {
      if (ec || !entry.is_regular_file()) {
        continue;
      }
      auto fileName = toLower(entry.path().filename().string());
      if (isCandidateVendorCsvFileName(fileName)) {
        addCsv(entry.path());
      }
    }
  }
  return csvFiles;
}

void parseOpSummaryCsv(const std::filesystem::path &file,
                       const std::vector<std::string> &enabledVendorMetrics,
                       VendorProfileArtifact &artifact) {
  std::ifstream in(file.string());
  if (!in.is_open()) {
    artifact.degradeReasons.push_back("Failed to open vendor CSV: " +
                                      file.string());
    return;
  }

  std::string headerLine;
  if (!std::getline(in, headerLine)) {
    return;
  }
  auto headers = splitCsvLine(headerLine);
  if (headers.empty()) {
    return;
  }

  const auto opNameIdx = findColumn(headers, {"opname", "op", "kernelname"});
  const auto taskIdIdx = findColumn(headers, {"taskid"});
  const auto corrIdIdx =
      findColumn(headers, {"correlationid", "correlation_id", "corrid"});
  const auto streamIdIdx = findColumn(headers, {"streamid"});
  const auto deviceIdIdx = findColumn(headers, {"deviceid", "device"});
  const auto startIdx =
      findColumn(headers, {"taskstarttimeus", "starttimeus", "taskstarttime"});
  const auto durationIdx =
      findColumn(headers, {"taskdurationus", "durationus", "taskduration"});
  const auto aiCoreTimeIdx =
      findColumn(headers, {"aicoretimems", "aicoretime"});
  const auto totalCycleIdx = findColumn(headers, {"totalcycle", "totalcycles"});

  size_t parsedRows = 0;
  std::string line;
  while (std::getline(in, line)) {
    auto row = splitCsvLine(line);
    if (row.empty()) {
      continue;
    }
    auto getCell = [&row](std::optional<size_t> idx) -> std::string {
      if (!idx.has_value() || idx.value() >= row.size()) {
        return {};
      }
      return trim(row[idx.value()]);
    };

    auto startRaw = getCell(startIdx);
    auto durationRaw = getCell(durationIdx);
    if (startRaw.empty() || durationRaw.empty()) {
      continue;
    }
    uint64_t startTimeNs =
        parseTimeToNs(startRaw, startIdx.has_value() ? headers[startIdx.value()]
                                                     : "start_time_us");
    uint64_t durationNs = parseTimeToNs(
        durationRaw,
        durationIdx.has_value() ? headers[durationIdx.value()] : "duration_us");
    if (startTimeNs == 0 || durationNs == 0) {
      continue;
    }

    VendorMetricAssociation association;
    association.state = VendorMetricState::Collected;
    association.source = "aclprof_op_summary";
    association.runtimeEvent.opName = getCell(opNameIdx);
    association.runtimeEvent.streamId =
        parseU64(getCell(streamIdIdx)).value_or(0);
    association.runtimeEvent.deviceId =
        parseU64(getCell(deviceIdIdx)).value_or(0);
    association.runtimeEvent.startTimeNs = startTimeNs;
    association.runtimeEvent.endTimeNs = startTimeNs + durationNs;

    if (auto taskId = parseU64(getCell(taskIdIdx)); taskId.has_value()) {
      association.runtimeEvent.taskId = taskId.value();
      association.metrics["task_id"] = taskId.value();
    }
    if (auto corrId = parseU64(getCell(corrIdIdx)); corrId.has_value()) {
      association.runtimeEvent.correlationId = corrId.value();
      association.metrics["correlation_id"] = corrId.value();
    }
    association.metrics["task_duration_us"] =
        static_cast<double>(durationNs) / 1000.0;
    association.metrics["runtime_base_source"] =
        std::string("aclprof_op_summary");
    association.metrics["input_file"] = file.string();

    if (std::find(enabledVendorMetrics.begin(), enabledVendorMetrics.end(),
                  "aicore") != enabledVendorMetrics.end()) {
      if (auto aiCoreTimeMs = parseDouble(getCell(aiCoreTimeIdx));
          aiCoreTimeMs.has_value()) {
        association.metrics["aicore_time_ms"] = aiCoreTimeMs.value();
      }
      if (auto totalCycles = parseU64(getCell(totalCycleIdx));
          totalCycles.has_value()) {
        association.metrics["total_cycles"] = totalCycles.value();
      }
    }

    for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
      auto key = normalizeMetricKey(headers[i]);
      auto value = trim(row[i]);
      if (key.empty() || value.empty() ||
          association.metrics.find(key) != association.metrics.end()) {
        continue;
      }
      if (auto u64Value = parseU64(value); u64Value.has_value()) {
        association.metrics[key] = u64Value.value();
      } else if (auto doubleValue = parseDouble(value);
                 doubleValue.has_value()) {
        association.metrics[key] = doubleValue.value();
      } else {
        association.metrics[key] = value;
      }
    }

    enrichBandwidthMetrics(association, enabledVendorMetrics);
    artifact.associations.push_back(std::move(association));
    ++parsedRows;
  }

  if (parsedRows == 0) {
    artifact.degradeReasons.push_back(
        "No rows with task timing fields found in " + file.string());
  }
}

void parseTaskTimeCsv(const std::filesystem::path &file,
                      std::vector<RuntimeTraceEventKey> &runtimeEvents,
                      std::vector<std::string> &degradeReasons) {
  std::ifstream in(file.string());
  if (!in.is_open()) {
    degradeReasons.push_back("Failed to open runtime task_time CSV: " +
                             file.string());
    return;
  }

  std::string headerLine;
  if (!std::getline(in, headerLine)) {
    return;
  }
  auto headers = splitCsvLine(headerLine);
  if (headers.empty()) {
    return;
  }

  const auto opNameIdx = findColumn(headers, {"opname", "op", "kernelname"});
  const auto taskIdIdx = findColumn(headers, {"taskid"});
  const auto corrIdIdx =
      findColumn(headers, {"correlationid", "correlation_id", "corrid"});
  const auto streamIdIdx = findColumn(headers, {"streamid"});
  const auto deviceIdIdx = findColumn(headers, {"deviceid", "device"});
  const auto startIdx = findColumn(headers, {"taskstarttimeus", "starttimeus",
                                             "taskstarttime", "starttime"});
  const auto durationIdx = findColumn(
      headers, {"taskdurationus", "durationus", "taskduration", "duration"});
  if (!startIdx.has_value() || !durationIdx.has_value()) {
    return;
  }

  size_t parsedRows = 0;
  std::string line;
  while (std::getline(in, line)) {
    auto row = splitCsvLine(line);
    if (row.empty()) {
      continue;
    }
    auto getCell = [&row](std::optional<size_t> idx) -> std::string {
      if (!idx.has_value() || idx.value() >= row.size()) {
        return {};
      }
      return trim(row[idx.value()]);
    };

    auto startRaw = getCell(startIdx);
    auto durationRaw = getCell(durationIdx);
    if (startRaw.empty() || durationRaw.empty()) {
      continue;
    }
    uint64_t startNs = parseTimeToNs(startRaw, headers[startIdx.value()]);
    uint64_t durationNs =
        parseTimeToNs(durationRaw, headers[durationIdx.value()]);
    if (startNs == 0 || durationNs == 0) {
      continue;
    }

    RuntimeTraceEventKey event;
    event.opName = getCell(opNameIdx);
    event.taskId = parseU64(getCell(taskIdIdx)).value_or(0);
    event.correlationId = parseU64(getCell(corrIdIdx)).value_or(0);
    event.streamId = parseU64(getCell(streamIdIdx)).value_or(0);
    event.deviceId = parseU64(getCell(deviceIdIdx)).value_or(0);
    event.startTimeNs = startNs;
    event.endTimeNs = startNs + durationNs;
    runtimeEvents.push_back(std::move(event));
    parsedRows++;
  }

  if (parsedRows == 0) {
    degradeReasons.push_back(
        "No runtime rows were parsed from task_time CSV: " + file.string());
  }
}

std::string classifySupplementalSource(const std::string &fileName) {
  auto lowerName = toLower(fileName);
  if (lowerName.find("op_statistic") != std::string::npos) {
    return "msprof_op_statistic";
  }
  if (lowerName.find("api_statistic") != std::string::npos) {
    return "msprof_api_statistic";
  }
  if (lowerName.find("step_trace") != std::string::npos) {
    return "msprof_step_trace";
  }
  if (lowerName.find("kernel_details") != std::string::npos) {
    return "msprof_kernel_details";
  }
  if (lowerName.find("mstx") != std::string::npos ||
      lowerName.find("msproftx") != std::string::npos ||
      lowerName.find("msprof_tx") != std::string::npos) {
    return "msprof_mstx";
  }
  if (lowerName.find("aicore") != std::string::npos) {
    return "msprof_aicore";
  }
  if (lowerName.find("aicpu") != std::string::npos) {
    return "msprof_aicpu";
  }
  if (lowerName.find("bandwidth") != std::string::npos ||
      lowerName.find("hbm") != std::string::npos ||
      lowerName.find("llc") != std::string::npos ||
      lowerName.find("memory") != std::string::npos ||
      lowerName.find("mem") != std::string::npos) {
    return "msprof_bandwidth";
  }
  return "msprof_summary";
}

bool populateRuntimeEventFromRow(const std::vector<std::string> &headers,
                                 const std::vector<std::string> &row,
                                 RuntimeTraceEventKey &event) {
  std::unordered_map<std::string, std::string> values;
  for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
    auto key = normalizeColumnName(headers[i]);
    auto value = trim(row[i]);
    if (!key.empty() && !value.empty()) {
      values[key] = value;
    }
  }

  auto getAny = [&](const std::vector<std::string> &keys) -> std::string {
    for (const auto &key : keys) {
      auto it = values.find(key);
      if (it != values.end()) {
        return it->second;
      }
    }
    return {};
  };

  event.opName = getAny({"opname", "op", "kernelname"});
  if (event.opName.empty()) {
    event.opName =
        getAny({"name", "message", "rangename", "range", "eventname"});
  }
  event.taskId = parseU64(getAny({"taskid"})).value_or(0);
  event.correlationId =
      parseU64(getAny({"correlationid", "correlation_id", "corrid"}))
          .value_or(0);
  event.deviceId = parseU64(getAny({"deviceid", "device"})).value_or(0);
  event.streamId = parseU64(getAny({"streamid"})).value_or(0);

  uint64_t startNs = 0;
  uint64_t endNs = 0;

  for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
    auto normalized = normalizeColumnName(headers[i]);
    auto value = trim(row[i]);
    if (value.empty()) {
      continue;
    }
    if (normalized == "taskstarttimeus" || normalized == "starttimeus" ||
        normalized == "taskstarttime" || normalized == "starttime" ||
        normalized == "starttimens" || normalized == "starttimestamp") {
      startNs = parseTimeToNs(value, headers[i]);
    } else if (normalized == "taskdurationus" || normalized == "durationus" ||
               normalized == "taskduration" || normalized == "duration" ||
               normalized == "durationns") {
      auto durationNs = parseTimeToNs(value, headers[i]);
      if (startNs > 0 && durationNs > 0) {
        endNs = startNs + durationNs;
      }
    } else if (normalized == "taskendtimeus" || normalized == "endtimeus" ||
               normalized == "taskendtime" || normalized == "endtime" ||
               normalized == "endtimens" || normalized == "endtimestamp") {
      endNs = parseTimeToNs(value, headers[i]);
    }
  }

  if (startNs > 0 && endNs > startNs) {
    event.startTimeNs = startNs;
    event.endTimeNs = endNs;
    return true;
  }
  return false;
}

void parseStructuredSupplementalCsv(
    const std::filesystem::path &file, const std::string &sourceTag,
    const std::vector<std::string> &enabledVendorMetrics,
    VendorProfileArtifact &artifact) {
  std::ifstream in(file.string());
  if (!in.is_open()) {
    artifact.degradeReasons.push_back("Failed to open supplemental CSV: " +
                                      file.string());
    return;
  }

  std::string headerLine;
  if (!std::getline(in, headerLine)) {
    return;
  }
  auto headers = splitCsvLine(headerLine);
  if (headers.empty()) {
    return;
  }

  constexpr size_t kMaxSupplementalRows = 1024;
  size_t parsedRows = 0;
  std::string line;
  while (parsedRows < kMaxSupplementalRows && std::getline(in, line)) {
    auto row = splitCsvLine(line);
    if (row.empty()) {
      continue;
    }

    VendorMetricAssociation association;
    association.state = VendorMetricState::Collected;
    association.source = sourceTag;
    association.note = "Supplemental msprof row (normalized).";
    association.metrics["input_file"] = file.string();
    association.metrics["summary_row_index"] =
        static_cast<uint64_t>(parsedRows);
    association.metrics["runtime_base_source"] = std::string("supplemental");

    (void)populateRuntimeEventFromRow(headers, row, association.runtimeEvent);
    if (association.runtimeEvent.taskId != 0) {
      association.metrics["task_id"] = association.runtimeEvent.taskId;
    }
    if (association.runtimeEvent.correlationId != 0) {
      association.metrics["correlation_id"] =
          association.runtimeEvent.correlationId;
    }
    if (association.runtimeEvent.startTimeNs > 0 &&
        association.runtimeEvent.endTimeNs >
            association.runtimeEvent.startTimeNs) {
      association.metrics["task_duration_us"] =
          static_cast<double>(association.runtimeEvent.endTimeNs -
                              association.runtimeEvent.startTimeNs) /
          1000.0;
    }

    bool hasMetric = false;
    for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
      auto key = normalizeMetricKey(headers[i]);
      auto value = trim(row[i]);
      if (key.empty() || value.empty()) {
        continue;
      }
      if (auto u64Value = parseU64(value); u64Value.has_value()) {
        association.metrics[key] = u64Value.value();
      } else if (auto doubleValue = parseDouble(value);
                 doubleValue.has_value()) {
        association.metrics[key] = doubleValue.value();
      } else {
        association.metrics[key] = value;
      }
      hasMetric = true;
    }
    if (hasMetric) {
      enrichBandwidthMetrics(association, enabledVendorMetrics);
      artifact.associations.push_back(std::move(association));
      parsedRows++;
    }
  }

  if (parsedRows == 0) {
    artifact.degradeReasons.push_back("No usable rows in supplemental CSV: " +
                                      file.string());
  }
}

uint64_t absDiffU64(uint64_t lhs, uint64_t rhs) {
  return lhs > rhs ? (lhs - rhs) : (rhs - lhs);
}

std::string canonicalKernelNameForCorrelation(const std::string &name) {
  auto normalized = trim(name);
  for (const auto &suffix : {" mix", " aiv", " aic"}) {
    if (normalized.size() > std::strlen(suffix) &&
        normalized.compare(normalized.size() - std::strlen(suffix),
                           std::strlen(suffix), suffix) == 0) {
      normalized.resize(normalized.size() - std::strlen(suffix));
      break;
    }
  }
  return normalized;
}

bool sameKernelForCorrelation(const std::string &lhs, const std::string &rhs) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  return canonicalKernelNameForCorrelation(lhs) ==
         canonicalKernelNameForCorrelation(rhs);
}

void correlateVendorToRuntimeEvents(
    VendorProfileArtifact &artifact,
    const std::vector<RuntimeTraceEventKey> &runtimeEvents) {
  constexpr uint64_t kStrictMatchWindowNs = 5'000'000; // 5ms
  constexpr uint64_t kFuzzyMatchWindowNs = 20'000'000; // 20ms
  std::map<std::string, size_t> nextRuntimeEventByKernel;
  std::map<std::string, size_t> nextLaunchRangeByKernel;

  auto matchRuntimeEventByKernelOrder =
      [&](const VendorMetricAssociation &association)
      -> const RuntimeTraceEventKey * {
    auto key =
        canonicalKernelNameForCorrelation(association.runtimeEvent.opName);
    if (key.empty()) {
      return nullptr;
    }
    auto &nextIndex = nextRuntimeEventByKernel[key];
    for (size_t index = nextIndex; index < runtimeEvents.size(); ++index) {
      const auto &runtimeEvent = runtimeEvents[index];
      if (!sameKernelForCorrelation(association.runtimeEvent.opName,
                                    runtimeEvent.opName)) {
        continue;
      }
      if (association.runtimeEvent.deviceId != 0 &&
          runtimeEvent.deviceId != 0 &&
          association.runtimeEvent.deviceId != runtimeEvent.deviceId) {
        continue;
      }
      nextIndex = index + 1;
      return &runtimeEvent;
    }
    return nullptr;
  };

  auto matchLaunchRangeByKernelOrder =
      [&](const VendorMetricAssociation &association)
      -> const VendorMetricAssociation * {
    auto key =
        canonicalKernelNameForCorrelation(association.runtimeEvent.opName);
    if (key.empty()) {
      return nullptr;
    }
    auto &nextIndex = nextLaunchRangeByKernel[key];
    for (size_t index = nextIndex; index < artifact.associations.size();
         ++index) {
      const auto &launchRange = artifact.associations[index];
      if (launchRange.source != "msprof_mstx" ||
          launchRange.runtimeEvent.endTimeNs <=
              launchRange.runtimeEvent.startTimeNs) {
        continue;
      }
      if (!sameKernelForCorrelation(association.runtimeEvent.opName,
                                    launchRange.runtimeEvent.opName)) {
        continue;
      }
      if (association.runtimeEvent.deviceId != 0 &&
          launchRange.runtimeEvent.deviceId != 0 &&
          association.runtimeEvent.deviceId !=
              launchRange.runtimeEvent.deviceId) {
        continue;
      }
      auto startDiff = absDiffU64(association.runtimeEvent.startTimeNs,
                                  launchRange.runtimeEvent.startTimeNs);
      if (startDiff > kStrictMatchWindowNs) {
        continue;
      }
      nextIndex = index + 1;
      return &launchRange;
    }
    return nullptr;
  };

  for (auto &association : artifact.associations) {
    if (association.source != "aclprof_op_summary" &&
        association.source != "msprof_mstx") {
      continue;
    }
    if (runtimeEvents.empty()) {
      association.state = VendorMetricState::Collected;
      association.note =
          "Runtime base event imported directly from aclprof summary.";
      continue;
    }

    if (association.source == "msprof_mstx") {
      if (auto runtimeEvent = matchRuntimeEventByKernelOrder(association)) {
        association.runtimeEvent.scopeId = runtimeEvent->scopeId;
        association.runtimeEvent.opName = runtimeEvent->opName;
        if (association.runtimeEvent.deviceId == 0) {
          association.runtimeEvent.deviceId = runtimeEvent->deviceId;
        }
        if (association.runtimeEvent.streamId == 0) {
          association.runtimeEvent.streamId = runtimeEvent->streamId;
        }
        association.metrics["runtime_scope_op_name"] = runtimeEvent->opName;
        association.metrics["matched_runtime_scope_id"] =
            static_cast<uint64_t>(runtimeEvent->scopeId);
        association.state = VendorMetricState::Collected;
        association.note = "Matched msprof range to Triton runtime scope by "
                           "kernel launch order.";
      }
      continue;
    }

    if (association.runtimeEvent.scopeId != 0 &&
        association.runtimeEvent.endTimeNs >
            association.runtimeEvent.startTimeNs) {
      if (association.note.empty()) {
        association.note = "Scope mapped without host timing correlation; "
                           "timing from aclprof summary.";
      }
      continue;
    }

    const RuntimeTraceEventKey *best = nullptr;
    uint64_t bestScore = std::numeric_limits<uint64_t>::max();
    std::string matchedTier = "";

    auto selectBy = [&](auto &&predicate, uint64_t windowNs) {
      best = nullptr;
      bestScore = std::numeric_limits<uint64_t>::max();
      for (const auto &runtimeEvent : runtimeEvents) {
        if (!predicate(runtimeEvent)) {
          continue;
        }
        auto startDiff = absDiffU64(association.runtimeEvent.startTimeNs,
                                    runtimeEvent.startTimeNs);
        auto endDiff = absDiffU64(association.runtimeEvent.endTimeNs,
                                  runtimeEvent.endTimeNs);
        if (startDiff > windowNs || endDiff > windowNs) {
          continue;
        }
        auto score = startDiff + endDiff;
        if (score < bestScore) {
          bestScore = score;
          best = &runtimeEvent;
        }
      }
      return best != nullptr;
    };

    // Tier-1 strict matching: device_id + stream_id + op_name + timestamp.
    auto assocCorrId = association.runtimeEvent.correlationId;
    if (assocCorrId == 0) {
      assocCorrId = parseMetricU64(association, "correlation_id").value_or(0);
    }
    auto assocTaskId = association.runtimeEvent.taskId;
    if (assocTaskId == 0) {
      assocTaskId = parseMetricU64(association, "task_id").value_or(0);
    }

    bool matched = false;
    if (assocCorrId != 0) {
      matched = selectBy(
          [&](const RuntimeTraceEventKey &runtimeEvent) {
            return runtimeEvent.correlationId != 0 &&
                   runtimeEvent.correlationId == assocCorrId;
          },
          std::numeric_limits<uint64_t>::max());
      if (matched) {
        matchedTier = "correlation_id";
      }
    }

    if (!matched && assocTaskId != 0) {
      matched = selectBy(
          [&](const RuntimeTraceEventKey &runtimeEvent) {
            return runtimeEvent.taskId != 0 &&
                   runtimeEvent.taskId == assocTaskId;
          },
          std::numeric_limits<uint64_t>::max());
      if (matched) {
        matchedTier = "task_id";
      }
    }

    if (!matched) {
      matched = selectBy(
          [&](const RuntimeTraceEventKey &runtimeEvent) {
            return sameKernelForCorrelation(association.runtimeEvent.opName,
                                            runtimeEvent.opName) &&
                   association.runtimeEvent.deviceId == runtimeEvent.deviceId &&
                   association.runtimeEvent.streamId == runtimeEvent.streamId;
          },
          kStrictMatchWindowNs);
      if (matched) {
        matchedTier = "strict";
      }
    }

    // Tier-2 relaxed matching: op_name + timestamp.
    if (!matched) {
      matched = selectBy(
          [&](const RuntimeTraceEventKey &runtimeEvent) {
            return sameKernelForCorrelation(association.runtimeEvent.opName,
                                            runtimeEvent.opName);
          },
          kStrictMatchWindowNs);
      if (matched) {
        matchedTier = "op_name";
      }
    }

    // Tier-3 fuzzy matching: timestamp only.
    if (!matched) {
      matched = selectBy([&](const RuntimeTraceEventKey &) { return true; },
                         kFuzzyMatchWindowNs);
      if (matched) {
        matchedTier = "fuzzy";
      }
    }

    if (!matched) {
      if (const auto *bestLaunchRange =
              matchLaunchRangeByKernelOrder(association)) {
        association.runtimeEvent.scopeId =
            bestLaunchRange->runtimeEvent.scopeId;
        association.runtimeEvent.opName = bestLaunchRange->runtimeEvent.opName;
        association.metrics["launch_range_op_name"] =
            bestLaunchRange->runtimeEvent.opName;
        association.metrics["merged_with_launch_range"] = std::string("true");
        association.state = VendorMetricState::Collected;
        association.note = "Matched CANN op_summary to msprof launch range by "
                           "kernel launch order.";
        continue;
      }
      association.state = VendorMetricState::Unmatched;
      association.note =
          "No runtime event matched by strict/fuzzy timestamp strategy";
      continue;
    }

    auto vendorEvent = association.runtimeEvent;
    if (vendorEvent.scopeId == 0) {
      vendorEvent.scopeId = best->scopeId;
    }
    if (vendorEvent.opName.empty()) {
      vendorEvent.opName = best->opName;
    }
    if (vendorEvent.deviceId == 0) {
      vendorEvent.deviceId = best->deviceId;
    }
    if (vendorEvent.streamId == 0) {
      vendorEvent.streamId = best->streamId;
    }
    if (vendorEvent.taskId == 0) {
      vendorEvent.taskId = best->taskId;
    }
    if (vendorEvent.correlationId == 0) {
      vendorEvent.correlationId = best->correlationId;
    }
    if (vendorEvent.endTimeNs <= vendorEvent.startTimeNs) {
      vendorEvent.startTimeNs = best->startTimeNs;
      vendorEvent.endTimeNs = best->endTimeNs;
    }
    association.runtimeEvent = std::move(vendorEvent);
    if (association.runtimeEvent.taskId != 0) {
      association.metrics["task_id"] = association.runtimeEvent.taskId;
    }
    if (association.runtimeEvent.correlationId != 0) {
      association.metrics["correlation_id"] =
          association.runtimeEvent.correlationId;
    }
    association.state = VendorMetricState::Collected;
    if (matchedTier == "correlation_id") {
      association.note = "Matched runtime scope by correlation_id; timing from "
                         "aclprof summary.";
    } else if (matchedTier == "task_id") {
      association.note =
          "Matched runtime scope by task_id; timing from aclprof summary.";
    } else if (matchedTier == "fuzzy") {
      association.note = "Fuzzy matched runtime scope by timestamp only; "
                         "timing from aclprof summary.";
    } else if (association.runtimeEvent.deviceId == 0 &&
               association.runtimeEvent.streamId == 0) {
      association.note = "Matched runtime scope by op_name/timestamp; timing "
                         "from aclprof summary.";
    } else {
      association.note = "Matched runtime scope by device_id/stream_id/op_name "
                         "and timestamp; timing from aclprof summary.";
    }
  }
}

void addRuntimeBaseFallbackAssociations(
    VendorProfileArtifact &artifact,
    const std::vector<RuntimeTraceEventKey> &runtimeEvents) {
  for (const auto &event : runtimeEvents) {
    if (event.endTimeNs <= event.startTimeNs) {
      continue;
    }
    VendorMetricAssociation association;
    association.runtimeEvent = event;
    association.state = VendorMetricState::Collected;
    association.source = "runtime_base_fallback";
    association.note =
        "aclprof summary unavailable; using runtime_base host-timing fallback.";
    association.metrics["runtime_base_source"] =
        std::string("host_timing_fallback");
    association.metrics["runtime_base_only"] = uint64_t(1);
    artifact.associations.push_back(std::move(association));
  }
}

void addRuntimeBaseNativeAssociations(
    VendorProfileArtifact &artifact,
    const std::vector<RuntimeTraceEventKey> &runtimeEvents,
    const std::string &sourceTag, const std::string &sourceMetricTag,
    const std::string &note) {
  for (const auto &event : runtimeEvents) {
    if (event.endTimeNs <= event.startTimeNs) {
      continue;
    }
    VendorMetricAssociation association;
    association.runtimeEvent = event;
    association.state = VendorMetricState::Collected;
    association.source = sourceTag;
    association.note = note;
    association.metrics["runtime_base_source"] = sourceMetricTag;
    association.metrics["task_duration_us"] =
        static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
    if (event.taskId != 0) {
      association.metrics["task_id"] = event.taskId;
    }
    if (event.correlationId != 0) {
      association.metrics["correlation_id"] = event.correlationId;
    }
    artifact.associations.push_back(std::move(association));
  }
}

void addAssociationQualityReasons(VendorProfileArtifact &artifact) {
  size_t unmatchedCount = 0;
  size_t collectedCount = 0;
  for (const auto &association : artifact.associations) {
    if (association.state == VendorMetricState::Unmatched) {
      unmatchedCount++;
    } else if (association.state == VendorMetricState::Collected) {
      collectedCount++;
    }
  }
  if (unmatchedCount > 0) {
    artifact.degradeReasons.push_back(
        "Vendor/runtime correlation unmatched entries: " +
        std::to_string(unmatchedCount) +
        " (collected=" + std::to_string(collectedCount) + ").");
  }
}

void appendUniqueReason(std::vector<std::string> &reasons,
                        const std::string &reason) {
  if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end()) {
    reasons.push_back(reason);
  }
}

#if defined(__linux__)
std::string shellQuote(const std::string &value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

bool looksLikeMsprofProfileDir(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec)) {
    return false;
  }
  if (std::filesystem::is_directory(path / "host", ec)) {
    return true;
  }
  for (const auto &entry : std::filesystem::directory_iterator(path, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_directory(ec)) {
      continue;
    }
    auto name = entry.path().filename().string();
    if (name.rfind("device_", 0) == 0) {
      return true;
    }
  }
  return false;
}

std::vector<std::filesystem::path>
collectMsprofExportRoots(const std::filesystem::path &outputPath) {
  std::vector<std::filesystem::path> roots;
  std::error_code ec;
  if (looksLikeMsprofProfileDir(outputPath)) {
    roots.push_back(outputPath);
    return roots;
  }
  if (!std::filesystem::is_directory(outputPath, ec)) {
    return roots;
  }
  for (const auto &entry :
       std::filesystem::directory_iterator(outputPath, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_directory(ec)) {
      continue;
    }
    auto name = entry.path().filename().string();
    if (name.rfind("PROF_", 0) == 0 &&
        looksLikeMsprofProfileDir(entry.path())) {
      roots.push_back(entry.path());
    }
  }
  std::sort(roots.begin(), roots.end());
  return roots;
}

void exportMsprofCsv(const std::string &outputPath,
                     std::vector<std::string> &reasons) {
  if (outputPath.empty()) {
    appendUniqueReason(
        reasons, "CANN aclprof auto-export skipped: output path is empty.");
    return;
  }
  auto exportRoots =
      collectMsprofExportRoots(std::filesystem::path(outputPath));
  if (exportRoots.empty()) {
    exportRoots.push_back(std::filesystem::path(outputPath));
  }
  for (const auto &exportRoot : exportRoots) {
    auto command =
        "msprof --export=on --type=text --summary-format=csv --output=" +
        shellQuote(exportRoot.string());
    auto ret = std::system(command.c_str());
    if (ret != 0) {
      appendUniqueReason(reasons,
                         "CANN aclprof auto-export failed with status " +
                             std::to_string(ret) + ": " + command);
    }
  }
}
#endif

} // namespace

CannProfiler::CannProfiler() = default;

CannProfiler::~CannProfiler() = default;

void CannProfiler::startOp(const Scope &scope) {
  MsProfTxRange range;
  bool enableMstx = false;
  void *domain = nullptr;
  void *stream = nullptr;

  {
    std::lock_guard<std::mutex> lock(mutex);
    if (hostTimingFallbackEnabled) {
      opStartTimesNs[scope.scopeId] = nowNs();
    }
    enableMstx = mstxActive && msproftxEnabled;
    domain = mstxDomain;
    stream = mstxStream;
  }

#if defined(__linux__)
  if (enableMstx) {
    auto &api = mstxApi();
    if (!api.rangeStartA || !api.rangeEnd) {
      std::lock_guard<std::mutex> lock(mutex);
      appendUniqueReason(
          runtimeDegradeReasons,
          "mstx APIs are unavailable; range annotations are disabled.");
    } else {
      if (domain && api.domainRangeStartA && api.domainRangeEnd) {
        range.rangeId =
            api.domainRangeStartA(domain, scope.name.c_str(), stream);
        range.mstxDomainRange = true;
      } else {
        range.rangeId = api.rangeStartA(scope.name.c_str(), stream);
        range.mstxDomainRange = false;
      }
      if (range.rangeId != 0) {
        range.active = true;
      } else {
        std::lock_guard<std::mutex> lock(mutex);
        appendUniqueReason(
            runtimeDegradeReasons,
            "mstx range start returned an invalid range id from " +
                api.loadedFrom + ".");
      }
    }
  }
#endif

  if (range.active) {
    std::lock_guard<std::mutex> lock(mutex);
    opRanges[scope.scopeId] = range;
  }
}

void CannProfiler::stopOp(const Scope &scope) {
  uint64_t startTimeNs = 0;
  uint64_t endTimeNs = 0;
  MsProfTxRange range;
  void *domain = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (hostTimingFallbackEnabled) {
      auto it = opStartTimesNs.find(scope.scopeId);
      if (it != opStartTimesNs.end()) {
        startTimeNs = it->second;
        opStartTimesNs.erase(it);
      }
      endTimeNs = nowNs();
    }
    auto rangeIt = opRanges.find(scope.scopeId);
    if (rangeIt != opRanges.end()) {
      range = rangeIt->second;
      opRanges.erase(rangeIt);
    }
    domain = mstxDomain;
  }

#if defined(__linux__)
  if (range.active) {
    auto &api = mstxApi();
    if (range.mstxDomainRange && domain && api.domainRangeEnd) {
      api.domainRangeEnd(domain, range.rangeId);
    } else if (api.rangeEnd) {
      api.rangeEnd(range.rangeId);
    } else {
      std::lock_guard<std::mutex> lock(mutex);
      appendUniqueReason(runtimeDegradeReasons,
                         "mstx range end API is unavailable.");
    }
  }
#endif

  if (startTimeNs > 0 && endTimeNs > startTimeNs) {
    RuntimeTraceEventKey runtimeEvent;
    runtimeEvent.scopeId = scope.scopeId;
    runtimeEvent.opName = scope.name;
    runtimeEvent.deviceId = 0;
    runtimeEvent.streamId = 0;
    runtimeEvent.startTimeNs = startTimeNs;
    runtimeEvent.endTimeNs = endTimeNs;
    std::lock_guard<std::mutex> lock(mutex);
    runtimeEvents.push_back(runtimeEvent);
  }
}

void CannProfiler::doStart() {
  std::lock_guard<std::mutex> lock(mutex);
  opStartTimesNs.clear();
  opRanges.clear();
  runtimeEvents.clear();
  runtimeDegradeReasons.clear();
  aclprofActive = false;
  aclprofConfig = nullptr;
  mstxActive = false;
  mstxDomain = nullptr;
  mstxStream = nullptr;
  mstxStreamOwned = false;

  std::error_code ec;
  if (!outputPath.empty()) {
    std::filesystem::create_directories(outputPath, ec);
  }

#if defined(__linux__)
  if (msproftxEnabled) {
    auto &mstx = mstxApi();
    if (!mstx.load()) {
      appendUniqueReason(runtimeDegradeReasons, mstx.loadError);
    } else {
      auto &aclRuntime = aclRuntimeApi();
      if (!aclRuntime.load()) {
        appendUniqueReason(runtimeDegradeReasons, aclRuntime.loadError);
      } else {
        if (aclRuntime.init) {
          (void)aclRuntime.init(nullptr);
        }
        auto setDeviceRet =
            aclRuntime.setDevice(static_cast<int32_t>(deviceId));
        if (setDeviceRet != 0) {
          appendUniqueReason(runtimeDegradeReasons,
                             "aclrtSetDevice failed while preparing mstx "
                             "stream with error code " +
                                 std::to_string(setDeviceRet));
        } else {
          void *stream = nullptr;
          auto createStreamRet = aclRuntime.createStream(&stream);
          if (createStreamRet == 0 && stream) {
            mstxStream = stream;
            mstxStreamOwned = true;
          } else {
            appendUniqueReason(runtimeDegradeReasons,
                               "aclrtCreateStream failed while preparing mstx "
                               "stream from " +
                                   aclRuntime.loadedFrom + " with error code " +
                                   std::to_string(createStreamRet));
          }
        }
      }
      if (!mstxDomainName.empty() && mstx.domainCreateA) {
        mstxDomain = mstx.domainCreateA(mstxDomainName.c_str());
        if (!mstxDomain) {
          appendUniqueReason(runtimeDegradeReasons,
                             "mstxDomainCreateA returned nullptr from " +
                                 mstx.loadedFrom +
                                 "; using default-domain mstx ranges.");
        }
      }
      mstxActive = true;
    }
  }

  if (!aclprofRuntimeEnabled) {
    return;
  }

  auto &api = aclProfApi();
  if (!api.load()) {
    runtimeDegradeReasons.push_back(
        "Optional legacy aclprof runtime path unavailable: " + api.loadError);
    aclprofActive = false;
    aclprofConfig = nullptr;
    return;
  }

  auto initRet = api.init(outputPath.c_str(), outputPath.size());
  if (initRet != 0) {
    runtimeDegradeReasons.push_back("aclprofInit failed with error code " +
                                    std::to_string(initRet));
    aclprofActive = false;
    aclprofConfig = nullptr;
    return;
  }

  uint32_t deviceList[1] = {deviceId};
  aclprofConfig = api.createConfig(deviceList, 1,
                                   static_cast<uint32_t>(aclprofAicoreMetricId),
                                   nullptr, aclprofDataTypeConfig);
  if (!aclprofConfig) {
    runtimeDegradeReasons.push_back("aclprofCreateConfig returned nullptr");
    api.finalize();
    aclprofActive = false;
    return;
  }

  if (api.setConfig && !memFreqHz.empty()) {
    auto setRet = api.setConfig(kAclProfSysHardwareMemFreq, memFreqHz.c_str(),
                                memFreqHz.size());
    if (setRet != 0) {
      runtimeDegradeReasons.push_back("aclprofSetConfig(ACL_PROF_SYS_HARDWARE_"
                                      "MEM_FREQ) failed with error code " +
                                      std::to_string(setRet));
    }
  }

  auto startRet =
      api.start(reinterpret_cast<const AclProfConfig *>(aclprofConfig));
  if (startRet != 0) {
    runtimeDegradeReasons.push_back("aclprofStart failed with error code " +
                                    std::to_string(startRet));
    api.destroyConfig(reinterpret_cast<const AclProfConfig *>(aclprofConfig));
    api.finalize();
    aclprofConfig = nullptr;
    aclprofActive = false;
    return;
  }

  aclprofActive = true;
#else
  if (msproftxEnabled) {
    runtimeDegradeReasons.push_back(
        "CANN mstx range annotations are only available on Linux.");
  }
  if (aclprofRuntimeEnabled) {
    runtimeDegradeReasons.push_back(
        "CANN legacy aclprof runtime path is only available on Linux.");
  }
  aclprofActive = false;
  aclprofConfig = nullptr;
#endif
}

void CannProfiler::doFlush() {}

void CannProfiler::doStop() {
  std::unordered_map<size_t, MsProfTxRange> ranges;
  void *domain = nullptr;
  void *stream = nullptr;
  bool streamOwned = false;
  void *profConfig = nullptr;
  bool profActive = false;
  bool autoExport = false;
  std::string exportPath;
  std::vector<std::string> localReasons;

  {
    std::lock_guard<std::mutex> lock(mutex);
    opStartTimesNs.clear();
    ranges.swap(opRanges);
    domain = mstxDomain;
    stream = mstxStream;
    streamOwned = mstxStreamOwned;
    mstxDomain = nullptr;
    mstxStream = nullptr;
    mstxStreamOwned = false;
    mstxActive = false;
    profConfig = aclprofConfig;
    profActive = aclprofActive;
    aclprofConfig = nullptr;
    aclprofActive = false;
    autoExport = aclprofAutoExportEnabled;
    exportPath = outputPath;
  }

#if defined(__linux__)
  auto &mstx = mstxApi();
  if (mstx.loaded) {
    for (auto &[scopeId, range] : ranges) {
      (void)scopeId;
      if (!range.active) {
        continue;
      }
      if (range.mstxDomainRange && domain && mstx.domainRangeEnd) {
        mstx.domainRangeEnd(domain, range.rangeId);
      } else if (mstx.rangeEnd) {
        mstx.rangeEnd(range.rangeId);
      }
    }
  }

  if (domain && mstx.loaded && mstx.domainDestroy) {
    mstx.domainDestroy(domain);
  }

  if (streamOwned && stream) {
    auto &aclRuntime = aclRuntimeApi();
    if (aclRuntime.loaded) {
      if (aclRuntime.synchronizeStream) {
        auto syncRet = aclRuntime.synchronizeStream(stream);
        if (syncRet != 0) {
          localReasons.push_back(
              "aclrtSynchronizeStream failed for mstx stream with error code " +
              std::to_string(syncRet));
        }
      }
      auto destroyRet = aclRuntime.destroyStream(stream);
      if (destroyRet != 0) {
        localReasons.push_back(
            "aclrtDestroyStream failed for mstx stream with error code " +
            std::to_string(destroyRet));
      }
    }
  }

  auto &api = aclProfApi();
  if (profActive && api.loaded) {
    auto stopRet =
        api.stop(reinterpret_cast<const AclProfConfig *>(profConfig));
    if (stopRet != 0) {
      localReasons.push_back("aclprofStop failed with error code " +
                             std::to_string(stopRet));
    }

    auto destroyRet =
        api.destroyConfig(reinterpret_cast<const AclProfConfig *>(profConfig));
    if (destroyRet != 0) {
      localReasons.push_back("aclprofDestroyConfig failed with error code " +
                             std::to_string(destroyRet));
    }

    auto finalizeRet = api.finalize();
    if (finalizeRet != 0) {
      localReasons.push_back("aclprofFinalize failed with error code " +
                             std::to_string(finalizeRet));
    }

    if (autoExport) {
      exportMsprofCsv(exportPath, localReasons);
    }
  }
#endif

  if (!localReasons.empty()) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &reason : localReasons) {
      appendUniqueReason(runtimeDegradeReasons, reason);
    }
  }
}

void CannProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  this->modeAndOptions = modeAndOptions;

  deviceId = 0;
  outputPath =
      getEnvOrDefault("PROTON_CANN_PROFILE_OUTPUT", "./proton_cann_profile");
  memFreqHz = "15";
  aclprofDataTypeConfig =
      kAclProfAclApi | kAclProfTaskTime | kAclProfRuntimeApi;
  aclprofAicoreMetricId = kAclAicoreArithmeticUtilization;
  msproftxEnabled = false;
  hostTimingFallbackEnabled = false;
  aclprofRuntimeEnabled = false;
  aclprofAutoExportEnabled = true;
  mstxDomainName = getEnvOrDefault("PROTON_CANN_MSTX_DOMAIN", "proton");

  bool wantsAicore = false;
  bool wantsBandwidth = false;
  bool msproftxExplicitlyConfigured = false;
  for (const auto &tokenRaw : modeAndOptions) {
    auto token = trim(tokenRaw);
    if (token.empty()) {
      continue;
    }
    if (token.find("vendor_metrics=") == 0) {
      auto value = trim(token.substr(std::string("vendor_metrics=").size()));
      for (const auto &metricRaw : split(value, ",")) {
        auto metric = toLower(trim(metricRaw));
        if (metric == "aicore") {
          wantsAicore = true;
        } else if (metric == "bandwidth") {
          wantsBandwidth = true;
        }
      }
      continue;
    }

    auto eqPos = token.find('=');
    if (eqPos == std::string::npos) {
      continue;
    }
    auto key = toLower(trim(token.substr(0, eqPos)));
    auto value = trim(token.substr(eqPos + 1));
    if (key == "device_id") {
      if (auto parsed = parseU64(value); parsed.has_value()) {
        deviceId = static_cast<uint32_t>(parsed.value());
      }
    } else if (key == "output_path" || key == "aclprof_output_path") {
      outputPath = value;
    } else if (key == "aclprof_set_config_mem_freq_hz") {
      memFreqHz = value;
    } else if (key == "aclprof_aicore_metric_id") {
      if (auto parsed = parseU64(value); parsed.has_value()) {
        aclprofAicoreMetricId = parsed.value();
      }
    } else if (key == "aclprof_data_type_flags") {
      auto flags = parseAclProfFlags(value);
      if (flags != 0) {
        aclprofDataTypeConfig = flags;
      }
    } else if (key == "aclprof_msproftx_enabled" || key == "msproftx_enabled" ||
               key == "mstx_enabled") {
      msproftxEnabled = parseBool(value);
      msproftxExplicitlyConfigured = true;
    } else if (key == "mstx_domain" || key == "mstx_domain_name") {
      mstxDomainName = value;
    } else if (key == "aclprof_runtime_enabled" ||
               key == "cann_aclprof_runtime") {
      aclprofRuntimeEnabled = parseBool(value);
    } else if (key == "aclprof_auto_export" ||
               key == "cann_aclprof_auto_export" ||
               key == "msprof_auto_export") {
      aclprofAutoExportEnabled = parseBool(value);
    } else if (key == "runtime_host_timing_fallback" ||
               key == "runtime_base_host_fallback") {
      hostTimingFallbackEnabled = parseBool(value);
    }
  }

  if (wantsAicore || wantsBandwidth) {
    aclprofDataTypeConfig |= kAclProfAicoreMetrics;
    if (!msproftxExplicitlyConfigured) {
      msproftxEnabled = true;
    }
  }
  if (msproftxEnabled) {
    aclprofDataTypeConfig |= kAclProfMsprofTx;
  }
  if (wantsBandwidth) {
    aclprofAicoreMetricId = kAclAicoreMemoryAccess;
  }

  if (!hostTimingFallbackEnabled) {
    auto envFlag =
        toLower(getEnvOrDefault("PROTON_CANN_RUNTIME_HOST_FALLBACK", "0"));
    hostTimingFallbackEnabled = parseBool(envFlag);
  }

  if (!aclprofRuntimeEnabled) {
    aclprofRuntimeEnabled =
        parseBool(getEnvOrDefault("PROTON_CANN_ACLPROF_RUNTIME", "0"));
  }
}

std::vector<std::string> CannProfiler::drainRuntimeDegradeReasons() {
  std::lock_guard<std::mutex> lock(mutex);
  auto reasons = runtimeDegradeReasons;
  runtimeDegradeReasons.clear();
  return reasons;
}

std::vector<RuntimeTraceEventKey> CannProfiler::drainRuntimeEvents() {
  std::lock_guard<std::mutex> lock(mutex);
  auto events = runtimeEvents;
  runtimeEvents.clear();
  return events;
}

VendorProfileArtifact
CannProfiler::importMsprofOutput(const SessionProfileMetadata &metadata,
                                 const VendorProfilePlan &plan) {
  auto runtimeReasons = CannProfiler::instance().drainRuntimeDegradeReasons();
  auto hostRuntimeEvents = CannProfiler::instance().drainRuntimeEvents();

  VendorProfileArtifact artifact;
  artifact.backend = metadata.backend;
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;
  artifact.degradeReasons = std::move(runtimeReasons);

  auto csvFiles = collectCandidateCsvFiles(metadata);
  std::vector<RuntimeTraceEventKey> nativeRuntimeEvents;
  for (const auto &file : csvFiles) {
    artifact.rawInputs.push_back(file.string());
    auto lowerName = toLower(file.filename().string());
    if (lowerName.find("task_time") != std::string::npos) {
      parseTaskTimeCsv(file, nativeRuntimeEvents, artifact.degradeReasons);
    } else if (lowerName.find("op_summary") != std::string::npos) {
      parseOpSummaryCsv(file, plan.enabledVendorMetrics, artifact);
    } else if (lowerName.find("summary") != std::string::npos ||
               lowerName.find("op_statistic") != std::string::npos ||
               lowerName.find("api_statistic") != std::string::npos ||
               lowerName.find("step_trace") != std::string::npos ||
               lowerName.find("kernel_details") != std::string::npos ||
               lowerName.find("mstx") != std::string::npos ||
               lowerName.find("msproftx") != std::string::npos ||
               lowerName.find("msprof_tx") != std::string::npos ||
               lowerName.find("aicore") != std::string::npos ||
               lowerName.find("aicpu") != std::string::npos ||
               lowerName.find("hbm") != std::string::npos ||
               lowerName.find("llc") != std::string::npos ||
               lowerName.find("bandwidth") != std::string::npos ||
               lowerName.find("memory") != std::string::npos ||
               lowerName.find("mem") != std::string::npos) {
      parseStructuredSupplementalCsv(file,
                                     classifySupplementalSource(lowerName),
                                     plan.enabledVendorMetrics, artifact);
    }
  }

  if (metricEnabled(plan.enabledVendorMetrics, "bandwidth")) {
    bool hasBandwidth = false;
    for (const auto &association : artifact.associations) {
      if (association.metrics.find("bandwidth_gb_s") !=
          association.metrics.end()) {
        hasBandwidth = true;
        break;
      }
    }
    if (!hasBandwidth) {
      artifact.degradeReasons.push_back(
          "Bandwidth requested, but no CANN memory/bandwidth CSV columns were "
          "imported. Enable msprof options such as --aic-metrics=MemoryAccess, "
          "--task-memory=on, and --sys-hardware-mem=on.");
    }
  }

  size_t opSummaryAssociationCount = 0;
  for (const auto &association : artifact.associations) {
    if (association.source == "aclprof_op_summary") {
      opSummaryAssociationCount++;
    }
  }

  if (opSummaryAssociationCount == 0 && !csvFiles.empty() &&
      !plan.enabledVendorMetrics.empty()) {
    artifact.degradeReasons.push_back(
        "Vendor CSV files were found, but no usable op_summary associations "
        "were parsed.");
  }
  if (opSummaryAssociationCount == 0 && csvFiles.empty() &&
      !plan.enabledVendorMetrics.empty()) {
    artifact.degradeReasons.push_back("No vendor summary CSV files were found. "
                                      "Expected msprof exports such as "
                                      "summary/op_summary*.csv.");
    if (adapterOptionEnabled(
            plan.requested.adapterOptions,
            {"mstx_enabled", "msproftx_enabled", "aclprof_msproftx_enabled"})) {
      artifact.degradeReasons.push_back(
          "msprof CSV exports were not visible at proton.finalize time. When "
          "using external `msprof --msproftx=on`, summaries may only be "
          "exported after the wrapped process exits.");
    }
  }

  if (opSummaryAssociationCount > 0) {
    if (!nativeRuntimeEvents.empty()) {
      correlateVendorToRuntimeEvents(artifact, nativeRuntimeEvents);
    } else if (!hostRuntimeEvents.empty()) {
      artifact.degradeReasons.push_back(
          "task_time runtime events unavailable; using host runtime events for "
          "op_summary correlation (runtime_host_timing_fallback=true).");
      correlateVendorToRuntimeEvents(artifact, hostRuntimeEvents);
    } else {
      const std::vector<RuntimeTraceEventKey> emptyRuntimeEvents{};
      correlateVendorToRuntimeEvents(artifact, emptyRuntimeEvents);
    }
  } else if (!nativeRuntimeEvents.empty()) {
    addRuntimeBaseNativeAssociations(
        artifact, nativeRuntimeEvents, "aclprof_task_time", "aclprof_task_time",
        "runtime_base imported from aclprof task_time events.");
  } else if (!hostRuntimeEvents.empty()) {
    artifact.degradeReasons.push_back(
        "aclprof summary unavailable; runtime_base falls back to host-timing "
        "events (runtime_host_timing_fallback=true).");
    addRuntimeBaseFallbackAssociations(artifact, hostRuntimeEvents);
  } else if (plan.runtimeBaseEnabled) {
    artifact.degradeReasons.push_back(
        "No runtime_base events could be imported from aclprof exports.");
  }
  addAssociationQualityReasons(artifact);

  return artifact;
}

} // namespace proton
