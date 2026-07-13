#include "Driver/Ascend/AscendApi.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace proton {

namespace ascend {

namespace {

// Reference: AscendCL aclrtDevAttr (CANN API docs).
constexpr int32_t kAclDevAttrComputeUnit = 1;
constexpr int32_t kAclDevAttrAiCoreCoreNum = 101;
constexpr int32_t kAclDevAttrL2CacheSize = 201;
constexpr int32_t kAclDevAttrTotalGlobalMemSize = 301;

#if defined(__linux__)
using AclError = int;
using FnGetDeviceCount = AclError (*)(uint32_t *);
using FnSetDevice = AclError (*)(int32_t);
using FnResetDevice = AclError (*)(int32_t);
using FnGetSocName = const char *(*)();
using FnGetDeviceInfo = AclError (*)(int32_t, int32_t, int64_t *);
using FnGetDeviceCapability = AclError (*)(uint32_t, int32_t, int64_t *);

struct AscendApi {
  void *lib{nullptr};
  FnGetDeviceCount getDeviceCount{nullptr};
  FnSetDevice setDevice{nullptr};
  FnResetDevice resetDevice{nullptr};
  FnGetSocName getSocName{nullptr};
  FnGetDeviceInfo getDeviceInfo{nullptr};
  FnGetDeviceCapability getDeviceCapability{nullptr};
  bool loaded{false};
  bool attempted{false};

  ~AscendApi() {
    if (lib) {
      dlclose(lib);
    }
  }

  bool load() {
    if (loaded) {
      return true;
    }
    if (attempted) {
      return false;
    }
    attempted = true;

    std::vector<std::string> candidates = {"libascendcl.so"};
    if (const char *root = std::getenv("ASCEND_TOOLKIT_PATH")) {
      std::string base(root);
      if (!base.empty()) {
        candidates.push_back(base + "/lib64/libascendcl.so");
      }
    }
    candidates.push_back(
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so");

    for (const auto &candidate : candidates) {
      lib = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (lib) {
        break;
      }
    }
    if (!lib) {
      return false;
    }

    getDeviceCount =
        reinterpret_cast<FnGetDeviceCount>(dlsym(lib, "aclrtGetDeviceCount"));
    setDevice = reinterpret_cast<FnSetDevice>(dlsym(lib, "aclrtSetDevice"));
    resetDevice =
        reinterpret_cast<FnResetDevice>(dlsym(lib, "aclrtResetDevice"));
    getSocName = reinterpret_cast<FnGetSocName>(dlsym(lib, "aclrtGetSocName"));
    getDeviceInfo =
        reinterpret_cast<FnGetDeviceInfo>(dlsym(lib, "aclrtGetDeviceInfo"));
    getDeviceCapability = reinterpret_cast<FnGetDeviceCapability>(
        dlsym(lib, "aclGetDeviceCapability"));

    loaded = getDeviceCount != nullptr;
    return loaded;
  }
};

AscendApi &api() {
  static AscendApi instance;
  return instance;
}
#endif

} // namespace

Device getDevice(uint64_t index) {
  uint64_t clockRate = 0;
  uint64_t memoryClockRate = 0;
  uint64_t busWidth = 0;
  uint64_t numSms = 0;
  std::string arch = "ascend";

#if defined(__linux__)
  auto &ascendApi = api();
  if (!ascendApi.load()) {
    return Device(DeviceType::ASCEND, index, clockRate, memoryClockRate,
                  busWidth, numSms, arch);
  }

  uint32_t deviceCount = 0;
  if (ascendApi.getDeviceCount && ascendApi.getDeviceCount(&deviceCount) == 0 &&
      index < deviceCount) {
    bool setOk = ascendApi.setDevice &&
                 ascendApi.setDevice(static_cast<int32_t>(index)) == 0;

    if (ascendApi.getSocName) {
      if (const char *soc = ascendApi.getSocName()) {
        if (*soc != '\0') {
          arch = std::string(soc);
        }
      }
    }

    if (ascendApi.getDeviceCapability) {
      int64_t value = 0;
      if (ascendApi.getDeviceCapability(static_cast<uint32_t>(index),
                                        kAclDevAttrAiCoreCoreNum,
                                        &value) == 0 &&
          value > 0) {
        numSms = static_cast<uint64_t>(value);
      }
      value = 0;
      if (ascendApi.getDeviceCapability(static_cast<uint32_t>(index),
                                        kAclDevAttrL2CacheSize, &value) == 0 &&
          value > 0) {
        busWidth = static_cast<uint64_t>(value);
      }
    }

    if (ascendApi.getDeviceInfo) {
      int64_t value = 0;
      if (ascendApi.getDeviceInfo(static_cast<int32_t>(index),
                                  kAclDevAttrAiCoreCoreNum, &value) == 0 &&
          value > 0) {
        numSms = static_cast<uint64_t>(value);
      } else if (ascendApi.getDeviceInfo(static_cast<int32_t>(index),
                                         kAclDevAttrComputeUnit, &value) == 0 &&
                 value > 0 && numSms == 0) {
        numSms = static_cast<uint64_t>(value);
      }

      value = 0;
      if (ascendApi.getDeviceInfo(static_cast<int32_t>(index),
                                  kAclDevAttrL2CacheSize, &value) == 0 &&
          value > 0) {
        busWidth = static_cast<uint64_t>(value);
      } else if (ascendApi.getDeviceInfo(static_cast<int32_t>(index),
                                         kAclDevAttrTotalGlobalMemSize,
                                         &value) == 0 &&
                 value > 0 && busWidth == 0) {
        // AscendCL does not expose a CUDA-style memory bus width directly.
        // Keep a stable non-zero capability snapshot in this field.
        busWidth = static_cast<uint64_t>(value);
      }
    }

    if (setOk && ascendApi.resetDevice) {
      (void)ascendApi.resetDevice(static_cast<int32_t>(index));
    }
  }
#endif

  return Device(DeviceType::ASCEND, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}

} // namespace ascend

} // namespace proton
