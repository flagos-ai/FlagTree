/**
 * Copyright 2026- FlagOS Contributors
 *
 * Host-side AICPU KFC launch for Triton raw device-HCCL (non-GE).
 *
 * Aligned 1:1 with official hcomm / ops-transformer:
 *   - AicpuUnfoldKernelLaunch ApiParamDef (5×u64) + separate tiling
 *   - AicpuAclKernelLaunch: ccl_kernel.json binHandle +
 *       aclrtKernelArgsAppend / AppendPlaceHolder / LaunchKernelWithConfig
 *   - kernelName RunAicpuRpcSrvLaunch (eager/non-GE)
 *   - DoAllReduceTiling: TASK_DELIVER, exitPolicy=0, preparePosition=1
 *   - CommonKFCMc2GenTask stream order: Record(main) → Wait(attach) → AICPU
 *       (AICore launched on main by caller after this returns)
 */

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <acl/acl_rt.h>
#include <hccl/hccl.h>
#include <hccl/hccl_mc2.h>
#include <hccl/hccl_types.h>
#include <hccl/hcomm_res_defs.h>

namespace {

constexpr uint8_t kAllReduceOpType = static_cast<uint8_t>(HCCL_CMD_ALLREDUCE);
constexpr uint8_t kCommEngineAicpu = static_cast<uint8_t>(COMM_ENGINE_AICPU);

// HcclKfcTaskType / buffer — match MatmulAllReduce DoAllReduceTiling.
constexpr uint8_t kTaskDeliver = 4;     // HCCL_KFC_TASK_HCC_TASK_DELIVER
constexpr uint8_t kPrepareOnKernel = 1; // TASK_PREPARE_KERNEL (device Hccl API)
constexpr uint8_t kBufferTypeOutput = 1;
constexpr uint8_t kBufferTypeWindowIn = 2;
constexpr uint8_t kAllReduceFuncId = 2;
constexpr uint8_t kAicpuFp16 = 3;
constexpr uint8_t kAicpuAllReduce = 2;
constexpr uint32_t kKfcMsgBodyBytes = 64U * 64U * 64U * 2U;
constexpr uint32_t kMc2WorkspaceOffset = 16U * 1024U * 1024U;
constexpr uint32_t kAicpuStreamMode = 1U;
constexpr uint16_t kAicpuTimeoutSec = 30;
constexpr size_t kHcomIdOffset = 16U + 4U + 4U + 8U + 32U * 8U + 32U * 8U;
constexpr size_t kHcomIdBytes = 128U;

using HcomMc2AiCpuStreamAllocAndGetFn = HcclResult (*)(
    const char *group, uint32_t streamMode, aclrtStream *aiCpuStream);

// Exact CANN 9.1.0 HcclKFCTilingData layout (aicpu_operator_pub.h).
struct KfcTilingData {
  uint32_t preparePosition;
  uint64_t sendOff;
  uint64_t recvOff;
  uint64_t tailSendOff;
  uint64_t tailRecvOff;
  uint64_t sendCnt;
  uint64_t recvCnt;
  uint64_t tailSendCnt;
  uint64_t tailRecvCnt;
  uint64_t totalCnt;
  uint32_t turnNum;
  uint32_t tailNum;
  uint32_t stride;
  uint32_t workspaceOff;
  uint32_t notifyOff;
  uint16_t notifyBeginCnt;
  uint16_t notifyEndCnt;
  uint8_t useBufferType;
  uint8_t funID;
  uint8_t dataType;
  uint8_t groupNum;
  uint8_t reuseMode;
  uint8_t commType;
  uint8_t reduceOp;
  uint8_t commOrder;
  uint8_t waitPolicy;
  uint8_t rspPolicy;
  uint8_t exitPolicy;
  uint8_t commAlg;
  uint8_t taskType;
  uint8_t debugMode;
  uint8_t stepSize;
  uint8_t sendArgIndex;
  uint8_t recvArgIndex;
  uint8_t commOutArgIndex;
  uint8_t hasCommOut;
};

static_assert(offsetof(KfcTilingData, sendOff) == 8);
static_assert(offsetof(KfcTilingData, turnNum) == 80);
static_assert(offsetof(KfcTilingData, notifyBeginCnt) == 100);
static_assert(offsetof(KfcTilingData, taskType) == 116);
static_assert(sizeof(KfcTilingData) == 128);

// Exact hcomm AicpuUnfoldKernelLaunch::ApiParamDef (tiling is NOT inline).
struct ApiParamDef {
  uint64_t x1 = 0; // sendbuffer
  uint64_t y = 0;  // unused when useBufferType drives gatherOut=y on aicpu
  uint64_t gatherOut = 0; // recvbuffer
  uint64_t context = 0;
  uint64_t workspace = 0;
};

static_assert(sizeof(ApiParamDef) == 5 * sizeof(uint64_t));

struct ProbeState {
  aclrtStream attachStream = nullptr;
  bool ownAttachStream = false;
  aclrtEvent startEvent = nullptr; // main→attach (MC2 Wait)
  aclError launchResult = ACL_SUCCESS;
};

aclrtBinHandle g_cclBinHandle = nullptr;
std::once_flag g_cclBinOnce;

int ReadWorkspace(uint64_t context, uint64_t *workspace) {
  if (context == 0 || workspace == nullptr) {
    return -1;
  }
  const aclError rc = aclrtMemcpy(
      workspace, sizeof(*workspace), reinterpret_cast<const void *>(context),
      sizeof(*workspace), ACL_MEMCPY_DEVICE_TO_HOST);
  return rc == ACL_SUCCESS ? 0 : static_cast<int>(rc);
}

int ReadHcomId(uint64_t context, char *hcomId, size_t hcomIdBytes) {
  if (context == 0 || hcomId == nullptr || hcomIdBytes < kHcomIdBytes) {
    return -1;
  }
  std::memset(hcomId, 0, hcomIdBytes);
  const aclError rc =
      aclrtMemcpy(hcomId, kHcomIdBytes,
                  reinterpret_cast<const void *>(context + kHcomIdOffset),
                  kHcomIdBytes, ACL_MEMCPY_DEVICE_TO_HOST);
  return rc == ACL_SUCCESS ? 0 : static_cast<int>(rc);
}

HcomMc2AiCpuStreamAllocAndGetFn ResolveMc2AiCpuStreamAlloc() {
  static HcomMc2AiCpuStreamAllocAndGetFn fn = nullptr;
  if (fn != nullptr) {
    return fn;
  }
  fn = reinterpret_cast<HcomMc2AiCpuStreamAllocAndGetFn>(
      dlsym(RTLD_DEFAULT, "HcomMc2AiCpuStreamAllocAndGet"));
  if (fn != nullptr) {
    return fn;
  }
  void *handle = dlopen("libhccl_fwk.so", RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    return nullptr;
  }
  fn = reinterpret_cast<HcomMc2AiCpuStreamAllocAndGetFn>(
      dlsym(handle, "HcomMc2AiCpuStreamAllocAndGet"));
  return fn;
}

int AllocHcclAiCpuStream(uint64_t context, aclrtStream *outStream) {
  if (outStream == nullptr) {
    return -1;
  }
  *outStream = nullptr;

  char hcomId[kHcomIdBytes];
  const int readRc = ReadHcomId(context, hcomId, sizeof(hcomId));
  if (readRc != 0 || hcomId[0] == '\0') {
    return readRc != 0 ? readRc : -2;
  }

  auto *allocFn = ResolveMc2AiCpuStreamAlloc();
  if (allocFn == nullptr) {
    return -3;
  }
  aclrtStream stream = nullptr;
  const HcclResult rc = allocFn(hcomId, kAicpuStreamMode, &stream);
  if (rc != HCCL_SUCCESS || stream == nullptr) {
    return rc == HCCL_SUCCESS ? -4 : static_cast<int>(rc);
  }
  *outStream = stream;
  return 0;
}

// hcomm GetKernelFilePath + LoadAICPUKernel / LoadBinaryFromFile.
std::string ResolveCclKernelJsonPath() {
  const char *home = std::getenv("ASCEND_HOME_PATH");
  if (home == nullptr || home[0] == '\0') {
    home = std::getenv("ASCEND_TOOLKIT_HOME");
  }
  std::string base = (home != nullptr && home[0] != '\0')
                         ? std::string(home)
                         : std::string("/usr/local/Ascend/cann");
  if (!base.empty() && base.back() == '/') {
    base.pop_back();
  }
  return base + "/opp/built-in/op_impl/aicpu/config/ccl_kernel.json";
}

int EnsureCclBinHandle() {
  std::call_once(g_cclBinOnce, []() {
    const std::string jsonPath = ResolveCclKernelJsonPath();
    char realPath[PATH_MAX] = {};
    if (realpath(jsonPath.c_str(), realPath) == nullptr) {
      std::fprintf(stderr, "[kfc] ccl_kernel.json realpath failed: %s\n",
                   jsonPath.c_str());
      return;
    }
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions loadOptions{};
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    const aclError rc =
        aclrtBinaryLoadFromFile(realPath, &loadOptions, &g_cclBinHandle);
    if (rc != ACL_SUCCESS || g_cclBinHandle == nullptr) {
      std::fprintf(stderr,
                   "[kfc] aclrtBinaryLoadFromFile(%s) failed acl_rc=%d\n",
                   realPath, static_cast<int>(rc));
      g_cclBinHandle = nullptr;
    }
  });
  return g_cclBinHandle == nullptr ? -7 : 0;
}

// Exact AicpuAclKernelLaunch TaskComm path (launch_aicpu.cc).
aclError AicpuAclKernelLaunchAligned(aclrtStream stm, void *apiParam,
                                     uint32_t apiParamSize, void *tilingDataPtr,
                                     uint32_t tilingDataSize) {
  if (g_cclBinHandle == nullptr || apiParam == nullptr || apiParamSize == 0 ||
      tilingDataPtr == nullptr || tilingDataSize == 0) {
    return ACL_ERROR_INVALID_PARAM;
  }

  aclrtFuncHandle funcHandle = nullptr;
  aclError aclRet = aclrtBinaryGetFunction(g_cclBinHandle,
                                           "RunAicpuRpcSrvLaunch", &funcHandle);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }

  aclrtArgsHandle argsHandle = nullptr;
  aclRet = aclrtKernelArgsInit(funcHandle, &argsHandle);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }

  aclrtParamHandle paraHandle = nullptr;
  aclRet =
      aclrtKernelArgsAppend(argsHandle, apiParam, apiParamSize, &paraHandle);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }

  aclRet = aclrtKernelArgsAppendPlaceHolder(argsHandle, &paraHandle);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }

  void *dataAddr = nullptr;
  aclRet = aclrtKernelArgsGetPlaceHolderBuffer(argsHandle, paraHandle,
                                               tilingDataSize, &dataAddr);
  if (aclRet != ACL_SUCCESS || dataAddr == nullptr) {
    return aclRet != ACL_SUCCESS ? aclRet : ACL_ERROR_INVALID_PARAM;
  }
  std::memcpy(dataAddr, tilingDataPtr, tilingDataSize);

  aclRet = aclrtKernelArgsFinalize(argsHandle);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }

  aclrtLaunchKernelAttr attr{};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  attr.value.timeout = kAicpuTimeoutSec;
  aclrtLaunchKernelCfg cfg{};
  cfg.numAttrs = 1;
  cfg.attrs = &attr;

  return aclrtLaunchKernelWithConfig(funcHandle, /*numBlocks=*/1, stm, &cfg,
                                     argsHandle, nullptr);
}

void CleanupCompleted(ProbeState *state) {
  if (state == nullptr) {
    return;
  }
  if (state->startEvent != nullptr) {
    (void)aclrtDestroyEvent(state->startEvent);
  }
  if (state->ownAttachStream && state->attachStream != nullptr) {
    (void)aclrtDestroyStream(state->attachStream);
  }
  delete state;
}

} // namespace

extern "C" {

int hccl_kfc_probe_create_ctx(void *comm, uint64_t count,
                              uint64_t *outContext) {
  if (comm == nullptr || outContext == nullptr) {
    return -1;
  }
  *outContext = 0;

  void *opArgs = nullptr;
  HcclResult rc = HcclKfcAllocOpArgs(&opArgs);
  if (rc != HCCL_SUCCESS || opArgs == nullptr) {
    return rc == HCCL_SUCCESS ? -2 : static_cast<int>(rc);
  }

  auto fail = [&](HcclResult error) {
    (void)HcclKfcFreeOpArgs(opArgs);
    return static_cast<int>(error);
  };
  if ((rc = HcclKfcOpArgsSetCommEngine(opArgs, kCommEngineAicpu)) !=
      HCCL_SUCCESS) {
    return fail(rc);
  }
  if ((rc = HcclKfcOpArgsSetSrcDataType(
           opArgs, static_cast<uint8_t>(HCCL_DATA_TYPE_FP16))) !=
      HCCL_SUCCESS) {
    return fail(rc);
  }
  if ((rc = HcclKfcOpArgsSetDstDataType(
           opArgs, static_cast<uint8_t>(HCCL_DATA_TYPE_FP16))) !=
      HCCL_SUCCESS) {
    return fail(rc);
  }
  if ((rc = HcclKfcOpArgsSetReduceType(
           opArgs, static_cast<uint32_t>(HCCL_REDUCE_SUM))) != HCCL_SUCCESS) {
    return fail(rc);
  }
  if ((rc = HcclKfcOpArgsSetCount(opArgs, count)) != HCCL_SUCCESS) {
    return fail(rc);
  }

  void *context = nullptr;
  rc = HcclCreateOpResCtx(reinterpret_cast<HcclComm>(comm), kAllReduceOpType,
                          opArgs, &context);
  (void)HcclKfcFreeOpArgs(opArgs);
  if (rc != HCCL_SUCCESS || context == nullptr) {
    return rc == HCCL_SUCCESS ? -3 : static_cast<int>(rc);
  }
  *outContext = reinterpret_cast<uint64_t>(context);
  return 0;
}

/**
 * Enqueue AICPU KFC on HCCL attach stream (CommonKFCMc2GenTask order):
 *   main:  RecordEvent
 *   attach: WaitEvent → AicpuAclKernelLaunch(RunAicpuRpcSrvLaunch, DELIVER)
 * AICore is launched on main after this returns.
 */
int hccl_kfc_probe_start(uint64_t context, void *mainStream, uint64_t sendPtr,
                         uint64_t recvPtr, uint64_t sendCnt, uint64_t turnNum,
                         uint32_t useBufferType, uint64_t *outHandle,
                         int *outLaunchResult) {
  if (context == 0 || mainStream == nullptr || sendPtr == 0 || recvPtr == 0 ||
      sendCnt == 0 || turnNum == 0 || outHandle == nullptr ||
      outLaunchResult == nullptr) {
    return -1;
  }
  if (useBufferType != kBufferTypeOutput &&
      useBufferType != kBufferTypeWindowIn && useBufferType != 0) {
    return -6;
  }
  *outHandle = 0;
  *outLaunchResult = -1;

  const int binRc = EnsureCclBinHandle();
  if (binRc != 0) {
    return binRc;
  }

  std::unique_ptr<ProbeState> state(new ProbeState());
  uint64_t workspace = 0;
  int rc = ReadWorkspace(context, &workspace);
  if (rc != 0 || workspace == 0) {
    return rc != 0 ? rc : -2;
  }

  const uint64_t turns = turnNum;
  const uint64_t perTile = sendCnt;
  const uint64_t total = perTile * turns;

  // DoAllReduceTiling (useHcclApi=true) + in-place C → OUTPUT buffer type.
  KfcTilingData tiling{};
  tiling.preparePosition = kPrepareOnKernel;
  tiling.sendOff = perTile * sizeof(uint16_t);
  tiling.recvOff = tiling.sendOff;
  tiling.sendCnt = perTile;
  tiling.recvCnt = perTile;
  tiling.totalCnt = total;
  tiling.turnNum = static_cast<uint32_t>(turns);
  tiling.tailNum = 0;
  tiling.stride = 0;
  tiling.workspaceOff = kMc2WorkspaceOffset;
  tiling.notifyOff = kKfcMsgBodyBytes;
  tiling.notifyBeginCnt = 1;
  tiling.notifyEndCnt = 1;
  tiling.useBufferType = static_cast<uint8_t>(
      useBufferType == 0 ? kBufferTypeOutput : useBufferType);
  tiling.funID = kAllReduceFuncId;
  tiling.dataType = kAicpuFp16;
  tiling.groupNum = 1;
  tiling.reuseMode = static_cast<uint8_t>(turns > 255 ? 255 : turns);
  tiling.commType = kAicpuAllReduce;
  tiling.reduceOp = 0;
  tiling.commOrder = 1;
  tiling.waitPolicy = 1;
  tiling.rspPolicy = 1;
  tiling.exitPolicy = 0;
  tiling.taskType = kTaskDeliver;
  tiling.hasCommOut = 1;

  // AicpuUnfoldKernelLaunch packing (tiling passed separately below).
  ApiParamDef apiParam{};
  apiParam.x1 = sendPtr;
  apiParam.y = recvPtr;
  apiParam.gatherOut = recvPtr;
  apiParam.context = context;
  apiParam.workspace = workspace;

  aclError aclRc = ACL_SUCCESS;
  {
    char hcomId[kHcomIdBytes] = {};
    const int hcomRc = ReadHcomId(context, hcomId, sizeof(hcomId));
    const int streamRc = AllocHcclAiCpuStream(context, &state->attachStream);
    if (streamRc == 0 && state->attachStream != nullptr) {
      (void)hcomRc;
      (void)hcomId;
    } else {
      aclRc = aclrtCreateStreamWithConfig(&state->attachStream, 0,
                                          ACL_STREAM_FAST_LAUNCH |
                                              ACL_STREAM_FAST_SYNC);
      if (aclRc != ACL_SUCCESS || state->attachStream == nullptr) {
        CleanupCompleted(state.release());
        return streamRc != 0
                   ? (200000 + streamRc)
                   : (aclRc == ACL_SUCCESS ? -4 : static_cast<int>(aclRc));
      }
      state->ownAttachStream = true;
      std::fprintf(stderr,
                   "[kfc] FALLBACK homemade stream=%p hcomId='%s' "
                   "stream_rc=%d\n",
                   state->attachStream, hcomId, streamRc);
    }
  }

  // CommonKFCMc2GenTask: HcomWait(attach) then AICPU; HcomRecord(main) then
  // AICore. ACL event approximates Wait/Record across streams for raw.
  aclRc = aclrtCreateEvent(&state->startEvent);
  if (aclRc != ACL_SUCCESS || state->startEvent == nullptr) {
    CleanupCompleted(state.release());
    return aclRc == ACL_SUCCESS ? -5 : static_cast<int>(aclRc);
  }
  aclRc = aclrtRecordEvent(state->startEvent,
                           reinterpret_cast<aclrtStream>(mainStream));
  if (aclRc != ACL_SUCCESS) {
    CleanupCompleted(state.release());
    return static_cast<int>(aclRc);
  }
  aclRc = aclrtStreamWaitEvent(state->attachStream, state->startEvent);
  if (aclRc != ACL_SUCCESS) {
    CleanupCompleted(state.release());
    return static_cast<int>(aclRc);
  }

  state->launchResult = AicpuAclKernelLaunchAligned(
      state->attachStream, &apiParam, static_cast<uint32_t>(sizeof(apiParam)),
      &tiling, static_cast<uint32_t>(sizeof(tiling)));
  *outLaunchResult = static_cast<int>(state->launchResult);

  if (state->launchResult != ACL_SUCCESS) {
    const int launchRc = static_cast<int>(state->launchResult);
    CleanupCompleted(state.release());
    return 100000 + launchRc;
  }

  *outHandle = reinterpret_cast<uint64_t>(state.release());
  return 0;
}

int hccl_kfc_probe_finish(uint64_t handle, int32_t timeoutMs) {
  auto *state = reinterpret_cast<ProbeState *>(handle);
  if (state == nullptr || state->attachStream == nullptr) {
    return -1;
  }
  // Host must sync main (Finalize) before this — see HcclKfcSession.finish.
  // Then drain attach (AICPU RunAicpuRpcSrvLaunch completed).
  const aclError rc =
      aclrtSynchronizeStreamWithTimeout(state->attachStream, timeoutMs);
  if (rc != ACL_SUCCESS) {
    return static_cast<int>(rc);
  }
  CleanupCompleted(state);
  return 0;
}

} // extern "C"
