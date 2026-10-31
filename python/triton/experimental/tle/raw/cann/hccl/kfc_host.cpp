/**
 * Copyright 2026- FlagOS Contributors
 *
 * Runtime-only probe for the CANN AICPU KFC launch contract.
 *
 * Launch packing matches cann-hccl HcclCommunicator::AicpuUnfoldKernelLaunch:
 *   ApiParamDef + hostInputInfo tiling fixup + KERNEL_TYPE_AICPU_KFC
 *   kernel = RunAicpuRpcSrvLaunch (eager/non-GE server entry)
 *
 * References:
 *   - cann-hccl hccl_communicator_host.cc::AicpuUnfoldKernelLaunch
 *   - hcomm aicpu_kfc_interface.cc::RunAicpuRpcSrvLaunch
 *   - aicpu_operator_pub.h::HcclKFCTilingData
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <vector>

#include <acl/acl_rt.h>
#include <hccl/hccl.h>
#include <hccl/hccl_mc2.h>
#include <hccl/hccl_types.h>
#include <hccl/hcomm_res_defs.h>

#include "runtime/rt_external_kernel.h"
#include "runtime/runtime/kernel.h"

namespace {

constexpr uint8_t kAllReduceOpType = static_cast<uint8_t>(HCCL_CMD_ALLREDUCE);
constexpr uint8_t kCommEngineAicpu = static_cast<uint8_t>(COMM_ENGINE_AICPU);

constexpr uint8_t kTaskStartServer = 2; // HCCL_KFC_TASK_HCC_START_SERVER
constexpr uint8_t kPrepareOnKernel = 1; // TASK_PREPARE_KERNEL
constexpr uint8_t kAllReduceFuncId = 2;
constexpr uint8_t kAicpuFp16 = 3;
constexpr uint8_t kAicpuAllReduce = 2;
constexpr uint32_t kKfcMsgBodyBytes = 64U * 64U * 64U * 2U;
constexpr uint32_t kMc2WorkspaceOffset = 16U * 1024U * 1024U;
constexpr uint32_t kAicpuStreamMode = 1U;
constexpr size_t kHcomIdOffset = 16U + 4U + 4U + 8U + 32U * 8U + 32U * 8U;
constexpr size_t kHcomIdBytes = 128U;
constexpr size_t kInlineTilingBytes = 2048U;

using HcomMc2AiCpuStreamAllocAndGetFn = HcclResult (*)(const char *group,
                                                       uint32_t streamMode,
                                                       rtStream_t *aiCpuStream);

// Exact CANN 9.1.0 HcclKFCTilingData layout.
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

// Mirrors cann-hccl AicpuUnfoldKernelLaunch::ApiParamDef.
// Layout overlays KFCTask: inputA, outputC, commOut, context, workSpace,
// tilingData.
struct ApiParamDef {
  uint64_t x1 = 0;        // inputA / sendbuffer
  uint64_t y = 0;         // outputC (unused when useBufferType==0)
  uint64_t gatherOut = 0; // commOut / recvbuffer
  uint64_t context = 0;
  uint64_t workspace = 0;
  uint64_t tilingDataPtr = 0;
  uint8_t tilingData[kInlineTilingBytes] = {};
  char soName[32] = "libccl_kernel.so";
  char kernelName[32] = "";
  char opName[32] = "HcclAicpuOp";
  char hostInputInfo[16] = {};
};

static_assert(offsetof(ApiParamDef, tilingDataPtr) == 5 * sizeof(void *));
static_assert(offsetof(ApiParamDef, tilingData) == 6 * sizeof(void *));

struct ProbeState {
  aclrtStream attachStream = nullptr;
  bool ownAttachStream = false;
  aclrtEvent startEvent = nullptr;
  std::vector<uint8_t> launchBuffer;
  rtError_t launchResult = RT_ERROR_NONE;
};

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
  rtStream_t stream = nullptr;
  const HcclResult rc = allocFn(hcomId, kAicpuStreamMode, &stream);
  if (rc != HCCL_SUCCESS || stream == nullptr) {
    return rc == HCCL_SUCCESS ? -4 : static_cast<int>(rc);
  }
  *outStream = reinterpret_cast<aclrtStream>(stream);
  return 0;
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
 * Enqueue on attach stream (after ACL wait):
 *   RunAicpuRpcSrvLaunch with ApiParamDef packing
 * Main stream records the start event before this wait.
 */
int hccl_kfc_probe_start(uint64_t context, void *mainStream, uint64_t scratch,
                         uint64_t count, uint64_t *outHandle,
                         int *outLaunchResult) {
  if (context == 0 || mainStream == nullptr || scratch == 0 ||
      outHandle == nullptr || outLaunchResult == nullptr) {
    return -1;
  }
  *outHandle = 0;
  *outLaunchResult = -1;

  std::unique_ptr<ProbeState> state(new ProbeState());
  uint64_t workspace = 0;
  int rc = ReadWorkspace(context, &workspace);
  if (rc != 0 || workspace == 0) {
    return rc != 0 ? rc : -2;
  }

  KfcTilingData tiling{};
  tiling.preparePosition = kPrepareOnKernel;
  tiling.sendCnt = count;
  tiling.recvCnt = count;
  tiling.totalCnt = count;
  tiling.turnNum = 1;
  tiling.workspaceOff = kMc2WorkspaceOffset;
  tiling.notifyOff = kKfcMsgBodyBytes;
  tiling.notifyBeginCnt = 1;
  tiling.notifyEndCnt = 1;
  tiling.useBufferType = 0;
  tiling.funID = kAllReduceFuncId;
  tiling.dataType = kAicpuFp16;
  tiling.groupNum = 1;
  tiling.reuseMode = 1;
  tiling.commType = kAicpuAllReduce;
  tiling.reduceOp = 0;
  tiling.commOrder = 1;
  tiling.waitPolicy = 1;
  tiling.rspPolicy = 1;
  tiling.exitPolicy = 2;
  tiling.taskType = kTaskStartServer;
  tiling.hasCommOut = 1;

  state->launchBuffer.assign(sizeof(ApiParamDef), 0);
  auto *apiParam = reinterpret_cast<ApiParamDef *>(state->launchBuffer.data());
  apiParam->x1 = scratch;
  apiParam->y = 0;
  apiParam->gatherOut = scratch;
  apiParam->context = context;
  apiParam->workspace = workspace;
  std::memcpy(apiParam->tilingData, &tiling, sizeof(tiling));
  apiParam->tilingDataPtr = reinterpret_cast<uint64_t>(apiParam->tilingData);
  std::snprintf(apiParam->soName, sizeof(apiParam->soName), "%s",
                "libccl_kernel.so");
  // Eager/non-GE path launches RunAicpuRpcSrvLaunch, not the GE wrapper
  // RunAicpuKfcSrvLaunch(void* args[]).
  std::snprintf(apiParam->kernelName, sizeof(apiParam->kernelName), "%s",
                "RunAicpuRpcSrvLaunch");
  std::snprintf(apiParam->opName, sizeof(apiParam->opName), "%s",
                "HcclAicpuOp");

  auto *hostInfo =
      reinterpret_cast<rtHostInputInfo_t *>(apiParam->hostInputInfo);
  hostInfo->addrOffset = static_cast<uint32_t>(5U * sizeof(void *));
  hostInfo->dataOffset = static_cast<uint32_t>(6U * sizeof(void *));

  aclError aclRc = ACL_SUCCESS;
  {
    char hcomId[kHcomIdBytes] = {};
    const int hcomRc = ReadHcomId(context, hcomId, sizeof(hcomId));
    const int streamRc = AllocHcclAiCpuStream(context, &state->attachStream);
    if (streamRc == 0 && state->attachStream != nullptr) {
      std::fprintf(stderr,
                   "[kfc-probe] HCCL AICPU stream=%p hcomId='%s' "
                   "pack=ApiParamDef kernel=RunAicpuRpcSrvLaunch\n",
                   state->attachStream, hcomId);
      (void)hcomRc;
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
                   "[kfc-probe] FALLBACK homemade stream=%p hcomId='%s' "
                   "stream_rc=%d\n",
                   state->attachStream, hcomId, streamRc);
    }
  }

  aclRc = aclrtCreateEvent(&state->startEvent);
  if (aclRc != ACL_SUCCESS || state->startEvent == nullptr) {
    CleanupCompleted(state.release());
    return aclRc == ACL_SUCCESS ? -5 : static_cast<int>(aclRc);
  }
  // ACL events require record before wait. AICore is enqueued after return,
  // so it still appears after this record on the main stream.
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

  rtAicpuArgsEx_t launchArgs{};
  launchArgs.args = state->launchBuffer.data();
  launchArgs.argsSize = static_cast<uint32_t>(sizeof(ApiParamDef));
  launchArgs.hostInputInfoPtr = hostInfo;
  launchArgs.hostInputInfoNum = 1;
  launchArgs.kernelOffsetInfoPtr = nullptr;
  launchArgs.kernelOffsetInfoNum = 0;
  launchArgs.soNameAddrOffset =
      static_cast<uint32_t>(reinterpret_cast<const char *>(apiParam->soName) -
                            reinterpret_cast<const char *>(apiParam));
  launchArgs.kernelNameAddrOffset = static_cast<uint32_t>(
      reinterpret_cast<const char *>(apiParam->kernelName) -
      reinterpret_cast<const char *>(apiParam));
  launchArgs.isNoNeedH2DCopy = false;
  launchArgs.timeout = 30;

  state->launchResult = rtAicpuKernelLaunchExWithArgs(
      KERNEL_TYPE_AICPU_KFC, apiParam->opName, 1, &launchArgs, nullptr,
      reinterpret_cast<rtStream_t>(state->attachStream), RT_KERNEL_DEFAULT);
  *outLaunchResult = static_cast<int>(state->launchResult);

  if (state->launchResult != RT_ERROR_NONE) {
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
  const aclError rc =
      aclrtSynchronizeStreamWithTimeout(state->attachStream, timeoutMs);
  if (rc != ACL_SUCCESS) {
    return static_cast<int>(rc);
  }
  CleanupCompleted(state);
  return 0;
}

} // extern "C"
