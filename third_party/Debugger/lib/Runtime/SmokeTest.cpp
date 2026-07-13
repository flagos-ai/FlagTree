#include "Debugger/Runtime/BufferLayout.h"
#include "Debugger/Runtime/TransferEngine.h"

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

mlir::flagtree::debugger::BufferMeta
makeDefaultBufferMeta(mlir::flagtree::debugger::BackendKind backendKind =
                          mlir::flagtree::debugger::BackendKind::UNKNOWN) {
  using namespace mlir::flagtree::debugger;
  BufferMeta meta{};
  meta.runId = 1;
  meta.deviceId = 0;
  meta.kernelId = 1;
  meta.protocolVer = kProtocolVersion;
  meta.recordLevel = RecordLevel::LEVEL_SUMMARY;
  meta.exportMode = ExportMode::POST_KERNEL_EXPORT;
  meta.backendKind = backendKind;
  return meta;
}

mlir::flagtree::debugger::DebugBufferPlan
makeDefaultBufferPlan(uint32_t recordCapacity = 1024, size_t payloadBytes = 0) {
  using namespace mlir::flagtree::debugger;
  DebugBufferPlan plan;
  plan.recordCapacity = recordCapacity;
  plan.recordSize = kDefaultRecordSize;
  plan.payloadBytes = payloadBytes;
  plan.exportMode = ExportMode::POST_KERNEL_EXPORT;
  return plan;
}

mlir::flagtree::debugger::DebugRuntimeMetadata makeDefaultRuntimeMetadata() {
  using namespace mlir::flagtree::debugger;
  DebugRuntimeMetadata metadata;

  BufferRegistrationInfo buffer;
  buffer.bufferId = 1;
  buffer.bufferName = "input";
  buffer.baseAddress = 0x1000;
  buffer.sizeBytes = 512;
  buffer.alignment = 16;
  metadata.buffers.push_back(buffer);

  LaunchTensorInfo tensor;
  tensor.argumentIndex = 0;
  tensor.logicalName = "x";
  tensor.dtype = "fp32";
  tensor.shape = {128};
  tensor.stride = {1};
  tensor.layout = "contiguous";
  tensor.bufferId = 1;
  tensor.baseAddress = 0x1000;
  tensor.sizeBytes = 512;
  metadata.tensors.push_back(tensor);

  return metadata;
}

int runLifecycleSmokeTest(mlir::flagtree::debugger::TransferEngine &engine) {
  using namespace mlir::flagtree::debugger;

  BufferMeta meta = makeDefaultBufferMeta();
  DebugBufferPlan plan = makeDefaultBufferPlan();
  DebugRuntimeMetadata runtimeMetadata = makeDefaultRuntimeMetadata();
  DebugLaunchContext ctx = engine.prepare(meta, plan, runtimeMetadata);
  engine.initHeader(ctx);

  auto *header = reinterpret_cast<RingBufferHeader *>(ctx.deviceCtrlPtr);
  if (!ctx.deviceCtrlPtr || !ctx.hostBufferPtr || !header) {
    return 1;
  }
  if (engine.hiddenArg(ctx) != reinterpret_cast<uint64_t>(ctx.deviceCtrlPtr)) {
    return 2;
  }

  SummaryRecord record{};
  record.header.recordKind = RecordKind::SUMMARY;
  record.header.opId = 9;
  record.header.logicalInstanceId = 3;
  record.collectorKind = CollectorKind::NAN_COUNT;
  record.resultType = ResultType::U64;
  record.resultData.u64Val = 4;
  header->writeIdx = 1;
  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 0),
              &record, sizeof(record));

  DebugExportedRun run = engine.syncExport(ctx);
  if (run.rawBuffer.size() < sizeof(RingBufferHeader)) {
    return 3;
  }

  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  if (exportedHeader->writeIdx != 1 || exportedHeader->capacity != 1024 ||
      exportedHeader->recordSize != kDefaultRecordSize) {
    return 4;
  }

  const auto *exportedRecord = reinterpret_cast<const SummaryRecord *>(
      run.rawBuffer.data() + getRecordSlotOffset(ctx.layout, 0));
  if (exportedRecord->header.opId != 9 ||
      exportedRecord->resultData.u64Val != 4) {
    return 5;
  }

  engine.release(ctx);
  if (ctx.deviceCtrlPtr != nullptr || ctx.hostBufferPtr != nullptr) {
    return 6;
  }

  return 0;
}

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
[[noreturn]] void failAcl(const char *call, int errorCode) {
  std::string message =
      std::string(call) + " failed with aclError=" + std::to_string(errorCode);
  if (const char *recent = aclGetRecentErrMsg(); recent && recent[0] != '\0') {
    message += ", recent_err=\"";
    message += recent;
    message += "\"";
  }
  std::fprintf(stderr, "FlagTree debugger runtime smoke test failed: %s\n",
               message.c_str());
  std::exit(250);
}

void checkAcl(aclError error, const char *call) {
  if (error != ACL_SUCCESS) {
    failAcl(call, error);
  }
}

class CannRuntimeScope {
public:
  ~CannRuntimeScope() {
    if (stream_ != nullptr) {
      (void)aclrtDestroyStream(stream_);
    }
    if (deviceSet_) {
      (void)aclrtResetDevice(deviceId_);
    }
  }

  void init(int32_t deviceId) {
    deviceId_ = deviceId;
    checkAcl(aclrtSetDevice(deviceId_), "aclrtSetDevice");
    deviceSet_ = true;
    checkAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
  }

  aclrtStream stream() const { return stream_; }

private:
  bool deviceSet_ = false;
  int32_t deviceId_ = 0;
  aclrtStream stream_ = nullptr;
};

int runCannLifecycleSmokeTest() {
  using namespace mlir::flagtree::debugger;

  CannRuntimeScope runtime;
  runtime.init(0);

  auto engine = createTransferEngine(
      BackendKind::CANN, reinterpret_cast<uint64_t>(runtime.stream()));

  BufferMeta meta = makeDefaultBufferMeta(BackendKind::CANN);
  DebugBufferPlan plan = makeDefaultBufferPlan();
  DebugRuntimeMetadata runtimeMetadata = makeDefaultRuntimeMetadata();
  DebugLaunchContext ctx = engine->prepare(meta, plan, runtimeMetadata);
  engine->initHeader(ctx);

  SummaryRecord record{};
  record.header.recordKind = RecordKind::SUMMARY;
  record.header.opId = 11;
  record.header.logicalInstanceId = 7;
  record.collectorKind = CollectorKind::NAN_COUNT;
  record.resultType = ResultType::U64;
  record.resultData.u64Val = 3;

  RingBufferHeader header{};
  header.writeIdx = 1;
  header.capacity = ctx.recordCapacity;
  header.overflowCount = 0;
  header.flags = RB_FLAG_NONE;
  header.recordSize = ctx.bufferPlan.recordSize;
  header.payloadOffset = static_cast<uint32_t>(ctx.layout.payloadOffset);

  auto *deviceBytes = reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr);
  checkAcl(aclrtMemcpy(deviceBytes, sizeof(header), &header, sizeof(header),
                       ACL_MEMCPY_HOST_TO_DEVICE),
           "aclrtMemcpy(header)");
  checkAcl(aclrtMemcpy(deviceBytes + getRecordSlotOffset(ctx.layout, 0),
                       sizeof(record), &record, sizeof(record),
                       ACL_MEMCPY_HOST_TO_DEVICE),
           "aclrtMemcpy(record)");

  DebugExportedRun run = engine->syncExport(ctx);
  engine->release(ctx);

  if (run.rawBuffer.size() <
      getRecordSlotOffset(ctx.layout, 0) + sizeof(SummaryRecord)) {
    return 201;
  }
  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  if (exportedHeader->writeIdx != 1) {
    return 202;
  }
  const auto *exportedRecord = reinterpret_cast<const SummaryRecord *>(
      run.rawBuffer.data() + getRecordSlotOffset(ctx.layout, 0));
  if (exportedRecord->header.opId != 11 ||
      exportedRecord->resultData.u64Val != 3) {
    return 203;
  }
  return 0;
}
#endif

} // namespace

int main() {
  using namespace mlir::flagtree::debugger;

  auto realEngine = createTransferEngine();
  if (int rc = runLifecycleSmokeTest(*realEngine)) {
    return 100 + rc;
  }

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
  if (const char *runCann =
          std::getenv("FLAGTREE_DEBUGGER_RUNTIME_RUN_CANN_SMOKE");
      runCann && runCann[0] != '\0' && std::strcmp(runCann, "0") != 0) {
    if (int rc = runCannLifecycleSmokeTest()) {
      return rc;
    }
  }
#endif

  return 0;
}
