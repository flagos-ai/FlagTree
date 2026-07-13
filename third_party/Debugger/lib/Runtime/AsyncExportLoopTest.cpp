#include "Debugger/Runtime/BufferLayout.h"
#include "Debugger/Runtime/TransferEngine.h"

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int readEnvInt(const char *name, int fallback) {
  if (const char *value = std::getenv(name)) {
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end != value && end && *end == '\0' && parsed > 0) {
      return static_cast<int>(parsed);
    }
  }
  return fallback;
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
  std::fprintf(stderr, "FlagTree debugger async export loop test failed: %s\n",
               message.c_str());
  std::exit(250);
}

void checkAcl(aclError error, const char *call) {
  if (error != ACL_SUCCESS) {
    failAcl(call, error);
  }
}

using namespace mlir::flagtree::debugger;

BufferMeta makeCannBufferMeta(uint32_t deviceId) {
  BufferMeta meta{};
  meta.runId = 1;
  meta.deviceId = deviceId;
  meta.kernelId = 1;
  meta.protocolVer = kProtocolVersion;
  meta.recordLevel = RecordLevel::LEVEL_SUMMARY;
  meta.exportMode = ExportMode::POST_KERNEL_EXPORT;
  meta.backendKind = BackendKind::CANN;
  return meta;
}

DebugBufferPlan makeCannBufferPlan(uint32_t recordCapacity) {
  DebugBufferPlan plan;
  plan.recordCapacity = recordCapacity;
  plan.recordSize = kDefaultRecordSize;
  plan.payloadBytes = 0;
  plan.exportMode = ExportMode::POST_KERNEL_EXPORT;
  return plan;
}

DebugRuntimeMetadata makeCannRuntimeMetadata() {
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

int runCannAsyncExportLoopTest() {
  using namespace mlir::flagtree::debugger;

  const int deviceId = readEnvInt("FLAGTREE_DEBUGGER_RUNTIME_ASYNC_DEVICE", 0);
  const int iterations =
      readEnvInt("FLAGTREE_DEBUGGER_RUNTIME_ASYNC_ITERS", 32);
  const int recordsPerIteration =
      readEnvInt("FLAGTREE_DEBUGGER_RUNTIME_ASYNC_RECORDS", 4);

  std::cout << "[async-loop] starting CANN async export loop test\n";
  std::cout << "[async-loop] config: device=" << deviceId
            << ", iterations=" << iterations
            << ", records_per_iteration=" << recordsPerIteration << "\n";

  CannRuntimeScope runtime;
  std::cout << "[async-loop] step 1/5: initializing ACL device + stream\n";
  runtime.init(deviceId);

  const uint32_t recordCapacity =
      static_cast<uint32_t>(std::max(recordsPerIteration, 16));

  BufferMeta meta = makeCannBufferMeta(static_cast<uint32_t>(deviceId));
  DebugBufferPlan plan = makeCannBufferPlan(recordCapacity);
  DebugRuntimeMetadata runtimeMetadata = makeCannRuntimeMetadata();

  std::cout << "[async-loop] step 2/5: creating CANN transfer engine\n";
  auto engine = createTransferEngine(
      BackendKind::CANN, reinterpret_cast<uint64_t>(runtime.stream()));
  std::cout << "[async-loop] step 3/5: preparing device/host buffers\n";
  DebugLaunchContext ctx = engine->prepare(meta, plan, runtimeMetadata);
  engine->initHeader(ctx);

  if (!ctx.deviceCtrlPtr || !ctx.hostBufferPtr || ctx.bufferSize == 0) {
    std::fprintf(stderr,
                 "FlagTree debugger async export loop test failed: invalid "
                 "launch context\n");
    return 1;
  }

  uint64_t observedValueSum = 0;
  uint64_t expectedValueSum = 0;
  uint64_t observedOpIdSum = 0;
  uint64_t expectedOpIdSum = 0;

  std::vector<uint8_t> staging(ctx.bufferSize, 0);

  const auto start = std::chrono::steady_clock::now();
  std::cout << "[async-loop] step 4/5: entering async export loop\n";
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::cout << "[async-loop] iteration " << (iteration + 1) << "/"
              << iterations << ": build summary records on host\n";
    std::fill(staging.begin(), staging.end(), 0);

    RingBufferHeader header{};
    header.writeIdx = static_cast<uint32_t>(recordsPerIteration);
    header.capacity = ctx.recordCapacity;
    header.overflowCount = 0;
    header.flags = RB_FLAG_NONE;
    header.recordSize = ctx.bufferPlan.recordSize;
    header.payloadOffset = static_cast<uint32_t>(ctx.layout.payloadOffset);
    std::memcpy(staging.data(), &header, sizeof(header));

    for (int recordIndex = 0; recordIndex < recordsPerIteration;
         ++recordIndex) {
      SummaryRecord record{};
      record.header.recordKind = RecordKind::SUMMARY;
      record.header.opId = static_cast<uint32_t>(
          1000 + iteration * recordsPerIteration + recordIndex);
      record.header.logicalInstanceId =
          static_cast<uint64_t>(iteration) * 100 + recordIndex;
      record.collectorKind = CollectorKind::ELEMENT_COUNT;
      record.resultType = ResultType::U64;
      record.resultData.u64Val =
          static_cast<uint64_t>(iteration + 1) * 10 + recordIndex;

      expectedOpIdSum += record.header.opId;
      expectedValueSum += record.resultData.u64Val;

      std::memcpy(staging.data() + getRecordSlotOffset(ctx.layout, recordIndex),
                  &record, sizeof(record));
    }

    std::cout << "[async-loop] iteration " << (iteration + 1) << "/"
              << iterations << ": async copy host->device\n";
    checkAcl(aclrtMemcpyAsync(ctx.deviceCtrlPtr, staging.size(), staging.data(),
                              staging.size(), ACL_MEMCPY_HOST_TO_DEVICE,
                              runtime.stream()),
             "aclrtMemcpyAsync(staging H2D)");

    std::cout << "[async-loop] iteration " << (iteration + 1) << "/"
              << iterations << ": submit async export device->host\n";
    engine->asyncExport(ctx);

    // Simulate lightweight CPU-side overlap while D2H is in flight.
    volatile uint64_t overlapAccumulator = 0;
    for (int i = 0; i < 256; ++i) {
      overlapAccumulator += static_cast<uint64_t>(iteration + i);
    }
    (void)overlapAccumulator;

    std::cout << "[async-loop] iteration " << (iteration + 1) << "/"
              << iterations << ": wait for host-side export\n";
    engine->waitAsyncExport(ctx);

    const auto *hostBytes =
        reinterpret_cast<const uint8_t *>(ctx.hostBufferPtr);
    const auto *exportedHeader =
        reinterpret_cast<const RingBufferHeader *>(hostBytes);
    if (exportedHeader->writeIdx !=
        static_cast<uint32_t>(recordsPerIteration)) {
      std::fprintf(stderr,
                   "unexpected writeIdx at iteration %d: got=%u expected=%d\n",
                   iteration, exportedHeader->writeIdx, recordsPerIteration);
      engine->release(ctx);
      return 2;
    }

    for (int recordIndex = 0; recordIndex < recordsPerIteration;
         ++recordIndex) {
      const auto *record = reinterpret_cast<const SummaryRecord *>(
          hostBytes + getRecordSlotOffset(ctx.layout, recordIndex));
      observedOpIdSum += record->header.opId;
      observedValueSum += record->resultData.u64Val;
    }

    std::cout << "[async-loop] iteration " << (iteration + 1) << "/"
              << iterations
              << ": host processed exported buffer, partial_value_sum="
              << observedValueSum << "\n";
  }
  const auto end = std::chrono::steady_clock::now();

  std::cout << "[async-loop] step 5/5: releasing buffers and validating sums\n";
  engine->release(ctx);

  if (observedOpIdSum != expectedOpIdSum ||
      observedValueSum != expectedValueSum) {
    std::fprintf(stderr,
                 "aggregation mismatch: "
                 "observedOpIdSum=%llu expectedOpIdSum=%llu "
                 "observedValueSum=%llu expectedValueSum=%llu\n",
                 static_cast<unsigned long long>(observedOpIdSum),
                 static_cast<unsigned long long>(expectedOpIdSum),
                 static_cast<unsigned long long>(observedValueSum),
                 static_cast<unsigned long long>(expectedValueSum));
    return 3;
  }

  const auto elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "FlagTree debugger async export loop passed\n";
  std::cout << "device=" << deviceId << "\n";
  std::cout << "iterations=" << iterations << "\n";
  std::cout << "records_per_iteration=" << recordsPerIteration << "\n";
  std::cout << "observed_value_sum=" << observedValueSum << "\n";
  std::cout << "observed_op_id_sum=" << observedOpIdSum << "\n";
  std::cout << "elapsed_ms=" << elapsedMs.count() << "\n";
  return 0;
}
#endif

} // namespace

int main() {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
  return runCannAsyncExportLoopTest();
#else
  std::fprintf(stderr,
               "FlagTree debugger async export loop test requires CANN runtime "
               "support in this build\n");
  return 200;
#endif
}
