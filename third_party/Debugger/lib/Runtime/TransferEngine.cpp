#include "Debugger/Runtime/TransferEngine.h"
#include "BackendAdapter.h"

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

const char *getDriverKindName(TransferDriverKind driverKind) {
  switch (driverKind) {
  case TransferDriverKind::HOST:
    return "host";
  case TransferDriverKind::CANN:
    return "cann";
  }
  return "unknown";
}

uint64_t resolveStreamHandle(const DebugLaunchContext &ctx,
                             const TransferEngineOptions &options) {
  return ctx.streamHandle != 0 ? ctx.streamHandle : options.streamHandle;
}

uint64_t saturatingMul(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs * rhs;
}

uint32_t saturatingU32(uint64_t value) {
  return static_cast<uint32_t>(
      std::min<uint64_t>(value, std::numeric_limits<uint32_t>::max()));
}

[[noreturn]] void failRuntime(const std::string &message) {
  std::fprintf(stderr, "FlagTree debugger runtime fatal error: %s\n",
               message.c_str());
  std::abort();
}

[[noreturn]] void
throwUnavailableBackend(const RuntimeBackendAdapter &adapter) {
  failRuntime(std::string("runtime backend adapter '") + adapter.name() +
              "' is not available in this build");
}

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
[[noreturn]] void throwAclError(const char *call, int errorCode) {
  std::string message =
      std::string(call) + " failed with aclError=" + std::to_string(errorCode);
  if (const char *recent = aclGetRecentErrMsg(); recent && recent[0] != '\0') {
    message += ", recent_err=\"";
    message += recent;
    message += "\"";
  }
  failRuntime(message);
}

void checkAcl(aclError error, const char *call) {
  if (error != ACL_SUCCESS) {
    throwAclError(call, error);
  }
}

aclrtStream toAclrtStream(uint64_t streamHandle) {
  return reinterpret_cast<aclrtStream>(streamHandle);
}
#endif

class HostRuntimeBackendAdapter final : public RuntimeBackendAdapter {
public:
  TransferDriverKind driverKind() const override {
    return TransferDriverKind::HOST;
  }

  const char *name() const override { return "host"; }

  bool isAvailable() const override { return true; }

  void *allocateDevice(size_t bytes) override { return std::malloc(bytes); }

  void freeDevice(void *ptr) override { std::free(ptr); }

  void *allocateHost(size_t bytes) override { return std::malloc(bytes); }

  void freeHost(void *ptr) override { std::free(ptr); }

  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t streamHandle) override {
    (void)streamHandle;
    if (!ptr || bytes == 0) {
      return;
    }
    std::memset(ptr, value, bytes);
  }

  void copyHostToDevice(void *deviceDst, const void *hostSrc, size_t bytes,
                        uint64_t streamHandle) override {
    (void)streamHandle;
    if (!deviceDst || !hostSrc || bytes == 0) {
      return;
    }
    std::memcpy(deviceDst, hostSrc, bytes);
  }

  void copyDeviceToHost(void *hostDst, const void *deviceSrc, size_t bytes,
                        uint64_t streamHandle) override {
    (void)streamHandle;
    if (!hostDst || !deviceSrc || bytes == 0) {
      return;
    }
    std::memcpy(hostDst, deviceSrc, bytes);
  }

  void synchronize(uint64_t streamHandle) override { (void)streamHandle; }
};

class CannRuntimeBackendAdapter final : public RuntimeBackendAdapter {
public:
  TransferDriverKind driverKind() const override {
    return TransferDriverKind::CANN;
  }

  const char *name() const override { return "cann"; }

  bool isAvailable() const override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    return true;
#else
    return false;
#endif
  }

  void *allocateDevice(size_t bytes) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    void *ptr = nullptr;
    checkAcl(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc");
    return ptr;
#else
    (void)bytes;
    throwUnavailableBackend(*this);
#endif
  }

  void freeDevice(void *ptr) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!ptr) {
      return;
    }
    checkAcl(aclrtFree(ptr), "aclrtFree");
#else
    (void)ptr;
    throwUnavailableBackend(*this);
#endif
  }

  void *allocateHost(size_t bytes) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    void *ptr = nullptr;
    checkAcl(aclrtMallocHost(&ptr, bytes), "aclrtMallocHost");
    return ptr;
#else
    (void)bytes;
    throwUnavailableBackend(*this);
#endif
  }

  void freeHost(void *ptr) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!ptr) {
      return;
    }
    checkAcl(aclrtFreeHost(ptr), "aclrtFreeHost");
#else
    (void)ptr;
    throwUnavailableBackend(*this);
#endif
  }

  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!ptr || bytes == 0) {
      return;
    }
    if (streamHandle != 0) {
      checkAcl(aclrtMemsetAsync(ptr, bytes, value, bytes,
                                toAclrtStream(streamHandle)),
               "aclrtMemsetAsync");
      return;
    }
    checkAcl(aclrtMemset(ptr, bytes, value, bytes), "aclrtMemset");
#else
    (void)ptr;
    (void)value;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void copyHostToDevice(void *deviceDst, const void *hostSrc, size_t bytes,
                        uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!deviceDst || !hostSrc || bytes == 0) {
      return;
    }
    if (streamHandle != 0) {
      checkAcl(aclrtMemcpyAsync(deviceDst, bytes, hostSrc, bytes,
                                ACL_MEMCPY_HOST_TO_DEVICE,
                                toAclrtStream(streamHandle)),
               "aclrtMemcpyAsync(H2D)");
      return;
    }
    checkAcl(aclrtMemcpy(deviceDst, bytes, hostSrc, bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(H2D)");
#else
    (void)deviceDst;
    (void)hostSrc;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void copyDeviceToHost(void *hostDst, const void *deviceSrc, size_t bytes,
                        uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!hostDst || !deviceSrc || bytes == 0) {
      return;
    }
    if (streamHandle != 0) {
      checkAcl(aclrtMemcpyAsync(hostDst, bytes, deviceSrc, bytes,
                                ACL_MEMCPY_DEVICE_TO_HOST,
                                toAclrtStream(streamHandle)),
               "aclrtMemcpyAsync(D2H)");
      return;
    }
    checkAcl(aclrtMemcpy(hostDst, bytes, deviceSrc, bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(D2H)");
#else
    (void)hostDst;
    (void)deviceSrc;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void synchronize(uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (streamHandle == 0) {
      return;
    }
    checkAcl(aclrtSynchronizeStream(toAclrtStream(streamHandle)),
             "aclrtSynchronizeStream");
#else
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }
};

struct BackendTransferAllocation {
  void *deviceBuffer = nullptr;
  void *hostBuffer = nullptr;
  size_t bytes = 0;
  bool asyncExportPending = false;
  bool asyncCopySubmitted = false;
};

void synthesizeExportHeader(const DebugLaunchContext &ctx,
                            BackendTransferAllocation &allocation) {
  if (!ctx.runtimeMetadata.hasLaunchGrid ||
      ctx.runtimeMetadata.recordsPerInstance == 0 || !allocation.hostBuffer ||
      allocation.bytes < sizeof(RingBufferHeader)) {
    return;
  }

  auto *header = reinterpret_cast<RingBufferHeader *>(allocation.hostBuffer);
  uint64_t totalSlots =
      saturatingMul(ctx.runtimeMetadata.gridX, ctx.runtimeMetadata.gridY);
  totalSlots = saturatingMul(totalSlots, ctx.runtimeMetadata.gridZ);
  totalSlots =
      saturatingMul(totalSlots, ctx.runtimeMetadata.recordsPerInstance);
  const uint64_t capacity = header->capacity;
  const uint64_t overflow = totalSlots > capacity ? totalSlots - capacity : 0;

  header->writeIdx = saturatingU32(totalSlots);
  header->overflowCount = saturatingU32(overflow);
  if (overflow != 0)
    header->flags |= RB_FLAG_OVERFLOW;
  else
    header->flags &= ~static_cast<uint32_t>(RB_FLAG_OVERFLOW);
}

class RealTransferEngine final : public TransferEngine {
public:
  explicit RealTransferEngine(const TransferEngineOptions &options)
      : options_(options), adapter_(createRuntimeBackendAdapter(options)) {}

  DebugLaunchContext
  prepare(const BufferMeta &meta, const DebugBufferPlan &plan,
          const DebugRuntimeMetadata &runtimeMetadata) override {
    ensureAdapterReady(meta);

    DebugLaunchContext ctx;
    ctx.meta = meta;
    ctx.bufferPlan = plan;
    ctx.runtimeMetadata = runtimeMetadata;
    ctx.recordCapacity = plan.recordCapacity;
    ctx.streamHandle = options_.streamHandle;
    ctx.layout = computeBufferLayout(plan.recordCapacity, plan.recordSize,
                                     plan.payloadBytes);
    ctx.bufferSize = ctx.layout.totalBytes;

    auto allocation = std::make_unique<BackendTransferAllocation>();
    allocation->bytes = ctx.bufferSize;
    allocation->deviceBuffer = adapter_->allocateDevice(ctx.bufferSize);
    if (!allocation->deviceBuffer) {
      failRuntime("failed to allocate runtime device buffer");
    }

    allocation->hostBuffer = adapter_->allocateHost(ctx.bufferSize);
    if (!allocation->hostBuffer) {
      adapter_->freeDevice(allocation->deviceBuffer);
      failRuntime("failed to allocate runtime host buffer");
    }

    std::memset(allocation->hostBuffer, 0, ctx.bufferSize);
    ctx.deviceCtrlPtr = allocation->deviceBuffer;
    ctx.hostBufferPtr = allocation->hostBuffer;

    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ctx.deviceCtrlPtr] = std::move(allocation);
    return ctx;
  }

  uint64_t hiddenArg(const DebugLaunchContext &ctx) override {
    return reinterpret_cast<uint64_t>(ctx.deviceCtrlPtr);
  }

  void initHeader(const DebugLaunchContext &ctx) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation) {
      return;
    }

    std::memset(allocation->hostBuffer, 0, allocation->bytes);
    // Header initialization is part of launch setup, not payload capture.  Keep
    // it synchronous so the instrumented kernel never observes stale allocation
    // contents when different CANN launch APIs are mixed in the same process.
    adapter_->memsetDevice(allocation->deviceBuffer, 0, allocation->bytes, 0);

    RingBufferHeader header{};
    header.writeIdx = 0;
    header.capacity = ctx.recordCapacity;
    header.overflowCount = 0;
    header.flags = RB_FLAG_NONE;
    header.recordSize = ctx.bufferPlan.recordSize;
    header.payloadOffset = static_cast<uint32_t>(ctx.layout.payloadOffset);
    header.reserved0 = 0;
    header.reserved1 = 0;

    std::memcpy(allocation->hostBuffer, &header, sizeof(header));
    adapter_->copyHostToDevice(allocation->deviceBuffer, &header,
                               sizeof(header), 0);
    allocation->asyncExportPending = false;
    allocation->asyncCopySubmitted = false;
  }

  DebugExportedRun syncExport(const DebugLaunchContext &ctx) override {
    DebugExportedRun run;
    run.meta = ctx.meta;
    run.runtimeMetadata = ctx.runtimeMetadata;

    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation) {
      return run;
    }

    copyDeviceToHostLocked(ctx, *allocation);
    auto *begin = reinterpret_cast<const uint8_t *>(allocation->hostBuffer);
    run.rawBuffer.assign(begin, begin + allocation->bytes);
    return run;
  }

  void asyncExport(const DebugLaunchContext &ctx) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation) {
      return;
    }
    const uint64_t streamHandle = resolveStreamHandle(ctx, options_);
    allocation->asyncCopySubmitted = false;
    if (options_.driverKind == TransferDriverKind::CANN && streamHandle != 0) {
      adapter_->copyDeviceToHost(allocation->hostBuffer,
                                 allocation->deviceBuffer, allocation->bytes,
                                 streamHandle);
      allocation->asyncCopySubmitted = true;
    }
    allocation->asyncExportPending = true;
  }

  void waitAsyncExport(const DebugLaunchContext &ctx) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation || !allocation->asyncExportPending) {
      return;
    }
    if (!allocation->asyncCopySubmitted) {
      copyDeviceToHostLocked(ctx, *allocation);
      return;
    }
    finalizeAsyncCopyLocked(ctx, *allocation);
  }

  void release(DebugLaunchContext &ctx) override {
    std::unique_ptr<BackendTransferAllocation> allocation;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = allocations_.find(ctx.deviceCtrlPtr);
      if (it != allocations_.end()) {
        allocation = std::move(it->second);
        allocations_.erase(it);
      }
    }

    if (allocation) {
      adapter_->freeHost(allocation->hostBuffer);
      adapter_->freeDevice(allocation->deviceBuffer);
    }

    ctx.deviceCtrlPtr = nullptr;
    ctx.hostBufferPtr = nullptr;
    ctx.bufferSize = 0;
    ctx.recordCapacity = 0;
    ctx.streamHandle = 0;
    ctx.layout = {};
  }

private:
  void ensureAdapterReady(const BufferMeta &meta) const {
    if (!adapter_ || !adapter_->isAvailable()) {
      failRuntime(std::string("transfer engine driver '") +
                  getDriverKindName(options_.driverKind) +
                  "' is not available");
    }
    if (options_.driverKind == TransferDriverKind::CANN &&
        meta.backendKind != BackendKind::CANN) {
      failRuntime("cann transfer driver requires BufferMeta.backendKind == "
                  "CANN");
    }
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (options_.driverKind == TransferDriverKind::CANN) {
      aclrtContext context = nullptr;
      checkAcl(aclrtGetCurrentContext(&context), "aclrtGetCurrentContext");
      if (context == nullptr) {
        failRuntime("cann transfer driver requires a current ACL context; "
                    "caller must set device/context before prepare()");
      }
    }
#endif
  }

  BackendTransferAllocation *
  findAllocationLocked(const DebugLaunchContext &ctx) {
    auto it = allocations_.find(ctx.deviceCtrlPtr);
    if (it == allocations_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  void copyDeviceToHostLocked(const DebugLaunchContext &ctx,
                              BackendTransferAllocation &allocation) {
    const uint64_t streamHandle = resolveStreamHandle(ctx, options_);
    adapter_->copyDeviceToHost(allocation.hostBuffer, allocation.deviceBuffer,
                               allocation.bytes, streamHandle);
    allocation.asyncCopySubmitted = streamHandle != 0;
    finalizeAsyncCopyLocked(ctx, allocation);
  }

  void finalizeAsyncCopyLocked(const DebugLaunchContext &ctx,
                               BackendTransferAllocation &allocation) {
    const uint64_t streamHandle = resolveStreamHandle(ctx, options_);
    adapter_->synchronize(streamHandle);
    synthesizeExportHeader(ctx, allocation);
    allocation.asyncCopySubmitted = false;
    allocation.asyncExportPending = false;
  }

  TransferEngineOptions options_;
  std::unique_ptr<RuntimeBackendAdapter> adapter_;
  std::mutex mutex_;
  std::unordered_map<void *, std::unique_ptr<BackendTransferAllocation>>
      allocations_;
};

} // namespace

TransferDriverKind resolveTransferDriverKind(BackendKind backendKind) {
  switch (backendKind) {
  case BackendKind::CANN:
    return TransferDriverKind::CANN;
  case BackendKind::UNKNOWN:
  case BackendKind::CUDA:
  case BackendKind::HIP:
  case BackendKind::MUSA:
    return TransferDriverKind::HOST;
  }
  return TransferDriverKind::HOST;
}

TransferEngineOptions makeTransferEngineOptions(BackendKind backendKind,
                                                uint64_t streamHandle) {
  TransferEngineOptions options;
  options.driverKind = resolveTransferDriverKind(backendKind);
  options.streamHandle = streamHandle;
  return options;
}

std::unique_ptr<RuntimeBackendAdapter>
createRuntimeBackendAdapter(const TransferEngineOptions &options) {
  switch (options.driverKind) {
  case TransferDriverKind::HOST:
    return std::make_unique<HostRuntimeBackendAdapter>();
  case TransferDriverKind::CANN:
    return std::make_unique<CannRuntimeBackendAdapter>();
  }
  failRuntime(std::string("unsupported transfer driver '") +
              getDriverKindName(options.driverKind) + "'");
}

std::unique_ptr<TransferEngine>
createTransferEngine(const TransferEngineOptions &options) {
  return std::make_unique<RealTransferEngine>(options);
}

std::unique_ptr<TransferEngine> createTransferEngine(BackendKind backendKind,
                                                     uint64_t streamHandle) {
  return createTransferEngine(
      makeTransferEngineOptions(backendKind, streamHandle));
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
