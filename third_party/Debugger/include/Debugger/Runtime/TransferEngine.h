#pragma once

#include "Debugger/Common/Protocol.h"
#include "Debugger/Runtime/BufferLayout.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {

struct DebugBufferPlan {
  uint32_t recordCapacity = 1024;
  uint32_t recordSize = kDefaultRecordSize;
  size_t payloadBytes = 0;
  ExportMode exportMode = ExportMode::POST_KERNEL_EXPORT;
};

struct BufferRegistrationInfo {
  uint32_t bufferId = 0;
  std::string bufferName;
  uint64_t baseAddress = 0;
  uint64_t sizeBytes = 0;
  uint32_t alignment = 0;
};

struct LaunchTensorInfo {
  uint32_t argumentIndex = 0;
  std::string logicalName;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string layout;
  uint32_t bufferId = 0;
  uint64_t baseAddress = 0;
  uint64_t sizeBytes = 0;
};

struct DebugRecordPlanEntry {
  uint32_t recordIndex = 0;
  uint32_t opId = 0;
  uint32_t scopeId = kInvalidScopeId;
  RecordKind recordKind = RecordKind::SUMMARY;
  CollectorKind collectorKind = CollectorKind::ELEMENT_COUNT;
  ResultType resultType = ResultType::U64;
  MemoryEventKind eventKind = MemoryEventKind::LAST_ALIGNED_ADDR;
};

struct FullDumpArtifactInfo {
  uint32_t opId = 0;
  uint64_t logicalInstanceId = 0;
  uint32_t payloadOffset = 0;
  uint32_t payloadLength = 0;
  std::string kind;
  std::string path;
};

struct DebugRuntimeMetadata {
  std::vector<BufferRegistrationInfo> buffers;
  std::vector<LaunchTensorInfo> tensors;
  std::string recordLayout;
  std::vector<DebugRecordPlanEntry> recordPlan;
  std::vector<FullDumpArtifactInfo> fullDumpArtifacts;
  bool hasLaunchGrid = false;
  uint32_t gridX = 1;
  uint32_t gridY = 1;
  uint32_t gridZ = 1;
  uint32_t recordsPerInstance = 0;
};

enum class TransferDriverKind : uint16_t {
  HOST = 1,
  CANN = 2,
};

TransferDriverKind resolveTransferDriverKind(BackendKind backendKind);

struct TransferEngineOptions {
  TransferDriverKind driverKind = TransferDriverKind::HOST;
  uint64_t streamHandle = 0;
};

TransferEngineOptions makeTransferEngineOptions(BackendKind backendKind,
                                                uint64_t streamHandle = 0);

// Module F runtime handoff.
// A prepares launches through this interface, C receives the resulting hidden
// argument inside the kernel as `__debug_ctrl_ptr`, and D consumes the bytes
// exported by `syncExport` / `asyncExport`.
struct DebugLaunchContext {
  BufferMeta meta{};
  DebugBufferPlan bufferPlan{};
  DebugRuntimeMetadata runtimeMetadata{};
  void *deviceCtrlPtr = nullptr;
  void *hostBufferPtr = nullptr;
  size_t bufferSize = 0;
  uint32_t recordCapacity = 0;
  uint64_t streamHandle = 0;
  BufferLayout layout;
};

struct DebugExportedRun {
  BufferMeta meta{};
  DebugRuntimeMetadata runtimeMetadata{};
  std::vector<uint8_t> rawBuffer;
};

class TransferEngine {
public:
  virtual ~TransferEngine() = default;

  virtual DebugLaunchContext
  prepare(const BufferMeta &meta, const DebugBufferPlan &plan,
          const DebugRuntimeMetadata &runtimeMetadata) = 0;
  virtual uint64_t hiddenArg(const DebugLaunchContext &ctx) = 0;
  virtual void initHeader(const DebugLaunchContext &ctx) = 0;
  virtual DebugExportedRun syncExport(const DebugLaunchContext &ctx) = 0;
  virtual void asyncExport(const DebugLaunchContext &ctx) = 0;
  virtual void waitAsyncExport(const DebugLaunchContext &ctx) = 0;
  virtual void release(DebugLaunchContext &ctx) = 0;
};

std::unique_ptr<TransferEngine>
createTransferEngine(const TransferEngineOptions &options = {});

std::unique_ptr<TransferEngine> createTransferEngine(BackendKind backendKind,
                                                     uint64_t streamHandle = 0);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
