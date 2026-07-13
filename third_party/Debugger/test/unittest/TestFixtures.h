#pragma once

#include "Debugger/Frontend/Bridge.h"
#include "Debugger/Runtime/BufferLayout.h"
#include "Debugger/Runtime/TransferEngine.h"

#include <cstring>
#include <string>

namespace mlir {
namespace flagtree {
namespace debugger {

struct TestRunOptions {
  uint64_t runId = 1;
  uint32_t deviceId = 0;
  uint32_t kernelId = 1;
  RecordLevel recordLevel = RecordLevel::LEVEL_SUMMARY;
  ExportMode exportMode = ExportMode::POST_KERNEL_EXPORT;
  BackendKind backendKind = BackendKind::CUDA;
  uint32_t recordCapacity = 1024;
  uint32_t recordSize = kDefaultRecordSize;
  size_t payloadBytes = 0;
};

inline const char *testBackendName(BackendKind backendKind) {
  switch (backendKind) {
  case BackendKind::CUDA:
    return "cuda";
  case BackendKind::HIP:
    return "hip";
  case BackendKind::MUSA:
    return "musa";
  case BackendKind::CANN:
    return "cann";
  case BackendKind::UNKNOWN:
  default:
    return "unknown";
  }
}

inline DebugCompileRequest
makeTestCompileRequest(const std::string &kernelName = "test_kernel",
                       BackendKind backendKind = BackendKind::CUDA) {
  DebugCompileRequest request;
  request.kernelName = kernelName;
  request.backendName = testBackendName(backendKind);
  request.targetName = "unit-target";
  request.options.enabled = true;
  request.options.recordLevel = RecordLevel::LEVEL_SUMMARY;
  request.options.exportMode = ExportMode::POST_KERNEL_EXPORT;
  request.options.recordCapacity = 1024;
  request.options.captureMemoryEvents = true;
  request.options.captureFullValues = false;
  return request;
}

inline BufferMeta makeTestBufferMeta(const TestRunOptions &options = {}) {
  BufferMeta meta{};
  meta.runId = options.runId;
  meta.deviceId = options.deviceId;
  meta.kernelId = options.kernelId;
  meta.protocolVer = kProtocolVersion;
  meta.recordLevel = options.recordLevel;
  meta.exportMode = options.exportMode;
  meta.backendKind = options.backendKind;
  return meta;
}

inline DebugBufferPlan makeTestBufferPlan(const TestRunOptions &options = {}) {
  DebugBufferPlan plan;
  plan.recordCapacity = options.recordCapacity;
  plan.recordSize = options.recordSize;
  plan.payloadBytes = options.payloadBytes;
  plan.exportMode = options.exportMode;
  return plan;
}

inline DebugRuntimeMetadata makeTestRuntimeMetadata() {
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

inline DebugLaunchContext
makeTestLaunchContext(const TestRunOptions &options = {}) {
  DebugLaunchContext ctx;
  ctx.meta = makeTestBufferMeta(options);
  ctx.bufferPlan = makeTestBufferPlan(options);
  ctx.runtimeMetadata = makeTestRuntimeMetadata();
  ctx.recordCapacity = options.recordCapacity;
  ctx.layout = computeBufferLayout(options.recordCapacity, options.recordSize,
                                   options.payloadBytes);
  ctx.bufferSize = ctx.layout.totalBytes;
  return ctx;
}

inline DebugExportedRun
makeTestExportedRun(const TestRunOptions &options = {}) {
  DebugLaunchContext ctx = makeTestLaunchContext(options);
  DebugExportedRun run;
  run.meta = ctx.meta;
  run.runtimeMetadata = ctx.runtimeMetadata;
  run.rawBuffer.resize(ctx.layout.totalBytes, 0);

  RingBufferHeader header{};
  header.writeIdx = 0;
  header.capacity = ctx.bufferPlan.recordCapacity;
  header.overflowCount = 0;
  header.flags = RB_FLAG_NONE;
  header.recordSize = ctx.bufferPlan.recordSize;
  header.payloadOffset = static_cast<uint32_t>(ctx.layout.payloadOffset);
  std::memcpy(run.rawBuffer.data(), &header, sizeof(header));

  return run;
}

inline TrackedOpEntry makeTestTrackedOpEntry(uint32_t opId,
                                             uint32_t scopeId = 1,
                                             bool isMemoryOp = false) {
  TrackedOpEntry entry;
  entry.opId = opId;
  entry.scopeId = scopeId;
  entry.resultIndex = 0;
  entry.isMemoryOp = isMemoryOp;
  entry.opCategory = isMemoryOp ? "load" : "";
  entry.role = isMemoryOp ? "load" : "";
  entry.mlirOpName = isMemoryOp ? "tt.load" : "arith.addf";
  entry.sourceLoc = "unit_test.py:1";
  entry.tritonStatement = isMemoryOp ? "x = tl.load(ptr)" : "y = x + 1";
  entry.result.valueKind = "tensor";
  entry.result.dtype = "fp32";
  entry.result.elementDtype = "fp32";
  entry.result.shape = "[128]";
  entry.result.stride = "[1]";
  entry.result.layout = "blocked";
  entry.result.addrSpace = isMemoryOp ? "global" : "";
  entry.addrSpace = isMemoryOp ? "global" : "";
  entry.accessType = isMemoryOp ? "load" : "";
  entry.accessBytes = isMemoryOp ? 4 : 0;
  entry.alignmentRequired = isMemoryOp ? 4 : 0;
  return entry;
}

inline KernelDebugMetadata
makeTestKernelDebugMetadata(uint32_t scopeCount = 1,
                            uint32_t trackedOpCount = 4,
                            bool includeMemoryOp = true) {
  KernelDebugMetadata metadata;
  metadata.debugKernelId = 1;
  metadata.kernelName = "test_kernel";
  metadata.backendName = "cuda";
  metadata.targetName = "unit-target";
  metadata.scopeCount = scopeCount;
  metadata.trackedOpCount = trackedOpCount;
  for (uint32_t i = 0; i < trackedOpCount; ++i) {
    bool isMemoryOp = includeMemoryOp && i == 0;
    metadata.trackedOps.push_back(makeTestTrackedOpEntry(i + 1, 1, isMemoryOp));
  }
  return metadata;
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
