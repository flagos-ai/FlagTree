#pragma once

#include "Debugger/Metadata/TrackedOpTable.h"
#include "Debugger/Runtime/TransferEngine.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace flagtree {
namespace debugger {

// Module A owns the Python-facing option plumbing and the runtime launch
// bridge. The goal of this header is to freeze the handoff points between:
// - Python frontend / launcher
// - B module metadata output
// - F module transfer engine
struct DebugFrontendOptions {
  bool enabled = false;
  RecordLevel recordLevel = RecordLevel::LEVEL_SUMMARY;
  ExportMode exportMode = ExportMode::POST_KERNEL_EXPORT;
  uint32_t recordCapacity = 1024;
  bool captureMemoryEvents = true;
  bool captureFullValues = false;
};

struct DebugCompileRequest {
  std::string kernelName;
  std::string backendName;
  std::string targetName;
  DebugFrontendOptions options;
};

struct DebugKernelArtifacts {
  DebugFrontendOptions options;
  DebugBufferPlan bufferPlan;
  KernelDebugMetadata metadata;
  std::string metadataJson;
  bool hiddenArgEnabled = false;
};

struct DebugLaunchRequest {
  BufferMeta bufferMeta{};
  DebugRuntimeMetadata runtimeMetadata{};
  DebugLaunchContext launchContext{};
  uint64_t hiddenArgValue = 0;
};

// Preferred launch-time handoff for real launcher paths. The bridge derives
// transfer options from BufferMeta/backend identity, creates the engine, and
// keeps it alive across prepare/export/release.
struct PreparedDebugLaunch {
  TransferEngineOptions transferOptions{};
  std::unique_ptr<TransferEngine> transferEngine;
  DebugLaunchRequest request{};
};

class FrontendBridge {
public:
  virtual ~FrontendBridge() = default;

  virtual DebugCompileRequest
  normalizeCompileRequest(const DebugCompileRequest &request) = 0;

  virtual void attachKernelMetadata(const DebugCompileRequest &request,
                                    DebugKernelArtifacts &artifacts,
                                    const KernelDebugMetadata &metadata) = 0;

  // `artifacts.bufferPlan` is the frozen A->F launch-time buffer contract.
  // Module C may update payload-related fields later if instrumentation decides
  // that extra payload space is required.
  virtual DebugLaunchRequest
  prepareLaunch(const DebugKernelArtifacts &artifacts, const BufferMeta &meta,
                const DebugRuntimeMetadata &runtimeMetadata,
                TransferEngine &transferEngine) = 0;

  virtual PreparedDebugLaunch
  prepareOwnedLaunch(const DebugKernelArtifacts &artifacts,
                     const BufferMeta &meta,
                     const DebugRuntimeMetadata &runtimeMetadata,
                     uint64_t streamHandle = 0) = 0;
};

std::unique_ptr<FrontendBridge> createFrontendBridge();

} // namespace debugger
} // namespace flagtree
} // namespace mlir
