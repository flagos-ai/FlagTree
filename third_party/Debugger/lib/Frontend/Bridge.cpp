#include "Debugger/Frontend/Bridge.h"

#include <cctype>
#include <string>
#include <string_view>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

BackendKind resolveBackendKindName(std::string_view backendName) {
  std::string lowered;
  lowered.reserve(backendName.size());
  for (char ch : backendName) {
    lowered.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }

  if (lowered == "cuda" || lowered == "nvidia") {
    return BackendKind::CUDA;
  }
  if (lowered == "hip" || lowered == "rocm" || lowered == "amd") {
    return BackendKind::HIP;
  }
  if (lowered == "musa") {
    return BackendKind::MUSA;
  }
  if (lowered == "cann" || lowered == "ascend") {
    return BackendKind::CANN;
  }
  return BackendKind::UNKNOWN;
}

DebugCompileRequest normalizeOptions(const DebugCompileRequest &request) {
  DebugCompileRequest out = request;
  if (!out.options.enabled) {
    return out;
  }
  if (out.options.recordCapacity < 64) {
    out.options.recordCapacity = 64;
  }
  if (out.options.recordLevel != RecordLevel::LEVEL_SUMMARY &&
      out.options.recordLevel != RecordLevel::LEVEL_TENSOR_FULL) {
    out.options.recordLevel = RecordLevel::LEVEL_SUMMARY;
  }
  if (out.options.exportMode != ExportMode::POST_KERNEL_EXPORT &&
      out.options.exportMode != ExportMode::STREAMING_EXPORT) {
    out.options.exportMode = ExportMode::POST_KERNEL_EXPORT;
  }
  return out;
}

BufferMeta normalizeBufferMeta(const DebugKernelArtifacts &artifacts,
                               const BufferMeta &meta) {
  BufferMeta normalized = meta;
  if (normalized.kernelId == kInvalidKernelId &&
      artifacts.metadata.debugKernelId != 0) {
    normalized.kernelId = artifacts.metadata.debugKernelId;
  }
  if (normalized.protocolVer == 0) {
    normalized.protocolVer = kProtocolVersion;
  }
  if (static_cast<uint16_t>(normalized.recordLevel) == 0) {
    normalized.recordLevel = artifacts.options.recordLevel;
  }
  if (static_cast<uint16_t>(normalized.exportMode) == 0) {
    normalized.exportMode = artifacts.bufferPlan.exportMode;
  }
  if (normalized.backendKind == BackendKind::UNKNOWN) {
    normalized.backendKind =
        resolveBackendKindName(artifacts.metadata.backendName);
  }
  return normalized;
}

class DefaultFrontendBridge final : public FrontendBridge {
public:
  DebugCompileRequest
  normalizeCompileRequest(const DebugCompileRequest &request) override {
    return normalizeOptions(request);
  }

  void attachKernelMetadata(const DebugCompileRequest &request,
                            DebugKernelArtifacts &artifacts,
                            const KernelDebugMetadata &metadata) override {
    artifacts.options = request.options;
    artifacts.bufferPlan.recordCapacity = request.options.recordCapacity;
    artifacts.bufferPlan.recordSize = kDefaultRecordSize;
    artifacts.bufferPlan.payloadBytes = 0;
    if (request.options.captureFullValues) {
      artifacts.bufferPlan.payloadBytes = 4096;
    }
    artifacts.bufferPlan.exportMode = request.options.exportMode;
    artifacts.metadata = metadata;
    if (!request.kernelName.empty()) {
      artifacts.metadata.kernelName = request.kernelName;
    }
    if (!request.backendName.empty()) {
      artifacts.metadata.backendName = request.backendName;
    }
    if (!request.targetName.empty()) {
      artifacts.metadata.targetName = request.targetName;
    }
    artifacts.metadataJson.clear();
    artifacts.hiddenArgEnabled = true;
  }

  DebugLaunchRequest prepareLaunch(const DebugKernelArtifacts &artifacts,
                                   const BufferMeta &meta,
                                   const DebugRuntimeMetadata &runtimeMetadata,
                                   TransferEngine &transferEngine) override {
    BufferMeta normalizedMeta = normalizeBufferMeta(artifacts, meta);
    DebugLaunchRequest request;
    request.bufferMeta = normalizedMeta;
    request.runtimeMetadata = runtimeMetadata;
    if (!artifacts.hiddenArgEnabled) {
      request.hiddenArgValue = 0;
      return request;
    }
    request.launchContext = transferEngine.prepare(
        normalizedMeta, artifacts.bufferPlan, runtimeMetadata);
    transferEngine.initHeader(request.launchContext);
    request.hiddenArgValue = transferEngine.hiddenArg(request.launchContext);
    return request;
  }

  PreparedDebugLaunch
  prepareOwnedLaunch(const DebugKernelArtifacts &artifacts,
                     const BufferMeta &meta,
                     const DebugRuntimeMetadata &runtimeMetadata,
                     uint64_t streamHandle) override {
    BufferMeta normalizedMeta = normalizeBufferMeta(artifacts, meta);

    PreparedDebugLaunch prepared;
    prepared.transferOptions =
        makeTransferEngineOptions(normalizedMeta.backendKind, streamHandle);
    prepared.transferEngine = createTransferEngine(prepared.transferOptions);
    prepared.request = prepareLaunch(artifacts, normalizedMeta, runtimeMetadata,
                                     *prepared.transferEngine);
    return prepared;
  }
};

} // namespace

std::unique_ptr<FrontendBridge> createFrontendBridge() {
  return std::make_unique<DefaultFrontendBridge>();
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
