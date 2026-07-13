#include "TestFixtures.h"

#include <gtest/gtest.h>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

TEST(FrontendBridgeTest, AttachKernelMetadataCopiesFrontendIdentity) {
  auto bridge = createFrontendBridge();

  DebugCompileRequest request =
      makeTestCompileRequest("cann_kernel", BackendKind::CANN);
  request.targetName = "ascend-910b";
  request.options.recordCapacity = 2048;
  request.options.exportMode = ExportMode::STREAMING_EXPORT;

  DebugKernelArtifacts artifacts;
  bridge->attachKernelMetadata(request, artifacts,
                               makeTestKernelDebugMetadata());

  EXPECT_EQ(artifacts.bufferPlan.recordCapacity, 2048u);
  EXPECT_EQ(artifacts.bufferPlan.recordSize, kDefaultRecordSize);
  EXPECT_EQ(artifacts.bufferPlan.exportMode, ExportMode::STREAMING_EXPORT);
  EXPECT_EQ(artifacts.metadata.kernelName, "cann_kernel");
  EXPECT_EQ(artifacts.metadata.backendName, "cann");
  EXPECT_EQ(artifacts.metadata.targetName, "ascend-910b");
  EXPECT_TRUE(artifacts.hiddenArgEnabled);
}

TEST(FrontendBridgeTest, PrepareLaunchNormalizesUnsetRuntimeFields) {
  auto bridge = createFrontendBridge();

  DebugCompileRequest request =
      makeTestCompileRequest("cuda_kernel", BackendKind::CUDA);
  request.options.exportMode = ExportMode::STREAMING_EXPORT;

  DebugKernelArtifacts artifacts;
  bridge->attachKernelMetadata(request, artifacts,
                               makeTestKernelDebugMetadata());

  BufferMeta meta{};
  meta.runId = 17;
  meta.deviceId = 3;

  auto engine = createTransferEngine(BackendKind::CUDA);
  DebugLaunchRequest launch = bridge->prepareLaunch(
      artifacts, meta, makeTestRuntimeMetadata(), *engine);

  EXPECT_EQ(launch.bufferMeta.runId, 17u);
  EXPECT_EQ(launch.bufferMeta.deviceId, 3u);
  EXPECT_EQ(launch.bufferMeta.kernelId, artifacts.metadata.debugKernelId);
  EXPECT_EQ(launch.bufferMeta.protocolVer, kProtocolVersion);
  EXPECT_EQ(launch.bufferMeta.recordLevel, RecordLevel::LEVEL_SUMMARY);
  EXPECT_EQ(launch.bufferMeta.exportMode, ExportMode::STREAMING_EXPORT);
  EXPECT_EQ(launch.bufferMeta.backendKind, BackendKind::CUDA);
  EXPECT_EQ(launch.hiddenArgValue,
            reinterpret_cast<uint64_t>(launch.launchContext.deviceCtrlPtr));

  engine->release(launch.launchContext);
}

TEST(FrontendBridgeTest, PrepareOwnedLaunchCreatesBackendAwareTransferEngine) {
  auto bridge = createFrontendBridge();

  DebugCompileRequest request =
      makeTestCompileRequest("cuda_kernel", BackendKind::CUDA);

  DebugKernelArtifacts artifacts;
  bridge->attachKernelMetadata(request, artifacts,
                               makeTestKernelDebugMetadata());

  BufferMeta meta = makeTestBufferMeta();
  meta.backendKind = BackendKind::UNKNOWN;
  meta.protocolVer = 0;
  meta.recordLevel = static_cast<RecordLevel>(0);
  meta.exportMode = static_cast<ExportMode>(0);

  PreparedDebugLaunch prepared = bridge->prepareOwnedLaunch(
      artifacts, meta, makeTestRuntimeMetadata(), 0x1234);

  ASSERT_TRUE(prepared.transferEngine != nullptr);
  EXPECT_EQ(prepared.transferOptions.driverKind, TransferDriverKind::HOST);
  EXPECT_EQ(prepared.transferOptions.streamHandle, 0x1234u);
  EXPECT_EQ(prepared.request.bufferMeta.backendKind, BackendKind::CUDA);
  EXPECT_EQ(prepared.request.bufferMeta.protocolVer, kProtocolVersion);
  EXPECT_EQ(prepared.request.bufferMeta.recordLevel,
            RecordLevel::LEVEL_SUMMARY);
  EXPECT_EQ(prepared.request.bufferMeta.exportMode,
            ExportMode::POST_KERNEL_EXPORT);
  EXPECT_EQ(prepared.request.launchContext.streamHandle, 0x1234u);
  EXPECT_EQ(
      prepared.request.hiddenArgValue,
      reinterpret_cast<uint64_t>(prepared.request.launchContext.deviceCtrlPtr));

  prepared.transferEngine->release(prepared.request.launchContext);
}

TEST(FrontendBridgeTest, PrepareLaunchCanDeriveCannBackendFromArtifacts) {
  auto bridge = createFrontendBridge();

  DebugCompileRequest request =
      makeTestCompileRequest("cann_kernel", BackendKind::CANN);

  DebugKernelArtifacts artifacts;
  bridge->attachKernelMetadata(request, artifacts,
                               makeTestKernelDebugMetadata());

  BufferMeta meta = makeTestBufferMeta();
  meta.backendKind = BackendKind::UNKNOWN;

  auto engine = createTransferEngine(BackendKind::CUDA);
  DebugLaunchRequest launch = bridge->prepareLaunch(
      artifacts, meta, makeTestRuntimeMetadata(), *engine);

  EXPECT_EQ(launch.bufferMeta.backendKind, BackendKind::CANN);

  engine->release(launch.launchContext);
}

} // namespace
} // namespace debugger
} // namespace flagtree
} // namespace mlir
