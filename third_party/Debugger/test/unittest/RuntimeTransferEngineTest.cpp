#include "Debugger/Runtime/BufferLayout.h"
#include "Debugger/Runtime/TransferEngine.h"
#include "TestFixtures.h"

#include <cstring>
#include <gtest/gtest.h>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

TransferEngineOptions makeHostTransferOptions() {
  return makeTransferEngineOptions(BackendKind::CUDA);
}

TEST(TransferDriverSelectionTest, ResolvesBackendKindsToDriverKinds) {
  EXPECT_EQ(resolveTransferDriverKind(BackendKind::CANN),
            TransferDriverKind::CANN);
  EXPECT_EQ(resolveTransferDriverKind(BackendKind::CUDA),
            TransferDriverKind::HOST);
  EXPECT_EQ(resolveTransferDriverKind(BackendKind::HIP),
            TransferDriverKind::HOST);
  EXPECT_EQ(resolveTransferDriverKind(BackendKind::MUSA),
            TransferDriverKind::HOST);
}

TEST(TransferDriverSelectionTest, BuildsTransferEngineOptionsFromBackendKind) {
  TransferEngineOptions options =
      makeTransferEngineOptions(BackendKind::CANN, 0x1234);

  EXPECT_EQ(options.driverKind, TransferDriverKind::CANN);
  EXPECT_EQ(options.streamHandle, 0x1234u);
}

TEST(BufferLayoutTest, ComputesOffsetsForRuntimeBuffer) {
  BufferLayout layout = computeBufferLayout(8, kDefaultRecordSize, 64);

  EXPECT_EQ(layout.headerBytes, sizeof(RingBufferHeader));
  EXPECT_EQ(layout.recordBytes, kDefaultRecordSize);
  EXPECT_EQ(layout.recordAreaBytes, 8u * kDefaultRecordSize);
  EXPECT_EQ(layout.payloadOffset,
            sizeof(RingBufferHeader) + 8u * kDefaultRecordSize);
  EXPECT_EQ(layout.totalBytes, layout.payloadOffset + 64u);
  EXPECT_EQ(getRecordSlotOffset(layout, 3),
            sizeof(RingBufferHeader) + 3u * kDefaultRecordSize);
}

TEST(TestFixturesTest, ProducesProtocolCompatibleExportBuffer) {
  TestRunOptions options;
  options.runId = 7;
  options.kernelId = 13;
  options.recordCapacity = 16;
  options.payloadBytes = 128;

  DebugExportedRun run = makeTestExportedRun(options);
  ASSERT_GE(run.rawBuffer.size(), sizeof(RingBufferHeader));

  const auto *header =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  EXPECT_EQ(run.meta.runId, 7u);
  EXPECT_EQ(run.meta.kernelId, 13u);
  EXPECT_EQ(header->writeIdx, 0u);
  EXPECT_EQ(header->capacity, 16u);
  EXPECT_EQ(header->overflowCount, 0u);
  EXPECT_EQ(header->flags, RB_FLAG_NONE);
  EXPECT_EQ(header->recordSize, kDefaultRecordSize);
  EXPECT_EQ(header->payloadOffset,
            sizeof(RingBufferHeader) + 16u * kDefaultRecordSize);
  EXPECT_EQ(run.runtimeMetadata.buffers.size(), 1u);
  EXPECT_EQ(run.runtimeMetadata.tensors.size(), 1u);
}

TEST(TransferEngineTest, PrepareAllocatesCpuBackedBuffers) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  BufferMeta meta = makeTestBufferMeta();
  DebugBufferPlan plan = makeTestBufferPlan();
  DebugRuntimeMetadata runtimeMetadata = makeTestRuntimeMetadata();

  DebugLaunchContext ctx = engine->prepare(meta, plan, runtimeMetadata);

  EXPECT_NE(ctx.deviceCtrlPtr, nullptr);
  EXPECT_NE(ctx.hostBufferPtr, nullptr);
  EXPECT_EQ(ctx.bufferSize, ctx.layout.totalBytes);
  EXPECT_EQ(ctx.recordCapacity, plan.recordCapacity);
  EXPECT_EQ(engine->hiddenArg(ctx),
            reinterpret_cast<uint64_t>(ctx.deviceCtrlPtr));

  engine->release(ctx);
}

TEST(TransferEngineTest, InitHeaderWritesProtocolFields) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  TestRunOptions options;
  options.recordCapacity = 16;
  options.payloadBytes = 96;

  DebugLaunchContext ctx =
      engine->prepare(makeTestBufferMeta(options), makeTestBufferPlan(options),
                      makeTestRuntimeMetadata());
  engine->initHeader(ctx);

  const auto *header =
      reinterpret_cast<const RingBufferHeader *>(ctx.deviceCtrlPtr);
  ASSERT_NE(header, nullptr);
  EXPECT_EQ(header->writeIdx, 0u);
  EXPECT_EQ(header->capacity, 16u);
  EXPECT_EQ(header->overflowCount, 0u);
  EXPECT_EQ(header->flags, RB_FLAG_NONE);
  EXPECT_EQ(header->recordSize, kDefaultRecordSize);
  EXPECT_EQ(header->payloadOffset,
            sizeof(RingBufferHeader) + 16u * kDefaultRecordSize);

  engine->release(ctx);
}

TEST(TransferEngineTest, SyncExportCopiesDeviceRecordsToHostBuffer) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  DebugLaunchContext ctx = engine->prepare(
      makeTestBufferMeta(), makeTestBufferPlan(), makeTestRuntimeMetadata());
  engine->initHeader(ctx);

  auto *header = reinterpret_cast<RingBufferHeader *>(ctx.deviceCtrlPtr);
  SummaryRecord record{};
  record.header.recordKind = RecordKind::SUMMARY;
  record.header.opId = 21;
  record.header.logicalInstanceId = 5;
  record.collectorKind = CollectorKind::MAX_FINITE;
  record.resultType = ResultType::F64;
  record.resultData.f64Val = 42.5;
  header->writeIdx = 1;
  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 0),
              &record, sizeof(record));

  DebugExportedRun run = engine->syncExport(ctx);

  ASSERT_GE(run.rawBuffer.size(),
            getRecordSlotOffset(ctx.layout, 0) + sizeof(SummaryRecord));
  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  EXPECT_EQ(exportedHeader->writeIdx, 1u);
  const auto *exportedRecord = reinterpret_cast<const SummaryRecord *>(
      run.rawBuffer.data() + getRecordSlotOffset(ctx.layout, 0));
  EXPECT_EQ(exportedRecord->header.opId, 21u);
  EXPECT_EQ(exportedRecord->header.logicalInstanceId, 5u);
  EXPECT_EQ(exportedRecord->collectorKind, CollectorKind::MAX_FINITE);
  EXPECT_DOUBLE_EQ(exportedRecord->resultData.f64Val, 42.5);

  engine->release(ctx);
}

TEST(TransferEngineTest, SyncExportSynthesizesHeaderFromLaunchGrid) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  TestRunOptions options;
  options.recordCapacity = 16;

  DebugRuntimeMetadata runtimeMetadata = makeTestRuntimeMetadata();
  runtimeMetadata.hasLaunchGrid = true;
  runtimeMetadata.gridX = 2;
  runtimeMetadata.gridY = 1;
  runtimeMetadata.gridZ = 1;
  runtimeMetadata.recordsPerInstance = 3;

  DebugLaunchContext ctx =
      engine->prepare(makeTestBufferMeta(options), makeTestBufferPlan(options),
                      runtimeMetadata);
  engine->initHeader(ctx);

  DebugExportedRun run = engine->syncExport(ctx);

  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  EXPECT_EQ(exportedHeader->writeIdx, 6u);
  EXPECT_EQ(exportedHeader->overflowCount, 0u);
  EXPECT_EQ(exportedHeader->flags & RB_FLAG_OVERFLOW, 0u);

  engine->release(ctx);
}

TEST(TransferEngineTest, SyncExportSynthesizesOverflowHeaderFromLaunchGrid) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  TestRunOptions options;
  options.recordCapacity = 16;

  DebugRuntimeMetadata runtimeMetadata = makeTestRuntimeMetadata();
  runtimeMetadata.hasLaunchGrid = true;
  runtimeMetadata.gridX = 2;
  runtimeMetadata.gridY = 3;
  runtimeMetadata.gridZ = 1;
  runtimeMetadata.recordsPerInstance = 4;

  DebugLaunchContext ctx =
      engine->prepare(makeTestBufferMeta(options), makeTestBufferPlan(options),
                      runtimeMetadata);
  engine->initHeader(ctx);

  DebugExportedRun run = engine->syncExport(ctx);

  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  EXPECT_EQ(exportedHeader->writeIdx, 24u);
  EXPECT_EQ(exportedHeader->overflowCount, 8u);
  EXPECT_NE(exportedHeader->flags & RB_FLAG_OVERFLOW, 0u);

  engine->release(ctx);
}

TEST(TransferEngineTest, AsyncExportCopiesOnWait) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  DebugLaunchContext ctx = engine->prepare(
      makeTestBufferMeta(), makeTestBufferPlan(), makeTestRuntimeMetadata());
  engine->initHeader(ctx);

  auto *header = reinterpret_cast<RingBufferHeader *>(ctx.deviceCtrlPtr);
  header->writeIdx = 3;

  auto *hostHeader =
      reinterpret_cast<const RingBufferHeader *>(ctx.hostBufferPtr);
  ASSERT_NE(hostHeader, nullptr);
  EXPECT_EQ(hostHeader->writeIdx, 0u);

  engine->asyncExport(ctx);
  EXPECT_EQ(hostHeader->writeIdx, 0u);

  engine->waitAsyncExport(ctx);
  EXPECT_EQ(hostHeader->writeIdx, 3u);

  engine->release(ctx);
}

TEST(TransferEngineTest, ReleaseClearsLaunchContextPointers) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  DebugLaunchContext ctx = engine->prepare(
      makeTestBufferMeta(), makeTestBufferPlan(), makeTestRuntimeMetadata());

  engine->release(ctx);

  EXPECT_EQ(ctx.deviceCtrlPtr, nullptr);
  EXPECT_EQ(ctx.hostBufferPtr, nullptr);
  EXPECT_EQ(ctx.bufferSize, 0u);
  EXPECT_EQ(ctx.recordCapacity, 0u);
  EXPECT_EQ(ctx.layout.totalBytes, sizeof(RingBufferHeader));
}

TEST(RealTransferEngineTest, PrepareAllocatesAdapterBackedBuffers) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  DebugBufferPlan plan = makeTestBufferPlan();
  DebugLaunchContext ctx =
      engine->prepare(makeTestBufferMeta(), plan, makeTestRuntimeMetadata());

  EXPECT_NE(ctx.deviceCtrlPtr, nullptr);
  EXPECT_NE(ctx.hostBufferPtr, nullptr);
  EXPECT_EQ(ctx.bufferSize, ctx.layout.totalBytes);
  EXPECT_EQ(ctx.recordCapacity, plan.recordCapacity);
  EXPECT_EQ(engine->hiddenArg(ctx),
            reinterpret_cast<uint64_t>(ctx.deviceCtrlPtr));

  engine->release(ctx);
}

TEST(RealTransferEngineTest, InitHeaderAndSyncExportCopyRecords) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  DebugLaunchContext ctx = engine->prepare(
      makeTestBufferMeta(), makeTestBufferPlan(), makeTestRuntimeMetadata());
  engine->initHeader(ctx);

  auto *header = reinterpret_cast<RingBufferHeader *>(ctx.deviceCtrlPtr);
  SummaryRecord record{};
  record.header.recordKind = RecordKind::SUMMARY;
  record.header.opId = 77;
  record.header.logicalInstanceId = 9;
  record.collectorKind = CollectorKind::ELEMENT_COUNT;
  record.resultType = ResultType::U64;
  record.resultData.u64Val = 128;
  header->writeIdx = 1;
  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 0),
              &record, sizeof(record));

  DebugExportedRun run = engine->syncExport(ctx);

  ASSERT_GE(run.rawBuffer.size(),
            getRecordSlotOffset(ctx.layout, 0) + sizeof(SummaryRecord));
  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  EXPECT_EQ(exportedHeader->writeIdx, 1u);
  const auto *exportedRecord = reinterpret_cast<const SummaryRecord *>(
      run.rawBuffer.data() + getRecordSlotOffset(ctx.layout, 0));
  EXPECT_EQ(exportedRecord->header.opId, 77u);
  EXPECT_EQ(exportedRecord->header.logicalInstanceId, 9u);
  EXPECT_EQ(exportedRecord->collectorKind, CollectorKind::ELEMENT_COUNT);
  EXPECT_EQ(exportedRecord->resultData.u64Val, 128u);

  engine->release(ctx);
}

TEST(RealTransferEngineTest, AsyncExportCopiesOnWait) {
  auto engine = createTransferEngine(makeHostTransferOptions());
  DebugLaunchContext ctx = engine->prepare(
      makeTestBufferMeta(), makeTestBufferPlan(), makeTestRuntimeMetadata());
  engine->initHeader(ctx);

  auto *header = reinterpret_cast<RingBufferHeader *>(ctx.deviceCtrlPtr);
  header->writeIdx = 2;

  auto *hostHeader =
      reinterpret_cast<const RingBufferHeader *>(ctx.hostBufferPtr);
  ASSERT_NE(hostHeader, nullptr);
  EXPECT_EQ(hostHeader->writeIdx, 0u);

  engine->asyncExport(ctx);
  EXPECT_EQ(hostHeader->writeIdx, 0u);

  engine->waitAsyncExport(ctx);
  EXPECT_EQ(hostHeader->writeIdx, 2u);

  engine->release(ctx);
}

TEST(RealTransferEngineTest, CannDriverRejectsNonCannBufferMeta) {
  auto engine = createTransferEngine(BackendKind::CANN);

  EXPECT_DEATH(
      {
        (void)engine->prepare(makeTestBufferMeta(), makeTestBufferPlan(),
                              makeTestRuntimeMetadata());
      },
      "BufferMeta\\.backendKind == CANN");
}

} // namespace
} // namespace debugger
} // namespace flagtree
} // namespace mlir
