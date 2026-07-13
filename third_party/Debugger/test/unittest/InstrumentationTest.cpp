#include "Debugger/Instrumentation/Collectors.h"
#include "Debugger/Instrumentation/RecordBuilder.h"
#include "Debugger/Instrumentation/Writer.h"
#include "Debugger/Runtime/BufferLayout.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

TEST(FlagTreeDebuggerInstrumentationTest, BuildsSummaryRecords) {
  const auto u64Record =
      buildSummaryU64Record(7, 17, CollectorKind::ELEMENT_COUNT, 64);
  EXPECT_EQ(u64Record.header.recordKind, RecordKind::SUMMARY);
  EXPECT_EQ(u64Record.header.opId, 7u);
  EXPECT_EQ(u64Record.header.logicalInstanceId, 17u);
  EXPECT_EQ(u64Record.collectorKind, CollectorKind::ELEMENT_COUNT);
  EXPECT_EQ(u64Record.resultType, ResultType::U64);
  EXPECT_EQ(u64Record.resultData.u64Val, 64u);

  const auto f32Record =
      buildSummaryF32Record(8, 18, CollectorKind::MEAN_FINITE, 3.5f);
  EXPECT_EQ(f32Record.resultType, ResultType::F32);
  EXPECT_FLOAT_EQ(f32Record.resultData.f32Val, 3.5f);

  const auto f64Record =
      buildSummaryF64Record(9, 19, CollectorKind::MAX_FINITE, 6.25);
  EXPECT_EQ(f64Record.resultType, ResultType::F64);
  EXPECT_DOUBLE_EQ(f64Record.resultData.f64Val, 6.25);

  const auto zeroRecord =
      buildSummaryU64Record(10, 20, CollectorKind::ZERO_COUNT, 2);
  EXPECT_EQ(zeroRecord.collectorKind, CollectorKind::ZERO_COUNT);
  EXPECT_EQ(zeroRecord.resultData.u64Val, 2u);

  const auto l2Record =
      buildSummaryF32Record(11, 21, CollectorKind::L2_NORM, 5.0f);
  EXPECT_EQ(l2Record.collectorKind, CollectorKind::L2_NORM);
  EXPECT_FLOAT_EQ(l2Record.resultData.f32Val, 5.0f);
}

TEST(FlagTreeDebuggerInstrumentationTest, BuildsMemoryAndFullValueRecords) {
  const auto memoryRecord = buildMemoryEventRecord(
      3, 99, 0x1234, MemoryEventKind::LAST_ALIGNED_ADDR, 7);
  EXPECT_EQ(memoryRecord.header.recordKind, RecordKind::MEMORY_EVENT);
  EXPECT_EQ(memoryRecord.header.opId, 3u);
  EXPECT_EQ(memoryRecord.header.logicalInstanceId, 99u);
  EXPECT_EQ(memoryRecord.addr, 0x1234u);
  EXPECT_EQ(memoryRecord.eventKind, MemoryEventKind::LAST_ALIGNED_ADDR);
  EXPECT_EQ(memoryRecord.ext0, 7u);

  const auto fullValueRecord = buildFullValueRefRecord(5, 11, 128, 256);
  EXPECT_EQ(fullValueRecord.header.recordKind, RecordKind::FULL_VALUE);
  EXPECT_EQ(fullValueRecord.payloadOffset, 128u);
  EXPECT_EQ(fullValueRecord.payloadLength, 256u);
}

TEST(FlagTreeDebuggerInstrumentationTest, ComputesLogicalInstanceId) {
  EXPECT_EQ(computeLogicalInstanceId(0, 0, 0, 8, 4), 0u);
  EXPECT_EQ(computeLogicalInstanceId(3, 0, 0, 8, 4), 3u);
  EXPECT_EQ(computeLogicalInstanceId(3, 2, 0, 8, 4), 19u);
  EXPECT_EQ(computeLogicalInstanceId(3, 2, 1, 8, 4), 51u);
}

TEST(FlagTreeDebuggerInstrumentationTest, CollectorLookupMatchesPhase1Set) {
  EXPECT_TRUE(isKnownCollector(CollectorKind::NAN_COUNT));
  EXPECT_TRUE(isKnownCollector(CollectorKind::ZERO_COUNT));
  EXPECT_TRUE(isKnownCollector(CollectorKind::L2_NORM));
  EXPECT_EQ(getCollectorName(CollectorKind::MEAN_FINITE), "mean_finite");
  EXPECT_EQ(getCollectorName(CollectorKind::ZERO_COUNT), "zero_count");
  EXPECT_EQ(getCollectorName(CollectorKind::L2_NORM), "l2_norm");

  const auto summaryCollectors =
      getEnabledCollectors(RecordLevel::LEVEL_SUMMARY);
  const auto fullCollectors =
      getEnabledCollectors(RecordLevel::LEVEL_TENSOR_FULL);

  EXPECT_EQ(summaryCollectors.size(), 8u);
  EXPECT_EQ(fullCollectors.size(), 8u);
  EXPECT_EQ(summaryCollectors, fullCollectors);
}

TEST(FlagTreeDebuggerInstrumentationTest,
     RingBufferWriterWritesAndTracksOverflow) {
  const BufferLayout layout = computeBufferLayout(2, sizeof(SummaryRecord), 0);
  std::vector<uint8_t> rawBuffer(layout.totalBytes, 0);
  ASSERT_TRUE(initializeRingBufferStorage(rawBuffer.data(), rawBuffer.size(), 2,
                                          sizeof(SummaryRecord)));

  const auto firstRecord =
      buildSummaryU64Record(1, 10, CollectorKind::NAN_COUNT, 1);
  const auto secondRecord =
      buildSummaryU64Record(2, 11, CollectorKind::INF_COUNT, 2);
  const auto thirdRecord =
      buildSummaryU64Record(3, 12, CollectorKind::ELEMENT_COUNT, 3);

  const auto first =
      appendSummaryRecord(rawBuffer.data(), rawBuffer.size(), firstRecord);
  const auto second =
      appendSummaryRecord(rawBuffer.data(), rawBuffer.size(), secondRecord);
  const auto third =
      appendSummaryRecord(rawBuffer.data(), rawBuffer.size(), thirdRecord);

  EXPECT_EQ(first.status, RecordWriteStatus::WRITTEN);
  EXPECT_EQ(first.slot, 0u);
  EXPECT_EQ(second.status, RecordWriteStatus::WRITTEN);
  EXPECT_EQ(second.slot, 1u);
  EXPECT_EQ(third.status, RecordWriteStatus::OVERFLOW);
  EXPECT_EQ(third.slot, 2u);

  const auto *header =
      reinterpret_cast<const RingBufferHeader *>(rawBuffer.data());
  EXPECT_EQ(header->writeIdx, 3u);
  EXPECT_EQ(header->overflowCount, 1u);
  EXPECT_TRUE((header->flags & RB_FLAG_OVERFLOW) != 0);

  SummaryRecord slot0{};
  SummaryRecord slot1{};
  std::memcpy(&slot0, rawBuffer.data() + sizeof(RingBufferHeader),
              sizeof(slot0));
  std::memcpy(&slot1,
              rawBuffer.data() + sizeof(RingBufferHeader) + sizeof(slot1),
              sizeof(slot1));
  EXPECT_EQ(slot0.header.opId, 1u);
  EXPECT_EQ(slot1.header.opId, 2u);
}

TEST(FlagTreeDebuggerInstrumentationTest, RejectsMismatchedRecordSize) {
  const BufferLayout layout = computeBufferLayout(1, sizeof(SummaryRecord), 0);
  std::vector<uint8_t> rawBuffer(layout.totalBytes, 0);
  ASSERT_TRUE(initializeRingBufferStorage(rawBuffer.data(), rawBuffer.size(), 1,
                                          sizeof(SummaryRecord)));

  const auto record =
      buildMemoryEventRecord(1, 1, 0x1000, MemoryEventKind::LAST_ALIGNED_ADDR);
  const auto result = appendRecordToRingBuffer(
      rawBuffer.data(), rawBuffer.size(), &record, sizeof(record) - 8);
  EXPECT_EQ(result.status, RecordWriteStatus::INVALID_ARGUMENT);
}

// ─── SummaryStats host-side computation (C-3 / C-4 analogues) ───────────────

TEST(FlagTreeDebuggerInstrumentationTest, ComputeSummaryStatsF32_Mixed) {
  // Array: two finite values, one NaN, one +Inf.
  const float data[] = {1.0f, 3.0f, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()};
  const SummaryStats stats = computeSummaryStatsF32(data, 4);

  EXPECT_EQ(stats.elementCount, 4u);
  EXPECT_EQ(stats.nanCount, 1u);
  EXPECT_EQ(stats.infCount, 1u);
  EXPECT_EQ(stats.zeroCount, 0u);
  EXPECT_DOUBLE_EQ(stats.mean, 2.0); // (1+3)/2
  EXPECT_DOUBLE_EQ(stats.min, 1.0);
  EXPECT_DOUBLE_EQ(stats.max, 3.0);
  EXPECT_DOUBLE_EQ(stats.l2Norm, std::sqrt(10.0));
}

TEST(FlagTreeDebuggerInstrumentationTest, ComputeSummaryStatsF32_Empty) {
  const SummaryStats stats = computeSummaryStatsF32(nullptr, 0);
  EXPECT_EQ(stats.elementCount, 0u);
  EXPECT_EQ(stats.nanCount, 0u);
  EXPECT_EQ(stats.infCount, 0u);
  EXPECT_EQ(stats.zeroCount, 0u);
  EXPECT_DOUBLE_EQ(stats.mean, 0.0);
  EXPECT_DOUBLE_EQ(stats.min, 0.0);
  EXPECT_DOUBLE_EQ(stats.max, 0.0);
  EXPECT_DOUBLE_EQ(stats.l2Norm, 0.0);
}

TEST(FlagTreeDebuggerInstrumentationTest, ComputeSummaryStatsF32_AllNaN) {
  const float data[] = {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
  const SummaryStats stats = computeSummaryStatsF32(data, 2);
  EXPECT_EQ(stats.elementCount, 2u);
  EXPECT_EQ(stats.nanCount, 2u);
  EXPECT_EQ(stats.infCount, 0u);
  EXPECT_EQ(stats.zeroCount, 0u);
  // No finite values → mean/min/max stay 0.0.
  EXPECT_DOUBLE_EQ(stats.mean, 0.0);
  EXPECT_DOUBLE_EQ(stats.min, 0.0);
  EXPECT_DOUBLE_EQ(stats.max, 0.0);
  EXPECT_DOUBLE_EQ(stats.l2Norm, 0.0);
}

TEST(FlagTreeDebuggerInstrumentationTest, ComputeSummaryStatsF64_Finite) {
  const double data[] = {-2.0, 0.0, 4.0};
  const SummaryStats stats = computeSummaryStatsF64(data, 3);
  EXPECT_EQ(stats.elementCount, 3u);
  EXPECT_EQ(stats.nanCount, 0u);
  EXPECT_EQ(stats.infCount, 0u);
  EXPECT_EQ(stats.zeroCount, 1u);
  EXPECT_DOUBLE_EQ(stats.mean, 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(stats.min, -2.0);
  EXPECT_DOUBLE_EQ(stats.max, 4.0);
  EXPECT_DOUBLE_EQ(stats.l2Norm, std::sqrt(20.0));
}

// ─── LinearAppendSink ────────────────────────────────────────────────────────

TEST(FlagTreeDebuggerInstrumentationTest, LinearAppendSink_WritesAndOverflows) {
  // Capacity for exactly 2 records.
  std::vector<uint8_t> buf(2 * kDefaultRecordSize, 0);
  auto sink = createLinearAppendSink(buf.data(), buf.size());

  const auto r1 = sink->writeSummary(
      buildSummaryU64Record(1, 10, CollectorKind::NAN_COUNT, 5));
  const auto r2 = sink->writeSummary(
      buildSummaryU64Record(2, 20, CollectorKind::INF_COUNT, 3));
  const auto r3 = sink->writeSummary(
      buildSummaryU64Record(3, 30, CollectorKind::ELEMENT_COUNT, 0));

  EXPECT_EQ(r1.status, RecordWriteStatus::WRITTEN);
  EXPECT_EQ(r1.slot, 0u);
  EXPECT_EQ(r2.status, RecordWriteStatus::WRITTEN);
  EXPECT_EQ(r2.slot, 1u);
  EXPECT_EQ(r3.status, RecordWriteStatus::OVERFLOW);
  EXPECT_EQ(r3.overflowCount, 1u);

  EXPECT_EQ(sink->recordCount(), 2u);

  // Verify raw bytes contain the correct records.
  SummaryRecord stored0{}, stored1{};
  std::memcpy(&stored0, buf.data(), sizeof(SummaryRecord));
  std::memcpy(&stored1, buf.data() + sizeof(SummaryRecord),
              sizeof(SummaryRecord));
  EXPECT_EQ(stored0.header.opId, 1u);
  EXPECT_EQ(stored0.resultData.u64Val, 5u);
  EXPECT_EQ(stored1.header.opId, 2u);
  EXPECT_EQ(stored1.resultData.u64Val, 3u);
}

TEST(FlagTreeDebuggerInstrumentationTest, LinearAppendSink_MixedRecordTypes) {
  std::vector<uint8_t> buf(3 * kDefaultRecordSize, 0);
  auto sink = createLinearAppendSink(buf.data(), buf.size());

  sink->writeSummary(buildSummaryU64Record(1, 0, CollectorKind::NAN_COUNT, 0));
  sink->writeMemoryEvent(
      buildMemoryEventRecord(2, 0, 0xABCD, MemoryEventKind::LAST_ALIGNED_ADDR));
  sink->writeFullValueRef(buildFullValueRefRecord(3, 0, 64, 128));

  EXPECT_EQ(sink->recordCount(), 3u);

  // Verify each slot by peeking at RecordKind via the shared header offset.
  RecordHeader hdr{};
  std::memcpy(&hdr, buf.data() + kDefaultRecordSize, sizeof(hdr));
  EXPECT_EQ(hdr.recordKind, RecordKind::MEMORY_EVENT);
  EXPECT_EQ(hdr.opId, 2u);
}

// ─── RingBufferSink (integration path via RecordSink interface)
// ───────────────

TEST(FlagTreeDebuggerInstrumentationTest, RingBufferSink_WritesAndOverflows) {
  const BufferLayout layout = computeBufferLayout(2, sizeof(SummaryRecord), 0);
  std::vector<uint8_t> buf(layout.totalBytes, 0);
  ASSERT_TRUE(initializeRingBufferStorage(buf.data(), buf.size(), 2,
                                          sizeof(SummaryRecord)));

  auto sink = createRingBufferSink(buf.data(), buf.size());

  sink->writeSummary(buildSummaryU64Record(1, 0, CollectorKind::NAN_COUNT, 1));
  sink->writeSummary(buildSummaryU64Record(2, 0, CollectorKind::INF_COUNT, 2));
  const auto overflow = sink->writeSummary(
      buildSummaryU64Record(3, 0, CollectorKind::ELEMENT_COUNT, 3));

  EXPECT_EQ(overflow.status, RecordWriteStatus::OVERFLOW);
  EXPECT_EQ(sink->recordCount(), 2u);

  const auto *header = reinterpret_cast<const RingBufferHeader *>(buf.data());
  EXPECT_EQ(header->overflowCount, 1u);
  EXPECT_TRUE((header->flags & RB_FLAG_OVERFLOW) != 0);
}

// ─── writeSummaryRecordsToSink end-to-end ────────────────────────────────────

TEST(FlagTreeDebuggerInstrumentationTest,
     WriteSummaryRecordsToSink_Phase1Core) {
  const float data[] = {1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()};
  const SummaryStats stats = computeSummaryStatsF32(data, 4);

  // Phase-1 summary level has 8 collectors.
  const size_t kExpectedRecords = 8;
  std::vector<uint8_t> buf(kExpectedRecords * kDefaultRecordSize, 0);
  auto sink = createLinearAppendSink(buf.data(), buf.size());

  writeSummaryRecordsToSink(42, 7, stats, RecordLevel::LEVEL_SUMMARY, *sink);

  EXPECT_EQ(sink->recordCount(), kExpectedRecords);

  // Every stored record must carry op_id=42, logical_instance_id=7 and
  // RecordKind::SUMMARY.
  for (size_t i = 0; i < kExpectedRecords; ++i) {
    SummaryRecord rec{};
    std::memcpy(&rec, buf.data() + i * kDefaultRecordSize, sizeof(rec));
    EXPECT_EQ(rec.header.recordKind, RecordKind::SUMMARY) << "record " << i;
    EXPECT_EQ(rec.header.opId, 42u) << "record " << i;
    EXPECT_EQ(rec.header.logicalInstanceId, 7u) << "record " << i;
  }

  // Spot-check individual collector values.
  // Order follows getEnabledCollectors: NAN_COUNT, INF_COUNT, ZERO_COUNT,
  // MEAN, MIN, MAX, L2_NORM, ELEMENT_COUNT.
  SummaryRecord nanRec{}, infRec{}, zeroRec{}, l2Rec{}, elemRec{};
  std::memcpy(&nanRec, buf.data() + 0 * kDefaultRecordSize, sizeof(nanRec));
  std::memcpy(&infRec, buf.data() + 1 * kDefaultRecordSize, sizeof(infRec));
  std::memcpy(&zeroRec, buf.data() + 2 * kDefaultRecordSize, sizeof(zeroRec));
  std::memcpy(&l2Rec, buf.data() + 6 * kDefaultRecordSize, sizeof(l2Rec));
  std::memcpy(&elemRec, buf.data() + 7 * kDefaultRecordSize, sizeof(elemRec));

  EXPECT_EQ(nanRec.collectorKind, CollectorKind::NAN_COUNT);
  EXPECT_EQ(nanRec.resultData.u64Val, 1u);

  EXPECT_EQ(infRec.collectorKind, CollectorKind::INF_COUNT);
  EXPECT_EQ(infRec.resultData.u64Val, 1u);

  EXPECT_EQ(zeroRec.collectorKind, CollectorKind::ZERO_COUNT);
  EXPECT_EQ(zeroRec.resultData.u64Val, 0u);

  EXPECT_EQ(l2Rec.collectorKind, CollectorKind::L2_NORM);
  EXPECT_DOUBLE_EQ(l2Rec.resultData.f64Val, std::sqrt(5.0));

  EXPECT_EQ(elemRec.collectorKind, CollectorKind::ELEMENT_COUNT);
  EXPECT_EQ(elemRec.resultData.u64Val, 4u);
}

} // namespace
} // namespace debugger
} // namespace flagtree
} // namespace mlir
