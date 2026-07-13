#include "Debugger/Decode/Decoder.h"
#include "Debugger/Decode/Reporter.h"
#include "Debugger/Runtime/BufferLayout.h"
#include "TestFixtures.h"

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace mlir::flagtree::debugger;

namespace {

template <typename T>
void writeObject(std::vector<uint8_t> &buffer, size_t offset, const T &value) {
  ASSERT_LE(offset + sizeof(T), buffer.size());
  std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

DebugExportedRun makeRunWithRecordSize(uint32_t capacity, uint32_t writeIdx,
                                       uint32_t recordSize,
                                       uint32_t overflowCount = 0,
                                       uint32_t flags = RB_FLAG_NONE) {
  TestRunOptions options;
  options.recordCapacity = capacity;
  options.recordSize = recordSize;
  options.payloadBytes = 64;

  DebugExportedRun run;
  run.meta = makeTestBufferMeta(options);
  run.runtimeMetadata = makeTestRuntimeMetadata();

  BufferLayout layout =
      computeBufferLayout(capacity, recordSize, options.payloadBytes);
  run.rawBuffer.assign(layout.totalBytes, 0);

  RingBufferHeader header{};
  header.writeIdx = writeIdx;
  header.capacity = capacity;
  header.overflowCount = overflowCount;
  header.flags = flags;
  header.recordSize = recordSize;
  header.payloadOffset = static_cast<uint32_t>(layout.payloadOffset);
  writeObject(run.rawBuffer, 0, header);
  return run;
}

DebugExportedRun makeRun(uint32_t capacity, uint32_t writeIdx,
                         uint32_t overflowCount = 0,
                         uint32_t flags = RB_FLAG_NONE) {
  return makeRunWithRecordSize(capacity, writeIdx, kDefaultRecordSize,
                               overflowCount, flags);
}

ReportOptions statementOnlyReportOptions() {
  ReportOptions options;
  options.includeStaticMetadata = false;
  options.includeOpLog = false;
  return options;
}

size_t slotOffset(uint32_t slotIndex,
                  uint32_t recordSize = kDefaultRecordSize) {
  return sizeof(RingBufferHeader) + static_cast<size_t>(slotIndex) * recordSize;
}

SummaryRecord makeSummary(uint32_t opId, uint64_t instance,
                          CollectorKind collector, uint64_t value) {
  SummaryRecord record{};
  record.header.recordKind = RecordKind::SUMMARY;
  record.header.opId = opId;
  record.header.logicalInstanceId = instance;
  record.collectorKind = collector;
  record.resultType = ResultType::U64;
  record.resultData.u64Val = value;
  return record;
}

MemoryEventRecord
makeMemory(uint32_t opId, uint64_t instance, uint64_t addr,
           MemoryEventKind kind = MemoryEventKind::LAST_ALIGNED_ADDR,
           uint32_t ext0 = 0) {
  MemoryEventRecord record{};
  record.header.recordKind = RecordKind::MEMORY_EVENT;
  record.header.opId = opId;
  record.header.logicalInstanceId = instance;
  record.addr = addr;
  record.eventKind = kind;
  record.ext0 = ext0;
  return record;
}

void writeAddressSummaryRecords(DebugExportedRun &run, uint32_t &slot,
                                uint32_t opId, uint64_t instance,
                                uint64_t baseAddr, uint32_t ext0 = 0) {
  const MemoryEventKind addressKinds[] = {
      MemoryEventKind::FIRST_ADDR,        MemoryEventKind::LAST_ADDR,
      MemoryEventKind::MIN_ADDR,          MemoryEventKind::MAX_ADDR,
      MemoryEventKind::ACTIVE_LANE_COUNT, MemoryEventKind::ADDRESS_SPAN_BYTES,
  };
  for (MemoryEventKind kind : addressKinds) {
    uint64_t value = baseAddr;
    if (kind == MemoryEventKind::ACTIVE_LANE_COUNT)
      value = 4;
    else if (kind == MemoryEventKind::ADDRESS_SPAN_BYTES)
      value = 16;
    writeObject(run.rawBuffer, slotOffset(slot++),
                makeMemory(opId, instance, value, kind, ext0));
  }
}

FullValueRefRecord makeFullValue(uint32_t opId, uint64_t instance,
                                 uint32_t payloadOffset,
                                 uint32_t payloadLength) {
  FullValueRefRecord record{};
  record.header.recordKind = RecordKind::FULL_VALUE;
  record.header.opId = opId;
  record.header.logicalInstanceId = instance;
  record.payloadOffset = payloadOffset;
  record.payloadLength = payloadLength;
  return record;
}

SummaryCountBundleRecord makeCountBundle(uint32_t opId, uint64_t instance) {
  SummaryCountBundleRecord record{};
  record.header.recordKind = RecordKind::SUMMARY_COUNT_BUNDLE_U64;
  record.header.opId = opId;
  record.header.logicalInstanceId = instance;
  record.nanCount = 1;
  record.infCount = 2;
  record.zeroCount = 3;
  record.elementCount = 16;
  return record;
}

SummaryValueBundleRecord makeValueBundle(uint32_t opId, uint64_t instance) {
  SummaryValueBundleRecord record{};
  record.header.recordKind = RecordKind::SUMMARY_VALUE_BUNDLE_F32;
  record.header.opId = opId;
  record.header.logicalInstanceId = instance;
  record.meanFinite = 4.0f;
  record.minFinite = -1.0f;
  record.maxFinite = 8.0f;
  record.l2Norm = 9.0f;
  return record;
}

DebugExportedRun makeGoldenReportRun() {
  DebugExportedRun run = makeRun(/*capacity=*/4, /*writeIdx=*/2,
                                 /*overflowCount=*/3, RB_FLAG_OVERFLOW);
  writeObject(run.rawBuffer, slotOffset(0),
              makeSummary(/*opId=*/2, /*instance=*/42,
                          CollectorKind::ELEMENT_COUNT, /*value=*/128));
  writeObject(run.rawBuffer, slotOffset(1),
              makeMemory(/*opId=*/1, /*instance=*/42, 0x1010));
  return run;
}

std::string expectedD4DynamicReportGolden() {
  return R"REPORT(FlagTree Debug Report
protocol_version: 2
run_id: 1
kernel_id: 1
kernel_name: test_kernel
device: 0
backend: CUDA
record_level: LEVEL_SUMMARY
export_mode: POST_KERNEL_EXPORT
record_count: 2
record_count_note: number of debug record slots written; not tensor element count
overflow_count: 3
flags: 1
overflow_warning: device debug buffer overflowed; report may be truncated

IR Op Log Records
op_id=1 scope_id=1
  static:
    mlir_op: tt.load
    role: load
    category: load
    source_loc: unit_test.py:1
    triton_statement: x = tl.load(ptr)
    dtype_in: <none>
    dtype_out: fp32
    shape: [128]
    stride: [1]
    layout: blocked
    memory_semantics: addr_space=global access_type=load access_bytes=4 alignment_required=4 has_mask=false boundary_check_policy=<none>

  instances: [42]
  memory_events_by_instance:
    logical_instance_id=42
      [0] kind=LAST_ALIGNED_ADDR addr=0x1010 ext0=0
        runtime_address: bufferId=1 name=input range=[0x1000,0x1200) sizeBytes=512 offset=16
        alignment_ok=true
        local_address_snapshot=not captured

op_id=2 scope_id=1
  static:
    mlir_op: arith.addf
    source_loc: unit_test.py:1
    triton_statement: y = x + 1
    dtype_in: <none>
    dtype_out: fp32
    shape: [128]
    stride: [1]
    layout: blocked

  instances: [42]
  summary:
    element_count: [128 (U64)]

IR Op Log Static Only Ops
op_id=3 scope_id=1
  dynamic_record_status: static_only
  dynamic_record_note: no runtime record was emitted for this op; static metadata is kept for producer/context analysis
  static:
    mlir_op: arith.addf
    source_loc: unit_test.py:1
    triton_statement: y = x + 1
    dtype_in: <none>
    dtype_out: fp32
    shape: [128]
    stride: [1]
    layout: blocked

op_id=4 scope_id=1
  dynamic_record_status: static_only
  dynamic_record_note: no runtime record was emitted for this op; static metadata is kept for producer/context analysis
  static:
    mlir_op: arith.addf
    source_loc: unit_test.py:1
    triton_statement: y = x + 1
    dtype_in: <none>
    dtype_out: fp32
    shape: [128]
    stride: [1]
    layout: blocked
)REPORT";
}

std::string expectedAggregateGolden() {
  return R"REPORT(Aggregates
kernel: total_records=2 summary_records=1 memory_event_records=1 full_value_ref_records=0
by_scope:
  scope_id=1 records=2
by_op:
  op_id=1 scope_id=1 total_records=1 summary_records=0 memory_event_records=1 full_value_ref_records=0
  op_id=2 scope_id=1 total_records=1 summary_records=1 memory_event_records=0 full_value_ref_records=0
    latest.element_count=128
)REPORT";
}

std::string expectedD5MetadataRunContextGolden() {
  return R"REPORT(FlagTree Debug Report
protocol_version: 2
run_id: 1
kernel_id: 1
kernel_name: test_kernel
device: 0
backend: CUDA
record_level: LEVEL_SUMMARY
export_mode: POST_KERNEL_EXPORT
record_count: 2
record_count_note: number of debug record slots written; not tensor element count
overflow_count: 3
flags: 1
overflow_warning: device debug buffer overflowed; report may be truncated

Runtime Inventory
buffers: 1
  bufferId=1 name=input base=0x1000 sizeBytes=512 alignment=16 range=[0x1000,0x1200)
tensors: 1
  arg=0 name=x dtype=fp32 shape=[128] stride=[1] layout=contiguous bufferId=1 base=0x1000 sizeBytes=512
)REPORT";
}

} // namespace

TEST(DebuggerDecodeTest, DecodesSummaryRecord) {
  DebugExportedRun run = makeRun(/*capacity=*/4, /*writeIdx=*/1);
  SummaryRecord raw = makeSummary(/*opId=*/2, /*instance=*/42,
                                  CollectorKind::NAN_COUNT, /*value=*/7);
  writeObject(run.rawBuffer, slotOffset(0), raw);

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  EXPECT_EQ(decoded.meta.runId, run.meta.runId);
  EXPECT_EQ(decoded.header.writeIdx, 1u);
  ASSERT_EQ(decoded.records.size(), 1u);
  const auto *summary = std::get_if<DecodedSummaryRecord>(&decoded.records[0]);
  ASSERT_NE(summary, nullptr);
  EXPECT_EQ(summary->raw.header.opId, 2u);
  EXPECT_EQ(summary->raw.header.logicalInstanceId, 42u);
  EXPECT_EQ(summary->raw.collectorKind, CollectorKind::NAN_COUNT);
  EXPECT_EQ(summary->raw.resultType, ResultType::U64);
  EXPECT_EQ(summary->raw.resultData.u64Val, 7u);
  EXPECT_EQ(decoded.runtimeMetadata.buffers.size(), 1u);
}

TEST(DebuggerDecodeTest, DecodesSummaryBundleRecords) {
  DebugExportedRun run =
      makeRunWithRecordSize(/*capacity=*/4, /*writeIdx=*/2, kBundleRecordSize);
  writeObject(run.rawBuffer, slotOffset(0, kBundleRecordSize),
              makeCountBundle(/*opId=*/2, /*instance=*/42));
  writeObject(run.rawBuffer, slotOffset(1, kBundleRecordSize),
              makeValueBundle(/*opId=*/2, /*instance=*/42));

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ASSERT_EQ(decoded.records.size(), 2u);
  const auto *count =
      std::get_if<DecodedSummaryCountBundleRecord>(&decoded.records[0]);
  ASSERT_NE(count, nullptr);
  EXPECT_EQ(count->raw.nanCount, 1u);
  EXPECT_EQ(count->raw.infCount, 2u);
  EXPECT_EQ(count->raw.zeroCount, 3u);
  EXPECT_EQ(count->raw.elementCount, 16u);

  const auto *value =
      std::get_if<DecodedSummaryValueBundleRecord>(&decoded.records[1]);
  ASSERT_NE(value, nullptr);
  EXPECT_FLOAT_EQ(value->raw.meanFinite, 4.0f);
  EXPECT_FLOAT_EQ(value->raw.minFinite, -1.0f);
  EXPECT_FLOAT_EQ(value->raw.maxFinite, 8.0f);
  EXPECT_FLOAT_EQ(value->raw.l2Norm, 9.0f);

  std::string report = renderTextReport(decoded, makeTestKernelDebugMetadata());
  EXPECT_NE(report.find("nan_count    : [1 (U64)]"), std::string::npos);
  EXPECT_NE(report.find("inf_count    : [2 (U64)]"), std::string::npos);
  EXPECT_NE(report.find("zero_count   : [3 (U64)]"), std::string::npos);
  EXPECT_NE(report.find("element_count: [16 (U64)]"), std::string::npos);
  EXPECT_NE(report.find("mean         : [4 (F32)]"), std::string::npos);
  EXPECT_NE(report.find("min          : [-1 (F32)]"), std::string::npos);
  EXPECT_NE(report.find("max          : [8 (F32)]"), std::string::npos);
  EXPECT_NE(report.find("l2_norm      : [9 (F32)]"), std::string::npos);
}

TEST(DebuggerDecodeTest, DecodesDeterministicCompactTimelineRecord) {
  DebugExportedRun run =
      makeRunWithRecordSize(/*capacity=*/1, /*writeIdx=*/1, kBundleRecordSize);
  run.runtimeMetadata.recordLayout = "deterministic_compact_v1";
  run.runtimeMetadata.recordsPerInstance = 1;
  DebugRecordPlanEntry plan;
  plan.recordIndex = 0;
  plan.opId = 7;
  plan.scopeId = 1;
  plan.recordKind = RecordKind::TIMELINE;
  run.runtimeMetadata.recordPlan.push_back(plan);

  const uint64_t startCycle = 100;
  const uint64_t endCycle = 145;
  const uint64_t durationCycle = 45;
  const size_t offset = slotOffset(0, kBundleRecordSize);
  writeObject(run.rawBuffer, offset + 16, startCycle);
  writeObject(run.rawBuffer, offset + 24, endCycle);
  writeObject(run.rawBuffer, offset + 32, durationCycle);

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ASSERT_EQ(decoded.records.size(), 1u);
  const auto *timeline =
      std::get_if<DecodedTimelineRecord>(&decoded.records[0]);
  ASSERT_NE(timeline, nullptr);
  EXPECT_EQ(timeline->raw.header.opId, 7u);
  EXPECT_EQ(timeline->raw.header.logicalInstanceId, 0u);
  EXPECT_EQ(timeline->raw.startCycle, startCycle);
  EXPECT_EQ(timeline->raw.endCycle, endCycle);
  EXPECT_EQ(timeline->raw.durationCycle, durationCycle);
}

TEST(DebuggerDecodeTest, DecodesMemoryEventRecord) {
  DebugExportedRun run = makeRun(/*capacity=*/4, /*writeIdx=*/1);
  MemoryEventRecord raw = makeMemory(/*opId=*/1, /*instance=*/11, 0x1010);
  writeObject(run.rawBuffer, slotOffset(0), raw);

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ASSERT_EQ(decoded.records.size(), 1u);
  const auto *memory =
      std::get_if<DecodedMemoryEventRecord>(&decoded.records[0]);
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->raw.header.opId, 1u);
  EXPECT_EQ(memory->raw.header.logicalInstanceId, 11u);
  EXPECT_EQ(memory->raw.addr, 0x1010u);
  EXPECT_EQ(memory->raw.eventKind, MemoryEventKind::LAST_ALIGNED_ADDR);
}

TEST(DebuggerDecodeTest, DecodesFullValueRefRecord) {
  DebugExportedRun run = makeRun(/*capacity=*/4, /*writeIdx=*/1);
  FullValueRefRecord raw = makeFullValue(/*opId=*/3, /*instance=*/9,
                                         /*payloadOffset=*/160,
                                         /*payloadLength=*/16);
  writeObject(run.rawBuffer, slotOffset(0), raw);

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ASSERT_EQ(decoded.records.size(), 1u);
  const auto *full =
      std::get_if<DecodedFullValueRefRecord>(&decoded.records[0]);
  ASSERT_NE(full, nullptr);
  EXPECT_EQ(full->raw.header.opId, 3u);
  EXPECT_EQ(full->raw.payloadOffset, 160u);
  EXPECT_EQ(full->raw.payloadLength, 16u);
}

TEST(DebuggerDecodeTest, RejectsProtocolMismatch) {
  DebugExportedRun run = makeRun(/*capacity=*/1, /*writeIdx=*/0);
  run.meta.protocolVer = 999;

  DecodedDebugRun decoded;
  std::string error;
  EXPECT_FALSE(decodeExportedRun(run, decoded, &error));
  EXPECT_NE(error.find("protocol version"), std::string::npos);
}

TEST(DebuggerDecodeTest, RejectsUnknownRecordKind) {
  DebugExportedRun run = makeRun(/*capacity=*/1, /*writeIdx=*/1);
  RecordHeader raw{};
  raw.recordKind = static_cast<RecordKind>(999);
  raw.opId = 1;
  raw.logicalInstanceId = 1;
  writeObject(run.rawBuffer, slotOffset(0), raw);

  DecodedDebugRun decoded;
  std::string error;
  EXPECT_FALSE(decodeExportedRun(run, decoded, &error));
  EXPECT_NE(error.find("Unknown recordKind"), std::string::npos);
}

TEST(DebuggerDecodeTest, RejectsInvalidRecordSize) {
  DebugExportedRun run = makeRun(/*capacity=*/1, /*writeIdx=*/0);
  RingBufferHeader header{};
  std::memcpy(&header, run.rawBuffer.data(), sizeof(header));
  header.recordSize = 16;
  writeObject(run.rawBuffer, 0, header);

  DecodedDebugRun decoded;
  std::string error;
  EXPECT_FALSE(decodeExportedRun(run, decoded, &error));
  EXPECT_NE(error.find("recordSize"), std::string::npos);
}

TEST(DebuggerDecodeTest, RejectsInvalidPayloadOffset) {
  DebugExportedRun run = makeRun(/*capacity=*/2, /*writeIdx=*/0);
  RingBufferHeader header{};
  std::memcpy(&header, run.rawBuffer.data(), sizeof(header));
  header.payloadOffset = sizeof(RingBufferHeader);
  writeObject(run.rawBuffer, 0, header);

  DecodedDebugRun decoded;
  std::string error;
  EXPECT_FALSE(decodeExportedRun(run, decoded, &error));
  EXPECT_NE(error.find("payloadOffset"), std::string::npos);
}

TEST(DebuggerDecodeTest, RendersReportWithDynamicRecords) {
  DebugExportedRun run = makeGoldenReportRun();

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  KernelDebugMetadata metadata = makeTestKernelDebugMetadata();
  std::string report = renderTextReport(decoded, metadata);

  EXPECT_NE(report.find("FlagTree Debug Report"), std::string::npos);
  EXPECT_NE(report.find("run_id: 1"), std::string::npos);
  EXPECT_NE(report.find("kernel_id: 1"), std::string::npos);
  EXPECT_NE(report.find("backend: CUDA"), std::string::npos);
  EXPECT_NE(report.find("record_count_note: number of debug record slots"),
            std::string::npos);
  EXPECT_NE(report.find("overflow_count: 3"), std::string::npos);
  EXPECT_NE(report.find("overflow_warning"), std::string::npos);
  EXPECT_NE(report.find("IR Op Log Records"), std::string::npos);
  EXPECT_NE(report.find("IR Op Log Static Only Ops"), std::string::npos);
  EXPECT_NE(report.find("dynamic_record_status: static_only"),
            std::string::npos);
  EXPECT_NE(report.find("summary:"), std::string::npos);
  EXPECT_NE(report.find("element_count: [128 (U64)]"), std::string::npos);
  EXPECT_NE(report.find("[0] kind=LAST_ALIGNED_ADDR"), std::string::npos);
  EXPECT_NE(report.find("mlir_op: arith.addf"), std::string::npos);
  EXPECT_NE(report.find("mlir_op: tt.load"), std::string::npos);
  EXPECT_NE(report.find("dtype_out: fp32"), std::string::npos);
  EXPECT_NE(report.find("shape: [128]"), std::string::npos);
  EXPECT_NE(report.find("stride: [1]"), std::string::npos);
  EXPECT_NE(report.find("layout: blocked"), std::string::npos);
  EXPECT_NE(report.find("runtime_address: bufferId=1"), std::string::npos);
  EXPECT_NE(report.find("offset=16"), std::string::npos);
  EXPECT_NE(report.find("alignment_ok=true"), std::string::npos);
  EXPECT_EQ(report.find("sample_value=not captured"), std::string::npos);
  EXPECT_NE(report.find("Runtime Inventory"), std::string::npos);
  EXPECT_EQ(report.find("Aggregates"), std::string::npos);
  EXPECT_EQ(report.find("latest.element_count=128"), std::string::npos);
}

TEST(DebuggerDecodeTest, RendersTritonStatementRecords) {
  DebugExportedRun run = makeRun(/*capacity=*/4, /*writeIdx=*/4);
  writeObject(run.rawBuffer, slotOffset(0),
              makeSummary(/*opId=*/2, /*instance=*/42,
                          CollectorKind::ELEMENT_COUNT, /*value=*/128));
  writeObject(run.rawBuffer, slotOffset(1),
              makeSummary(/*opId=*/5, /*instance=*/42,
                          CollectorKind::ELEMENT_COUNT, /*value=*/64));
  writeObject(run.rawBuffer, slotOffset(2),
              makeFullValue(/*opId=*/2, /*instance=*/42,
                            /*payloadOffset=*/160, /*payloadLength=*/16));
  writeObject(run.rawBuffer, slotOffset(3),
              makeFullValue(/*opId=*/5, /*instance=*/42,
                            /*payloadOffset=*/176, /*payloadLength=*/16));
  run.runtimeMetadata.fullDumpArtifacts.push_back(FullDumpArtifactInfo{
      /*opId=*/2,
      /*logicalInstanceId=*/42,
      /*payloadOffset=*/160,
      /*payloadLength=*/16,
      /*kind=*/"value",
      /*path=*/"/tmp/flagtree_debugger_artifacts/op2_inst42_value.npy",
  });
  run.runtimeMetadata.fullDumpArtifacts.push_back(FullDumpArtifactInfo{
      /*opId=*/5,
      /*logicalInstanceId=*/42,
      /*payloadOffset=*/176,
      /*payloadLength=*/16,
      /*kind=*/"value",
      /*path=*/"/tmp/flagtree_debugger_artifacts/op5_inst42_operand.npy",
  });

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  KernelDebugMetadata metadata = makeTestKernelDebugMetadata(
      /*scopeCount=*/1, /*trackedOpCount=*/0, /*includeMemoryOp=*/false);
  TrackedOpEntry anchor = makeTestTrackedOpEntry(/*opId=*/2);
  anchor.statementId = 1007;
  anchor.statementResultName = "y";
  anchor.tritonStatement = "y = a + b";

  StatementValueInfo result;
  result.sourceName = "y";
  result.sourceRole = "result";
  result.captureOpId = 2;
  result.capturePolicy = "captured_current_op";
  result.value = anchor.result;
  anchor.statementValues.push_back(result);

  StatementValueInfo operand;
  operand.sourceName = "a";
  operand.sourceRole = "operand";
  operand.hasOperandIndex = true;
  operand.operandIndex = 0;
  operand.captureOpId = 5;
  operand.capturePolicy = "captured_at_current_statement";
  operand.value = anchor.result;
  anchor.statementValues.push_back(operand);

  TrackedOpEntry synthetic = makeTestTrackedOpEntry(/*opId=*/5);
  synthetic.statementId = anchor.statementId;
  synthetic.isSyntheticStatementCapture = true;
  synthetic.mlirOpName = "flagtree.debug.operand_capture";
  synthetic.role = "operand";
  synthetic.tritonStatement = anchor.tritonStatement;
  synthetic.statementResultName = "a";

  metadata.trackedOpCount = 2;
  metadata.trackedOps = {anchor, synthetic};

  std::string report = renderTextReport(decoded, metadata);
  EXPECT_NE(report.find("Triton Statement Records"), std::string::npos);
  EXPECT_NE(report.find("IR Op Log Records"), std::string::npos);
  EXPECT_NE(report.find("statement_id: 1007"), std::string::npos);
  EXPECT_NE(report.find("statement: y = a + b"), std::string::npos);
  EXPECT_NE(report.find("[result y]:"), std::string::npos);
  EXPECT_NE(report.find("<operand a>:"), std::string::npos);
  EXPECT_EQ(report.find("capture_status:"), std::string::npos);
  EXPECT_EQ(report.find("capture_op_id:"), std::string::npos);
  EXPECT_EQ(report.find("producer_op_id:"), std::string::npos);
  EXPECT_NE(report.find("element_count: [64 (U64)]"), std::string::npos);

  ReportOptions statementOptions = statementOnlyReportOptions();
  std::string statementReport =
      renderTextReport(decoded, metadata, statementOptions);
  EXPECT_NE(statementReport.find("full_value_file: op2_inst42_value.npy"),
            std::string::npos);
  EXPECT_NE(statementReport.find("full_value_file: op5_inst42_operand.npy"),
            std::string::npos);
  EXPECT_EQ(statementReport.find("full_value_refs:"), std::string::npos);
  EXPECT_EQ(statementReport.find("/tmp/flagtree_debugger_artifacts/"),
            std::string::npos);

  std::string json = renderJsonReport(decoded, metadata);
  EXPECT_NE(json.find("\"records_by_op\""), std::string::npos);
  EXPECT_NE(json.find("\"op_log\""), std::string::npos);
  EXPECT_NE(json.find("\"source_name\":\"a\""), std::string::npos);
  EXPECT_NE(json.find("\"capture_policy\":\"captured_at_current_statement\""),
            std::string::npos);

  std::string statementJson =
      renderJsonReport(decoded, metadata, statementOnlyReportOptions());
  EXPECT_NE(statementJson.find("\"source_name\":\"a\""), std::string::npos);
  EXPECT_EQ(statementJson.find("\"capture_policy\""), std::string::npos);
  EXPECT_EQ(statementJson.find("\"capture_op_id\""), std::string::npos);
  EXPECT_EQ(statementJson.find("\"producer_op_id\""), std::string::npos);
  EXPECT_EQ(statementJson.find("\"operand_index\""), std::string::npos);
  EXPECT_EQ(statementJson.find("\"anchor_op_id\""), std::string::npos);
  EXPECT_NE(statementJson.find("\"full_value_refs_by_instance\""),
            std::string::npos);
  EXPECT_NE(statementJson.find("\"path\":\"/tmp/flagtree_debugger_artifacts/"
                               "op2_inst42_value.npy\""),
            std::string::npos);
}

TEST(DebuggerDecodeTest, RendersStatementMemoryAccesses) {
  DebugExportedRun run = makeRun(/*capacity=*/16, /*writeIdx=*/13);
  const MemoryEventKind addressKinds[] = {
      MemoryEventKind::FIRST_ADDR,        MemoryEventKind::LAST_ADDR,
      MemoryEventKind::MIN_ADDR,          MemoryEventKind::MAX_ADDR,
      MemoryEventKind::ACTIVE_LANE_COUNT, MemoryEventKind::ADDRESS_SPAN_BYTES,
  };
  uint32_t slot = 0;
  for (MemoryEventKind kind : addressKinds)
    writeObject(run.rawBuffer, slotOffset(slot++),
                makeMemory(/*opId=*/1, /*instance=*/0,
                           kind == MemoryEventKind::ACTIVE_LANE_COUNT ? 4
                           : kind == MemoryEventKind::ADDRESS_SPAN_BYTES
                               ? 16
                               : 0x1000,
                           kind));
  writeObject(run.rawBuffer, slotOffset(slot++),
              makeSummary(/*opId=*/2, /*instance=*/0,
                          CollectorKind::ELEMENT_COUNT, /*value=*/4));
  for (MemoryEventKind kind : addressKinds)
    writeObject(run.rawBuffer, slotOffset(slot++),
                makeMemory(/*opId=*/4, /*instance=*/0,
                           kind == MemoryEventKind::ACTIVE_LANE_COUNT ? 4
                           : kind == MemoryEventKind::ADDRESS_SPAN_BYTES
                               ? 16
                               : 0x1100,
                           kind));

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  KernelDebugMetadata metadata = makeTestKernelDebugMetadata(
      /*scopeCount=*/1, /*trackedOpCount=*/0, /*includeMemoryOp=*/false);

  TrackedOpEntry load = makeTestTrackedOpEntry(/*opId=*/1, /*scopeId=*/1,
                                               /*isMemoryOp=*/true);
  load.statementId = 1001;
  load.statementResultName = "x";
  load.tritonStatement = "x = tl.load(ptr)";
  StatementValueInfo loadResult;
  loadResult.sourceName = "x";
  loadResult.sourceRole = "result";
  loadResult.captureOpId = 1;
  loadResult.capturePolicy = "captured_current_op";
  loadResult.value = load.result;
  load.statementValues.push_back(loadResult);

  TrackedOpEntry add = makeTestTrackedOpEntry(/*opId=*/2);
  add.statementId = 1002;
  add.statementResultName = "z";
  add.tritonStatement = "z = x + 1";
  StatementValueInfo addResult;
  addResult.sourceName = "z";
  addResult.sourceRole = "result";
  addResult.captureOpId = 2;
  addResult.capturePolicy = "captured_current_op";
  addResult.value = add.result;
  add.statementValues.push_back(addResult);

  TrackedOpEntry store = makeTestTrackedOpEntry(/*opId=*/4, /*scopeId=*/1,
                                                /*isMemoryOp=*/true);
  store.opCategory = "store";
  store.role = "store";
  store.mlirOpName = "tt.store";
  store.accessType = "store";
  store.statementId = 1003;
  store.statementResultName = "";
  store.tritonStatement = "tl.store(ptr, z)";
  OperandStaticInfo ptrOperand;
  ptrOperand.operandIndex = 0;
  ptrOperand.operandRole = "ptr";
  OperandStaticInfo valueOperand;
  valueOperand.operandIndex = 1;
  valueOperand.operandRole = "value";
  valueOperand.producerOpId = 2;
  store.operands = {ptrOperand, valueOperand};

  StatementValueInfo ptrValue;
  ptrValue.sourceName = "ptr";
  ptrValue.sourceRole = "operand";
  ptrValue.hasOperandIndex = true;
  ptrValue.operandIndex = 0;
  ptrValue.capturePolicy = "kernel_argument";
  ptrValue.value = store.result;
  store.statementValues.push_back(ptrValue);

  StatementValueInfo zOperand;
  zOperand.sourceName = "z";
  zOperand.sourceRole = "operand";
  zOperand.hasOperandIndex = true;
  zOperand.operandIndex = 1;
  zOperand.producerOpId = 2;
  zOperand.captureOpId = 2;
  zOperand.capturePolicy = "reused_producer";
  zOperand.value = add.result;
  store.statementValues.push_back(zOperand);

  metadata.trackedOpCount = 3;
  metadata.trackedOps = {load, add, store};

  std::string report = renderTextReport(decoded, metadata);
  EXPECT_NE(report.find("[result x]:"), std::string::npos);
  EXPECT_NE(report.find("[result z]:"), std::string::npos);
  EXPECT_NE(report.find("memory_access:"), std::string::npos);
  EXPECT_NE(report.find("address_summary(load from):"), std::string::npos);
  EXPECT_NE(report.find("address_summary(store to):"), std::string::npos);
  EXPECT_EQ(report.find("access_op_id:"), std::string::npos);
  EXPECT_EQ(report.find("access_type:"), std::string::npos);
  EXPECT_NE(report.find("statement: tl.store(ptr, z)"), std::string::npos);
  EXPECT_NE(report.find("<operand z>: [result z]"), std::string::npos);
  EXPECT_EQ(report.find("loaded_from"), std::string::npos);
  EXPECT_EQ(report.find("stored_to"), std::string::npos);
  EXPECT_EQ(report.find("[memory "), std::string::npos);

  std::string json =
      renderJsonReport(decoded, metadata, statementOnlyReportOptions());
  EXPECT_NE(json.find("\"memory_accesses\""), std::string::npos);
  EXPECT_EQ(json.find("\"access_op_id\""), std::string::npos);
  EXPECT_EQ(json.find("\"access_type\""), std::string::npos);
  EXPECT_EQ(json.find("\"address_role\""), std::string::npos);
  EXPECT_EQ(json.find("\"address_ext0\""), std::string::npos);
  EXPECT_EQ(json.find("\"memory_uses\""), std::string::npos);
  EXPECT_EQ(json.find("\"kind\":\"loaded_from\""), std::string::npos);
  EXPECT_EQ(json.find("\"kind\":\"stored_to\""), std::string::npos);
  EXPECT_EQ(json.find("\"label\":\"[memory store:001]\""), std::string::npos);
}

TEST(DebuggerDecodeTest, RendersAtomicAndDescriptorMemoryUses) {
  DebugExportedRun run = makeRun(/*capacity=*/32, /*writeIdx=*/24);
  uint32_t slot = 0;
  writeAddressSummaryRecords(run, slot, /*opId=*/10, /*instance=*/0,
                             /*baseAddr=*/0x1000);
  writeAddressSummaryRecords(run, slot, /*opId=*/11, /*instance=*/0,
                             /*baseAddr=*/0x1040);
  writeAddressSummaryRecords(run, slot, /*opId=*/12, /*instance=*/0,
                             /*baseAddr=*/0x1080);
  writeAddressSummaryRecords(run, slot, /*opId=*/14, /*instance=*/0,
                             /*baseAddr=*/0x10c0);
  ASSERT_EQ(slot, 24u);

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  KernelDebugMetadata metadata = makeTestKernelDebugMetadata(
      /*scopeCount=*/1, /*trackedOpCount=*/0, /*includeMemoryOp=*/false);

  TrackedOpEntry atomicRmw = makeTestTrackedOpEntry(/*opId=*/10, /*scopeId=*/1,
                                                    /*isMemoryOp=*/true);
  atomicRmw.opCategory = "atomic";
  atomicRmw.role = "atomic";
  atomicRmw.mlirOpName = "tt.atomic_rmw";
  atomicRmw.accessType = "atomic_rmw";
  atomicRmw.statementId = 1010;
  atomicRmw.statementResultName = "old";
  atomicRmw.tritonStatement = "old = tl.atomic_add(ptr, val)";
  StatementValueInfo rmwResult;
  rmwResult.sourceName = "old";
  rmwResult.sourceRole = "result";
  rmwResult.captureOpId = 10;
  rmwResult.capturePolicy = "captured_current_op";
  rmwResult.value = atomicRmw.result;
  atomicRmw.statementValues.push_back(rmwResult);

  TrackedOpEntry atomicCas = makeTestTrackedOpEntry(/*opId=*/11, /*scopeId=*/1,
                                                    /*isMemoryOp=*/true);
  atomicCas.opCategory = "atomic";
  atomicCas.role = "atomic";
  atomicCas.mlirOpName = "tt.atomic_cas";
  atomicCas.accessType = "atomic_cas";
  atomicCas.statementId = 1011;
  atomicCas.statementResultName = "cas";
  atomicCas.tritonStatement = "cas = tl.atomic_cas(ptr, cmp, val)";
  StatementValueInfo casResult;
  casResult.sourceName = "cas";
  casResult.sourceRole = "result";
  casResult.captureOpId = 11;
  casResult.capturePolicy = "captured_current_op";
  casResult.value = atomicCas.result;
  atomicCas.statementValues.push_back(casResult);

  TrackedOpEntry descLoad = makeTestTrackedOpEntry(/*opId=*/12, /*scopeId=*/1,
                                                   /*isMemoryOp=*/true);
  descLoad.opCategory = "load";
  descLoad.role = "load";
  descLoad.mlirOpName = "tt.descriptor_load";
  descLoad.accessType = "descriptor_load";
  descLoad.statementId = 1012;
  descLoad.statementResultName = "tile";
  descLoad.tritonStatement = "tile = tl.load(desc)";
  StatementValueInfo loadResult;
  loadResult.sourceName = "tile";
  loadResult.sourceRole = "result";
  loadResult.captureOpId = 12;
  loadResult.capturePolicy = "captured_current_op";
  loadResult.value = descLoad.result;
  descLoad.statementValues.push_back(loadResult);

  TrackedOpEntry producer = makeTestTrackedOpEntry(/*opId=*/13);
  producer.statementId = 1013;
  producer.statementResultName = "z";
  producer.tritonStatement = "z = tile + 1";
  StatementValueInfo producerResult;
  producerResult.sourceName = "z";
  producerResult.sourceRole = "result";
  producerResult.captureOpId = 13;
  producerResult.capturePolicy = "captured_current_op";
  producerResult.value = producer.result;
  producer.statementValues.push_back(producerResult);

  TrackedOpEntry descStore = makeTestTrackedOpEntry(/*opId=*/14, /*scopeId=*/1,
                                                    /*isMemoryOp=*/true);
  descStore.opCategory = "store";
  descStore.role = "store";
  descStore.mlirOpName = "tt.descriptor_store";
  descStore.accessType = "descriptor_store";
  descStore.statementId = 1014;
  descStore.statementResultName = "";
  descStore.tritonStatement = "tl.store(desc, z)";
  OperandStaticInfo descOperand;
  descOperand.operandIndex = 0;
  descOperand.operandRole = "descriptor";
  OperandStaticInfo valueOperand;
  valueOperand.operandIndex = 1;
  valueOperand.operandRole = "value";
  valueOperand.producerOpId = 13;
  descStore.operands = {descOperand, valueOperand};

  StatementValueInfo descValue;
  descValue.sourceName = "desc";
  descValue.sourceRole = "operand";
  descValue.hasOperandIndex = true;
  descValue.operandIndex = 0;
  descValue.capturePolicy = "kernel_argument";
  descValue.value = descStore.result;
  descStore.statementValues.push_back(descValue);

  StatementValueInfo zOperand;
  zOperand.sourceName = "z";
  zOperand.sourceRole = "operand";
  zOperand.hasOperandIndex = true;
  zOperand.operandIndex = 1;
  zOperand.producerOpId = 13;
  zOperand.captureOpId = 13;
  zOperand.capturePolicy = "reused_producer";
  zOperand.value = producer.result;
  descStore.statementValues.push_back(zOperand);

  metadata.trackedOpCount = 5;
  metadata.trackedOps = {atomicRmw, atomicCas, descLoad, producer, descStore};

  std::string report = renderTextReport(decoded, metadata);
  EXPECT_NE(report.find("memory_access:"), std::string::npos);
  EXPECT_NE(report.find("address_summary(load from):"), std::string::npos);
  EXPECT_NE(report.find("address_summary(store to):"), std::string::npos);
  EXPECT_EQ(report.find("access_type:"), std::string::npos);
  EXPECT_EQ(report.find("atomic_at [memory atomic_rmw:001]"),
            std::string::npos);
  EXPECT_EQ(report.find("atomic_at [memory atomic_cas:001]"),
            std::string::npos);
  EXPECT_EQ(report.find("loaded_from [memory descriptor_load:001]"),
            std::string::npos);
  EXPECT_EQ(report.find("stored_to [memory descriptor_store:001]"),
            std::string::npos);
  EXPECT_NE(report.find("<operand z>: [result z]"), std::string::npos);

  std::string json =
      renderJsonReport(decoded, metadata, statementOnlyReportOptions());
  EXPECT_NE(json.find("\"memory_accesses\""), std::string::npos);
  EXPECT_EQ(json.find("\"kind\":\"atomic_at\""), std::string::npos);
  EXPECT_EQ(json.find("\"access_op_id\""), std::string::npos);
  EXPECT_EQ(json.find("\"access_type\""), std::string::npos);
  EXPECT_EQ(json.find("\"address_role\""), std::string::npos);
  EXPECT_EQ(json.find("\"address_ext0\""), std::string::npos);
}

TEST(DebuggerDecodeTest, RendersStatementLevelMultiAddressMemoryAccesses) {
  DebugExportedRun run = makeRun(/*capacity=*/4, /*writeIdx=*/2);
  writeObject(run.rawBuffer, slotOffset(0),
              makeMemory(/*opId=*/20, /*instance=*/0, /*addr=*/0x1000,
                         MemoryEventKind::LAST_ALIGNED_ADDR, /*ext0=*/0));
  writeObject(run.rawBuffer, slotOffset(1),
              makeMemory(/*opId=*/20, /*instance=*/0, /*addr=*/0x1100,
                         MemoryEventKind::LAST_ALIGNED_ADDR, /*ext0=*/1));

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  KernelDebugMetadata metadata = makeTestKernelDebugMetadata(
      /*scopeCount=*/1, /*trackedOpCount=*/0, /*includeMemoryOp=*/false);

  TrackedOpEntry copy = makeTestTrackedOpEntry(/*opId=*/20, /*scopeId=*/1,
                                               /*isMemoryOp=*/true);
  copy.opCategory = "copy";
  copy.role = "copy";
  copy.mlirOpName = "memref.copy";
  copy.accessType = "memref_copy";
  copy.statementId = 1020;
  copy.statementResultName = "";
  copy.tritonStatement = "memref.copy %src, %dst";

  OperandStaticInfo srcOperand;
  srcOperand.operandIndex = 0;
  srcOperand.operandRole = "src";
  OperandStaticInfo dstOperand;
  dstOperand.operandIndex = 1;
  dstOperand.operandRole = "dst";
  copy.operands = {srcOperand, dstOperand};

  StatementValueInfo srcValue;
  srcValue.sourceName = "src";
  srcValue.sourceRole = "operand";
  srcValue.hasOperandIndex = true;
  srcValue.operandIndex = 0;
  srcValue.capturePolicy = "kernel_argument";
  srcValue.value = copy.result;
  copy.statementValues.push_back(srcValue);

  StatementValueInfo dstValue;
  dstValue.sourceName = "dst";
  dstValue.sourceRole = "operand";
  dstValue.hasOperandIndex = true;
  dstValue.operandIndex = 1;
  dstValue.capturePolicy = "kernel_argument";
  dstValue.value = copy.result;
  copy.statementValues.push_back(dstValue);

  metadata.trackedOpCount = 1;
  metadata.trackedOps = {copy};

  std::string report = renderTextReport(decoded, metadata);
  EXPECT_NE(report.find("statement: memref.copy %src, %dst"),
            std::string::npos);
  EXPECT_NE(report.find("memory_accesses:"), std::string::npos);
  EXPECT_NE(report.find("src:"), std::string::npos);
  EXPECT_NE(report.find("dst:"), std::string::npos);
  EXPECT_EQ(report.find("access_type:"), std::string::npos);
  EXPECT_EQ(report.find("copied_from [memory memref_copy.src:001]"),
            std::string::npos);
  EXPECT_EQ(report.find("copied_to [memory memref_copy.dst:001]"),
            std::string::npos);
  EXPECT_EQ(report.find("address_role:"), std::string::npos);
  EXPECT_NE(report.find("[0] kind=LAST_ALIGNED_ADDR addr=0x1000 ext0=0"),
            std::string::npos);
  EXPECT_NE(report.find("[0] kind=LAST_ALIGNED_ADDR addr=0x1100 ext0=1"),
            std::string::npos);

  std::string json =
      renderJsonReport(decoded, metadata, statementOnlyReportOptions());
  EXPECT_NE(json.find("\"memory_accesses\""), std::string::npos);
  EXPECT_EQ(json.find("\"memory_uses\""), std::string::npos);
  EXPECT_NE(json.find("\"label\":\"src\""), std::string::npos);
  EXPECT_NE(json.find("\"label\":\"dst\""), std::string::npos);
  EXPECT_EQ(json.find("\"address_role\""), std::string::npos);
  EXPECT_EQ(json.find("\"address_ext0\""), std::string::npos);
  EXPECT_EQ(json.find("\"access_op_id\""), std::string::npos);
  EXPECT_EQ(json.find("\"access_type\""), std::string::npos);
  EXPECT_NE(json.find("\"memory_events_by_instance\""), std::string::npos);
}

TEST(DebuggerDecodeTest, MarksEmptyRuntimeMetadataInReport) {
  DebugExportedRun run = makeRun(/*capacity=*/1, /*writeIdx=*/0);
  run.runtimeMetadata = DebugRuntimeMetadata{};

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ReportOptions options;
  options.includeDynamicRecords = false;
  options.includeStaticMetadata = true;
  options.includeAggregates = false;

  std::string report =
      renderTextReport(decoded, makeTestKernelDebugMetadata(), options);
  EXPECT_NE(report.find("buffers: 0"), std::string::npos);
  EXPECT_NE(report.find("tensors: 0"), std::string::npos);
  EXPECT_NE(report.find("runtime metadata not captured"), std::string::npos);
}

TEST(DebuggerDecodeTest, RendersZeroAndL2SummaryMetrics) {
  DebugExportedRun run = makeRun(/*capacity=*/2, /*writeIdx=*/2);
  writeObject(run.rawBuffer, slotOffset(0),
              makeSummary(/*opId=*/2, /*instance=*/0, CollectorKind::ZERO_COUNT,
                          /*value=*/3));
  SummaryRecord l2{};
  l2.header.recordKind = RecordKind::SUMMARY;
  l2.header.opId = 2;
  l2.collectorKind = CollectorKind::L2_NORM;
  l2.resultType = ResultType::F32;
  l2.resultData.f32Val = 5.0f;
  writeObject(run.rawBuffer, slotOffset(1), l2);

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  std::string report = renderTextReport(decoded, makeTestKernelDebugMetadata());
  EXPECT_NE(report.find("zero_count: [3 (U64)]"), std::string::npos);
  EXPECT_NE(report.find("l2_norm   : [5 (F32)]"), std::string::npos);
  EXPECT_EQ(report.find("latest.zero_count=3"), std::string::npos);
  EXPECT_EQ(report.find("latest.l2_norm=5"), std::string::npos);
}

TEST(DebuggerDecodeTest, D4GoldenSeparatesDynamicRecords) {
  DebugExportedRun run = makeGoldenReportRun();

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ReportOptions options;
  options.includeDynamicRecords = true;
  options.includeStaticMetadata = false;
  options.includeAggregates = false;

  std::string report =
      renderTextReport(decoded, makeTestKernelDebugMetadata(), options);
  EXPECT_EQ(expectedD4DynamicReportGolden(), report);
}

TEST(DebuggerDecodeTest, RendersAggregatesOnlyWhenRequested) {
  DebugExportedRun run = makeGoldenReportRun();

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ReportOptions options;
  options.includeDynamicRecords = false;
  options.includeStaticMetadata = false;
  options.includeAggregates = true;

  std::string report =
      renderTextReport(decoded, makeTestKernelDebugMetadata(), options);
  EXPECT_NE(report.find(expectedAggregateGolden()), std::string::npos);
}

TEST(DebuggerDecodeTest, D5GoldenShowsMetadataAndRunContext) {
  DebugExportedRun run = makeGoldenReportRun();

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ReportOptions options;
  options.includeDynamicRecords = false;
  options.includeStaticMetadata = true;
  options.includeAggregates = false;

  std::string report =
      renderTextReport(decoded, makeTestKernelDebugMetadata(), options);
  EXPECT_EQ(expectedD5MetadataRunContextGolden(), report);
}

TEST(DebuggerDecodeTest, StaticOpCatalogIsExplicitTextReportOption) {
  DebugExportedRun run = makeGoldenReportRun();

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  ReportOptions options;
  options.includeDynamicRecords = false;
  options.includeStaticMetadata = false;
  options.includeStaticOpCatalog = true;
  options.includeAggregates = false;

  std::string report =
      renderTextReport(decoded, makeTestKernelDebugMetadata(), options);
  EXPECT_NE(report.find("Static Op Catalog"), std::string::npos);
  EXPECT_NE(report.find("tracked_ops: 4"), std::string::npos);
  EXPECT_EQ(report.find("Runtime Inventory"), std::string::npos);
}

TEST(DebuggerDecodeTest, MarksMissingTrackedOpInReport) {
  DebugExportedRun run = makeRun(/*capacity=*/1, /*writeIdx=*/1);
  writeObject(run.rawBuffer, slotOffset(0),
              makeSummary(/*opId=*/999, /*instance=*/1,
                          CollectorKind::NAN_COUNT, /*value=*/1));

  DecodedDebugRun decoded;
  std::string error;
  ASSERT_TRUE(decodeExportedRun(run, decoded, &error)) << error;

  KernelDebugMetadata metadata = makeTestKernelDebugMetadata();
  std::string report = renderTextReport(decoded, metadata);
  EXPECT_NE(report.find("static: <missing>"), std::string::npos);
}
