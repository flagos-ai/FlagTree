#include "Debugger/Runtime/BufferLayout.h"
#include "Debugger/Runtime/TransferEngine.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using namespace mlir::flagtree::debugger;

std::string toString(RecordLevel level) {
  switch (level) {
  case RecordLevel::LEVEL_SUMMARY:
    return "LEVEL_SUMMARY";
  case RecordLevel::LEVEL_TENSOR_FULL:
    return "LEVEL_TENSOR_FULL";
  }
  return "UNKNOWN_RECORD_LEVEL";
}

std::string toString(ExportMode mode) {
  switch (mode) {
  case ExportMode::POST_KERNEL_EXPORT:
    return "POST_KERNEL_EXPORT";
  case ExportMode::STREAMING_EXPORT:
    return "STREAMING_EXPORT";
  }
  return "UNKNOWN_EXPORT_MODE";
}

std::string toString(BackendKind backend) {
  switch (backend) {
  case BackendKind::UNKNOWN:
    return "UNKNOWN";
  case BackendKind::CUDA:
    return "CUDA";
  case BackendKind::HIP:
    return "HIP";
  case BackendKind::MUSA:
    return "MUSA";
  case BackendKind::CANN:
    return "CANN";
  }
  return "UNKNOWN_BACKEND";
}

std::string toString(RecordKind kind) {
  switch (kind) {
  case RecordKind::SUMMARY:
    return "SUMMARY";
  case RecordKind::MEMORY_EVENT:
    return "MEMORY_EVENT";
  case RecordKind::FULL_VALUE:
    return "FULL_VALUE";
  case RecordKind::SUMMARY_COUNT_BUNDLE_U64:
    return "SUMMARY_COUNT_BUNDLE_U64";
  case RecordKind::SUMMARY_VALUE_BUNDLE_F32:
    return "SUMMARY_VALUE_BUNDLE_F32";
  case RecordKind::TIMELINE:
    return "TIMELINE";
  }
  return "UNKNOWN_RECORD_KIND";
}

std::string toString(CollectorKind collector) {
  switch (collector) {
  case CollectorKind::NAN_COUNT:
    return "NAN_COUNT";
  case CollectorKind::INF_COUNT:
    return "INF_COUNT";
  case CollectorKind::ZERO_COUNT:
    return "ZERO_COUNT";
  case CollectorKind::MEAN_FINITE:
    return "MEAN_FINITE";
  case CollectorKind::MIN_FINITE:
    return "MIN_FINITE";
  case CollectorKind::MAX_FINITE:
    return "MAX_FINITE";
  case CollectorKind::L2_NORM:
    return "L2_NORM";
  case CollectorKind::ELEMENT_COUNT:
    return "ELEMENT_COUNT";
  }
  return "UNKNOWN_COLLECTOR";
}

std::string toString(ResultType type) {
  switch (type) {
  case ResultType::U64:
    return "U64";
  case ResultType::F32:
    return "F32";
  case ResultType::F64:
    return "F64";
  }
  return "UNKNOWN_RESULT_TYPE";
}

std::string toString(MemoryEventKind kind) {
  switch (kind) {
  case MemoryEventKind::LAST_ALIGNED_ADDR:
    return "LAST_ALIGNED_ADDR";
  case MemoryEventKind::BASE_ALIGNED_ADDR:
    return "BASE_ALIGNED_ADDR";
  case MemoryEventKind::FIRST_ADDR:
    return "FIRST_ADDR";
  case MemoryEventKind::LAST_ADDR:
    return "LAST_ADDR";
  case MemoryEventKind::MIN_ADDR:
    return "MIN_ADDR";
  case MemoryEventKind::MAX_ADDR:
    return "MAX_ADDR";
  case MemoryEventKind::ACTIVE_LANE_COUNT:
    return "ACTIVE_LANE_COUNT";
  case MemoryEventKind::ADDRESS_SPAN_BYTES:
    return "ADDRESS_SPAN_BYTES";
  }
  return "UNKNOWN_MEMORY_EVENT";
}

void printDivider(const std::string &title) {
  std::cout << "\n==== " << title << " ====\n";
}

void printBufferMeta(const BufferMeta &meta) {
  std::cout << "run_id=" << meta.runId << "\n";
  std::cout << "device_id=" << meta.deviceId << "\n";
  std::cout << "kernel_id=" << meta.kernelId << "\n";
  std::cout << "protocol_ver=" << meta.protocolVer << "\n";
  std::cout << "record_level=" << toString(meta.recordLevel) << "\n";
  std::cout << "export_mode=" << toString(meta.exportMode) << "\n";
  std::cout << "backend=" << toString(meta.backendKind) << "\n";
}

void printBufferPlan(const DebugBufferPlan &plan, const BufferLayout &layout) {
  std::cout << "record_capacity=" << plan.recordCapacity << "\n";
  std::cout << "record_size=" << plan.recordSize << "\n";
  std::cout << "payload_bytes=" << plan.payloadBytes << "\n";
  std::cout << "layout.header_bytes=" << layout.headerBytes << "\n";
  std::cout << "layout.record_area_bytes=" << layout.recordAreaBytes << "\n";
  std::cout << "layout.payload_offset=" << layout.payloadOffset << "\n";
  std::cout << "layout.total_bytes=" << layout.totalBytes << "\n";
}

void printRuntimeMetadata(const DebugRuntimeMetadata &metadata) {
  std::cout << "registered_buffers=" << metadata.buffers.size() << "\n";
  for (const auto &buffer : metadata.buffers) {
    std::cout << "  buffer[id=" << buffer.bufferId
              << "] name=" << buffer.bufferName << " base=0x" << std::hex
              << buffer.baseAddress << std::dec << " size=" << buffer.sizeBytes
              << " alignment=" << buffer.alignment << "\n";
  }

  std::cout << "launch_tensors=" << metadata.tensors.size() << "\n";
  for (const auto &tensor : metadata.tensors) {
    std::cout << "  tensor[arg=" << tensor.argumentIndex
              << "] name=" << tensor.logicalName << " dtype=" << tensor.dtype
              << " layout=" << tensor.layout << " buffer_id=" << tensor.bufferId
              << " base=0x" << std::hex << tensor.baseAddress << std::dec
              << " size=" << tensor.sizeBytes << "\n";
  }
}

void printHeader(const RingBufferHeader &header) {
  std::cout << "write_idx=" << header.writeIdx << "\n";
  std::cout << "capacity=" << header.capacity << "\n";
  std::cout << "overflow_count=" << header.overflowCount << "\n";
  std::cout << "flags=" << header.flags << "\n";
  std::cout << "record_size=" << header.recordSize << "\n";
  std::cout << "payload_offset=" << header.payloadOffset << "\n";
}

void printRecord(const uint8_t *base, const BufferLayout &layout,
                 uint32_t slot) {
  const auto *recordHeader = reinterpret_cast<const RecordHeader *>(
      base + getRecordSlotOffset(layout, slot));
  std::cout << "slot[" << slot
            << "] kind=" << toString(recordHeader->recordKind)
            << " op_id=" << recordHeader->opId
            << " logical_instance_id=" << recordHeader->logicalInstanceId
            << "\n";

  if (recordHeader->recordKind == RecordKind::SUMMARY) {
    const auto *summary = reinterpret_cast<const SummaryRecord *>(recordHeader);
    std::cout << "  collector=" << toString(summary->collectorKind)
              << " result_type=" << toString(summary->resultType);
    if (summary->resultType == ResultType::U64) {
      std::cout << " value=" << summary->resultData.u64Val << "\n";
    } else if (summary->resultType == ResultType::F64) {
      std::cout << " value=" << summary->resultData.f64Val << "\n";
    } else {
      std::cout << " value=" << summary->resultData.f32Val << "\n";
    }
    return;
  }

  if (recordHeader->recordKind == RecordKind::MEMORY_EVENT) {
    const auto *event =
        reinterpret_cast<const MemoryEventRecord *>(recordHeader);
    std::cout << "  event_kind=" << toString(event->eventKind) << " addr=0x"
              << std::hex << event->addr << std::dec << "\n";
  }
}

struct TraceRunOptions {
  uint64_t runId = 42;
  uint32_t deviceId = 0;
  uint32_t kernelId = 9001;
  uint32_t recordCapacity = 8;
  size_t payloadBytes = 64;
};

BufferMeta makeTraceBufferMeta(const TraceRunOptions &options) {
  BufferMeta meta{};
  meta.runId = options.runId;
  meta.deviceId = options.deviceId;
  meta.kernelId = options.kernelId;
  meta.protocolVer = kProtocolVersion;
  meta.recordLevel = RecordLevel::LEVEL_SUMMARY;
  meta.exportMode = ExportMode::POST_KERNEL_EXPORT;
  meta.backendKind = BackendKind::UNKNOWN;
  return meta;
}

DebugBufferPlan makeTraceBufferPlan(const TraceRunOptions &options) {
  DebugBufferPlan plan;
  plan.recordCapacity = options.recordCapacity;
  plan.recordSize = kDefaultRecordSize;
  plan.payloadBytes = options.payloadBytes;
  plan.exportMode = ExportMode::POST_KERNEL_EXPORT;
  return plan;
}

DebugRuntimeMetadata makeTraceRuntimeMetadata() {
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

} // namespace

int main() {
  using namespace mlir::flagtree::debugger;

  TraceRunOptions options;
  auto engine = createTransferEngine();
  BufferMeta meta = makeTraceBufferMeta(options);
  DebugBufferPlan plan = makeTraceBufferPlan(options);
  DebugRuntimeMetadata runtimeMetadata = makeTraceRuntimeMetadata();

  printDivider("1. Input");
  std::cout << "This demo uses the host transfer driver.\n";
  printBufferMeta(meta);
  printBufferPlan(plan,
                  computeBufferLayout(plan.recordCapacity, plan.recordSize,
                                      plan.payloadBytes));
  printRuntimeMetadata(runtimeMetadata);

  printDivider("2. Prepare");
  DebugLaunchContext ctx = engine->prepare(meta, plan, runtimeMetadata);
  std::cout << "device_ctrl_ptr=0x" << std::hex
            << reinterpret_cast<uintptr_t>(ctx.deviceCtrlPtr) << std::dec
            << "\n";
  std::cout << "host_buffer_ptr=0x" << std::hex
            << reinterpret_cast<uintptr_t>(ctx.hostBufferPtr) << std::dec
            << "\n";
  std::cout << "hidden_arg=0x" << std::hex << engine->hiddenArg(ctx) << std::dec
            << "\n";
  std::cout << "buffer_size=" << ctx.bufferSize << "\n";

  printDivider("3. initHeader");
  engine->initHeader(ctx);
  const auto *deviceHeader =
      reinterpret_cast<const RingBufferHeader *>(ctx.deviceCtrlPtr);
  printHeader(*deviceHeader);

  printDivider("4. Simulate Kernel Write");
  auto *mutableHeader = reinterpret_cast<RingBufferHeader *>(ctx.deviceCtrlPtr);

  SummaryRecord record0{};
  record0.header.recordKind = RecordKind::SUMMARY;
  record0.header.opId = 101;
  record0.header.logicalInstanceId = 0;
  record0.collectorKind = CollectorKind::NAN_COUNT;
  record0.resultType = ResultType::U64;
  record0.resultData.u64Val = 2;

  SummaryRecord record1{};
  record1.header.recordKind = RecordKind::SUMMARY;
  record1.header.opId = 102;
  record1.header.logicalInstanceId = 0;
  record1.collectorKind = CollectorKind::MAX_FINITE;
  record1.resultType = ResultType::F64;
  record1.resultData.f64Val = 18.75;

  MemoryEventRecord record2{};
  record2.header.recordKind = RecordKind::MEMORY_EVENT;
  record2.header.opId = 201;
  record2.header.logicalInstanceId = 0;
  record2.addr = 0x1000;
  record2.eventKind = MemoryEventKind::LAST_ALIGNED_ADDR;

  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 0),
              &record0, sizeof(record0));
  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 1),
              &record1, sizeof(record1));
  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 2),
              &record2, sizeof(record2));
  mutableHeader->writeIdx = 3;

  std::cout << "device buffer now contains 3 records:\n";
  printRecord(reinterpret_cast<const uint8_t *>(ctx.deviceCtrlPtr), ctx.layout,
              0);
  printRecord(reinterpret_cast<const uint8_t *>(ctx.deviceCtrlPtr), ctx.layout,
              1);
  printRecord(reinterpret_cast<const uint8_t *>(ctx.deviceCtrlPtr), ctx.layout,
              2);

  printDivider("5. syncExport");
  const auto *hostHeaderBefore =
      reinterpret_cast<const RingBufferHeader *>(ctx.hostBufferPtr);
  std::cout << "host write_idx before export=" << hostHeaderBefore->writeIdx
            << "\n";

  DebugExportedRun run = engine->syncExport(ctx);
  const auto *exportedHeader =
      reinterpret_cast<const RingBufferHeader *>(run.rawBuffer.data());
  std::cout << "exported raw_buffer bytes=" << run.rawBuffer.size() << "\n";
  printHeader(*exportedHeader);
  printRecord(run.rawBuffer.data(), ctx.layout, 0);
  printRecord(run.rawBuffer.data(), ctx.layout, 1);
  printRecord(run.rawBuffer.data(), ctx.layout, 2);

  printDivider("6. asyncExport semantics");
  mutableHeader->writeIdx = 4;
  SummaryRecord record3{};
  record3.header.recordKind = RecordKind::SUMMARY;
  record3.header.opId = 103;
  record3.header.logicalInstanceId = 1;
  record3.collectorKind = CollectorKind::ELEMENT_COUNT;
  record3.resultType = ResultType::U64;
  record3.resultData.u64Val = 128;
  std::memcpy(reinterpret_cast<uint8_t *>(ctx.deviceCtrlPtr) +
                  getRecordSlotOffset(ctx.layout, 3),
              &record3, sizeof(record3));

  engine->asyncExport(ctx);
  const auto *hostHeaderAfterAsyncQueued =
      reinterpret_cast<const RingBufferHeader *>(ctx.hostBufferPtr);
  std::cout << "host write_idx immediately after asyncExport="
            << hostHeaderAfterAsyncQueued->writeIdx << "\n";
  engine->waitAsyncExport(ctx);
  const auto *hostHeaderAfterWait =
      reinterpret_cast<const RingBufferHeader *>(ctx.hostBufferPtr);
  std::cout << "host write_idx after waitAsyncExport="
            << hostHeaderAfterWait->writeIdx << "\n";

  printDivider("7. release");
  engine->release(ctx);
  std::cout << "device_ctrl_ptr="
            << reinterpret_cast<uintptr_t>(ctx.deviceCtrlPtr) << "\n";
  std::cout << "host_buffer_ptr="
            << reinterpret_cast<uintptr_t>(ctx.hostBufferPtr) << "\n";
  std::cout << "buffer_size=" << ctx.bufferSize << "\n";
  std::cout << "record_capacity=" << ctx.recordCapacity << "\n";

  printDivider("8. Summary");
  std::cout << "Input: BufferMeta + DebugBufferPlan + DebugRuntimeMetadata\n";
  std::cout << "Process: prepare -> initHeader -> kernel writes records -> "
               "export -> release\n";
  std::cout
      << "Output: DebugExportedRun { meta, runtimeMetadata, rawBuffer }\n";

  return 0;
}
