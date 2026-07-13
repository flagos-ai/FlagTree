#include "Debugger/Decode/Decoder.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <type_traits>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

std::string enumValue(uint64_t value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename EnumT> std::string enumValue(EnumT value) {
  using UnderlyingT = typename std::underlying_type<EnumT>::type;
  return enumValue(static_cast<uint64_t>(static_cast<UnderlyingT>(value)));
}

bool fail(std::string *errorMessage, const std::string &message) {
  if (errorMessage) {
    *errorMessage = message;
  }
  return false;
}

template <typename T>
bool readObject(const std::vector<uint8_t> &buffer, size_t offset, T &out,
                std::string *errorMessage, const char *what) {
  if (offset > buffer.size() || sizeof(T) > buffer.size() - offset) {
    std::ostringstream os;
    os << "Cannot decode " << what << ": need " << sizeof(T)
       << " bytes at offset " << offset << ", rawBuffer has " << buffer.size()
       << " bytes";
    return fail(errorMessage, os.str());
  }
  std::memcpy(&out, buffer.data() + offset, sizeof(T));
  return true;
}

bool isValidRecordLevel(RecordLevel level) {
  switch (level) {
  case RecordLevel::LEVEL_SUMMARY:
  case RecordLevel::LEVEL_TENSOR_FULL:
    return true;
  }
  return false;
}

bool isValidExportMode(ExportMode mode) {
  switch (mode) {
  case ExportMode::POST_KERNEL_EXPORT:
  case ExportMode::STREAMING_EXPORT:
    return true;
  }
  return false;
}

bool validateMeta(const BufferMeta &meta, std::string *errorMessage) {
  if (meta.protocolVer != 1 && meta.protocolVer != kProtocolVersion) {
    std::ostringstream os;
    os << "Unsupported debug protocol version " << meta.protocolVer
       << "; expected 1 or " << kProtocolVersion;
    return fail(errorMessage, os.str());
  }
  if (!isValidRecordLevel(meta.recordLevel)) {
    return fail(errorMessage,
                "Unsupported recordLevel " + enumValue(meta.recordLevel));
  }
  if (!isValidExportMode(meta.exportMode)) {
    return fail(errorMessage,
                "Unsupported exportMode " + enumValue(meta.exportMode));
  }
  return true;
}

bool validateHeader(const RingBufferHeader &header, size_t rawBufferSize,
                    std::string *errorMessage) {
  if (header.recordSize != kDefaultRecordSize &&
      header.recordSize != kBundleRecordSize) {
    std::ostringstream os;
    os << "Unsupported recordSize " << header.recordSize << "; expected "
       << kDefaultRecordSize << " or " << kBundleRecordSize;
    return fail(errorMessage, os.str());
  }
  if (header.recordSize < sizeof(RecordHeader)) {
    std::ostringstream os;
    os << "Invalid recordSize " << header.recordSize
       << ": smaller than RecordHeader size " << sizeof(RecordHeader);
    return fail(errorMessage, os.str());
  }

  const size_t headerBytes = sizeof(RingBufferHeader);
  const size_t recordBytes = header.recordSize;
  const size_t capacity = header.capacity;
  if (capacity > (static_cast<size_t>(-1) - headerBytes) / recordBytes) {
    return fail(errorMessage,
                "Invalid ring buffer header: capacity * recordSize overflows");
  }

  const size_t recordAreaEnd = headerBytes + capacity * recordBytes;
  if (header.payloadOffset < recordAreaEnd) {
    std::ostringstream os;
    os << "Invalid payloadOffset " << header.payloadOffset
       << ": record area ends at " << recordAreaEnd;
    return fail(errorMessage, os.str());
  }
  if (header.payloadOffset > rawBufferSize) {
    std::ostringstream os;
    os << "Invalid payloadOffset " << header.payloadOffset << ": rawBuffer has "
       << rawBufferSize << " bytes";
    return fail(errorMessage, os.str());
  }
  if (recordAreaEnd > rawBufferSize) {
    std::ostringstream os;
    os << "Raw buffer is truncated: record area ends at " << recordAreaEnd
       << ", rawBuffer has " << rawBufferSize << " bytes";
    return fail(errorMessage, os.str());
  }
  if (header.capacity == 0 && header.writeIdx != 0) {
    return fail(
        errorMessage,
        "Invalid ring buffer header: writeIdx is non-zero with zero capacity");
  }
  return true;
}

bool decodeRecordAt(const std::vector<uint8_t> &buffer, size_t slotOffset,
                    uint32_t slotIndex, DecodedRecord &record,
                    std::string *errorMessage) {
  RecordHeader recordHeader{};
  if (!readObject(buffer, slotOffset, recordHeader, errorMessage,
                  "RecordHeader")) {
    return false;
  }

  switch (recordHeader.recordKind) {
  case RecordKind::SUMMARY: {
    SummaryRecord raw{};
    if (!readObject(buffer, slotOffset, raw, errorMessage, "SummaryRecord")) {
      return false;
    }
    record = DecodedSummaryRecord{raw};
    return true;
  }
  case RecordKind::SUMMARY_COUNT_BUNDLE_U64: {
    SummaryCountBundleRecord raw{};
    if (!readObject(buffer, slotOffset, raw, errorMessage,
                    "SummaryCountBundleRecord")) {
      return false;
    }
    record = DecodedSummaryCountBundleRecord{raw};
    return true;
  }
  case RecordKind::SUMMARY_VALUE_BUNDLE_F32: {
    SummaryValueBundleRecord raw{};
    if (!readObject(buffer, slotOffset, raw, errorMessage,
                    "SummaryValueBundleRecord")) {
      return false;
    }
    record = DecodedSummaryValueBundleRecord{raw};
    return true;
  }
  case RecordKind::MEMORY_EVENT: {
    MemoryEventRecord raw{};
    if (!readObject(buffer, slotOffset, raw, errorMessage,
                    "MemoryEventRecord")) {
      return false;
    }
    record = DecodedMemoryEventRecord{raw};
    return true;
  }
  case RecordKind::FULL_VALUE: {
    FullValueRefRecord raw{};
    if (!readObject(buffer, slotOffset, raw, errorMessage,
                    "FullValueRefRecord")) {
      return false;
    }
    record = DecodedFullValueRefRecord{raw};
    return true;
  }
  case RecordKind::TIMELINE: {
    TimelineRecord raw{};
    if (!readObject(buffer, slotOffset, raw, errorMessage, "TimelineRecord")) {
      return false;
    }
    record = DecodedTimelineRecord{raw};
    return true;
  }
  }

  std::ostringstream os;
  os << "Unknown recordKind " << enumValue(recordHeader.recordKind)
     << " at slot " << slotIndex << " (offset " << slotOffset << ")";
  return fail(errorMessage, os.str());
}

const DebugRecordPlanEntry *
findRecordPlanEntry(const DebugRuntimeMetadata &runtimeMetadata,
                    uint32_t recordIndex) {
  if (recordIndex < runtimeMetadata.recordPlan.size() &&
      runtimeMetadata.recordPlan[recordIndex].recordIndex == recordIndex) {
    return &runtimeMetadata.recordPlan[recordIndex];
  }
  for (const DebugRecordPlanEntry &entry : runtimeMetadata.recordPlan) {
    if (entry.recordIndex == recordIndex)
      return &entry;
  }
  return nullptr;
}

bool isDeterministicCompactRun(const DebugRuntimeMetadata &runtimeMetadata) {
  return runtimeMetadata.recordLayout == "deterministic_compact_v1" &&
         runtimeMetadata.recordsPerInstance != 0 &&
         !runtimeMetadata.recordPlan.empty();
}

bool decodeCompactRecordAt(const std::vector<uint8_t> &buffer,
                           size_t slotOffset, uint32_t slotIndex,
                           const DebugRuntimeMetadata &runtimeMetadata,
                           DecodedRecord &record, std::string *errorMessage) {
  const uint32_t recordIndex = slotIndex % runtimeMetadata.recordsPerInstance;
  const uint64_t logicalInstanceId =
      slotIndex / runtimeMetadata.recordsPerInstance;
  const DebugRecordPlanEntry *plan =
      findRecordPlanEntry(runtimeMetadata, recordIndex);
  if (!plan) {
    std::ostringstream os;
    os << "Missing deterministic compact record plan for record_index "
       << recordIndex << " at slot " << slotIndex;
    return fail(errorMessage, os.str());
  }

  switch (plan->recordKind) {
  case RecordKind::SUMMARY: {
    SummaryRecord raw{};
    raw.header.recordKind = RecordKind::SUMMARY;
    raw.header.opId = plan->opId;
    raw.header.logicalInstanceId = logicalInstanceId;
    raw.collectorKind = plan->collectorKind;
    raw.resultType = plan->resultType;
    if (!readObject(buffer, slotOffset + 24, raw.resultData, errorMessage,
                    "SummaryRecord.resultData")) {
      return false;
    }
    record = DecodedSummaryRecord{raw};
    return true;
  }
  case RecordKind::SUMMARY_COUNT_BUNDLE_U64: {
    SummaryCountBundleRecord raw{};
    raw.header.recordKind = RecordKind::SUMMARY_COUNT_BUNDLE_U64;
    raw.header.opId = plan->opId;
    raw.header.logicalInstanceId = logicalInstanceId;
    if (!readObject(buffer, slotOffset + 16, raw.nanCount, errorMessage,
                    "SummaryCountBundleRecord.nanCount") ||
        !readObject(buffer, slotOffset + 24, raw.infCount, errorMessage,
                    "SummaryCountBundleRecord.infCount") ||
        !readObject(buffer, slotOffset + 32, raw.zeroCount, errorMessage,
                    "SummaryCountBundleRecord.zeroCount") ||
        !readObject(buffer, slotOffset + 40, raw.elementCount, errorMessage,
                    "SummaryCountBundleRecord.elementCount")) {
      return false;
    }
    record = DecodedSummaryCountBundleRecord{raw};
    return true;
  }
  case RecordKind::SUMMARY_VALUE_BUNDLE_F32: {
    SummaryValueBundleRecord raw{};
    raw.header.recordKind = RecordKind::SUMMARY_VALUE_BUNDLE_F32;
    raw.header.opId = plan->opId;
    raw.header.logicalInstanceId = logicalInstanceId;
    if (!readObject(buffer, slotOffset + 16, raw.meanFinite, errorMessage,
                    "SummaryValueBundleRecord.meanFinite") ||
        !readObject(buffer, slotOffset + 20, raw.minFinite, errorMessage,
                    "SummaryValueBundleRecord.minFinite") ||
        !readObject(buffer, slotOffset + 24, raw.maxFinite, errorMessage,
                    "SummaryValueBundleRecord.maxFinite") ||
        !readObject(buffer, slotOffset + 28, raw.l2Norm, errorMessage,
                    "SummaryValueBundleRecord.l2Norm")) {
      return false;
    }
    record = DecodedSummaryValueBundleRecord{raw};
    return true;
  }
  case RecordKind::MEMORY_EVENT: {
    MemoryEventRecord raw{};
    raw.header.recordKind = RecordKind::MEMORY_EVENT;
    raw.header.opId = plan->opId;
    raw.header.logicalInstanceId = logicalInstanceId;
    raw.eventKind = plan->eventKind;
    if (!readObject(buffer, slotOffset + 16, raw.addr, errorMessage,
                    "MemoryEventRecord.addr")) {
      return false;
    }
    readObject(buffer, slotOffset + 28, raw.ext0, nullptr,
               "MemoryEventRecord.ext0");
    record = DecodedMemoryEventRecord{raw};
    return true;
  }
  case RecordKind::FULL_VALUE: {
    FullValueRefRecord raw{};
    raw.header.recordKind = RecordKind::FULL_VALUE;
    raw.header.opId = plan->opId;
    raw.header.logicalInstanceId = logicalInstanceId;
    readObject(buffer, slotOffset + 16, raw.payloadOffset, nullptr,
               "FullValueRefRecord.payloadOffset");
    readObject(buffer, slotOffset + 20, raw.payloadLength, nullptr,
               "FullValueRefRecord.payloadLength");
    record = DecodedFullValueRefRecord{raw};
    return true;
  }
  case RecordKind::TIMELINE: {
    TimelineRecord raw{};
    raw.header.recordKind = RecordKind::TIMELINE;
    raw.header.opId = plan->opId;
    raw.header.logicalInstanceId = logicalInstanceId;
    if (!readObject(buffer, slotOffset + 16, raw.startCycle, errorMessage,
                    "TimelineRecord.startCycle") ||
        !readObject(buffer, slotOffset + 24, raw.endCycle, errorMessage,
                    "TimelineRecord.endCycle") ||
        !readObject(buffer, slotOffset + 32, raw.durationCycle, errorMessage,
                    "TimelineRecord.durationCycle")) {
      return false;
    }
    record = DecodedTimelineRecord{raw};
    return true;
  }
  }

  std::ostringstream os;
  os << "Unsupported deterministic compact record kind "
     << enumValue(plan->recordKind) << " at slot " << slotIndex;
  return fail(errorMessage, os.str());
}

} // namespace

bool decodeExportedRun(const DebugExportedRun &run, DecodedDebugRun &decoded,
                       std::string *errorMessage) {
  decoded = {};
  decoded.meta = run.meta;
  decoded.runtimeMetadata = run.runtimeMetadata;

  if (!validateMeta(run.meta, errorMessage)) {
    decoded = {};
    return false;
  }

  RingBufferHeader header{};
  if (!readObject(run.rawBuffer, 0, header, errorMessage, "RingBufferHeader")) {
    decoded = {};
    return false;
  }
  if (!validateHeader(header, run.rawBuffer.size(), errorMessage)) {
    decoded = {};
    return false;
  }

  decoded.meta = run.meta;
  decoded.header = header;
  decoded.runtimeMetadata = run.runtimeMetadata;

  const uint32_t recordCount = std::min(header.writeIdx, header.capacity);
  const bool compact = isDeterministicCompactRun(run.runtimeMetadata);
  decoded.records.reserve(recordCount);
  for (uint32_t slotIndex = 0; slotIndex < recordCount; ++slotIndex) {
    const size_t slotOffset =
        sizeof(RingBufferHeader) +
        static_cast<size_t>(slotIndex) * header.recordSize;
    if (compact) {
      DecodedRecord record;
      if (!decodeCompactRecordAt(run.rawBuffer, slotOffset, slotIndex,
                                 run.runtimeMetadata, record, errorMessage)) {
        decoded = {};
        return false;
      }
      decoded.records.push_back(record);
      continue;
    }

    RecordHeader recordHeader{};
    if (!readObject(run.rawBuffer, slotOffset, recordHeader, errorMessage,
                    "RecordHeader")) {
      decoded = {};
      return false;
    }
    if (static_cast<uint16_t>(recordHeader.recordKind) == 0 &&
        recordHeader.opId == 0 && recordHeader.logicalInstanceId == 0) {
      continue;
    }

    DecodedRecord record;
    if (!decodeRecordAt(run.rawBuffer, slotOffset, slotIndex, record,
                        errorMessage)) {
      decoded = {};
      return false;
    }
    decoded.records.push_back(record);
  }

  if (errorMessage) {
    errorMessage->clear();
  }
  return true;
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
