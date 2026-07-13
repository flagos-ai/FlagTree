#include "Debugger/Instrumentation/Writer.h"

#include <algorithm>
#include <cstring>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

size_t getRequiredRecordAreaBytes(const RingBufferHeader &header) {
  return sizeof(RingBufferHeader) +
         static_cast<size_t>(header.capacity) * header.recordSize;
}

RecordWriteResult makeResult(RecordWriteStatus status, uint32_t slot,
                             const RingBufferHeader &header) {
  RecordWriteResult result;
  result.status = status;
  result.slot = slot;
  result.writeIdx = header.writeIdx;
  result.overflowCount = header.overflowCount;
  return result;
}

// ─── LinearAppendSink ────────────────────────────────────────────────────────
// Records are laid out sequentially at [slot * kDefaultRecordSize] inside the
// caller-owned buffer.  There is NO RingBufferHeader prefix.  The caller can
// cast the raw bytes back to a concrete record type for validation.
class LinearAppendSink final : public RecordSink {
public:
  LinearAppendSink(void *buffer, size_t sizeBytes)
      : buffer_(static_cast<uint8_t *>(buffer)), sizeBytes_(sizeBytes) {}

  RecordWriteResult writeSummary(const SummaryRecord &record) override {
    return writeRaw(&record, sizeof(record));
  }

  RecordWriteResult writeMemoryEvent(const MemoryEventRecord &record) override {
    return writeRaw(&record, sizeof(record));
  }

  RecordWriteResult
  writeFullValueRef(const FullValueRefRecord &record) override {
    return writeRaw(&record, sizeof(record));
  }

  uint32_t recordCount() const override { return count_; }

private:
  RecordWriteResult writeRaw(const void *data, size_t recordSize) {
    // All protocol records are kDefaultRecordSize bytes; enforce it.
    if (recordSize != kDefaultRecordSize) {
      RecordWriteResult r;
      r.status = RecordWriteStatus::INVALID_ARGUMENT;
      return r;
    }

    const size_t offset = static_cast<size_t>(count_) * kDefaultRecordSize;
    if (offset + kDefaultRecordSize > sizeBytes_) {
      RecordWriteResult r;
      r.status = RecordWriteStatus::OVERFLOW;
      r.slot = count_;
      r.writeIdx = count_;
      r.overflowCount = ++overflowCount_;
      return r;
    }

    std::memcpy(buffer_ + offset, data, kDefaultRecordSize);
    RecordWriteResult r;
    r.status = RecordWriteStatus::WRITTEN;
    r.slot = count_++;
    r.writeIdx = count_;
    r.overflowCount = 0;
    return r;
  }

  uint8_t *buffer_;
  size_t sizeBytes_;
  uint32_t count_ = 0;
  uint32_t overflowCount_ = 0;
};

// ─── RingBufferSink ──────────────────────────────────────────────────────────
// Wraps the existing appendRecord* helpers.  `ctrlPtr` must point to an
// already-initialised RingBufferHeader (call initializeRingBufferStorage
// first). Overflow counter and RB_FLAG_OVERFLOW semantics are preserved.
class RingBufferSink final : public RecordSink {
public:
  RingBufferSink(void *ctrlPtr, size_t bufferSize)
      : ctrlPtr_(ctrlPtr), bufferSize_(bufferSize) {}

  RecordWriteResult writeSummary(const SummaryRecord &record) override {
    auto r = appendSummaryRecord(ctrlPtr_, bufferSize_, record);
    if (r.status == RecordWriteStatus::WRITTEN)
      ++count_;
    return r;
  }

  RecordWriteResult writeMemoryEvent(const MemoryEventRecord &record) override {
    auto r = appendMemoryEventRecord(ctrlPtr_, bufferSize_, record);
    if (r.status == RecordWriteStatus::WRITTEN)
      ++count_;
    return r;
  }

  RecordWriteResult
  writeFullValueRef(const FullValueRefRecord &record) override {
    auto r = appendFullValueRefRecord(ctrlPtr_, bufferSize_, record);
    if (r.status == RecordWriteStatus::WRITTEN)
      ++count_;
    return r;
  }

  uint32_t recordCount() const override { return count_; }

private:
  void *ctrlPtr_;
  size_t bufferSize_;
  uint32_t count_ = 0;
};

} // namespace

uint64_t computeLogicalInstanceId(uint32_t pid0, uint32_t pid1, uint32_t pid2,
                                  uint32_t numPrograms0,
                                  uint32_t numPrograms1) {
  return static_cast<uint64_t>(pid0) +
         static_cast<uint64_t>(pid1) * numPrograms0 +
         static_cast<uint64_t>(pid2) * numPrograms0 * numPrograms1;
}

RingBufferHeader makeRingBufferHeader(uint32_t capacity, uint32_t recordSize,
                                      uint32_t payloadOffset) {
  RingBufferHeader header{};
  header.writeIdx = 0;
  header.capacity = capacity;
  header.overflowCount = 0;
  header.flags = RB_FLAG_NONE;
  header.recordSize = recordSize;
  header.payloadOffset = payloadOffset != 0
                             ? payloadOffset
                             : static_cast<uint32_t>(sizeof(RingBufferHeader) +
                                                     capacity * recordSize);
  header.reserved0 = 0;
  header.reserved1 = 0;
  return header;
}

bool initializeRingBufferStorage(void *ctrlPtr, size_t bufferSize,
                                 uint32_t capacity, uint32_t recordSize,
                                 uint32_t payloadOffset) {
  if (!ctrlPtr || recordSize == 0) {
    return false;
  }

  RingBufferHeader header =
      makeRingBufferHeader(capacity, recordSize, payloadOffset);
  if (bufferSize < getRequiredRecordAreaBytes(header)) {
    return false;
  }

  std::memcpy(ctrlPtr, &header, sizeof(header));
  return true;
}

RecordWriteResult appendRecordToRingBuffer(void *ctrlPtr, size_t bufferSize,
                                           const void *record,
                                           size_t recordSize) {
  if (!ctrlPtr || !record || recordSize == 0 ||
      bufferSize < sizeof(RingBufferHeader)) {
    return {};
  }

  auto *header = reinterpret_cast<RingBufferHeader *>(ctrlPtr);
  if (header->recordSize != recordSize ||
      bufferSize < getRequiredRecordAreaBytes(*header)) {
    return {};
  }

  const uint32_t slot = header->writeIdx++;
  if (slot >= header->capacity) {
    ++header->overflowCount;
    header->flags |= RB_FLAG_OVERFLOW;
    return makeResult(RecordWriteStatus::OVERFLOW, slot, *header);
  }

  auto *recordBase =
      reinterpret_cast<uint8_t *>(ctrlPtr) + sizeof(RingBufferHeader);
  auto *dest = recordBase + static_cast<size_t>(slot) * header->recordSize;
  std::memcpy(dest, record, recordSize);
  return makeResult(RecordWriteStatus::WRITTEN, slot, *header);
}

RecordWriteResult appendSummaryRecord(void *ctrlPtr, size_t bufferSize,
                                      const SummaryRecord &record) {
  return appendRecordToRingBuffer(ctrlPtr, bufferSize, &record, sizeof(record));
}

RecordWriteResult appendMemoryEventRecord(void *ctrlPtr, size_t bufferSize,
                                          const MemoryEventRecord &record) {
  return appendRecordToRingBuffer(ctrlPtr, bufferSize, &record, sizeof(record));
}

RecordWriteResult appendFullValueRefRecord(void *ctrlPtr, size_t bufferSize,
                                           const FullValueRefRecord &record) {
  return appendRecordToRingBuffer(ctrlPtr, bufferSize, &record, sizeof(record));
}

std::unique_ptr<RecordSink> createLinearAppendSink(void *buffer,
                                                   size_t sizeBytes) {
  return std::make_unique<LinearAppendSink>(buffer, sizeBytes);
}

std::unique_ptr<RecordSink> createRingBufferSink(void *ctrlPtr,
                                                 size_t bufferSize) {
  return std::make_unique<RingBufferSink>(ctrlPtr, bufferSize);
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
