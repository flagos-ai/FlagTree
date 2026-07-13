#pragma once

#include "Debugger/Common/Protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace mlir {
namespace flagtree {
namespace debugger {

// Host-side helpers that mirror the device write protocol described in
// debugger_design.md. Module C uses them for decoupled testing until module F's
// runtime-managed ring buffer is fully wired in.

enum class RecordWriteStatus : uint8_t {
  WRITTEN = 0,
  OVERFLOW = 1,
  INVALID_ARGUMENT = 2,
};

struct RecordWriteResult {
  RecordWriteStatus status = RecordWriteStatus::INVALID_ARGUMENT;
  uint32_t slot = 0;
  uint32_t writeIdx = 0;
  uint32_t overflowCount = 0;
};

// ─── RecordSink ──────────────────────────────────────────────────────────────
// Abstract record-writing interface for decoupled C-module development.
//
// Standalone development (before F's ring buffer is wired in):
//   createLinearAppendSink() — writes records sequentially into a caller-owned
//   buffer; returns OVERFLOW when the buffer is full.  Use the raw buffer to
//   verify record format and field semantics in unit tests.
//
// Integration (once F's control block is ready):
//   createRingBufferSink() — wraps appendRecordToRingBuffer with full
//   ring-buffer overflow semantics.
class RecordSink {
public:
  virtual ~RecordSink() = default;

  virtual RecordWriteResult writeSummary(const SummaryRecord &record) = 0;
  virtual RecordWriteResult
  writeMemoryEvent(const MemoryEventRecord &record) = 0;
  virtual RecordWriteResult
  writeFullValueRef(const FullValueRefRecord &record) = 0;

  // Number of records that have been successfully written (WRITTEN status).
  virtual uint32_t recordCount() const = 0;
};

uint64_t computeLogicalInstanceId(uint32_t pid0, uint32_t pid1, uint32_t pid2,
                                  uint32_t numPrograms0, uint32_t numPrograms1);

RingBufferHeader makeRingBufferHeader(uint32_t capacity,
                                      uint32_t recordSize = kDefaultRecordSize,
                                      uint32_t payloadOffset = 0);

bool initializeRingBufferStorage(void *ctrlPtr, size_t bufferSize,
                                 uint32_t capacity,
                                 uint32_t recordSize = kDefaultRecordSize,
                                 uint32_t payloadOffset = 0);

RecordWriteResult appendRecordToRingBuffer(void *ctrlPtr, size_t bufferSize,
                                           const void *record,
                                           size_t recordSize);

RecordWriteResult appendSummaryRecord(void *ctrlPtr, size_t bufferSize,
                                      const SummaryRecord &record);
RecordWriteResult appendMemoryEventRecord(void *ctrlPtr, size_t bufferSize,
                                          const MemoryEventRecord &record);
RecordWriteResult appendFullValueRefRecord(void *ctrlPtr, size_t bufferSize,
                                           const FullValueRefRecord &record);

// ─── RecordSink factory functions ────────────────────────────────────────────

// Linear-append sink.  Records are stored sequentially from offset 0 of
// `buffer` (no RingBufferHeader prefix).  Returns OVERFLOW when the buffer
// cannot accommodate another kDefaultRecordSize-sized record.
// The caller retains ownership of `buffer` and can cast its bytes back to
// concrete record structs for test validation.
std::unique_ptr<RecordSink> createLinearAppendSink(void *buffer,
                                                   size_t sizeBytes);

// Ring-buffer sink.  `ctrlPtr` must point to an already-initialised
// RingBufferHeader (call initializeRingBufferStorage first).
// Delegates every write to appendRecordToRingBuffer, preserving the
// overflow-counter / RB_FLAG_OVERFLOW semantics required by F.
std::unique_ptr<RecordSink> createRingBufferSink(void *ctrlPtr,
                                                 size_t bufferSize);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
