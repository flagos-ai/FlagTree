#pragma once

#include <cstddef>
#include <cstdint>

namespace mlir {
namespace flagtree {
namespace debugger {

constexpr uint16_t kProtocolVersion = 2;
constexpr uint32_t kDefaultRecordSize = 32;
constexpr uint32_t kBundleRecordSize = 64;
constexpr uint32_t kDefaultRingBufferHeaderSize = 32;
constexpr uint32_t kInvalidKernelId = 0;
constexpr uint32_t kInvalidScopeId = 0;
constexpr uint32_t kInvalidOpId = 0;

enum class RecordLevel : uint16_t {
  LEVEL_SUMMARY = 1,
  LEVEL_TENSOR_FULL = 2,
};

enum class ExportMode : uint16_t {
  POST_KERNEL_EXPORT = 1,
  STREAMING_EXPORT = 2,
};

enum class BackendKind : uint16_t {
  UNKNOWN = 0,
  CUDA = 1,
  HIP = 2,
  MUSA = 3,
  CANN = 4,
};

enum class RecordKind : uint16_t {
  SUMMARY = 1,
  MEMORY_EVENT = 2,
  FULL_VALUE = 3,
  SUMMARY_COUNT_BUNDLE_U64 = 4,
  SUMMARY_VALUE_BUNDLE_F32 = 5,
  TIMELINE = 6,
};

enum class CollectorKind : uint16_t {
  NAN_COUNT = 1,
  INF_COUNT = 2,
  MEAN_FINITE = 3,
  MIN_FINITE = 4,
  MAX_FINITE = 5,
  ELEMENT_COUNT = 6,
  ZERO_COUNT = 7,
  L2_NORM = 8,
};

enum class ResultType : uint16_t {
  U64 = 1,
  F32 = 2,
  F64 = 3,
};

enum class MemoryEventKind : uint16_t {
  LAST_ALIGNED_ADDR = 1,
  BASE_ALIGNED_ADDR = 2,
  FIRST_ADDR = 3,
  LAST_ADDR = 4,
  MIN_ADDR = 5,
  MAX_ADDR = 6,
  ACTIVE_LANE_COUNT = 7,
  ADDRESS_SPAN_BYTES = 8,
};

struct RecordHeader {
  RecordKind recordKind;
  uint16_t reserved0;
  uint32_t opId;
  uint64_t logicalInstanceId;
};
static_assert(sizeof(RecordHeader) == 16, "RecordHeader must be 16 bytes");

struct SummaryRecord {
  RecordHeader header;
  CollectorKind collectorKind;
  ResultType resultType;
  uint32_t reserved1;
  union {
    uint64_t u64Val;
    double f64Val;
    float f32Val;
  } resultData;
};
static_assert(sizeof(SummaryRecord) == 32, "SummaryRecord must be 32 bytes");

struct MemoryEventRecord {
  RecordHeader header;
  uint64_t addr;
  MemoryEventKind eventKind;
  uint16_t reserved1;
  uint32_t ext0;
};
static_assert(sizeof(MemoryEventRecord) == 32,
              "MemoryEventRecord must be 32 bytes");

struct FullValueRefRecord {
  RecordHeader header;
  uint32_t payloadOffset;
  uint32_t payloadLength;
  uint64_t reserved1;
};
static_assert(sizeof(FullValueRefRecord) == 32,
              "FullValueRefRecord must be 32 bytes");

struct SummaryCountBundleRecord {
  RecordHeader header;
  uint64_t nanCount;
  uint64_t infCount;
  uint64_t zeroCount;
  uint64_t elementCount;
  uint64_t reserved0;
  uint64_t reserved1;
};
static_assert(sizeof(SummaryCountBundleRecord) == 64,
              "SummaryCountBundleRecord must be 64 bytes");

struct SummaryValueBundleRecord {
  RecordHeader header;
  float meanFinite;
  float minFinite;
  float maxFinite;
  float l2Norm;
  uint32_t reserved[8];
};
static_assert(sizeof(SummaryValueBundleRecord) == 64,
              "SummaryValueBundleRecord must be 64 bytes");

struct TimelineRecord {
  RecordHeader header;
  uint64_t startCycle;
  uint64_t endCycle;
  uint64_t durationCycle;
  uint64_t reserved[3];
};
static_assert(sizeof(TimelineRecord) == 64, "TimelineRecord must be 64 bytes");

struct RingBufferHeader {
  uint32_t writeIdx;
  uint32_t capacity;
  uint32_t overflowCount;
  uint32_t flags;
  uint32_t recordSize;
  uint32_t payloadOffset;
  uint32_t reserved0;
  uint32_t reserved1;
};
static_assert(sizeof(RingBufferHeader) == 32,
              "RingBufferHeader must be 32 bytes");

enum RingBufferFlags : uint32_t {
  RB_FLAG_NONE = 0,
  RB_FLAG_OVERFLOW = 1u << 0,
  RB_FLAG_FROZEN = 1u << 1,
};

struct BufferMeta {
  uint64_t runId;
  uint32_t deviceId;
  uint32_t kernelId;
  uint16_t protocolVer;
  RecordLevel recordLevel;
  ExportMode exportMode;
  BackendKind backendKind;
};
static_assert(sizeof(BufferMeta) == 24, "BufferMeta must be 24 bytes");

} // namespace debugger
} // namespace flagtree
} // namespace mlir
