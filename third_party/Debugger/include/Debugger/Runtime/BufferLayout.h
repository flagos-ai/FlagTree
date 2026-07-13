#pragma once

#include "Debugger/Common/Protocol.h"

#include <cstddef>
#include <cstdint>

namespace mlir {
namespace flagtree {
namespace debugger {

struct BufferLayout {
  size_t headerBytes = sizeof(RingBufferHeader);
  size_t recordBytes = kDefaultRecordSize;
  size_t recordAreaBytes = 0;
  size_t payloadOffset = sizeof(RingBufferHeader);
  size_t totalBytes = sizeof(RingBufferHeader);
};

BufferLayout computeBufferLayout(uint32_t recordCapacity,
                                 uint32_t recordSize = kDefaultRecordSize,
                                 size_t payloadBytes = 0);

size_t getRecordSlotOffset(const BufferLayout &layout, uint32_t slotIndex);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
