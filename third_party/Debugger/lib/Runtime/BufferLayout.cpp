#include "Debugger/Runtime/BufferLayout.h"

namespace mlir {
namespace flagtree {
namespace debugger {

BufferLayout computeBufferLayout(uint32_t recordCapacity, uint32_t recordSize,
                                 size_t payloadBytes) {
  BufferLayout layout;
  layout.headerBytes = sizeof(RingBufferHeader);
  layout.recordBytes = recordSize;
  layout.recordAreaBytes = static_cast<size_t>(recordCapacity) * recordSize;
  layout.payloadOffset = layout.headerBytes + layout.recordAreaBytes;
  layout.totalBytes = layout.payloadOffset + payloadBytes;
  return layout;
}

size_t getRecordSlotOffset(const BufferLayout &layout, uint32_t slotIndex) {
  return layout.headerBytes +
         static_cast<size_t>(slotIndex) * layout.recordBytes;
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
