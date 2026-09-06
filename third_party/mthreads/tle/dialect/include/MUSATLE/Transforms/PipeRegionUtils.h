#ifndef MUSATLE_TRANSFORMS_PIPEREGIONUTILS_H
#define MUSATLE_TRANSFORMS_PIPEREGIONUTILS_H

#ifdef __TLE__

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include <cstdint>

namespace mlir::triton::musa_tle {

struct PipeByteInterval {
  int64_t byteOffset = 0;
  int64_t byteSize = 0;
};

struct PipeResolvedRegion {
  Value memdescRoot;
  Value stage;
  PipeByteInterval interval;
  bool exact = false;
};

FailureOr<PipeResolvedRegion> resolvePipeMemDescRegion(Value memdesc);

// Return the static bytes in one field.  A LocalAllocOp whose first shape
// dimension is the pipe capacity is interpreted as a ring and returns the
// bytes in one indexed slot.  An already-indexed descriptor returns the bytes
// of its result type.
FailureOr<int64_t> getStaticPipeFieldBytes(Value field);

bool intervalsOverlap(const PipeByteInterval &, const PipeByteInterval &);

} // namespace mlir::triton::musa_tle

#endif // __TLE__

#endif // MUSATLE_TRANSFORMS_PIPEREGIONUTILS_H
