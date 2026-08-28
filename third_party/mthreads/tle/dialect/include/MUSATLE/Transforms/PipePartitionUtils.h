#ifndef MUSATLE_TRANSFORMS_PIPEPARTITIONUTILS_H
#define MUSATLE_TRANSFORMS_PIPEPARTITIONUTILS_H

#ifdef __TLE__

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <cstdint>
#include <optional>

namespace mlir::triton::musa_tle {

enum class PipePartitionKind {
  CTA,
  WarpSpecializeDefault,
  WarpSpecializeWorker,
};

struct PipeStaticPartitionInfo {
  triton::gpu::WarpSpecializeOp owner;
  Region *region = nullptr;
  PipePartitionKind kind = PipePartitionKind::CTA;
  std::optional<unsigned> workerIndex;
  unsigned partitionIndex = 0;
  int32_t warpBegin = 0;
  int32_t warpCount = 0;
};

FailureOr<std::optional<PipeStaticPartitionInfo>>
resolvePipeStaticPartition(Operation *op);

FailureOr<int32_t> getPipePartitionStartThread(Operation *op);

bool samePipeStaticPartition(const PipeStaticPartitionInfo &lhs,
                             const PipeStaticPartitionInfo &rhs);

} // namespace mlir::triton::musa_tle

#endif // __TLE__

#endif // MUSATLE_TRANSFORMS_PIPEPARTITIONUTILS_H
