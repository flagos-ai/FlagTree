// MIT License
//
// Copyright (c) 2026 The FlagOS Contributors

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <limits>
#include <set>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLECOALESCEBARRIERINITIALIZATION
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

FailureOr<int64_t> resolveStaticAllocationOffset(Value descriptor,
                                                 Operation *diagnosticOp) {
  Operation *definition = descriptor.getDefiningOp();
  if (!definition)
    return diagnosticOp->emitOpError()
           << "requires a statically allocated barrier descriptor";

  if (auto offsetAttr =
          definition->getAttrOfType<IntegerAttr>("allocation.offset"))
    return offsetAttr.getInt();

  if (auto index = dyn_cast<gpu::MemDescIndexOp>(definition)) {
    FailureOr<int64_t> sourceOffset =
        resolveStaticAllocationOffset(index.getSrc(), diagnosticOp);
    if (failed(sourceOffset))
      return failure();

    APInt indexValue;
    if (!matchPattern(index.getIndex(), m_ConstantInt(&indexValue)))
      return diagnosticOp->emitOpError()
             << "requires a constant memdesc_index for barrier allocation";
    int64_t signedIndex = indexValue.getSExtValue();
    if (signedIndex < 0)
      return diagnosticOp->emitOpError()
             << "requires a non-negative memdesc_index; got " << signedIndex;

    gpu::MemDescType resultType = index.getResult().getType();
    if (isa<gpu::PaddedSharedEncodingAttr>(resultType.getEncoding()))
      return diagnosticOp->emitOpError()
             << "does not support padded shared encodings for grouped barrier "
                "initialization";

    int64_t strideElements = 1;
    for (int64_t extent : gpu::getAllocationShapePerCTA(
             resultType.getEncoding(), resultType.getShape())) {
      if (extent <= 0 ||
          strideElements > std::numeric_limits<int64_t>::max() / extent)
        return diagnosticOp->emitOpError()
               << "has an invalid or overflowing memdesc_index stride";
      strideElements *= extent;
    }
    int64_t elementBits = resultType.getElementTypeBitWidth();
    if (elementBits <= 0 ||
        strideElements > std::numeric_limits<int64_t>::max() / elementBits)
      return diagnosticOp->emitOpError()
             << "has an overflowing memdesc_index element stride";
    int64_t strideBits = strideElements * elementBits;
    if (strideBits % 8 != 0)
      return diagnosticOp->emitOpError()
             << "has a memdesc_index stride that is not byte aligned";
    int64_t strideBytes = strideBits / 8;
    if (signedIndex != 0 &&
        strideBytes > std::numeric_limits<int64_t>::max() / signedIndex)
      return diagnosticOp->emitOpError()
             << "has an overflowing memdesc_index byte offset";
    int64_t viewOffset = signedIndex * strideBytes;
    if (*sourceOffset > std::numeric_limits<int64_t>::max() - viewOffset)
      return diagnosticOp->emitOpError()
             << "has an overflowing static barrier allocation offset";
    return *sourceOffset + viewOffset;
  }

  if (auto alias = dyn_cast<tle::MemDescAliasOp>(definition)) {
    FailureOr<int64_t> sourceOffset =
        resolveStaticAllocationOffset(alias.getSrc(), diagnosticOp);
    if (failed(sourceOffset))
      return failure();
    int64_t aliasOffset = alias.getOffsetBytesAttr().getInt();
    if (aliasOffset < 0 ||
        *sourceOffset > std::numeric_limits<int64_t>::max() - aliasOffset)
      return diagnosticOp->emitOpError()
             << "has an invalid memdesc_alias barrier allocation offset";
    return *sourceOffset + aliasOffset;
  }

  return diagnosticOp->emitOpError()
         << "cannot resolve a static barrier allocation offset through "
         << definition->getName();
}

LogicalResult coalesceBarrierInitialization(triton::FuncOp function,
                                            int32_t minimumBarrierCount) {
  if (function.getBody().empty())
    return success();

  Block &entry = function.getBody().front();
  SmallVector<triton::nvidia_gpu::InitBarrierOp> barriers;
  bool hasExistingGroup = false;
  for (Operation &operation : entry) {
    if (auto barrier =
            dyn_cast<triton::nvidia_gpu::InitBarrierOp>(&operation))
      barriers.push_back(barrier);
    hasExistingGroup |= isa<tle::InitBarrierGroupOp>(operation);
  }

  if (barriers.empty())
    return success();
  if (hasExistingGroup)
    return function.emitOpError()
           << "contains both scalar and grouped barrier initialization";
  if (barriers.size() < static_cast<size_t>(minimumBarrierCount))
    return success();

  ModuleOp module = function->getParentOfType<ModuleOp>();
  int32_t workerCount =
      gpu::lookupNumWarps(function) *
      gpu::TritonGPUDialect::getThreadsPerWarp(module);
  if (workerCount <= 0)
    return function.emitOpError()
           << "computed a non-positive barrier initialization worker count";

  SmallVector<std::pair<int32_t, int32_t>> records;
  records.reserve(barriers.size());
  std::set<int32_t> uniqueOffsets;
  for (triton::nvidia_gpu::InitBarrierOp barrier : barriers) {
    FailureOr<int64_t> resolvedOffset =
        resolveStaticAllocationOffset(barrier.getAlloc(), barrier);
    if (failed(resolvedOffset))
      return failure();
    int64_t offset = *resolvedOffset;
    if (offset < 0 || offset > std::numeric_limits<int32_t>::max() ||
        offset % 8 != 0)
      return barrier.emitOpError()
             << "requires a non-negative, 8-byte-aligned i32 allocation "
                "offset; got "
             << offset;
    if (!uniqueOffsets.insert(static_cast<int32_t>(offset)).second)
      return barrier.emitOpError()
             << "shares allocation offset " << offset
             << " with another initialized barrier";

    int64_t count = barrier.getCount();
    if (count <= 0 || count > std::numeric_limits<int32_t>::max())
      return barrier.emitOpError()
             << "requires a positive i32 participant count; got " << count;
    records.emplace_back(static_cast<int32_t>(count),
                         static_cast<int32_t>(offset));
  }

  llvm::sort(records);
  SmallVector<int32_t> offsets;
  SmallVector<int32_t> counts;
  offsets.reserve(records.size());
  counts.reserve(records.size());
  for (auto [count, offset] : records) {
    offsets.push_back(offset);
    counts.push_back(count);
  }

  OpBuilder builder(barriers.front());
  builder.create<tle::InitBarrierGroupOp>(
      barriers.front().getLoc(), builder.getDenseI32ArrayAttr(offsets),
      builder.getDenseI32ArrayAttr(counts),
      builder.getI32IntegerAttr(workerCount));
  for (triton::nvidia_gpu::InitBarrierOp barrier : barriers)
    barrier.erase();
  return success();
}

class CoalesceBarrierInitializationPass
    : public impl::TritonTleCoalesceBarrierInitializationBase<
          CoalesceBarrierInitializationPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (minBarrierCount <= 0) {
      getOperation().emitOpError()
          << "min-barrier-count must be positive, got " << minBarrierCount;
      return signalPassFailure();
    }

    SmallVector<triton::FuncOp> functions;
    getOperation().walk(
        [&](triton::FuncOp function) { functions.push_back(function); });
    for (triton::FuncOp function : functions) {
      if (failed(coalesceBarrierInitialization(function, minBarrierCount))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
