#ifdef __TLE__

#include "MUSATLE/Transforms/PipeRegionUtils.h"

#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace mlir::triton::musa_tle {

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;

static Value canonicalizeCapture(Value value) {
  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!partitions)
      break;
    auto ws = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!ws ||
        blockArg.getArgNumber() >= partitions.getExplicitCaptures().size())
      break;
    value = partitions.getExplicitCaptures()[blockArg.getArgNumber()];
  }
  return value;
}

static FailureOr<int64_t> getStaticBytes(ttg::MemDescType type,
                                         bool dropRingDimension) {
  ArrayRef<int64_t> shape = type.getShape();
  if (dropRingDimension) {
    if (shape.empty())
      return failure();
    shape = shape.drop_front();
  }
  if (shape.empty())
    return failure();
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
      return failure();
    elements *= dim;
  }
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 ||
      elements > std::numeric_limits<int64_t>::max() / bitWidth)
    return failure();
  int64_t bits = elements * bitWidth;
  if (bits % 8 != 0)
    return failure();
  int64_t bytes = bits / 8;
  if (bytes <= 0)
    return failure();
  return bytes;
}

static FailureOr<int64_t> getSubsliceByteOffset(ttg::MemDescSubsliceOp op) {
  auto srcType = dyn_cast<ttg::MemDescType>(op.getSrc().getType());
  if (!srcType)
    return failure();
  ArrayRef<int32_t> offsets = op.getOffsets();
  if (offsets.empty())
    return int64_t(0);
  if (offsets.size() != static_cast<size_t>(srcType.getRank()))
    return failure();

  int64_t elementBytes = srcType.getElementType().getIntOrFloatBitWidth() / 8;
  if (elementBytes <= 0)
    return failure();

  // Use the same linear-layout mapping as shared-memory lowering.  It maps a
  // logical subview origin to the physical contiguous offset while preserving
  // swizzle/padding semantics.  Invalid layouts are rejected by the caller's
  // stable contiguous-region diagnostic rather than guessed as row-major.
  LinearLayout layout;
  if (auto padded =
          dyn_cast<ttg::PaddedSharedEncodingAttr>(srcType.getEncoding()))
    layout = padded.getLinearComponent();
  else if (srcType.getEncoding())
    layout = ttg::toLinearLayout(srcType);
  else {
    int64_t linear = 0;
    for (auto [dim, offset] : llvm::enumerate(offsets)) {
      ArrayRef<int64_t> shape = srcType.getShape();
      int64_t stride = 1;
      for (size_t next = dim + 1; next < shape.size(); ++next) {
        if (shape[next] <= 0 ||
            stride > std::numeric_limits<int64_t>::max() / shape[next])
          return failure();
        stride *= shape[next];
      }
      if (offset < 0 || offset > std::numeric_limits<int64_t>::max() / stride)
        return failure();
      linear += static_cast<int64_t>(offset) * stride;
    }
    if (linear > std::numeric_limits<int64_t>::max() / elementBytes)
      return failure();
    return linear * elementBytes;
  }

  MLIRContext *ctx = op->getContext();
  SmallVector<StringAttr> dimNames =
      triton::standardOutDimNames(ctx, srcType.getRank());
  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  logicalOffsets.reserve(offsets.size());
  for (auto &&[dimName, offset] : llvm::zip_equal(dimNames, offsets))
    logicalOffsets.push_back({dimName, offset});

  StringAttr offsetDim = StringAttr::get(ctx, "offset");
  layout = layout.sublayout({offsetDim}, dimNames);
  LinearLayout inverse = layout.invert();
  auto mapped = inverse.apply(logicalOffsets);
  if (mapped.size() != 1 || mapped[0].first != offsetDim ||
      mapped[0].second < 0 ||
      mapped[0].second > std::numeric_limits<int64_t>::max() / elementBytes)
    return failure();
  int64_t byteOffset = static_cast<int64_t>(mapped[0].second) * elementBytes;

  if (auto padded =
          dyn_cast<ttg::PaddedSharedEncodingAttr>(srcType.getEncoding())) {
    int64_t padBytes = 0;
    for (auto &&[interval, padding] :
         llvm::zip_equal(padded.getIntervals(), padded.getPaddings())) {
      if (interval == 0 || padding == 0)
        continue;
      int64_t intervalBytes = static_cast<int64_t>(interval) * elementBytes;
      int64_t paddingBytes = static_cast<int64_t>(padding) * elementBytes;
      if (intervalBytes <= 0 || paddingBytes <= 0 ||
          !llvm::isPowerOf2_64(static_cast<uint64_t>(intervalBytes)) ||
          !llvm::isPowerOf2_64(static_cast<uint64_t>(paddingBytes)))
        return failure();
      unsigned intervalLog2 = llvm::Log2_64(intervalBytes);
      unsigned paddingLog2 = llvm::Log2_64(paddingBytes);
      if (byteOffset > (std::numeric_limits<int64_t>::max() >> paddingLog2))
        return failure();
      padBytes += (byteOffset >> intervalLog2) << paddingLog2;
    }
    if (byteOffset > std::numeric_limits<int64_t>::max() - padBytes)
      return failure();
    byteOffset += padBytes;
  }
  return byteOffset;
}

} // namespace

FailureOr<PipeResolvedRegion> resolvePipeMemDescRegion(Value memdesc) {
  Value current = canonicalizeCapture(memdesc);
  Value stage;
  int64_t byteOffset = 0;
  bool exact = true;
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      if (stage)
        return failure();
      stage = index.getIndex();
      current = canonicalizeCapture(index.getSrc());
      continue;
    }
    if (auto alias = current.getDefiningOp<tle::MemDescAliasOp>()) {
      int64_t offset = alias.getOffsetBytesAttr().getInt();
      if (offset < 0 ||
          byteOffset > std::numeric_limits<int64_t>::max() - offset)
        return failure();
      byteOffset += offset;
      current = canonicalizeCapture(alias.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      FailureOr<int64_t> offset = getSubsliceByteOffset(subslice);
      if (failed(offset) || *offset < 0 ||
          byteOffset > std::numeric_limits<int64_t>::max() - *offset)
        return failure();
      byteOffset += *offset;
      current = canonicalizeCapture(subslice.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(trans.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(current.getType());
      FailureOr<int64_t> srcBytes =
          srcTy ? getStaticBytes(srcTy, false) : FailureOr<int64_t>(failure());
      FailureOr<int64_t> dstBytes =
          dstTy ? getStaticBytes(dstTy, false) : FailureOr<int64_t>(failure());
      if (failed(srcBytes) || failed(dstBytes) || *srcBytes != *dstBytes)
        return failure();
      current = canonicalizeCapture(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(reshape.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(current.getType());
      FailureOr<int64_t> srcBytes =
          srcTy ? getStaticBytes(srcTy, false) : FailureOr<int64_t>(failure());
      FailureOr<int64_t> dstBytes =
          dstTy ? getStaticBytes(dstTy, false) : FailureOr<int64_t>(failure());
      if (failed(srcBytes) || failed(dstBytes) || *srcBytes != *dstBytes)
        return failure();
      current = canonicalizeCapture(reshape.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(reinterpret.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(current.getType());
      FailureOr<int64_t> srcBytes =
          srcTy ? getStaticBytes(srcTy, false) : FailureOr<int64_t>(failure());
      FailureOr<int64_t> dstBytes =
          dstTy ? getStaticBytes(dstTy, false) : FailureOr<int64_t>(failure());
      if (failed(srcBytes) || failed(dstBytes) || *srcBytes != *dstBytes)
        return failure();
      current = canonicalizeCapture(reinterpret.getSrc());
      continue;
    }
    if (auto wgmma = current.getDefiningOp<tle::MemDescWGMMAViewOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(wgmma.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(current.getType());
      FailureOr<int64_t> srcBytes =
          srcTy ? getStaticBytes(srcTy, false) : FailureOr<int64_t>(failure());
      FailureOr<int64_t> dstBytes =
          dstTy ? getStaticBytes(dstTy, false) : FailureOr<int64_t>(failure());
      if (failed(srcBytes) || failed(dstBytes) || *srcBytes != *dstBytes)
        return failure();
      current = canonicalizeCapture(wgmma.getSrc());
      continue;
    }
    break;
  }

  if (!stage)
    return failure();
  auto type = dyn_cast<ttg::MemDescType>(memdesc.getType());
  if (!type)
    return failure();
  FailureOr<int64_t> byteSize = getStaticBytes(type, false);
  if (failed(byteSize) || byteOffset < 0 ||
      byteOffset > std::numeric_limits<int64_t>::max() - *byteSize)
    return failure();
  return PipeResolvedRegion{current, stage, {byteOffset, *byteSize}, exact};
}

FailureOr<int64_t> getStaticPipeFieldBytes(Value field) {
  auto type = dyn_cast<ttg::MemDescType>(field.getType());
  if (!type)
    return failure();
  bool isRingAllocation =
      isa_and_nonnull<ttg::LocalAllocOp>(field.getDefiningOp());
  return getStaticBytes(type, isRingAllocation);
}

bool intervalsOverlap(const PipeByteInterval &lhs,
                      const PipeByteInterval &rhs) {
  if (lhs.byteOffset < 0 || rhs.byteOffset < 0 || lhs.byteSize <= 0 ||
      rhs.byteSize <= 0)
    return true;
  if (lhs.byteOffset > std::numeric_limits<int64_t>::max() - lhs.byteSize ||
      rhs.byteOffset > std::numeric_limits<int64_t>::max() - rhs.byteSize)
    return true;
  return lhs.byteOffset < rhs.byteOffset + rhs.byteSize &&
         rhs.byteOffset < lhs.byteOffset + lhs.byteSize;
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
