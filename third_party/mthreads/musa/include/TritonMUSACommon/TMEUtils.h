#ifndef TRITONMUSA_COMMON_TME_UTILS_H
#define TRITONMUSA_COMMON_TME_UTILS_H

#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::triton::musa {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

enum class TMECopyKind {
  GlobalToLocal,
  LocalToGlobal,
};

struct ResolvedTMESwizzleConfig {
  TMESwizzleGranularity swizzleGranularity = TMESwizzleGranularity::SG_NONE;
  TMESwizzleStride swizzleStride = TMESwizzleStride::SS_256B;
  TMESwizzleLine swizzleLine = TMESwizzleLine::SL_256B;
};

struct FinalTMECopyConfig {
  DenseI32ArrayAttr blockShape;
  TMESwizzleGranularity swizzleGranularity = TMESwizzleGranularity::SG_NONE;
  TMESwizzleStride swizzleStride = TMESwizzleStride::SS_256B;
  TMESwizzleLine swizzleLine = TMESwizzleLine::SL_256B;
  TMEL2CachePolicy cachePolicy = TMEL2CachePolicy::NEW_ALLOC;
  TMEPersistence innerPersistence = TMEPersistence::CACHE_NORMAL;
  TMEPersistence outerPersistence = TMEPersistence::CACHE_NORMAL;
  std::optional<TMEPrefetchSize> prefetchSize;
};

inline std::optional<int32_t> toI32(int64_t value) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  return static_cast<int32_t>(value);
}

template <typename BuilderT>
Value materializeI32Value(Value value, Location loc, BuilderT &builder) {
  Type ty = value.getType();
  if (ty.isInteger(32))
    return value;
  if (isa<IndexType>(ty))
    return arith::IndexCastOp::create(builder, loc, builder.getI32Type(),
                                      value);
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() > 32)
      return arith::TruncIOp::create(builder, loc, builder.getI32Type(), value);
    if (intTy.getWidth() < 32)
      return arith::ExtSIOp::create(builder, loc, builder.getI32Type(), value);
    return value;
  }
  return {};
}

inline FailureOr<DenseI32ArrayAttr>
materializeTMEBlockShapeAttr(MLIRContext *ctx, ArrayRef<int64_t> dims) {
  if (dims.empty() || dims.size() > 5)
    return failure();

  SmallVector<int32_t> i32Dims;
  i32Dims.reserve(dims.size());
  for (int64_t dimVal : dims) {
    if (dimVal <= 0)
      return failure();
    auto dim = toI32(dimVal);
    if (!dim)
      return failure();
    i32Dims.push_back(*dim);
  }
  return DenseI32ArrayAttr::get(ctx, i32Dims);
}

template <typename BuilderT>
FailureOr<SmallVector<Value>>
materializeTMECoordValues(Location loc, ValueRange indices, BuilderT &builder) {
  unsigned rank = indices.size();
  if (rank == 0 || rank > 5)
    return failure();

  SmallVector<Value> coords;
  coords.reserve(rank);
  for (Value val : indices) {
    Value i32Val = materializeI32Value(val, loc, builder);
    if (!i32Val)
      return failure();
    coords.push_back(i32Val);
  }
  return coords;
}

inline std::optional<int64_t>
inferElemBytesFromMemDescType(ttg::MemDescType ty) {
  int bitWidth = ty.getElementTypeBitWidth();
  if (bitWidth <= 0)
    return std::nullopt;
  return static_cast<int64_t>((bitWidth + 7) / 8);
}

inline std::optional<bool> inferRowMajorFromMemDescType(ttg::MemDescType ty) {
  auto order = ttg::getOrder(ty);
  if (order.empty())
    return std::nullopt;
  return static_cast<int64_t>(order.front() + 1) ==
         static_cast<int64_t>(ty.getShape().size());
}

inline int64_t getSwizzleGranularityBytes(TMESwizzleGranularity value) {
  switch (value) {
  case TMESwizzleGranularity::SG_NONE:
    return 0;
  case TMESwizzleGranularity::SG_16B:
    return 16;
  case TMESwizzleGranularity::SG_32B:
    return 32;
  case TMESwizzleGranularity::SG_64B:
    return 64;
  case TMESwizzleGranularity::SG_128B:
    return 128;
  }
  return 0;
}

inline int64_t getSwizzleStrideBytes(TMESwizzleStride value) {
  switch (value) {
  case TMESwizzleStride::SS_32B:
    return 32;
  case TMESwizzleStride::SS_64B:
    return 64;
  case TMESwizzleStride::SS_128B:
    return 128;
  case TMESwizzleStride::SS_256B:
    return 256;
  }
  return 0;
}

inline int64_t getSwizzleLineBytes(TMESwizzleLine value) {
  switch (value) {
  case TMESwizzleLine::SL_128B:
    return 128;
  case TMESwizzleLine::SL_256B:
    return 256;
  }
  return 0;
}

inline bool isValidTMESwizzleConfig(TMESwizzleGranularity granularity,
                                    TMESwizzleStride stride,
                                    TMESwizzleLine line) {
  int64_t granularityBytes = getSwizzleGranularityBytes(granularity);
  int64_t strideBytes = getSwizzleStrideBytes(stride);
  int64_t lineBytes = getSwizzleLineBytes(line);
  return strideBytes >= granularityBytes && strideBytes <= lineBytes;
}

inline SmallVector<unsigned> getDefaultTMEOrder(unsigned rank) {
  SmallVector<unsigned> order;
  order.reserve(rank);
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i)
    order.push_back(i);
  return order;
}

inline ttg::CGAEncodingAttr getDefaultTMECGALayout(MLIRContext *ctx,
                                                   unsigned rank,
                                                   unsigned numCTAs,
                                                   ArrayRef<unsigned> order) {
  SmallVector<unsigned> ctasPerCGA(rank, 1);
  if (!ctasPerCGA.empty())
    ctasPerCGA.back() = std::max(1u, numCTAs);
  return ttg::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA, ctasPerCGA,
                                               order);
}

inline ttg::CGAEncodingAttr canonicalizeTMECGALayoutForShape(
    MLIRContext *ctx, ttg::CGAEncodingAttr cgaLayout, ArrayRef<int64_t> shape,
    ArrayRef<unsigned> order, unsigned numCTAs) {
  if (!cgaLayout) {
    assert(ctx && "missing MLIR context for default TME CGA layout");
    return getDefaultTMECGALayout(ctx, shape.size(), numCTAs, order);
  }
  if (cgaLayout.getRank() != shape.size())
    return ttng::updateCGALayoutForShape(cgaLayout, shape);
  return cgaLayout;
}

inline ttg::SwizzledSharedEncodingAttr getDefaultTMECompatibleSharedEncoding(
    RankedTensorType tensorTy, ttg::CGAEncodingAttr cgaLayout = {},
    ArrayRef<int64_t> usageShape = {}, unsigned numCTAs = 1) {
  auto *ctx = tensorTy.getContext();
  SmallVector<int64_t> shape =
      usageShape.empty()
          ? SmallVector<int64_t>(tensorTy.getShape())
          : SmallVector<int64_t>(usageShape.begin(), usageShape.end());
  auto order = getDefaultTMEOrder(tensorTy.getRank());
  cgaLayout =
      cgaLayout
          ? canonicalizeTMECGALayoutForShape(ctx, cgaLayout, shape, order,
                                             numCTAs)
          : getDefaultTMECGALayout(ctx, tensorTy.getRank(), numCTAs, order);
  return ttg::SwizzledSharedEncodingAttr::get(ctx, /*vec=*/1, /*perPhase=*/1,
                                              /*maxPhase=*/1, order, cgaLayout);
}

inline std::optional<ttg::SwizzledSharedEncodingAttr>
tryMapTMECompatibleSharedEncodingToCanonicalSwizzled(
    Operation *op, RankedTensorType tensorTy, Attribute encoding,
    ArrayRef<int64_t> usageShape, unsigned numCTAs) {
  auto *ctx = tensorTy.getContext();
  ttg::SwizzledSharedEncodingAttr candidate;
  if (auto swizzled =
          dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(encoding)) {
    candidate = swizzled;
  } else if (auto nvmma =
                 dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(encoding)) {
    if (nvmma.getTransposed() || nvmma.getFp4Padded())
      return std::nullopt;
    SmallVector<unsigned> order = ttg::getOrder(nvmma, usageShape);
    candidate = ttg::SwizzledSharedEncodingAttr::get(
        ctx, nvmma.getVec(), nvmma.getPerPhase(), nvmma.getMaxPhase(), order,
        nvmma.getCGALayout());
  } else {
    return std::nullopt;
  }

  auto updated =
      cast<ttg::SwizzledSharedEncodingAttr>(ttng::updateEncodingForShape(
          op, cast<ttg::SharedEncodingTrait>(candidate), tensorTy));
  auto shape = usageShape.empty() ? tensorTy.getShape() : usageShape;
  auto cgaLayout =
      updated.getCGALayout()
          ? canonicalizeTMECGALayoutForShape(ctx, updated.getCGALayout(), shape,
                                             updated.getOrder(), numCTAs)
          : getDefaultTMECGALayout(ctx, tensorTy.getRank(), numCTAs,
                                   updated.getOrder());
  return ttg::SwizzledSharedEncodingAttr::get(
      ctx, updated.getVec(), updated.getPerPhase(), updated.getMaxPhase(),
      updated.getOrder(), cgaLayout);
}

inline int64_t applyPH1TMESwizzleToByteAddress(int64_t addrBytes,
                                               TMESwizzleGranularity sg,
                                               TMESwizzleStride ss,
                                               TMESwizzleLine sl) {
  if (sg == TMESwizzleGranularity::SG_NONE)
    return addrBytes;
  int64_t sgBytes = getSwizzleGranularityBytes(sg);
  int64_t ssBytes = getSwizzleStrideBytes(ss);
  int64_t slBytes = getSwizzleLineBytes(sl);
  assert(sgBytes > 0 && ssBytes > 0 && slBytes > 0);
  int64_t lineOffset = addrBytes % slBytes;
  int64_t lineId = addrBytes / slBytes;
  int64_t swizzleGroup = ssBytes / sgBytes;
  int64_t swizzleLineId = lineId % swizzleGroup;
  int64_t sectorInLine = lineOffset / sgBytes;
  int64_t offsetInSector = lineOffset % sgBytes;
  int64_t targetSectorInLine = sectorInLine ^ swizzleLineId;
  return lineId * slBytes + targetSectorInLine * sgBytes + offsetInSector;
}

inline SmallVector<int32_t>
decodeLinearOffsetToCoords(int64_t linearOffset, ArrayRef<int64_t> shape,
                           ArrayRef<unsigned> order) {
  SmallVector<int32_t> coords(shape.size(), 0);
  for (unsigned dim : order) {
    int64_t dimSize = shape[dim];
    if (dimSize <= 0)
      return {};
    coords[dim] = static_cast<int32_t>(linearOffset % dimSize);
    linearOffset /= dimSize;
  }
  return coords;
}

struct PH1TMELeadingDimGrouping {
  int64_t numGroups = 1;
  int64_t elemsPerGroupInLeadingDim = 0;
  int64_t elemsPerGroup = 0;
};

inline FailureOr<PH1TMELeadingDimGrouping>
getPH1TMELeadingDimGrouping(ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
                            unsigned elemBytes) {
  if (shape.size() != 2 || order.size() < 2)
    return PH1TMELeadingDimGrouping{};

  int64_t leadingDim = shape[order[0]];
  int64_t leadingWidthBytes = leadingDim * static_cast<int64_t>(elemBytes);
  if (leadingDim <= 0 || leadingWidthBytes <= 0)
    return failure();

  if (leadingWidthBytes <= 256) {
    return PH1TMELeadingDimGrouping{/*numGroups=*/1,
                                    /*elemsPerGroupInLeadingDim=*/leadingDim,
                                    /*elemsPerGroup=*/shape[order[1]] *
                                        leadingDim};
  }

  int64_t maxColsPerGroup = 256 / static_cast<int64_t>(elemBytes);
  if (maxColsPerGroup <= 0)
    return failure();
  int64_t numGroups = (leadingDim + maxColsPerGroup - 1) / maxColsPerGroup;
  if (numGroups <= 0 || (leadingDim % numGroups) != 0)
    return failure();

  int64_t elemsPerGroupInLeadingDim = leadingDim / numGroups;
  int64_t totalElems = 1;
  for (int64_t dim : shape) {
    if (dim <= 0)
      return failure();
    totalElems *= dim;
  }
  if ((totalElems % numGroups) != 0)
    return failure();

  return PH1TMELeadingDimGrouping{numGroups, elemsPerGroupInLeadingDim,
                                  totalElems / numGroups};
}

inline SmallVector<int32_t>
decodePH1TMELinearOffsetToCoords(int64_t linearOffset, ArrayRef<int64_t> shape,
                                 ArrayRef<unsigned> order, unsigned elemBytes) {
  auto grouping = getPH1TMELeadingDimGrouping(shape, order, elemBytes);
  if (failed(grouping))
    return {};
  if (grouping->numGroups == 1)
    return decodeLinearOffsetToCoords(linearOffset, shape, order);

  SmallVector<int32_t> coords(shape.size(), 0);
  int64_t groupId = linearOffset / grouping->elemsPerGroup;
  int64_t offsetInGroup = linearOffset % grouping->elemsPerGroup;
  int64_t row = offsetInGroup / grouping->elemsPerGroupInLeadingDim;
  int64_t colInGroup = offsetInGroup % grouping->elemsPerGroupInLeadingDim;
  int64_t col = groupId * grouping->elemsPerGroupInLeadingDim + colInGroup;

  if (groupId < 0 || row < 0 || col < 0 || row >= shape[order[1]] ||
      col >= shape[order[0]])
    return {};

  coords[order[0]] = static_cast<int32_t>(col);
  coords[order[1]] = static_cast<int32_t>(row);
  return coords;
}

inline FailureOr<int64_t> linearizePH1TMELinearCoords(ArrayRef<int64_t> coords,
                                                      ArrayRef<int64_t> shape,
                                                      ArrayRef<unsigned> order,
                                                      unsigned elemBytes) {
  auto grouping = getPH1TMELeadingDimGrouping(shape, order, elemBytes);
  if (failed(grouping))
    return failure();
  if (coords.size() != shape.size() || order.size() < 2)
    return failure();

  int64_t leading = coords[order[0]];
  int64_t row = coords[order[1]];
  if (leading < 0 || row < 0 || leading >= shape[order[0]] ||
      row >= shape[order[1]])
    return failure();

  if (grouping->numGroups == 1)
    return row * shape[order[0]] + leading;

  int64_t groupId = leading / grouping->elemsPerGroupInLeadingDim;
  int64_t colInGroup = leading % grouping->elemsPerGroupInLeadingDim;
  return groupId * grouping->elemsPerGroup +
         row * grouping->elemsPerGroupInLeadingDim + colInGroup;
}

inline FailureOr<Value> linearizePH1TMELinearCoords(TritonLLVMOpBuilder &b,
                                                    ArrayRef<Value> coords,
                                                    ArrayRef<int64_t> shape,
                                                    ArrayRef<unsigned> order,
                                                    unsigned elemBytes) {
  auto grouping = getPH1TMELeadingDimGrouping(shape, order, elemBytes);
  if (failed(grouping))
    return failure();
  if (coords.size() != shape.size() || order.size() < 2)
    return failure();

  Value leading = coords[order[0]];
  Value row = coords[order[1]];
  if (grouping->numGroups == 1) {
    Value linearOffset = b.add(
        b.mul(row, b.i32_val(static_cast<int32_t>(shape[order[0]]))), leading);
    return linearOffset;
  }

  Value elemsPerGroupInLeadingDim =
      b.i32_val(static_cast<int32_t>(grouping->elemsPerGroupInLeadingDim));
  Value elemsPerGroup =
      b.i32_val(static_cast<int32_t>(grouping->elemsPerGroup));
  Value groupId = b.udiv(leading, elemsPerGroupInLeadingDim);
  Value colInGroup = b.urem(leading, elemsPerGroupInLeadingDim);
  Value linearOffset = b.add(b.add(b.mul(groupId, elemsPerGroup),
                                   b.mul(row, elemsPerGroupInLeadingDim)),
                             colInGroup);
  return linearOffset;
}

inline LinearLayout
combineTMECtaLayoutWithCGA(LinearLayout ctaLayout,
                           ttg::CGAEncodingAttr cgaLayoutAttr,
                           ArrayRef<int64_t> shape) {
  int rank = shape.size();
  assert(ctaLayout.getNumOutDims() == rank);
  assert(cgaLayoutAttr.getCTAOrder().size() == rank);
  MLIRContext *ctx = cgaLayoutAttr.getContext();
  auto outDimNames = triton::standardOutDimNames(ctx, rank);

  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (auto [dim, size] : llvm::zip(outDimNames, shape))
    labeledShape[dim] = size;

  LinearLayout cgaLayout =
      ensureLayoutNotLargerThan(cgaLayoutAttr.getLinearLayout(), labeledShape)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  for (auto dim : ctaLayout.getOutDimNames()) {
    ctaShape[dim] =
        std::max(int64_t{1}, labeledShape[dim] / cgaLayout.getOutDimSize(dim));
  }

  ctaLayout = ensureLayoutNotSmallerThan(ctaLayout, ctaShape);
  ctaLayout = ensureLayoutNotLargerThan(ctaLayout, ctaShape);

  LinearLayout result =
      (ctaLayout * cgaLayout)
          .transposeOuts(triton::standardOutDimNames(ctx, rank));
  return result;
}

inline LinearLayout buildPH1TMESharedLinearLayout(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> allocShape, unsigned elemBytes,
    ArrayRef<unsigned> order, ttg::CGAEncodingAttr cgaLayout,
    TMESwizzleGranularity sg, TMESwizzleStride ss, TMESwizzleLine sl) {
  assert(!shape.empty() && "PH1 TME shared layout requires non-zero rank");
  assert(cgaLayout && "PH1 TME shared layout requires canonical CGA layout");
  auto physicalShape =
      allocShape.empty()
          ? SmallVector<int64_t>(shape.begin(), shape.end())
          : SmallVector<int64_t>(allocShape.take_back(shape.size()).begin(),
                                 allocShape.take_back(shape.size()).end());
  MLIRContext *ctx = cgaLayout.getContext();
  auto shapePerCTA =
      ttg::getShapePerCTA(cgaLayout.getCTASplitNum(), physicalShape);

  int64_t totalElems = 1;
  for (int64_t dim : shapePerCTA)
    totalElems *= dim;

  auto outDimNames = triton::standardOutDimNames(ctx, shape.size());
  SmallVector<std::vector<int32_t>> offsetBases;
  for (int64_t elemOffset = 1; elemOffset < totalElems; elemOffset <<= 1) {
    int64_t logicalByteOffset =
        applyPH1TMESwizzleToByteAddress(elemOffset * elemBytes, sg, ss, sl);
    assert((logicalByteOffset % elemBytes) == 0 &&
           "PH1 TME swizzle must preserve element alignment");
    auto coords = decodePH1TMELinearOffsetToCoords(
        logicalByteOffset / elemBytes, shapePerCTA, order, elemBytes);
    offsetBases.emplace_back(coords.begin(), coords.end());
  }

  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.reserve(shape.size());
  for (auto [dimName, dimSize] : llvm::zip(outDimNames, shapePerCTA))
    outDims.emplace_back(dimName, static_cast<int32_t>(dimSize));

  LinearLayout::BasesT ctaBases;
  ctaBases[StringAttr::get(ctx, "offset")] =
      std::vector<std::vector<int32_t>>(offsetBases.begin(), offsetBases.end());
  LinearLayout ctaLayout(ctaBases, outDims, /*requireSurjective=*/false);
  return combineTMECtaLayoutWithCGA(ctaLayout, cgaLayout, physicalShape);
}

inline std::optional<ResolvedTMESwizzleConfig>
getCanonicalPH1TMESwizzleConfigForGranularityBytes(int64_t sgBytes) {
  switch (sgBytes) {
  case 16:
    return ResolvedTMESwizzleConfig{TMESwizzleGranularity::SG_16B,
                                    TMESwizzleStride::SS_256B,
                                    TMESwizzleLine::SL_256B};
  case 32:
    return ResolvedTMESwizzleConfig{TMESwizzleGranularity::SG_32B,
                                    TMESwizzleStride::SS_256B,
                                    TMESwizzleLine::SL_256B};
  case 64:
    return ResolvedTMESwizzleConfig{TMESwizzleGranularity::SG_64B,
                                    TMESwizzleStride::SS_256B,
                                    TMESwizzleLine::SL_256B};
  default:
    return std::nullopt;
  }
}

inline FailureOr<ResolvedTMESwizzleConfig>
resolveCanonicalPH1TMESharedCarrierConfig(ttg::MemDescType localType) {
  auto swizzled =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(localType.getEncoding());
  auto maybeElemBytes = inferElemBytesFromMemDescType(localType);
  if (!swizzled || !maybeElemBytes || *maybeElemBytes <= 0)
    return failure();

  auto order = swizzled.getOrder();
  if (localType.getShape().size() != 2 || order.size() < 2)
    return failure();

  if (swizzled.getVec() == 1 && swizzled.getPerPhase() == 1 &&
      swizzled.getMaxPhase() == 1) {
    return ResolvedTMESwizzleConfig{TMESwizzleGranularity::SG_NONE,
                                    TMESwizzleStride::SS_256B,
                                    TMESwizzleLine::SL_256B};
  }

  constexpr int64_t kLineBytes = 256;
  int64_t elemBytes = *maybeElemBytes;
  int64_t leadingWidthBytes = localType.getShape()[order.front()] * elemBytes;
  if (leadingWidthBytes <= 0)
    return failure();

  int64_t sgBytes = 0;
  if (swizzled.getPerPhase() > 1) {
    if ((kLineBytes % swizzled.getPerPhase()) != 0 ||
        leadingWidthBytes != (kLineBytes / swizzled.getPerPhase()))
      return failure();
    sgBytes = kLineBytes / swizzled.getMaxPhase();
    int64_t expectedVec = sgBytes / elemBytes;
    if (expectedVec != swizzled.getVec())
      return failure();
  } else {
    if (leadingWidthBytes < kLineBytes || (leadingWidthBytes % kLineBytes) != 0)
      return failure();
    int64_t factor = leadingWidthBytes / kLineBytes;
    if (factor <= 0 || !llvm::isPowerOf2_64(static_cast<uint64_t>(factor)))
      return failure();
    sgBytes = kLineBytes / (factor * swizzled.getMaxPhase());
    int64_t expectedVec = factor * (sgBytes / elemBytes);
    if (sgBytes <= 0 || expectedVec != swizzled.getVec())
      return failure();
  }

  auto config = getCanonicalPH1TMESwizzleConfigForGranularityBytes(sgBytes);
  if (!config)
    return failure();
  return *config;
}

inline LinearLayout
getMUSASharedLinearLayoutOrGeneric(ttg::MemDescType localType) {
  auto maybeElemBytes = inferElemBytesFromMemDescType(localType);
  auto localEncoding =
      dyn_cast<ttg::SharedEncodingTrait>(localType.getEncoding());
  if (!maybeElemBytes || *maybeElemBytes <= 0 || !localEncoding)
    return ttg::toLinearLayout(localType);

  auto order = ttg::getOrder(localType);
  if (order.empty())
    return ttg::toLinearLayout(localType);
  auto cgaLayout = ttg::getCGALayout(localEncoding);
  if (!cgaLayout)
    cgaLayout = getDefaultTMECGALayout(localType.getContext(),
                                       localType.getShape().size(), 1, order);

  auto carrierConfig = resolveCanonicalPH1TMESharedCarrierConfig(localType);
  if (failed(carrierConfig))
    return ttg::toLinearLayout(localType);

  return buildPH1TMESharedLinearLayout(
      localType.getShape(), localType.getAllocShape(),
      static_cast<unsigned>(*maybeElemBytes), order, cgaLayout,
      carrierConfig->swizzleGranularity, carrierConfig->swizzleStride,
      carrierConfig->swizzleLine);
}

inline FailureOr<ResolvedTMESwizzleConfig>
resolveTMESwizzleConfigFromEncoding(ttg::MemDescType localType) {
  auto localEncoding =
      dyn_cast<ttg::SharedEncodingTrait>(localType.getEncoding());
  if (!localEncoding)
    return failure();
  auto maybeElemBytes = inferElemBytesFromMemDescType(localType);
  if (!maybeElemBytes || *maybeElemBytes <= 0)
    return failure();
  auto order = ttg::getOrder(localType);
  if (order.empty())
    return failure();
  auto cgaLayout = ttg::getCGALayout(localEncoding);
  if (!cgaLayout)
    cgaLayout = getDefaultTMECGALayout(localType.getContext(),
                                       localType.getShape().size(), 1, order);

  if (auto carrierConfig = resolveCanonicalPH1TMESharedCarrierConfig(localType);
      succeeded(carrierConfig))
    return *carrierConfig;

  auto targetLayout = getMUSASharedLinearLayoutOrGeneric(localType);
  auto sharedSpace = ttg::SharedMemorySpaceAttr::get(localType.getContext());
  auto canonicalNoSwizzle = ttg::MemDescType::get(
      localType.getShape(), localType.getElementType(),
      ttg::SwizzledSharedEncodingAttr::get(localType.getContext(), 1, 1, 1,
                                           order, cgaLayout),
      sharedSpace, localType.getMutableMemory(), localType.getAllocShape());
  if (getMUSASharedLinearLayoutOrGeneric(canonicalNoSwizzle) == targetLayout) {
    return ResolvedTMESwizzleConfig{TMESwizzleGranularity::SG_NONE,
                                    TMESwizzleStride::SS_256B,
                                    TMESwizzleLine::SL_256B};
  }

  SmallVector<ResolvedTMESwizzleConfig> matches;
  constexpr TMESwizzleGranularity granularityOptions[] = {
      TMESwizzleGranularity::SG_16B, TMESwizzleGranularity::SG_32B,
      TMESwizzleGranularity::SG_64B, TMESwizzleGranularity::SG_128B};
  constexpr TMESwizzleStride strideOptions[] = {
      TMESwizzleStride::SS_32B, TMESwizzleStride::SS_64B,
      TMESwizzleStride::SS_128B, TMESwizzleStride::SS_256B};
  constexpr TMESwizzleLine lineOptions[] = {TMESwizzleLine::SL_128B,
                                            TMESwizzleLine::SL_256B};

  for (TMESwizzleGranularity sg : granularityOptions) {
    for (TMESwizzleStride ss : strideOptions) {
      for (TMESwizzleLine sl : lineOptions) {
        if (!isValidTMESwizzleConfig(sg, ss, sl))
          continue;
        auto candidateLayout = buildPH1TMESharedLinearLayout(
            localType.getShape(), localType.getAllocShape(),
            static_cast<unsigned>(*maybeElemBytes), order, cgaLayout, sg, ss,
            sl);
        if (candidateLayout == targetLayout) {
          matches.push_back(ResolvedTMESwizzleConfig{sg, ss, sl});
          if (matches.size() > 1)
            return failure();
        }
      }
    }
  }
  if (matches.size() != 1)
    return failure();
  return matches.front();
}

inline ttg::SwizzledSharedEncodingAttr
normalizeTMECompatibleSharedEncodingOrDefault(
    Operation *op, RankedTensorType tensorTy, Attribute encoding,
    ttg::CGAEncodingAttr preferredCGA, ArrayRef<int64_t> usageShape,
    ArrayRef<int64_t> allocShape, unsigned numCTAs) {
  auto shape = usageShape.empty() ? tensorTy.getShape() : usageShape;
  auto tryCandidate = tryMapTMECompatibleSharedEncodingToCanonicalSwizzled(
      op, tensorTy, encoding, shape, numCTAs);
  if (tryCandidate) {
    auto sharedSpace = ttg::SharedMemorySpaceAttr::get(tensorTy.getContext());
    auto candidateMemDesc = ttg::MemDescType::get(
        shape, tensorTy.getElementType(), *tryCandidate, sharedSpace,
        /*mutableMemory=*/true, allocShape.empty() ? shape : allocShape);
    if (succeeded(resolveTMESwizzleConfigFromEncoding(candidateMemDesc)))
      return *tryCandidate;
    preferredCGA = preferredCGA ? preferredCGA : tryCandidate->getCGALayout();
  }
  return getDefaultTMECompatibleSharedEncoding(tensorTy, preferredCGA, shape,
                                               numCTAs);
}

inline FailureOr<FinalTMECopyConfig>
resolveFinalTMECopyConfig(ttg::MemDescType localType,
                          ArrayRef<int64_t> descBlockShape, TMECopyKind kind) {
  auto blockShape =
      materializeTMEBlockShapeAttr(localType.getContext(), descBlockShape);
  if (failed(blockShape))
    return failure();

  FinalTMECopyConfig config;
  config.blockShape = *blockShape;
  if (kind == TMECopyKind::GlobalToLocal)
    config.prefetchSize = TMEPrefetchSize::SZ_NONE;

  auto swizzle = resolveTMESwizzleConfigFromEncoding(localType);
  if (failed(swizzle))
    return failure();
  config.swizzleGranularity = swizzle->swizzleGranularity;
  config.swizzleStride = swizzle->swizzleStride;
  config.swizzleLine = swizzle->swizzleLine;

  if (!isValidTMESwizzleConfig(config.swizzleGranularity, config.swizzleStride,
                               config.swizzleLine))
    return failure();
  return config;
}

template <typename BuilderT>
AsyncTMECopyGlobalToLocalOp
createAsyncTMECopyGlobalToLocal(BuilderT &builder, Location loc, Value desc,
                                ValueRange coord, Value barId, Value result,
                                Value pred, const FinalTMECopyConfig &config) {
  OperationState state(loc, AsyncTMECopyGlobalToLocalOp::getOperationName());
  state.addOperands(desc);
  state.addOperands(coord);
  state.addOperands({barId, result, pred});
  state.addAttribute("blockShape", config.blockShape);
  state.addAttribute("swizzleGranularity",
                     TMESwizzleGranularityAttr::get(builder.getContext(),
                                                    config.swizzleGranularity));
  state.addAttribute(
      "swizzleStride",
      TMESwizzleStrideAttr::get(builder.getContext(), config.swizzleStride));
  state.addAttribute(
      "swizzleLine",
      TMESwizzleLineAttr::get(builder.getContext(), config.swizzleLine));
  state.addAttribute(
      "prefetchSize",
      TMEPrefetchSizeAttr::get(builder.getContext(), *config.prefetchSize));
  state.addAttribute(
      "cachePolicy",
      TMEL2CachePolicyAttr::get(builder.getContext(), config.cachePolicy));
  state.addAttribute(
      "innerPersistence",
      TMEPersistenceAttr::get(builder.getContext(), config.innerPersistence));
  state.addAttribute(
      "outerPersistence",
      TMEPersistenceAttr::get(builder.getContext(), config.outerPersistence));
  return cast<AsyncTMECopyGlobalToLocalOp>(builder.create(state));
}

template <typename BuilderT>
AsyncTMECopyLocalToGlobalOp
createAsyncTMECopyLocalToGlobal(BuilderT &builder, Location loc, Value desc,
                                ValueRange coord, Value src, Value pred,
                                const FinalTMECopyConfig &config) {
  OperationState state(loc, AsyncTMECopyLocalToGlobalOp::getOperationName());
  state.addOperands(desc);
  state.addOperands(coord);
  state.addOperands({src, pred});
  state.addAttribute("blockShape", config.blockShape);
  state.addAttribute("swizzleGranularity",
                     TMESwizzleGranularityAttr::get(builder.getContext(),
                                                    config.swizzleGranularity));
  state.addAttribute(
      "swizzleStride",
      TMESwizzleStrideAttr::get(builder.getContext(), config.swizzleStride));
  state.addAttribute(
      "swizzleLine",
      TMESwizzleLineAttr::get(builder.getContext(), config.swizzleLine));
  state.addAttribute(
      "cachePolicy",
      TMEL2CachePolicyAttr::get(builder.getContext(), config.cachePolicy));
  state.addAttribute(
      "innerPersistence",
      TMEPersistenceAttr::get(builder.getContext(), config.innerPersistence));
  state.addAttribute(
      "outerPersistence",
      TMEPersistenceAttr::get(builder.getContext(), config.outerPersistence));
  return cast<AsyncTMECopyLocalToGlobalOp>(builder.create(state));
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_TME_UTILS_H
