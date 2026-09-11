#include "Gluon/Targets/GluonC500Layout.h"

#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/GluonC500PhysicalRegisterSlice.h"

#include "mlir/IR/Block.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>

#define DEBUG_TYPE "metax-gluon-c500-register-slice"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gpu::metax::gluon::c500 {

using ::mlir::triton::gpu::metax::gluon::decodeRegisterSliceIndices;

static bool haveCompatibleRegisterSliceEncodingClasses(Attribute fullEncoding,
                                                       Attribute subEncoding) {
  return (isa<ttg::BlockedEncodingAttr>(fullEncoding) &&
          isa<ttg::BlockedEncodingAttr>(subEncoding)) ||
         (isa<ttg::DotOperandEncodingAttr>(fullEncoding) &&
          isa<ttg::DotOperandEncodingAttr>(subEncoding)) ||
         (isa<ttg::MACAMmaEncodingAttr>(fullEncoding) &&
          isa<ttg::MACAMmaEncodingAttr>(subEncoding));
}

static FailureOr<int64_t>
linearizeRegisterSliceIndex(ArrayRef<unsigned> coordinate,
                            ArrayRef<unsigned> shape,
                            ArrayRef<unsigned> order) {
  if (coordinate.size() != 2 || shape.size() != 2 || order.size() != 2 ||
      order[0] >= 2 || order[1] >= 2 || order[0] == order[1])
    return failure();
  uint64_t minorCoordinate = coordinate[order[0]];
  uint64_t majorCoordinate = coordinate[order[1]];
  uint64_t minorExtent = shape[order[0]];
  if (minorCoordinate >= minorExtent ||
      majorCoordinate >= shape[order[1]] ||
      (majorCoordinate != 0 &&
       minorExtent > std::numeric_limits<uint64_t>::max() / majorCoordinate))
    return failure();
  uint64_t linear = majorCoordinate * minorExtent + minorCoordinate;
  if (linear > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return failure();
  return static_cast<int64_t>(linear);
}

static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
deriveC500RegisterSliceIndexAttributes(
    RankedTensorType fullType, RankedTensorType subType,
    ArrayRef<int64_t> logicalSubTensorIndex) {
  if (fullType.getRank() != 2 || subType.getRank() != 2 ||
      logicalSubTensorIndex.size() != 2)
    return failure();

  Attribute fullEncoding = unwrapNoVerifyEncoding(fullType.getEncoding());
  Attribute subEncoding = unwrapNoVerifyEncoding(subType.getEncoding());
  SmallVector<unsigned, 2> fullSizePerThread;
  SmallVector<unsigned, 2> subSizePerThread;
  SmallVector<unsigned, 2> order;
  if (auto full = dyn_cast<ttg::BlockedEncodingAttr>(fullEncoding)) {
    auto sub = dyn_cast<ttg::BlockedEncodingAttr>(subEncoding);
    if (!sub)
      return failure();
    llvm::append_range(fullSizePerThread, full.getSizePerThread());
    llvm::append_range(subSizePerThread, sub.getSizePerThread());
    llvm::append_range(order, full.getOrder());
  } else if (auto full = dyn_cast<ttg::MACAMmaEncodingAttr>(fullEncoding)) {
    auto sub = dyn_cast<ttg::MACAMmaEncodingAttr>(subEncoding);
    if (!sub || full.getElementsMNK().size() != 3 ||
        sub.getElementsMNK().size() != 3)
      return failure();
    fullSizePerThread = {full.getElementsMNK()[0],
                         full.getElementsMNK()[1]};
    subSizePerThread = {sub.getElementsMNK()[0],
                        sub.getElementsMNK()[1]};
    order = full.getColMajor() ? SmallVector<unsigned, 2>{0, 1}
                               : SmallVector<unsigned, 2>{1, 0};
  } else if (auto full =
                 dyn_cast<ttg::DotOperandEncodingAttr>(fullEncoding)) {
    auto sub = dyn_cast<ttg::DotOperandEncodingAttr>(subEncoding);
    if (!sub || full.getOpIdx() != sub.getOpIdx())
      return failure();
    llvm::append_range(fullSizePerThread,
                       ttg::toLinearEncoding(fullType).getSizePerThread());
    llvm::append_range(subSizePerThread,
                       ttg::toLinearEncoding(subType).getSizePerThread());
    order = full.getOpIdx() == 0 ? SmallVector<unsigned, 2>{1, 0}
                                 : SmallVector<unsigned, 2>{0, 1};
  } else {
    return failure();
  }
  if (fullSizePerThread.size() != 2 || subSizePerThread.size() != 2 ||
      order.size() != 2 || llvm::is_contained(fullSizePerThread, 0u) ||
      llvm::is_contained(subSizePerThread, 0u))
    return failure();

  SmallVector<unsigned> shapePerCTA = ttg::getShapePerCTATile(fullType);
  if (shapePerCTA.size() != 2 || llvm::is_contained(shapePerCTA, 0u))
    return failure();
  SmallVector<unsigned, 2> numCTAs;
  for (auto [fullDim, ctaDim] :
       llvm::zip(fullType.getShape(), shapePerCTA)) {
    if (fullDim <= 0 ||
        static_cast<uint64_t>(fullDim) >
            std::numeric_limits<unsigned>::max() ||
        fullDim % ctaDim != 0)
      return failure();
    numCTAs.push_back(static_cast<unsigned>(fullDim / ctaDim));
  }

  SmallVector<SmallVector<unsigned, 4>, 2> ctaByDim(2);
  SmallVector<SmallVector<unsigned, 4>, 2> elementByDim(2);
  constexpr uint64_t kMaxPlannedIndexElements = 1ull << 20;
  for (unsigned dim = 0; dim < 2; ++dim) {
    if (logicalSubTensorIndex[dim] < 0)
      return failure();
    uint64_t logicalIndex = logicalSubTensorIndex[dim];
    if (fullSizePerThread[dim] == subSizePerThread[dim]) {
      int64_t subDim = subType.getShape()[dim];
      if (subDim <= 0 || subDim % shapePerCTA[dim] != 0)
        return failure();
      uint64_t subCtas = static_cast<uint64_t>(subDim) / shapePerCTA[dim];
      if (subCtas == 0 || subCtas > numCTAs[dim] ||
          subCtas > kMaxPlannedIndexElements ||
          fullSizePerThread[dim] > kMaxPlannedIndexElements ||
          logicalIndex > numCTAs[dim] / subCtas ||
          logicalIndex * subCtas > numCTAs[dim] - subCtas)
        return failure();
      unsigned first = static_cast<unsigned>(logicalIndex * subCtas);
      for (unsigned i = 0; i < subCtas; ++i)
        ctaByDim[dim].push_back(first + i);
      for (unsigned i = 0; i < fullSizePerThread[dim]; ++i)
        elementByDim[dim].push_back(i);
    } else {
      if (subSizePerThread[dim] > fullSizePerThread[dim] ||
          numCTAs[dim] > kMaxPlannedIndexElements ||
          subSizePerThread[dim] > kMaxPlannedIndexElements ||
          logicalIndex >
              static_cast<uint64_t>(fullSizePerThread[dim]) /
                  subSizePerThread[dim] ||
          logicalIndex * subSizePerThread[dim] >
              fullSizePerThread[dim] - subSizePerThread[dim])
        return failure();
      for (unsigned i = 0; i < numCTAs[dim]; ++i)
        ctaByDim[dim].push_back(i);
      unsigned first =
          static_cast<unsigned>(logicalIndex * subSizePerThread[dim]);
      for (unsigned i = 0; i < subSizePerThread[dim]; ++i)
        elementByDim[dim].push_back(first + i);
    }
  }

  SmallVector<int64_t> ctaIndices;
  SmallVector<int64_t> elementIndices;
  auto hasBoundedProduct = [&](size_t lhs, size_t rhs) {
    return lhs == 0 ||
           (rhs <= kMaxPlannedIndexElements / lhs &&
            lhs * rhs <= kMaxPlannedIndexElements);
  };
  if (!hasBoundedProduct(ctaByDim[0].size(), ctaByDim[1].size()) ||
      !hasBoundedProduct(elementByDim[0].size(),
                         elementByDim[1].size()))
    return failure();
  for (unsigned first : ctaByDim[0]) {
    for (unsigned second : ctaByDim[1]) {
      SmallVector<unsigned, 2> coordinate{first, second};
      FailureOr<int64_t> linear = linearizeRegisterSliceIndex(
          coordinate, numCTAs, order);
      if (failed(linear))
        return failure();
      ctaIndices.push_back(*linear);
    }
  }
  for (unsigned first : elementByDim[0]) {
    for (unsigned second : elementByDim[1]) {
      SmallVector<unsigned, 2> coordinate{first, second};
      FailureOr<int64_t> linear = linearizeRegisterSliceIndex(
          coordinate, fullSizePerThread, order);
      if (failed(linear))
        return failure();
      elementIndices.push_back(*linear);
    }
  }
  llvm::sort(ctaIndices);
  llvm::sort(elementIndices);
  return std::make_pair(std::move(ctaIndices), std::move(elementIndices));
}

static FailureOr<C500RegisterSlicePlan>
buildPhysicalC500RegisterSlicePlan(RankedTensorType fullType,
                                   RankedTensorType subType,
                                   ArrayRef<int64_t> offsets) {
  if (!fullType || !subType || fullType.getRank() != 2 ||
      subType.getRank() != 2 ||
      offsets.size() != static_cast<size_t>(fullType.getRank()))
    return failure();

  SmallVector<int64_t> logicalSubTensorIndex;
  logicalSubTensorIndex.reserve(offsets.size());
  for (auto [fullDim, subDim, offset] :
       llvm::zip(fullType.getShape(), subType.getShape(), offsets)) {
    if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
        fullDim <= 0 || subDim <= 0 || offset < 0 || subDim > fullDim ||
        offset > fullDim - subDim || offset % subDim != 0 ||
        static_cast<uint64_t>(fullDim) >
            std::numeric_limits<unsigned>::max() ||
        static_cast<uint64_t>(subDim) >
            std::numeric_limits<unsigned>::max())
      return failure();
    logicalSubTensorIndex.push_back(offset / subDim);
  }

  Attribute fullEncoding = unwrapNoVerifyEncoding(fullType.getEncoding());
  Attribute subEncoding = unwrapNoVerifyEncoding(subType.getEncoding());
  if (!haveCompatibleRegisterSliceEncodingClasses(fullEncoding, subEncoding))
    return failure();

  fullType = cast<RankedTensorType>(
      cloneTypeWithEncoding(fullType, fullEncoding));
  subType =
      cast<RankedTensorType>(cloneTypeWithEncoding(subType, subEncoding));
  auto indices = deriveC500RegisterSliceIndexAttributes(
      fullType, subType, logicalSubTensorIndex);
  if (failed(indices))
    return failure();
  auto [ctaIndices, elementIndices] = std::move(*indices);
  FailureOr<SmallVector<unsigned>> selected = decodeRegisterSliceIndices(
      fullType, ctaIndices, elementIndices);
  if (failed(selected))
    return failure();
  return C500RegisterSlicePlan{std::move(ctaIndices),
                               std::move(elementIndices), std::move(*selected)};
}

FailureOr<RankedTensorType>
inferC500RegisterSlicePhysicalType(RankedTensorType fullType,
                                   const C500RegisterSlicePlan &plan,
                                   Location loc) {
  ttg::ExtractTensorOp::Properties properties;
  properties.ctaIdx = DenseI64ArrayAttr::get(
      fullType.getContext(), plan.ctaIndices);
  properties.elemIdx = DenseI64ArrayAttr::get(
      fullType.getContext(), plan.elementIndices);

  Block typeOnlyBlock;
  BlockArgument source = typeOnlyBlock.addArgument(fullType, loc);
  SmallVector<Type, 1> inferredTypes;
  if (failed(ttg::ExtractTensorOp::inferReturnTypes(
          fullType.getContext(), loc, ValueRange(source), DictionaryAttr(),
          OpaqueProperties(&properties), RegionRange(), inferredTypes)) ||
      inferredTypes.size() != 1)
    return failure();
  auto inferredType = dyn_cast<RankedTensorType>(inferredTypes.front());
  if (!inferredType)
    return failure();
  return inferredType;
}

FailureOr<RegisterSliceMaterializationPlan>
planC500RegisterSliceMaterialization(RankedTensorType fullType,
                                     RankedTensorType logicalSubType,
                                     ArrayRef<int64_t> offsets,
                                     Operation *anchor) {
  if (!fullType || !logicalSubType || !anchor ||
      fullType.getContext() != anchor->getContext() ||
      logicalSubType.getContext() != anchor->getContext() ||
      fullType.getRank() != logicalSubType.getRank() ||
      offsets.size() != static_cast<size_t>(fullType.getRank()))
    return failure();

  Attribute fullEncoding = unwrapNoVerifyEncoding(fullType.getEncoding());
  Attribute subEncoding =
      unwrapNoVerifyEncoding(logicalSubType.getEncoding());
  if (!fullEncoding || !subEncoding)
    return failure();
  fullType =
      cast<RankedTensorType>(cloneTypeWithEncoding(fullType, fullEncoding));
  logicalSubType = cast<RankedTensorType>(
      cloneTypeWithEncoding(logicalSubType, subEncoding));

  auto tryCarrier = [&](RankedTensorType physicalFull)
      -> FailureOr<RegisterSliceMaterializationPlan> {
    Attribute encoding =
        unwrapNoVerifyEncoding(physicalFull.getEncoding());
    if (!encoding)
      return failure();
    RankedTensorType physicalSub =
        logicalSubType.cloneWithEncoding(encoding);
    FailureOr<C500RegisterSlicePlan> slice =
        planC500RegisterSlice(physicalFull, physicalSub, offsets);
    if (failed(slice))
      return failure();
    FailureOr<RankedTensorType> inferred =
        inferC500RegisterSlicePhysicalType(physicalFull, *slice,
                                          anchor->getLoc());
    if (failed(inferred) || *inferred != physicalSub)
      return failure();
    return RegisterSliceMaterializationPlan{
        physicalFull, physicalSub, std::move(*slice)};
  };

  if (auto direct = tryCarrier(fullType); succeeded(direct))
    return direct;

  // A same-encoding logical slice need not be a physical subview of the
  // original register carrier. For blocked tensors, build a lowering-only
  // carrier whose native CTA tile is exactly the logical subview. Explicit
  // convert_layout boundaries preserve the source-level layout relation.
  if (auto blocked =
          dyn_cast<ttg::BlockedEncodingAttr>(fullEncoding)) {
    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    int numWarps = ttg::lookupNumWarps(anchor);
    int threadsPerWarp =
        module ? ttg::TritonGPUDialect::getThreadsPerWarp(module) : 0;
    int numCTAs = ttg::lookupNumCTAs(anchor);
    if (numWarps > 0 && threadsPerWarp > 0 && numCTAs > 0) {
      SmallVector<unsigned, 2> sizePerThread(
          logicalSubType.getRank(), 1);
      auto carrier = ttg::BlockedEncodingAttr::get(
          anchor->getContext(), logicalSubType.getShape(), sizePerThread,
          blocked.getOrder(), numWarps, threadsPerWarp, numCTAs);
      auto carrierFull =
          fullType.cloneWithEncoding(carrier);
      if (auto plan = tryCarrier(carrierFull); succeeded(plan))
        return plan;
    }
  }

  if (subEncoding != fullEncoding) {
    auto subCarrierFull = fullType.cloneWithEncoding(subEncoding);
    if (auto plan = tryCarrier(subCarrierFull); succeeded(plan))
      return plan;
  }
  return anchor->emitError()
         << "C500 cannot form a complete physical register partition for "
            "full type "
         << fullType << ", logical sub type " << logicalSubType
         << ", offsets " << offsets;
}

static bool infersRequestedExtractType(
    RankedTensorType fullType, RankedTensorType subType,
    const C500RegisterSlicePlan &plan) {
  FailureOr<RankedTensorType> inferred =
      inferC500RegisterSlicePhysicalType(
          fullType, plan, UnknownLoc::get(fullType.getContext()));
  return succeeded(inferred) && *inferred == subType;
}

static bool haveCompatibleMacaFragmentABI(ttg::MACAMmaEncodingAttr full,
                                          ttg::MACAMmaEncodingAttr sub) {
  ArrayRef<unsigned> fullMNK = full.getElementsMNK();
  ArrayRef<unsigned> subMNK = sub.getElementsMNK();
  return full.getVersionMajor() == 2 && sub.getVersionMajor() == 2 &&
         !full.getIsATrans() && !full.getIsBTrans() &&
         !sub.getIsATrans() && !sub.getIsBTrans() &&
         full.getVersionMajor() == sub.getVersionMajor() &&
         full.getVersionMinor() == sub.getVersionMinor() &&
         full.getWarpsPerCTA() == sub.getWarpsPerCTA() &&
         full.getColMajor() == sub.getColMajor() &&
         full.getIsATrans() == sub.getIsATrans() &&
         full.getIsBTrans() == sub.getIsBTrans() &&
         full.getElementsStride() == sub.getElementsStride() &&
         full.getCTALayout() == sub.getCTALayout() && fullMNK.size() == 3 &&
         subMNK.size() == 3 && subMNK[0] != 0 && subMNK[1] != 0 &&
         fullMNK[0] % subMNK[0] == 0 && fullMNK[1] % subMNK[1] == 0 &&
         fullMNK[2] == subMNK[2];
}

/// Check that a logical C500 accumulator slice is carried by the same physical
/// axes as the MMA lowering. The minor output dimension is partitioned through
/// per-thread instruction-result elements; the major dimension is partitioned
/// through repeated MMA tiles. Register cardinality alone cannot distinguish
/// these axes and would accept same-sized but logically unrelated fragments.
static bool haveCompatibleMacaFragmentCarriers(
    RankedTensorType fullType, RankedTensorType subType,
    ttg::MACAMmaEncodingAttr full, ttg::MACAMmaEncodingAttr sub) {
  if (fullType.getRank() != 2 || subType.getRank() != 2)
    return false;

  SmallVector<uint64_t, 2> factors;
  for (auto [fullDim, subDim] :
       llvm::zip(fullType.getShape(), subType.getShape())) {
    if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
        fullDim <= 0 || subDim <= 0 || fullDim % subDim != 0 ||
        fullDim > std::numeric_limits<int>::max() ||
        subDim > std::numeric_limits<int>::max())
      return false;
    factors.push_back(static_cast<uint64_t>(fullDim / subDim));
  }

  ArrayRef<unsigned> fullElements = full.getElementsMNK();
  ArrayRef<unsigned> subElements = sub.getElementsMNK();
  ArrayRef<unsigned> warps = full.getWarpsPerCTA();
  if (fullElements.size() != 3 || subElements.size() != 3 ||
      warps.size() != 2)
    return false;

  unsigned packedDimension = full.getColMajor() ? 0 : 1;
  unsigned replicaDimension = 1 - packedDimension;
  if (fullElements[replicaDimension] != subElements[replicaDimension] ||
      static_cast<uint64_t>(subElements[packedDimension]) *
              factors[packedDimension] !=
          fullElements[packedDimension])
    return false;

  auto getRepetitions = [&](RankedTensorType type,
                            ttg::MACAMmaEncodingAttr encoding) {
    ArrayRef<unsigned> encodingElements = encoding.getElementsMNK();
    return SmallVector<uint64_t, 2>{
        static_cast<uint64_t>(ttg::getNumRepM(
            static_cast<int>(type.getShape()[0]), warps,
            encodingElements[0])),
        static_cast<uint64_t>(ttg::getNumRepN(
            static_cast<int>(type.getShape()[1]), warps,
            encodingElements[1]))};
  };
  SmallVector<uint64_t, 2> fullRepetitions =
      getRepetitions(fullType, full);
  SmallVector<uint64_t, 2> subRepetitions =
      getRepetitions(subType, sub);
  return fullRepetitions[packedDimension] ==
             subRepetitions[packedDimension] &&
         subRepetitions[replicaDimension] * factors[replicaDimension] ==
             fullRepetitions[replicaDimension];
}

/// Validate the C500 accumulator fragment ABI independently of mathematical
/// LinearLayout slicing. MACA elementsMNK[0:1] select repeated MMA instruction
/// fragments in the View/Dot ABI; they are not always a same-lane logical
/// rectangle. Requiring all fragment slots to partition the per-thread
/// LinearLayout element-slot domain
/// prevents this target-specific path from becoming a permissive escape hatch.
static FailureOr<C500RegisterSlicePlan>
planMacaMmaFragmentSlice(RankedTensorType fullType, RankedTensorType subType,
                         ArrayRef<int64_t> offsets) {
  auto fullEncoding = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
      unwrapNoVerifyEncoding(fullType.getEncoding()));
  auto subEncoding = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
      unwrapNoVerifyEncoding(subType.getEncoding()));
  if (!fullEncoding || !subEncoding || fullType.getRank() != 2 ||
      subType.getRank() != 2 ||
      fullType.getElementType() != subType.getElementType() ||
      !haveCompatibleMacaFragmentABI(fullEncoding, subEncoding) ||
      !haveCompatibleMacaFragmentCarriers(fullType, subType, fullEncoding,
                                          subEncoding)) {
    LDBG("[maca-fragment-carrier-reject] full="
         << fullType << ", sub=" << subType
         << ", reason=incompatible replica/element carrier");
    return failure();
  }

  // Do not pass the deferred wrapper to the target LinearLayout converter.
  // The register-slice pass normally runs after placeholder resolution, but
  // this local normalization keeps the analysis total under standalone use
  // and lets the pass driver diagnose an invalid pipeline without asserting.
  fullType = cast<RankedTensorType>(
      cloneTypeWithEncoding(fullType, fullEncoding));
  subType = cast<RankedTensorType>(
      cloneTypeWithEncoding(subType, subEncoding));

  SmallVector<int64_t, 2> fragmentGrid;
  for (auto [fullDim, subDim, offset] :
       llvm::zip(fullType.getShape(), subType.getShape(), offsets)) {
    if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
        fullDim <= 0 || subDim <= 0 || fullDim % subDim != 0 || offset < 0 ||
        subDim > fullDim || offset > fullDim - subDim ||
        offset % subDim != 0)
      return failure();
    fragmentGrid.push_back(fullDim / subDim);
  }

  FailureOr<C500RegisterSlicePlan> requested =
      buildPhysicalC500RegisterSlicePlan(fullType, subType, offsets);
  if (failed(requested))
    return failure();

  FailureOr<triton::LinearLayout> fullLayout = getC500LinearLayout(fullType);
  FailureOr<triton::LinearLayout> subLayout = getC500LinearLayout(subType);
  if (failed(fullLayout) || failed(subLayout))
    return failure();
  StringAttr reg = StringAttr::get(fullType.getContext(), "register");
  if (!fullLayout->hasInDim(reg) || !subLayout->hasInDim(reg))
    return failure();
  uint64_t fullElementSlotCount = fullLayout->getInDimSize(reg);
  uint64_t subElementSlotCount = subLayout->getInDimSize(reg);
  if (subElementSlotCount == 0 ||
      static_cast<uint64_t>(fragmentGrid[0]) >
          std::numeric_limits<uint64_t>::max() /
              static_cast<uint64_t>(fragmentGrid[1]))
    return failure();
  uint64_t fragmentCount = static_cast<uint64_t>(fragmentGrid[0]) *
                           static_cast<uint64_t>(fragmentGrid[1]);
  if (fragmentCount >
          std::numeric_limits<uint64_t>::max() / subElementSlotCount ||
      fragmentCount * subElementSlotCount != fullElementSlotCount)
    return failure();

  DenseSet<unsigned> partition;
  for (int64_t m = 0; m < fragmentGrid[0]; ++m) {
    for (int64_t n = 0; n < fragmentGrid[1]; ++n) {
      SmallVector<int64_t, 2> slotOffsets = {
          m * subType.getShape()[0], n * subType.getShape()[1]};
      FailureOr<C500RegisterSlicePlan> slot =
          buildPhysicalC500RegisterSlicePlan(fullType, subType, slotOffsets);
      if (failed(slot) ||
          slot->sourceElementSlots.size() != subElementSlotCount)
        return failure();
      for (unsigned sourceElementSlot : slot->sourceElementSlots)
        if (!partition.insert(sourceElementSlot).second)
          return failure();
    }
  }
  if (partition.size() != fullElementSlotCount)
    return failure();
  return requested;
}

/// Plan one repeated C500 dot-operand fragment.  The physical dot ABI stores
/// each instruction fragment as one consecutive LinearLayout element-slot
/// replica. Logical
/// fragment coordinates are linearized in operand order and are accepted only
/// when every slot forms a disjoint, complete partition of the source
/// element-slot domain. Support follows this target ABI proof and is never inferred from a
/// particular workload or benchmark shape.
static FailureOr<C500RegisterSlicePlan>
planMacaDotOperandFragmentSlice(RankedTensorType fullType,
                                RankedTensorType subType,
                                ArrayRef<int64_t> offsets) {
  auto fullEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      unwrapNoVerifyEncoding(fullType.getEncoding()));
  auto subEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      unwrapNoVerifyEncoding(subType.getEncoding()));
  if (!fullEncoding || !subEncoding ||
      fullType.getRank() != 2 || subType.getRank() != 2 ||
      fullType.getElementType() != subType.getElementType() ||
      fullEncoding.getOpIdx() != subEncoding.getOpIdx() ||
      fullEncoding.getParent() != subEncoding.getParent() ||
      (fullEncoding.getOpIdx() != 0 && fullEncoding.getOpIdx() != 1) ||
      !isa<ttg::MACAMmaEncodingAttr>(fullEncoding.getParent()))
    return failure();

  fullType = cast<RankedTensorType>(
      cloneTypeWithEncoding(fullType, fullEncoding));
  subType = cast<RankedTensorType>(
      cloneTypeWithEncoding(subType, subEncoding));

  SmallVector<unsigned, 2> fragmentGrid;
  SmallVector<unsigned, 2> fragmentCoordinate;
  for (auto [fullDim, subDim, offset] :
       llvm::zip(fullType.getShape(), subType.getShape(), offsets)) {
    if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
        fullDim <= 0 || subDim <= 0 || fullDim % subDim != 0 || offset < 0 ||
        subDim > fullDim || offset > fullDim - subDim ||
        offset % subDim != 0 ||
        static_cast<uint64_t>(fullDim / subDim) >
            std::numeric_limits<unsigned>::max() ||
        static_cast<uint64_t>(offset / subDim) >
            std::numeric_limits<unsigned>::max())
      return failure();
    fragmentGrid.push_back(static_cast<unsigned>(fullDim / subDim));
    fragmentCoordinate.push_back(static_cast<unsigned>(offset / subDim));
  }
  if (fragmentGrid.size() != 2 || fragmentCoordinate.size() != 2)
    return failure();

  FailureOr<triton::LinearLayout> fullLayout = getC500LinearLayout(fullType);
  FailureOr<triton::LinearLayout> subLayout = getC500LinearLayout(subType);
  if (failed(fullLayout) || failed(subLayout))
    return failure();
  StringAttr reg = StringAttr::get(fullType.getContext(), "register");
  if (!fullLayout->hasInDim(reg) || !subLayout->hasInDim(reg))
    return failure();
  uint64_t fullElementSlotCount = fullLayout->getInDimSize(reg);
  uint64_t subElementSlotCount = subLayout->getInDimSize(reg);
  if (subElementSlotCount == 0)
    return failure();

  std::optional<uint64_t> fragmentCount = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(fragmentGrid[0]),
      static_cast<uint64_t>(fragmentGrid[1]));
  std::optional<uint64_t> partitionSize = fragmentCount
                                              ? llvm::checkedMulUnsigned(
                                                    *fragmentCount,
                                                    subElementSlotCount)
                                              : std::nullopt;
  if (!fragmentCount ||
      *fragmentCount > std::numeric_limits<unsigned>::max() ||
      !partitionSize || *partitionSize != fullElementSlotCount ||
      subElementSlotCount > static_cast<uint64_t>(
                             std::numeric_limits<int64_t>::max()))
    return failure();

  SmallVector<unsigned, 2> order =
      fullEncoding.getOpIdx() == 0 ? SmallVector<unsigned, 2>{1, 0}
                                   : SmallVector<unsigned, 2>{0, 1};
  FailureOr<int64_t> fragmentIndex = linearizeRegisterSliceIndex(
      fragmentCoordinate, fragmentGrid, order);
  if (failed(fragmentIndex))
    return failure();

  SmallVector<int64_t> ctaIndices{*fragmentIndex};
  SmallVector<int64_t> elementIndices;
  elementIndices.reserve(subElementSlotCount);
  for (uint64_t index = 0; index < subElementSlotCount; ++index)
    elementIndices.push_back(static_cast<int64_t>(index));

  FailureOr<SmallVector<unsigned>> selected = decodeRegisterSliceIndices(
      fullType, ctaIndices, elementIndices);
  if (failed(selected) || selected->size() != subElementSlotCount)
    return failure();
  C500RegisterSlicePlan requested{std::move(ctaIndices),
                                  std::move(elementIndices),
                                  std::move(*selected)};
  FailureOr<RankedTensorType> inferred =
      inferC500RegisterSlicePhysicalType(
          fullType, requested, UnknownLoc::get(fullType.getContext()));
  if (failed(inferred) || inferred->getShape() != subType.getShape() ||
      inferred->getElementType() != subType.getElementType())
    return failure();
  auto inferredDot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      inferred->getEncoding());
  if (!inferredDot || inferredDot.getOpIdx() != subEncoding.getOpIdx() ||
      inferredDot.getParent() != subEncoding.getParent())
    return failure();

  DenseSet<unsigned> partition;
  for (unsigned fragment = 0; fragment < *fragmentCount; ++fragment) {
    SmallVector<int64_t, 1> slotCta{static_cast<int64_t>(fragment)};
    FailureOr<SmallVector<unsigned>> slot = decodeRegisterSliceIndices(
        fullType, slotCta, requested.elementIndices);
    if (failed(slot) || slot->size() != subElementSlotCount)
      return failure();
    for (unsigned sourceRegister : *slot)
      if (!partition.insert(sourceRegister).second)
        return failure();
  }
  if (partition.size() != fullElementSlotCount)
    return failure();
  return requested;
}

FailureOr<C500RegisterSlicePlan>
planC500RegisterSlice(RankedTensorType fullType, RankedTensorType subType,
                      ArrayRef<int64_t> offsets) {
  if (!fullType || !subType)
    return failure();
  Attribute fullEncoding = unwrapNoVerifyEncoding(fullType.getEncoding());
  Attribute subEncoding = unwrapNoVerifyEncoding(subType.getEncoding());
  fullType = cast<RankedTensorType>(
      cloneTypeWithEncoding(fullType, fullEncoding));
  subType = cast<RankedTensorType>(
      cloneTypeWithEncoding(subType, subEncoding));
  if (isa_and_nonnull<ttg::DotOperandEncodingAttr>(fullEncoding) &&
      isa_and_nonnull<ttg::DotOperandEncodingAttr>(subEncoding))
    // Dot operands have a target fragment ABI. Failure to prove that ABI must
    // not be reinterpreted as an ordinary same-lane logical subview.
    return planMacaDotOperandFragmentSlice(fullType, subType, offsets);
  if (fullEncoding != subEncoding &&
      isa_and_nonnull<ttg::MACAMmaEncodingAttr>(fullEncoding) &&
      isa_and_nonnull<ttg::MACAMmaEncodingAttr>(subEncoding))
    // Distinct accumulator encodings describe target MMA fragment carriers.
    // If that contract fails, a count-compatible generic register selection
    // must not relabel the pair as a logical subview: it has not proved that
    // the selected registers carry the requested logical coordinates.
    return planMacaMmaFragmentSlice(fullType, subType, offsets);

  FailureOr<C500RegisterSlicePlan> physical =
      buildPhysicalC500RegisterSlicePlan(fullType, subType, offsets);
  if (succeeded(physical) &&
      infersRequestedExtractType(fullType, subType, *physical))
    return physical;
  return planMacaMmaFragmentSlice(fullType, subType, offsets);
}

} // namespace mlir::triton::gpu::metax::gluon::c500

namespace mlir::triton::gpu::metax::gluon {
namespace {

using ::mlir::triton::gpu::metax::gluon::decodeRegisterSliceIndices;

template <typename T>
static std::string formatIntegerArray(ArrayRef<T> values) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
  return storage;
}

static LogicalResult verifyRegisterSelectionCardinality(
    Operation *op, StringRef opName, RankedTensorType fullType,
    RankedTensorType subType, ArrayRef<int64_t> ctaIndices,
    ArrayRef<int64_t> elementIndices) {
  FailureOr<SmallVector<unsigned>> selected =
      decodeRegisterSliceIndices(fullType, ctaIndices, elementIndices);
  if (failed(selected))
    return op->emitError()
           << "cannot decode " << opName << " register indices for "
           << fullType;

  FailureOr<triton::LinearLayout> subLayout = getC500LinearLayout(subType);
  if (failed(subLayout))
    return op->emitError()
           << "cannot derive a safe C500 LinearLayout for " << opName
           << " sub-tensor " << subType;
  StringAttr reg = StringAttr::get(op->getContext(), "register");
  if (!subLayout->hasInDim(reg) ||
      selected->size() !=
          static_cast<size_t>(subLayout->getInDimSize(reg)))
    return op->emitError()
           << opName
           << " physical selection count does not match the sub-tensor "
              "register-domain element-slot count; selected="
           << selected->size() << ", sub=" << subType;
  DenseSet<unsigned> unique(selected->begin(), selected->end());
  if (unique.size() != selected->size())
    return op->emitError()
           << opName << " physical register selection contains duplicates: "
           << formatIntegerArray(ArrayRef<unsigned>(*selected));
  return success();
}

} // namespace

LogicalResult verifyExtractTensorContract(ttg::ExtractTensorOp extractOp) {
  auto sourceTy = dyn_cast<RankedTensorType>(extractOp.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(extractOp.getType());
  if (!sourceTy || !resultTy)
    return success();

  if (failed(verifyRegisterSelectionCardinality(
          extractOp, "extract_tensor", sourceTy, resultTy,
          extractOp.getCtaIdx(), extractOp.getElemIdx())))
    return failure();

  ttg::ExtractTensorOp::Properties properties;
  properties.ctaIdx =
      DenseI64ArrayAttr::get(extractOp.getContext(), extractOp.getCtaIdx());
  properties.elemIdx =
      DenseI64ArrayAttr::get(extractOp.getContext(), extractOp.getElemIdx());

  SmallVector<Type, 1> inferredTypes;
  if (failed(ttg::ExtractTensorOp::inferReturnTypes(
          extractOp.getContext(), extractOp.getLoc(),
          ValueRange(extractOp.getSource()), DictionaryAttr(),
          OpaqueProperties(&properties), RegionRange(), inferredTypes)) ||
      inferredTypes.empty())
    return extractOp.emitError()
           << "cannot infer ttg.extract_tensor result type from source layout "
           << sourceTy.getEncoding();

  if (inferredTypes.front() != resultTy)
    return extractOp.emitError()
           << "ttg.extract_tensor result type does not match the physical "
              "subview inferred from its source layout; inferred="
           << inferredTypes.front() << ", actual=" << resultTy;

  return success();
}

LogicalResult verifyInsertTensorContract(ttg::InsertTensorOp insertOp) {
  auto fullTy = dyn_cast<RankedTensorType>(insertOp.getInserted().getType());
  auto subTy = dyn_cast<RankedTensorType>(insertOp.getInsert().getType());
  auto resultTy = dyn_cast<RankedTensorType>(insertOp.getType());
  if (!fullTy || !subTy || !resultTy)
    return success();
  if (fullTy != resultTy)
    return insertOp.emitError()
           << "ttg.insert_tensor result must preserve the full tensor type; "
              "full="
           << fullTy << ", result=" << resultTy;

  if (failed(verifyRegisterSelectionCardinality(
          insertOp, "insert_tensor", fullTy, subTy, insertOp.getCtaIdx(),
          insertOp.getElemIdx())))
    return failure();

  c500::C500RegisterSlicePlan physicalSlice{
      SmallVector<int64_t>(insertOp.getCtaIdx()),
      SmallVector<int64_t>(insertOp.getElemIdx()), {}};
  FailureOr<RankedTensorType> inferred =
      c500::inferC500RegisterSlicePhysicalType(fullTy, physicalSlice,
                                              insertOp.getLoc());
  if (failed(inferred))
    return insertOp.emitError()
           << "cannot infer ttg.insert_tensor input type from destination "
              "layout "
           << fullTy.getEncoding();
  if (*inferred != subTy)
    return insertOp.emitError()
           << "ttg.insert_tensor input type does not match the physical "
              "subview inferred from its destination layout; inferred="
           << *inferred << ", actual=" << subTy;
  return success();
}

} // namespace mlir::triton::gpu::metax::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
