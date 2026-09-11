#include "Gluon/Targets/GluonC500Layout.h"

#include "Gluon/GluonLayoutPlaceholders.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>

#define DEBUG_TYPE "metax-gluon-c500-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace c500 {

namespace {

struct SharedTransferBankMetrics {
  unsigned vectorElements = 1;
  unsigned readConflictDegree = 1;
  unsigned writeConflictDegree = 1;
};

struct RepeatedSharedCandidate {
  LayoutTransferMetrics genericMetrics;
  LayoutTransferMetrics repeatedMetrics;
  SharedTransferBankMetrics genericBanks;
  SharedTransferBankMetrics repeatedBanks;
  unsigned elementBitWidth = 8;
};

static constexpr uint64_t kC500SharedMemoryCapacityBytes = 64 * 1024;

/// Measure the exact shared mapping selected by Triton's transfer lowerer.
/// GenericSwizzling models the target shared-memory segment as 32 banks of
/// 32 bits and treats same-address accesses as broadcasts. Its conflict result
/// is reported as the number of extra serialized accesses, so add one to emit
/// the more actionable conflict degree.
static SharedTransferBankMetrics
analyzeSharedTransferBanks(const tt::LinearLayout &sourceLayout,
                           const tt::LinearLayout &resultLayout,
                           const tt::LinearLayout &sharedLayout,
                           int32_t bitWidth, Operation *anchor) {
  StringAttr vector = StringAttr::get(anchor->getContext(), "vector");
  auto [readConflicts, writeConflicts] = ttg::bankConflictsLdSt(
      sourceLayout, resultLayout, sharedLayout, bitWidth);
  return {static_cast<unsigned>(sharedLayout.getInDimSize(vector)),
          static_cast<unsigned>(readConflicts + 1),
          static_cast<unsigned>(writeConflicts + 1)};
}

/// Analyze whether both lowering variants implement the fixed type contract.
/// This is a legality and metric query only; it deliberately contains no
/// profitability policy.
static std::optional<RepeatedSharedCandidate>
analyzeRepeatedSharedCandidate(const LayoutTransferRequirement &requirement,
                               Operation *anchor) {
  Attribute sourceEncoding =
      unwrapNoVerifyEncoding(requirement.sourceType.getEncoding());
  Attribute resultEncoding =
      unwrapNoVerifyEncoding(requirement.resultType.getEncoding());
  if (!isa<ttg::MACAMmaEncodingAttr>(sourceEncoding) ||
      !isa<ttg::BlockedEncodingAttr>(resultEncoding))
    return std::nullopt;

  // Keep this proof aligned with ConvertLayoutOpSwizzlingConversion. The
  // specialized lowering is CTA-local: a block component in the minimal
  // conversion requires a cross-CTA implementation that it does not provide.
  tt::LinearLayout minimalTransfer =
      mlir::minimalCvtLayout(requirement.sourceType, requirement.resultType);
  StringAttr block = StringAttr::get(anchor->getContext(), "block");
  if (llvm::is_contained(minimalTransfer.getInDimNames(), block) ||
      !mlir::cvtNeedsSharedMemory(requirement.sourceType,
                                  requirement.resultType))
    return std::nullopt;

  tt::LinearLayout sourceLayout = ttg::toLinearLayout(requirement.sourceType);
  tt::LinearLayout resultLayout = ttg::toLinearLayout(requirement.resultType);
  sourceLayout =
      tt::actionRemoveBroadcastedRegs(sourceLayout).apply(sourceLayout);
  resultLayout =
      tt::actionRemoveBroadcastedRegs(resultLayout).apply(resultLayout);
  int32_t bitWidth = tt::getBitwidth(requirement.sourceType);
  // The C500 repeated-shared implementation moves byte-addressable scalar
  // elements. Leave sub-byte transfers on the generic path instead of
  // silently rounding their scratch footprint or bank width.
  if (bitWidth <= 0 || bitWidth % 8 != 0)
    return std::nullopt;
  tt::LinearLayout genericShared = ttg::optimalSwizzlingLdSt(
      sourceLayout, resultLayout, bitWidth, /*forceNoVec=*/false);
  tt::LinearLayout repeatedShared = ttg::optimalSwizzlingLdSt(
      sourceLayout, resultLayout, bitWidth, /*forceNoVec=*/true);
  StringAttr repetitions = StringAttr::get(anchor->getContext(), "reps");

  return RepeatedSharedCandidate{
      {/*scratchElements=*/tt::getNumScratchElemsSwizzledCvt(
           requirement.sourceType, requirement.resultType,
           /*forceNoVectorize=*/false),
       /*repetitions=*/
           static_cast<unsigned>(genericShared.getInDimSize(repetitions))},
      {/*scratchElements=*/tt::getNumScratchElemsSwizzledCvt(
           requirement.sourceType, requirement.resultType,
           /*forceNoVectorize=*/true),
       /*repetitions=*/
           static_cast<unsigned>(repeatedShared.getInDimSize(repetitions))},
      analyzeSharedTransferBanks(sourceLayout, resultLayout, genericShared,
                                 bitWidth, anchor),
      analyzeSharedTransferBanks(sourceLayout, resultLayout, repeatedShared,
                                 bitWidth, anchor),
      static_cast<unsigned>(bitWidth)};
}

static uint64_t getScratchBytes(const LayoutTransferMetrics &metrics,
                                unsigned elementBitWidth) {
  // Candidate analysis admits only positive byte-addressable widths before
  // constructing RepeatedSharedCandidate.
  return static_cast<uint64_t>(metrics.scratchElements) *
         (elementBitWidth / 8);
}

/// One scratch allocation larger than the complete C500 shared-memory
/// capacity is unconditionally illegal, independently of allocation aliasing
/// or occupancy. A repeated form that brings that single allocation under the
/// hard limit is therefore a legality rescue, not a profitability heuristic.
/// Final Allocation analysis still validates the complete kernel footprint.
static bool isCapacityRescue(const RepeatedSharedCandidate &candidate) {
  return getScratchBytes(candidate.genericMetrics,
                         candidate.elementBitWidth) >
             kC500SharedMemoryCapacityBytes &&
         getScratchBytes(candidate.repeatedMetrics,
                         candidate.elementBitWidth) <=
             kC500SharedMemoryCapacityBytes;
}

/// Return the first conservative reason why temporal decomposition must not be
/// selected. Scratch reduction is considered only after preserving the shared
/// transaction width and both modeled bank-conflict degrees. This is a local
/// Pareto test; it intentionally makes no occupancy claim because this query
/// does not know the module's complete static and dynamic shared allocation.
static StringRef getRepeatedSharedRejectionReason(
    const LayoutTransferRequirement &requirement,
    const RepeatedSharedCandidate &candidate) {
  // A semantically valid implementation that rescues an otherwise impossible
  // allocation takes precedence over profitability-only path restrictions.
  if (isCapacityRescue(candidate))
    return {};
  if (!requirement.isTerminalGlobalStore)
    return "not-terminal-global-store";
  if (!requirement.isOutsideLoop)
    return "inside-loop";
  if (candidate.genericMetrics.repetitions != 1)
    return "generic-already-repeated";
  if (candidate.repeatedMetrics.repetitions <= 1)
    return "no-temporal-decomposition";
  if (candidate.repeatedMetrics.scratchElements >=
      candidate.genericMetrics.scratchElements)
    return "no-scratch-reduction";
  if (candidate.repeatedBanks.vectorElements <
      candidate.genericBanks.vectorElements)
    return "reduced-shared-vector-width";
  if (candidate.repeatedBanks.readConflictDegree >
      candidate.genericBanks.readConflictDegree)
    return "higher-shared-read-conflict";
  if (candidate.repeatedBanks.writeConflictDegree >
      candidate.genericBanks.writeConflictDegree)
    return "higher-shared-write-conflict";
  return {};
}

} // namespace

FailureOr<LayoutTransferPlan>
planLayoutTransfer(const LayoutTransferRequirement &requirement,
                   Operation *anchor) {
  if (!anchor || !requirement.sourceType || !requirement.resultType ||
      requirement.sourceType.getContext() != anchor->getContext() ||
      requirement.resultType.getContext() != anchor->getContext() ||
      requirement.sourceType.getShape() != requirement.resultType.getShape() ||
      requirement.sourceType.getElementType() !=
          requirement.resultType.getElementType()) {
    if (anchor)
      anchor->emitError()
          << "C500 layout-transfer planner received an incompatible "
             "source/result contract";
    return failure();
  }

  LayoutTransferPlan plan;
  Attribute sourceEncoding =
      unwrapNoVerifyEncoding(requirement.sourceType.getEncoding());
  Attribute resultEncoding =
      unwrapNoVerifyEncoding(requirement.resultType.getEncoding());
  std::optional<RepeatedSharedCandidate> candidate =
      analyzeRepeatedSharedCandidate(requirement, anchor);
  if (!candidate) {
    LDBG("[layout-transfer-candidate] loc="
         << anchor->getLoc()
         << ", applicable=false, reason=unsupported-or-nonlocal-transfer"
         << ", source=" << sourceEncoding << ", result=" << resultEncoding);
    return plan;
  }

  plan.genericMetrics = candidate->genericMetrics;
  plan.selectedMetrics = candidate->genericMetrics;
  StringRef rejectionReason =
      getRepeatedSharedRejectionReason(requirement, *candidate);
  if (rejectionReason.empty()) {
    plan.implementation = LayoutTransferImplementation::RepeatedShared;
    plan.selectedMetrics = candidate->repeatedMetrics;
  }
  bool capacityRescue = isCapacityRescue(*candidate);
  StringRef selectionBasis =
      plan.implementation == LayoutTransferImplementation::Generic
          ? "generic"
          : capacityRescue ? "capacity-rescue" : "pareto";

  LDBG("[layout-transfer-candidate] loc="
       << anchor->getLoc() << ", applicable=true, source="
       << sourceEncoding << ", result=" << resultEncoding
       << ", generic-scratch-elements="
       << candidate->genericMetrics.scratchElements
       << ", repeated-scratch-elements="
       << candidate->repeatedMetrics.scratchElements
       << ", generic-scratch-bytes="
       << getScratchBytes(candidate->genericMetrics,
                          candidate->elementBitWidth)
       << ", repeated-scratch-bytes="
       << getScratchBytes(candidate->repeatedMetrics,
                          candidate->elementBitWidth)
       << ", shared-capacity-bytes=" << kC500SharedMemoryCapacityBytes
       << ", generic-repetitions=" << candidate->genericMetrics.repetitions
       << ", repeated-repetitions="
       << candidate->repeatedMetrics.repetitions
       << ", bank-count=32, bank-width-bytes=4"
       << ", generic-vector-elements="
       << candidate->genericBanks.vectorElements
       << ", generic-read-conflict-degree="
       << candidate->genericBanks.readConflictDegree
       << ", generic-write-conflict-degree="
       << candidate->genericBanks.writeConflictDegree
       << ", repeated-vector-elements="
       << candidate->repeatedBanks.vectorElements
       << ", repeated-read-conflict-degree="
       << candidate->repeatedBanks.readConflictDegree
       << ", repeated-write-conflict-degree="
       << candidate->repeatedBanks.writeConflictDegree);
  LDBG("[layout-transfer-selection] loc="
       << anchor->getLoc()
       << ", terminal-global-store=" << requirement.isTerminalGlobalStore
       << ", outside-loop=" << requirement.isOutsideLoop
       << ", vector-width-not-worse="
       << (candidate->repeatedBanks.vectorElements >=
           candidate->genericBanks.vectorElements)
       << ", read-conflict-not-worse="
       << (candidate->repeatedBanks.readConflictDegree <=
           candidate->genericBanks.readConflictDegree)
       << ", write-conflict-not-worse="
       << (candidate->repeatedBanks.writeConflictDegree <=
           candidate->genericBanks.writeConflictDegree)
       << ", capacity-rescue=" << capacityRescue
       << ", selection-basis=" << selectionBasis
       << ", rejection-reason="
       << (rejectionReason.empty() ? "none" : rejectionReason)
       << ", implementation="
       << (plan.implementation == LayoutTransferImplementation::RepeatedShared
               ? "repeated-shared"
               : "generic"));
  return plan;
}

} // namespace c500

namespace {

static ttg::MACAMmaEncodingAttr
getCanonicalMacaAccumulatorEncoding(ttg::MACAMmaEncodingAttr encoding) {
  if (!encoding.getIsATrans() && !encoding.getIsBTrans())
    return encoding;
  return ttg::MACAMmaEncodingAttr::get(
      encoding.getContext(), encoding.getVersionMajor(),
      encoding.getVersionMinor(), encoding.getWarpsPerCTA(),
      encoding.getElementsMNK(), encoding.getColMajor(),
      encoding.getCTALayout(), /*isATrans=*/false, /*isBTrans=*/false,
      encoding.getElementsStride());
}

static std::optional<std::string>
getMacaEncodingContractError(ttg::MACAMmaEncodingAttr encoding) {
  ArrayRef<unsigned> warpsPerCTA = encoding.getWarpsPerCTA();
  ArrayRef<unsigned> elementsMNK = encoding.getElementsMNK();
  ArrayRef<unsigned> elementsStride = encoding.getElementsStride();
  if (encoding.getVersionMajor() != 2)
    return (Twine("versionMajor must be 2, got ") +
            Twine(encoding.getVersionMajor()))
        .str();
  if (warpsPerCTA.size() != 2)
    return (Twine("warpsPerCTA must have rank 2, got ") +
            Twine(warpsPerCTA.size()))
        .str();
  if (elementsMNK.size() != 3)
    return (Twine("elementsMNK must contain M, N, and K, got ") +
            Twine(elementsMNK.size()) + " elements")
        .str();
  if (elementsStride.size() != 2)
    return (Twine("elementsStride must contain A and B strides, got ") +
            Twine(elementsStride.size()) + " elements")
        .str();
  if (!encoding.getCTALayout() ||
      encoding.getCTALayout().getRank() != warpsPerCTA.size())
    return "CTA layout rank must match warpsPerCTA rank";
  if (encoding.getColMajor() > 1)
    return (Twine("colMajor must be 0 or 1, got ") +
            Twine(encoding.getColMajor()))
        .str();
  auto hasInvalidPowerOfTwo = [](ArrayRef<unsigned> values) {
    return llvm::any_of(values, [](unsigned value) {
      return value == 0 || !llvm::isPowerOf2_64(value);
    });
  };
  if (hasInvalidPowerOfTwo(warpsPerCTA))
    return "warpsPerCTA values must be nonzero powers of two";
  if (hasInvalidPowerOfTwo(elementsMNK))
    return "elementsMNK values must be nonzero powers of two";
  if (llvm::is_contained(elementsStride, 0u))
    return "elementsStride values must be nonzero";

  unsigned elementsK = elementsMNK[2];
  if (encoding.getIsATrans() && elementsK % elementsStride[0] != 0)
    return (Twine("A transpose stride ") + Twine(elementsStride[0]) +
            " must divide elementsMNK K " + Twine(elementsK))
        .str();
  if (encoding.getIsBTrans() && elementsK % elementsStride[1] != 0)
    return (Twine("B transpose stride ") + Twine(elementsStride[1]) +
            " must divide elementsMNK K " + Twine(elementsK))
        .str();
  return std::nullopt;
}

static LogicalResult verifyMacaEncodingInType(Type type, Operation *owner,
                                              StringRef boundary,
                                              StringRef typeOwner) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getEncoding() ||
      isa<gluon_dialect::AutoEncodingAttr>(tensorType.getEncoding()))
    return success();

  Attribute encoding = unwrapNoVerifyEncoding(tensorType.getEncoding());
  ttg::MACAMmaEncodingAttr mma;
  bool isDotOperand = false;
  if (auto direct = dyn_cast<ttg::MACAMmaEncodingAttr>(encoding)) {
    mma = direct;
  } else if (auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(encoding)) {
    mma = dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent());
    isDotOperand = static_cast<bool>(mma);
  }
  if (!mma)
    return success();

  if (std::optional<std::string> error = getMacaEncodingContractError(mma))
    return owner->emitError()
           << boundary << " rejected malformed MACA MMA encoding in "
           << typeOwner << ": " << *error << "; type=" << tensorType;
  if (tensorType.getRank() != 2)
    return owner->emitError()
           << boundary << " requires rank-2 MACA tensors in " << typeOwner
           << ", got rank " << tensorType.getRank() << "; type="
           << tensorType;
  if (llvm::any_of(tensorType.getShape(), [](int64_t extent) {
        return ShapedType::isDynamic(extent) || extent <= 0 ||
               static_cast<uint64_t>(extent) >
                   std::numeric_limits<unsigned>::max();
      }))
    return owner->emitError()
           << boundary << " requires positive static MACA tensor dimensions "
           << "representable by the C500 layout model in " << typeOwner
           << "; type=" << tensorType;
  if (isDotOperand && (mma.getIsATrans() || mma.getIsBTrans()))
    return owner->emitError()
           << boundary
           << " does not support MACA dot-operand layouts whose parent uses "
              "isATrans/isBTrans: the current dot LinearLayout and MetaX "
              "local-load lowering do not model that physical LDS-trans "
              "contract; type="
           << tensorType;
  return success();
}

static FailureOr<triton::LinearLayout>
getRegisterToSharedConversion(RankedTensorType registerType,
                              ttg::MemDescType memDescType) {
  if (!registerType.getEncoding() || !memDescType.getEncoding())
    return failure();
  FailureOr<triton::LinearLayout> registerLayout =
      getC500LinearLayout(registerType);
  if (failed(registerLayout))
    return failure();
  triton::LinearLayout sharedLayout = triton::LinearLayout::empty();
  Attribute sharedEncoding = unwrapNoVerifyEncoding(memDescType.getEncoding());
  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(sharedEncoding))
    sharedLayout = padded.getLinearComponent();
  else {
    FailureOr<triton::LinearLayout> linearShared =
        getC500LinearLayout(memDescType);
    if (failed(linearShared))
      return failure();
    sharedLayout = std::move(*linearShared);
  }
  return registerLayout->invertAndCompose(sharedLayout);
}

struct AtomicRmwOwnershipInfo {
  uint64_t logicalElements = 0;
  uint64_t elementSlotsPerThread = 0;
  uint64_t activeThreads = 0;
  uint64_t physicalThreads = 0;
  uint64_t blocks = 0;
};

static FailureOr<AtomicRmwOwnershipInfo>
analyzePredicatedAtomicRmwOwnership(RankedTensorType valueTy,
                                    const triton::LinearLayout &layout) {
  MLIRContext *context = valueTy.getContext();
  StringAttr reg = StringAttr::get(context, "register");
  StringAttr lane = StringAttr::get(context, "lane");
  StringAttr warp = StringAttr::get(context, "warp");
  StringAttr block = StringAttr::get(context, "block");

  if (llvm::any_of(layout.getInDimNames(), [&](StringAttr dim) {
        return dim != reg && dim != lane && dim != warp && dim != block;
      }))
    return failure();

  auto getInputSize = [&](StringAttr dim) -> uint64_t {
    return layout.hasInDim(dim) ? layout.getInDimSize(dim) : 1;
  };

  AtomicRmwOwnershipInfo info;
  info.logicalElements = valueTy.getNumElements();
  info.elementSlotsPerThread = getInputSize(reg);
  uint64_t lanes = getInputSize(lane);
  uint64_t warps = getInputSize(warp);
  info.blocks = getInputSize(block);
  std::optional<uint64_t> physicalThreads =
      llvm::checkedMulUnsigned(lanes, warps);
  if (!physicalThreads || info.logicalElements == 0 ||
      info.elementSlotsPerThread == 0)
    return failure();
  info.physicalThreads = *physicalThreads;

  if (info.logicalElements % info.elementSlotsPerThread != 0)
    return failure();
  info.activeThreads = info.logicalElements / info.elementSlotsPerThread;
  if (!llvm::isPowerOf2_64(info.activeThreads) ||
      info.activeThreads > info.physicalThreads)
    return failure();

  std::optional<uint64_t> activeOwners = llvm::checkedMulUnsigned(
      info.activeThreads, info.elementSlotsPerThread);
  activeOwners = activeOwners
                     ? llvm::checkedMulUnsigned(*activeOwners, info.blocks)
                     : std::nullopt;
  if (!activeOwners || *activeOwners != info.logicalElements ||
      layout.getTotalOutDimSize() != info.logicalElements)
    return failure();

  unsigned activeThreadBits = llvm::Log2_64(info.activeThreads);
  unsigned laneBits = llvm::Log2_64(lanes);
  unsigned warpBits = llvm::Log2_64(warps);
  unsigned activeLaneBits = std::min(activeThreadBits, laneBits);
  unsigned activeWarpBits = activeThreadBits - activeLaneBits;
  if (activeWarpBits > warpBits)
    return failure();

  triton::LinearLayout::BasesT restrictedBases = layout.getBases();
  if (layout.hasInDim(lane))
    restrictedBases[lane].resize(activeLaneBits);
  if (layout.hasInDim(warp))
    restrictedBases[warp].resize(activeWarpBits);

  triton::LinearLayout activeLayout(std::move(restrictedBases),
                                    layout.getOutDims(),
                                    /*requireSurjective=*/false);
  if (!activeLayout.isInjective() || !activeLayout.isSurjective() ||
      activeLayout.getTotalInDimSize() != info.logicalElements ||
      activeLayout.getTotalOutDimSize() != info.logicalElements)
    return failure();
  return info;
}

} // namespace

unsigned normalizeMacaAccumulatorEncodings(Operation *root) {
  unsigned replacements = 0;
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](RankedTensorType type) -> std::optional<Type> {
        auto encoding = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
            unwrapNoVerifyEncoding(type.getEncoding()));
        if (!encoding || (!encoding.getIsATrans() && !encoding.getIsBTrans()))
          return std::nullopt;
        ++replacements;
        return type.cloneWithEncoding(
            getCanonicalMacaAccumulatorEncoding(encoding));
      });
  replacer.recursivelyReplaceElementsIn(
      root, /*replaceAttrs=*/true, /*replaceLocs=*/false,
      /*replaceTypes=*/true);
  return replacements;
}

LogicalResult verifyMacaEncodingContracts(ModuleOp module,
                                          StringRef boundary) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    for (auto [index, type] : llvm::enumerate(op->getResultTypes()))
      if (failed(verifyMacaEncodingInType(
              type, op, boundary,
              (Twine("operation result #") + Twine(index)).str())))
        return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (failed(verifyMacaEncodingInType(
                  argument.getType(), op, boundary,
                  (Twine("block argument #") + Twine(argument.getArgNumber()))
                      .str())))
            return WalkResult::interrupt();
    if (auto func = dyn_cast<tt::FuncOp>(op)) {
      for (Type type : func.getFunctionType().getInputs())
        if (failed(verifyMacaEncodingInType(type, op, boundary,
                                            "function input")))
          return WalkResult::interrupt();
      for (Type type : func.getFunctionType().getResults())
        if (failed(verifyMacaEncodingInType(type, op, boundary,
                                            "function result")))
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

FailureOr<triton::LinearLayout>
getC500LinearLayout(RankedTensorType tensorType) {
  if (!tensorType || !tensorType.getEncoding() || tensorType.getRank() == 0 ||
      llvm::any_of(tensorType.getShape(), [](int64_t extent) {
        return ShapedType::isDynamic(extent) || extent <= 0 ||
               static_cast<uint64_t>(extent) >
                   std::numeric_limits<unsigned>::max();
      }))
    return failure();

  Attribute encoding = unwrapNoVerifyEncoding(tensorType.getEncoding());
  tensorType = tensorType.cloneWithEncoding(encoding);
  if (auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(encoding)) {
    if (getMacaEncodingContractError(mma) || tensorType.getRank() != 2)
      return failure();
    tensorType = tensorType.cloneWithEncoding(
        getCanonicalMacaAccumulatorEncoding(mma));
  } else if (auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(encoding)) {
    if (auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent())) {
      if (getMacaEncodingContractError(mma) || tensorType.getRank() != 2 ||
          mma.getIsATrans() || mma.getIsBTrans())
        return failure();
    }
  }
  return ttg::toLinearLayout(tensorType);
}

FailureOr<triton::LinearLayout>
getC500LinearLayout(ttg::MemDescType memDescType) {
  if (!memDescType || !memDescType.getEncoding() ||
      memDescType.getRank() == 0 ||
      llvm::any_of(memDescType.getShape(), [](int64_t extent) {
        return ShapedType::isDynamic(extent) || extent <= 0 ||
               static_cast<uint64_t>(extent) >
                   std::numeric_limits<unsigned>::max();
      }))
    return failure();
  memDescType = cast<ttg::MemDescType>(cloneTypeWithEncoding(
      memDescType, unwrapNoVerifyEncoding(memDescType.getEncoding())));
  return ttg::toLinearLayout(memDescType);
}

bool isSimpleDotLocalLoad(ttg::LocalLoadOp loadOp) {
  if (!loadOp || loadOp.getToken())
    return false;
  auto tensorType = dyn_cast<RankedTensorType>(loadOp.getType());
  return tensorType &&
         isa<ttg::DotOperandEncodingAttr>(
             unwrapNoVerifyEncoding(tensorType.getEncoding()));
}

LogicalResult checkAtomicRmwOwnership(RankedTensorType valueTy) {
  if (!valueTy || !valueTy.getEncoding())
    return failure();
  valueTy = cast<RankedTensorType>(cloneTypeWithEncoding(
      valueTy, unwrapNoVerifyEncoding(valueTy.getEncoding())));
  FailureOr<triton::LinearLayout> layout = getC500LinearLayout(valueTy);
  if (failed(layout))
    return failure();
  return success(layout->isInjective() ||
                 succeeded(analyzePredicatedAtomicRmwOwnership(valueTy,
                                                               *layout)));
}

LogicalResult verifyAtomicRmwOwnership(tt::AtomicRMWOp atomicOp) {
  auto valueTy = dyn_cast<RankedTensorType>(atomicOp.getVal().getType());
  if (!valueTy)
    return success();
  valueTy = cast<RankedTensorType>(cloneTypeWithEncoding(
      valueTy, unwrapNoVerifyEncoding(valueTy.getEncoding())));
  FailureOr<triton::LinearLayout> layout = getC500LinearLayout(valueTy);
  if (failed(layout))
    return atomicOp.emitError()
           << "cannot derive a safe C500 ownership layout for tensor "
              "atomic_rmw value type "
           << valueTy;
  if (layout->isInjective()) {
    LDBG("[atomic-rmw-ownership] verified raw injective layout="
         << valueTy.getEncoding() << ", physical-owners="
         << layout->getTotalInDimSize() << ", logical-elements="
         << layout->getTotalOutDimSize()
         << ", location=" << atomicOp.getLoc());
    return success();
  }

  FailureOr<AtomicRmwOwnershipInfo> predicatedOwnership =
      analyzePredicatedAtomicRmwOwnership(valueTy, *layout);
  if (failed(predicatedOwnership))
    return atomicOp.emitError()
           << "tensor atomic_rmw has replicated distributed ownership that "
              "is not made injective by the C500 tensor-atomic active-thread "
              "predicate; layout "
           << valueTy.getEncoding() << " maps "
           << layout->getTotalInDimSize() << " physical owners onto "
           << layout->getTotalOutDimSize()
           << " logical tensor elements; require either a raw injective "
              "layout or an exact injective-and-surjective active-owner "
              "subdomain";

  LDBG("[atomic-rmw-ownership] verified predicated exact cover layout="
       << valueTy.getEncoding() << ", raw-physical-owners="
       << layout->getTotalInDimSize()
       << ", logical-elements=" << predicatedOwnership->logicalElements
       << ", ownership-element-slots-per-thread="
       << predicatedOwnership->elementSlotsPerThread
       << ", active-threads=" << predicatedOwnership->activeThreads << "/"
       << predicatedOwnership->physicalThreads
       << ", blocks=" << predicatedOwnership->blocks
       << ", location=" << atomicOp.getLoc());
  return success();
}

FailureOr<RegisterToSharedContractInfo>
checkRegisterToSharedContract(RankedTensorType registerType,
                              ttg::MemDescType memDescType) {
  RegisterToSharedContractInfo info;
  if (!registerType || !memDescType || !registerType.getEncoding() ||
      !memDescType.getEncoding())
    return failure();
  registerType = cast<RankedTensorType>(cloneTypeWithEncoding(
      registerType, unwrapNoVerifyEncoding(registerType.getEncoding())));
  memDescType = cast<ttg::MemDescType>(cloneTypeWithEncoding(
      memDescType, unwrapNoVerifyEncoding(memDescType.getEncoding())));

  FailureOr<triton::LinearLayout> conversion =
      getRegisterToSharedConversion(registerType, memDescType);
  if (failed(conversion))
    return failure();

  MLIRContext *context = registerType.getContext();
  StringAttr reg = StringAttr::get(context, "register");
  StringAttr lane = StringAttr::get(context, "lane");
  StringAttr warp = StringAttr::get(context, "warp");
  StringAttr block = StringAttr::get(context, "block");
  info.lowerable = conversion->isTrivialOver({block});
  if (!info.lowerable)
    return info;
  triton::LinearLayout threadConversion =
      conversion->sublayout({reg, lane, warp},
                            {StringAttr::get(context, "offset")});
  info.completeCoverage = threadConversion.isSurjective();
  auto freeVariables = threadConversion.getFreeVariableMasks();
  info.freeWarpMask = freeVariables.lookup(warp);
  return info;
}

static StringRef stringifyLayoutContractRole(LayoutContractRole role) {
  switch (role) {
  case LayoutContractRole::DotOperand:
    return "dot operand";
  case LayoutContractRole::RegisterToSharedSink:
    return "register-to-shared sink";
  case LayoutContractRole::RegisterSubview:
    return "register subview";
  }
  llvm_unreachable("unknown Gluon layout contract role");
}

LogicalResult verifyRegisterToSharedContract(
    Operation *op, RankedTensorType registerType, ttg::MemDescType memDescType,
    LayoutContractRole role) {
  FailureOr<RegisterToSharedContractInfo> info =
      checkRegisterToSharedContract(registerType, memDescType);
  if (failed(info))
    return op->emitError()
           << stringifyLayoutContractRole(role)
           << " requires ranked tensor and memdesc types with concrete layouts";

  if (!info->lowerable)
    return op->emitError()
           << stringifyLayoutContractRole(role)
           << " register layout is not linearly lowerable to shared layout; "
           << "complete-coverage=" << info->completeCoverage
           << ", source=" << registerType.getEncoding()
           << ", shared=" << memDescType.getEncoding();

  LDBG(stringifyLayoutContractRole(role)
       << " verified: source=" << registerType.getEncoding()
       << ", shared=" << memDescType.getEncoding()
       << ", complete-coverage=" << info->completeCoverage
       << ", free-warp-mask=" << info->freeWarpMask);
  return success();
}

} // namespace mlir::triton::gpu::metax::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
