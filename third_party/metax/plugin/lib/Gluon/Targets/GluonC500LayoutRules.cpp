#include "Gluon/Targets/GluonC500LayoutRules.h"

#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"
#include "Gluon/Targets/GluonC500Layout.h"
#include "TritonMETAXGPUTransforms/MACACommon.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;

#define DEBUG_TYPE "metax-gluon-c500-dot-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu::metax::gluon {

struct C500DotLayoutOptions {
  int computeCapability = 0;
};

bool isTunableC500Dot(const DotPipelineFacts &facts) {
  if (!facts.dot ||
      llvm::any_of(facts.operands,
                   [](const DotPipelineOperandFacts &operand) {
                     return operand.requiresBsmPermutation();
                   }))
    return false;
  auto isAuto = [](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type &&
           isa_and_nonnull<gd::AutoEncodingAttr>(
               unwrapNoVerifyEncoding(type.getEncoding()));
  };
  tt::DotOp dot = facts.dot;
  // C/D own the candidate accumulator contract. A/B may already be concrete:
  // in that case the selected DotOperand requirement becomes a local
  // convert_layout boundary rather than retagging the fixed producer.
  return isAuto(dot.getC()) && isAuto(dot.getD());
}

LogicalResult checkBsmPermPhysicalContract(RankedTensorType sourceType,
                                           RankedTensorType resultType);

bool supportsC500MacaAccumulatorOrder(Type lhsElementType,
                                     Type rhsElementType,
                                     unsigned colMajor) {
  if (colMajor == 0)
    return true;
  if (colMajor != 1)
    return false;
  auto isSupportedElementType = [](Type type) {
    return type.isF16() || type.isBF16();
  };
  return isSupportedElementType(lhsElementType) &&
         isSupportedElementType(rhsElementType);
}

namespace {

static int getMACAMmaVersionMajor(int computeCapability) {
  if (computeCapability < 70)
    return 0;
  if (computeCapability < 80)
    return 1;
  if (computeCapability < 90)
    return 2;
  if (computeCapability < 100)
    return 3;
  return 4;
}

static int getMACAMmaVersionMinor(int computeCapability) {
  return computeCapability % 10;
}

static constexpr unsigned getDefaultMACAMmaColMajor() { return 0; }

constexpr unsigned kC500MmaTileM = 16;
constexpr unsigned kC500MmaTileN = 16;
constexpr unsigned kC500MmaTileK = 4;
constexpr unsigned kC500BsmPhysicalRegisters = 32;
constexpr unsigned kC500MaxRegisterTransferBits = 128;

struct AsyncWriterAccess {
  ttg::AsyncCopyGlobalToLocalOp copy;
  MemDescAliasPath destination;
};

static FailureOr<Attribute> inferTransEncoding(Attribute encoding,
                                               ArrayRef<int64_t> shape,
                                               ArrayRef<int32_t> order,
                                               Location location) {
  if (!encoding)
    return failure();
  auto *interface =
      cast<tt::DialectInferLayoutInterface>(&encoding.getDialect());
  Attribute result;
  if (failed(interface->inferTransOpEncoding(encoding, shape, order, result,
                                             location)))
    return failure();
  return result;
}

static std::optional<SmallVector<unsigned>>
getSharedMemoryOrder(ttg::MemDescType type) {
  auto shared = dyn_cast_or_null<ttg::SharedEncodingTrait>(
      unwrapNoVerifyEncoding(type.getEncoding()));
  if (shared)
    return ttg::getOrder(shared, type.getShape());
  if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
          unwrapNoVerifyEncoding(type.getEncoding())))
    return SmallVector<unsigned>(blocked.getOrder());
  return std::nullopt;
}

static FailureOr<SmallVector<AsyncWriterAccess, 8>>
collectAsyncWriters(ModuleOp module, GluonMemDescAliasAnalysis &aliases) {
  SmallVector<AsyncWriterAccess, 8> writers;
  WalkResult result = module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
    FailureOr<MemDescAliasPath> destination =
        aliases.getPath(copy->getOperand(1));
    if (failed(destination)) {
      copy.emitError()
          << "async-copy destination has no proven layout view path";
      return WalkResult::interrupt();
    }
    writers.push_back({copy, std::move(*destination)});
    return WalkResult::advance();
  });
  return result.wasInterrupted()
             ? FailureOr<SmallVector<AsyncWriterAccess, 8>>(failure())
             : FailureOr<SmallVector<AsyncWriterAccess, 8>>(
                   std::move(writers));
}

struct BsmPermPhysicalContractInfo {
  unsigned sourceElementSlots = 0;
  unsigned resultElementSlots = 0;
  unsigned elementsN = 0;
  unsigned elementsK = 0;
};

static FailureOr<unsigned>
getRegisterElementSlotCount(RankedTensorType tensorType) {
  FailureOr<tt::LinearLayout> layout = getC500LinearLayout(tensorType);
  if (failed(layout))
    return failure();
  StringAttr registerDim =
      StringAttr::get(tensorType.getContext(), "register");
  if (!layout->hasInDim(registerDim))
    return failure();
  int32_t elementSlots = layout->getInDimSize(registerDim);
  if (elementSlots <= 0)
    return failure();
  return static_cast<unsigned>(elementSlots);
}

static FailureOr<BsmPermPhysicalContractInfo>
analyzeBsmPermPhysicalContract(RankedTensorType sourceType,
                               RankedTensorType resultType) {
  if (!sourceType || !resultType ||
      sourceType.getShape() != resultType.getShape())
    return failure();

  auto srcDot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      unwrapNoVerifyEncoding(sourceType.getEncoding()));
  auto resultDot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      unwrapNoVerifyEncoding(resultType.getEncoding()));
  auto mma = resultDot ? dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
                             resultDot.getParent())
                       : nullptr;
  if (!srcDot || !resultDot || !mma || srcDot.getOpIdx() != 1 ||
      resultDot.getOpIdx() != 1 || srcDot.getParent() != resultDot.getParent() ||
      !sourceType.getElementType().isInteger(32) ||
      (!resultType.getElementType().isF16() &&
       !resultType.getElementType().isBF16()))
    return failure();

  ArrayRef<unsigned> elements = mma.getElementsMNK();
  if (elements.size() != 3 || elements[1] == 0 || elements[2] == 0)
    return failure();
  uint64_t elementsN = elements[1];
  uint64_t elementsK = elements[2];
  std::optional<uint64_t> elementsSize =
      llvm::checkedMulUnsigned(elementsN, elementsK);
  if (!elementsSize ||
      *elementsSize >
          static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      elementsN % 2 != 0 || elementsK % 2 != 0)
    return failure();

  FailureOr<unsigned> sourceElementSlots =
      getRegisterElementSlotCount(sourceType);
  FailureOr<unsigned> resultElementSlots =
      getRegisterElementSlotCount(resultType);
  if (failed(sourceElementSlots) || failed(resultElementSlots) ||
      *sourceElementSlots != kC500BsmPhysicalRegisters ||
      *resultElementSlots != kC500BsmPhysicalRegisters)
    return failure();

  SmallVector<unsigned, kC500BsmPhysicalRegisters> writes(
      kC500BsmPhysicalRegisters, 0);
  for (uint64_t j = 0; j < *elementsSize / 2; j += 2 * elementsN) {
    for (uint64_t vector = 0; vector < elementsN / 2; ++vector) {
      uint64_t source = (j / elementsN + vector) * elementsK;
      uint64_t output = 2 * vector + elementsN * (j / elementsK);
      SmallVector<uint64_t, 4> sourceIndices = {
          source, source + 1, source + elementsN, source + elementsN + 1};
      if (llvm::any_of(sourceIndices, [&](uint64_t index) {
            return index >= *sourceElementSlots;
          }))
        return failure();
      SmallVector<uint64_t, 8> outputIndices = {
          output,
          output + 1,
          output + elementsK,
          output + elementsK + 1,
          output + 2 * elementsK,
          output + 2 * elementsK + 1,
          output + 3 * elementsK,
          output + 3 * elementsK + 1};
      for (uint64_t index : outputIndices) {
        if (index >= *resultElementSlots)
          return failure();
        ++writes[index];
      }
    }
  }
  if (llvm::any_of(writes, [](unsigned count) { return count != 1; }))
    return failure();

  return BsmPermPhysicalContractInfo{
      *sourceElementSlots, *resultElementSlots,
      static_cast<unsigned>(elementsN), static_cast<unsigned>(elementsK)};
}

static FailureOr<BsmPermPhysicalContractInfo>
analyzeBsmPermPhysicalContract(ttg::BsmPermOp op) {
  auto sourceType = dyn_cast<RankedTensorType>(op.getSrc1().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!sourceType || !resultType)
    return failure();
  return analyzeBsmPermPhysicalContract(sourceType, resultType);
}

/// The reduction extent is a semantic property of the current dot. Parent
/// tensor shapes are merely storage/view context and cannot enlarge K without
/// an explicit reduction-tiling contract.
static FailureOr<int64_t> getLogicalK(tt::DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  int64_t aK = aType.getShape()[1];
  int64_t bK = bType.getShape()[0];
  if (ShapedType::isDynamic(aK) || ShapedType::isDynamic(bK) || aK != bK)
    return failure();
  return aK;
}

static std::optional<SmallVector<unsigned, 2>> getEffectiveDotOperandOrder(
    const DotOperandMemoryFacts &memoryFacts) {
  if (!memoryFacts.hasOrderSources ||
      !memoryFacts.hasProvenPhysicalOrder() ||
      *memoryFacts.contiguousDimension >= 2)
    return std::nullopt;
  unsigned contiguousDimension = *memoryFacts.contiguousDimension;
  return SmallVector<unsigned, 2>{contiguousDimension,
                                  1 - contiguousDimension};
}

static bool isSupportedRank2Order(ArrayRef<unsigned> order) {
  return order == ArrayRef<unsigned>({0, 1}) ||
         order == ArrayRef<unsigned>({1, 0});
}

static std::optional<unsigned> getOperandElementsPerMma(
    Type inputType, Type resultType, tt::InputPrecision inputPrecision,
    int computeCapability) {
  if (resultType.isF32()) {
    if (inputType.isF16() || inputType.isBF16())
      return 4;
    if (inputType.isF32() && inputPrecision == tt::InputPrecision::TF32)
      return 2;
    if (inputType.isF32())
      return 1;
    if (isa<Float8E4M3FNType, Float8E5M2Type>(inputType) &&
        computeCapability >= 86)
      return 8;
  }
  if (resultType.isF64() && inputType.isF64())
    return 1;
  if (resultType.isSignlessInteger(32) && inputType.isSignlessInteger(8))
    return computeCapability >= 86 ? 8 : 4;
  return std::nullopt;
}

static ttg::MACAMmaEncodingAttr buildMmaEncoding(
    MLIRContext *context, int computeCapability, Type aElementType,
    Type bElementType, ArrayRef<unsigned> aOrder, ArrayRef<unsigned> bOrder,
    ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> elementsMNK,
    unsigned colMajor) {
  int versionMajor = getMACAMmaVersionMajor(computeCapability);
  int versionMinor = getMACAMmaVersionMinor(computeCapability);
  SmallVector<unsigned, 3> elements(elementsMNK.begin(), elementsMNK.end());
  bool isATrans = getIfLdsTrans(elements, versionMajor, versionMinor, aOrder,
                                /*isA=*/true, aElementType);
  bool isBTrans = getIfLdsTrans(elements, versionMajor, versionMinor, bOrder,
                                /*isA=*/false, bElementType);
  SmallVector<unsigned, 2> strides{
      isATrans ? ttg::getLdsTransVec(aElementType) : 1,
      isBTrans ? ttg::getLdsTransVec(bElementType) : 1};
  return ttg::MACAMmaEncodingAttr::get(
      context, versionMajor, versionMinor, warpsPerCTA, elements,
      colMajor, isATrans, isBTrans, strides);
}

static LogicalResult validateBsmLayout(RankedTensorType bType,
                                       Attribute bEncoding) {
  if ((!bType.getElementType().isF16() &&
       !bType.getElementType().isBF16()) ||
      !bEncoding)
    return failure();

  auto resultType = bType.cloneWithEncoding(bEncoding);
  auto sourceType = RankedTensorType::get(
      bType.getShape(), IntegerType::get(bType.getContext(), 32), bEncoding);
  return success(succeeded(
      checkBsmPermPhysicalContract(sourceType, resultType)));
}

struct DotPhysicalFacts {
  RankedTensorType aType;
  RankedTensorType bType;
  RankedTensorType dType;
  SmallVector<int64_t, 2> tileShape;
  int64_t logicalK = 0;
  unsigned numWarps = 0;
  unsigned operandElementsPerMma = 0;
  bool usesBsm = false;
};

/// Build the finite exact K-grouping domain. Physical K contiguity selects the
/// fallback order; it does not make the remaining exact groupings illegal.
static FailureOr<SmallVector<unsigned, 4>>
buildMmaKGroupingDomain(const DotPhysicalFacts &facts,
                        ArrayRef<unsigned> aOrder,
                        ArrayRef<unsigned> bOrder) {
  unsigned native = facts.operandElementsPerMma;
  if (!native || facts.logicalK % kC500MmaTileK)
    return failure();
  unsigned aBits = facts.aType.getElementType().getIntOrFloatBitWidth();
  unsigned bBits = facts.bType.getElementType().getIntOrFloatBitWidth();
  unsigned transferBits = std::max(aBits, bBits);
  if (!aBits || !bBits || transferBits > kC500MaxRegisterTransferBits)
    return failure();

  uint64_t reductionAtoms =
      static_cast<uint64_t>(facts.logicalK) / kC500MmaTileK;
  if (reductionAtoms < native ||
      facts.logicalK % (kC500MmaTileK * native))
    return failure();
  SmallVector<unsigned, 4> groupings;
  uint64_t maximum = std::min(
      reductionAtoms,
      static_cast<uint64_t>(kC500MaxRegisterTransferBits / transferBits));
  for (uint64_t grouping = llvm::bit_floor(maximum); grouping >= native;
       grouping >>= 1) {
    if (grouping % native == 0 &&
        facts.logicalK % (kC500MmaTileK * grouping) == 0)
      groupings.push_back(static_cast<unsigned>(grouping));
    if (grouping == native)
      break;
  }
  if (!llvm::is_contained(groupings, native))
    return failure();

  bool kContiguous =
      llvm::equal(aOrder,
                  ttg::getOrderForDotOperand(
                      /*opIdx=*/0, facts.aType.getRank(), /*kContig=*/true)) &&
      llvm::equal(bOrder,
                  ttg::getOrderForDotOperand(
                      /*opIdx=*/1, facts.bType.getRank(), /*kContig=*/true));
  if (!kContiguous && groupings.front() != native) {
    llvm::erase(groupings, native);
    groupings.insert(groupings.begin(), native);
  }
  return groupings;
}

static std::optional<std::string>
validatePlan(const DotPhysicalFacts &facts, const DotLayoutPlan &plan) {
  auto dotMma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
      plan.mma);
  auto aEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      plan.operandA);
  auto bEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      plan.operandB);
  if (!dotMma || !aEncoding || !bEncoding)
    return "plan does not contain a complete MACA dot assignment";
  if (aEncoding.getOpIdx() != 0 || bEncoding.getOpIdx() != 1 ||
      aEncoding.getParent() != dotMma || bEncoding.getParent() != dotMma)
    return "dot operand roles or parent encodings are inconsistent";
  if (!supportsC500MacaAccumulatorOrder(facts.aType.getElementType(),
                                        facts.bType.getElementType(),
                                        dotMma.getColMajor()))
    return "accumulator order is not supported by the C500 MMA lowering for "
           "the operand element types";

  ArrayRef<unsigned> warps = dotMma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = dotMma.getElementsMNK();
  if (warps.size() != 2 || elements.size() != 3 || warps[0] == 0 ||
      warps[1] == 0 || elements[0] == 0 || elements[1] == 0 ||
      elements[2] == 0 ||
      static_cast<uint64_t>(warps[0]) * warps[1] != facts.numWarps)
    return "warp grid or elementsMNK violates the C500 rank-2 contract";
  if (elements[2] < facts.operandElementsPerMma ||
      elements[2] % facts.operandElementsPerMma != 0)
    return "elementsK is not an integer number of C500 MMA operands";

  auto dotType = RankedTensorType::get(facts.dType.getShape(),
                                       facts.dType.getElementType(), dotMma);
  auto aType = RankedTensorType::get(facts.aType.getShape(),
                                     facts.aType.getElementType(), aEncoding);
  auto bType = RankedTensorType::get(facts.bType.getShape(),
                                     facts.bType.getElementType(), bEncoding);
  if (failed(getC500LinearLayout(dotType)) ||
      failed(getC500LinearLayout(aType)) ||
      failed(getC500LinearLayout(bType)))
    return "plan is not representable by the C500 LinearLayout model";

  int64_t m = facts.dType.getShape()[0];
  int64_t n = facts.dType.getShape()[1];
  int64_t k = facts.aType.getShape()[1];
  int64_t repM = ttg::getNumRepM(m, warps, elements[0]);
  int64_t repN = ttg::getNumRepN(n, warps, elements[1]);
  int64_t repK = ttg::getNumRepK(k, elements[2]);
  int64_t coveredM = repM * static_cast<int64_t>(warps[0]) *
                     kC500MmaTileM * elements[0];
  int64_t coveredN = repN * static_cast<int64_t>(warps[1]) *
                     kC500MmaTileN * elements[1];
  int64_t coveredK = repK * kC500MmaTileK * elements[2];
  if (coveredM < m || coveredN < n || coveredK != k)
    return "plan does not exactly cover the lowering's M/N/K domain";

  if (facts.usesBsm &&
      failed(validateBsmLayout(facts.bType, bEncoding)))
    return "plan violates the C500 32-register BSM permutation ABI";
  return std::nullopt;
}

static DotLayoutPlan
makeDotLayoutPlan(tt::DotOp dotOp,
                  ttg::MACAMmaEncodingAttr accumulatorEncoding) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  auto aEncoding = ttg::DotOperandEncodingAttr::get(
      dotOp.getContext(), /*opIdx=*/0, accumulatorEncoding,
      aType.getElementType());
  auto bEncoding = ttg::DotOperandEncodingAttr::get(
      dotOp.getContext(), /*opIdx=*/1, accumulatorEncoding,
      bType.getElementType());
  DotLayoutPlan result;
  result.mma = accumulatorEncoding;
  result.operandA = aEncoding;
  result.operandB = bEncoding;
  return result;
}

/// Construct the canonical non-BSM layout directly. Square output tiles prefer
/// a balanced warp grid so the dot layout can also carry exact-same-encoding
/// accumulator slices. Other shapes retain the lowering's legacy order.
static FailureOr<DotLayoutPlan> buildNonBsmDotLayout(
    tt::DotOp dotOp, const DotPhysicalFacts &facts,
    ArrayRef<unsigned> aOrder, ArrayRef<unsigned> bOrder, unsigned elementsK,
    const C500DotLayoutOptions &options) {
  unsigned colMajor = getDefaultMACAMmaColMajor();
  unsigned packedDimension = colMajor ? 0 : 1;
  unsigned replicaDimension = 1 - packedDimension;

  uint64_t atoms[2]{
      llvm::divideCeil(static_cast<uint64_t>(facts.tileShape[0]),
                       static_cast<uint64_t>(kC500MmaTileM)),
      llvm::divideCeil(static_cast<uint64_t>(facts.tileShape[1]),
                       static_cast<uint64_t>(kC500MmaTileN))};
  uint64_t replicaAtomSize =
      replicaDimension == 0 ? kC500MmaTileM : kC500MmaTileN;
  uint64_t replicaAtoms = llvm::divideCeil(
      static_cast<uint64_t>(facts.tileShape[replicaDimension]),
      replicaAtomSize);

  auto deriveElements = [](uint64_t atomCount, unsigned warpPartitions,
                           unsigned structuralFactor) {
    uint64_t required =
        llvm::divideCeil(atomCount, uint64_t(warpPartitions));
    return static_cast<unsigned>(llvm::PowerOf2Ceil(std::max<uint64_t>(
        {required, static_cast<uint64_t>(structuralFactor), 1})));
  };

  SmallVector<unsigned, 4> replicaWarpCandidates;
  for (unsigned replicaWarps = 1; replicaWarps <= facts.numWarps;
       replicaWarps <<= 1) {
    if (facts.numWarps % replicaWarps ||
        static_cast<uint64_t>(replicaWarps) > replicaAtoms)
      continue;
    replicaWarpCandidates.push_back(replicaWarps);
  }
  if (atoms[0] == atoms[1])
    llvm::sort(replicaWarpCandidates, [&](unsigned lhs, unsigned rhs) {
      auto imbalance = [&](unsigned replicaWarps) {
        unsigned packedWarps = facts.numWarps / replicaWarps;
        return std::max(replicaWarps, packedWarps) -
               std::min(replicaWarps, packedWarps);
      };
      unsigned lhsImbalance = imbalance(lhs);
      unsigned rhsImbalance = imbalance(rhs);
      return lhsImbalance != rhsImbalance ? lhsImbalance < rhsImbalance
                                          : lhs < rhs;
    });

  for (unsigned replicaWarps : replicaWarpCandidates) {
    unsigned packedWarps = facts.numWarps / replicaWarps;
    unsigned replicaElements =
        deriveElements(replicaAtoms, replicaWarps, 1);
    if (static_cast<uint64_t>(replicaWarps) * replicaElements >
        replicaAtoms)
      continue;
    unsigned packedElements =
        deriveElements(atoms[packedDimension], packedWarps, 1);

    SmallVector<unsigned, 2> warps(2);
    SmallVector<unsigned, 3> elements(3);
    warps[replicaDimension] = replicaWarps;
    warps[packedDimension] = packedWarps;
    elements[replicaDimension] = replicaElements;
    elements[packedDimension] = packedElements;
    elements[2] = elementsK;
    auto encoding = buildMmaEncoding(
        dotOp.getContext(), options.computeCapability,
        facts.aType.getElementType(), facts.bType.getElementType(), aOrder,
        bOrder, warps, elements, colMajor);
    DotLayoutPlan plan = makeDotLayoutPlan(dotOp, encoding);
    if (validatePlan(facts, plan))
      continue;
    return plan;
  }
  return dotOp.emitError()
         << "no C500 warp grid satisfies accumulator coverage and "
            "LinearLayout contracts";
}

static SmallVector<unsigned, 4> getPowerOfTwoGroupings(unsigned maximum) {
  SmallVector<unsigned, 4> result;
  for (unsigned value = maximum; value != 0; value >>= 1)
    result.push_back(value);
  return result;
}

/// Enumerate the same finite physical axes consumed by the C500 lowering. The
/// deterministic constructor above remains entry zero; every alternative is a
/// complete MACA parent plus both DotOperand encodings and passes the same
/// LinearLayout/coverage verifier.
static FailureOr<SmallVector<DotLayoutPlan, 8>> buildNonBsmDotLayoutDomain(
    tt::DotOp dotOp, const DotPhysicalFacts &facts,
    ArrayRef<unsigned> aOrder, ArrayRef<unsigned> bOrder,
    ArrayRef<unsigned> elementsKDomain,
    const C500DotLayoutOptions &options) {
  if (elementsKDomain.empty())
    return failure();
  FailureOr<DotLayoutPlan> fallback = buildNonBsmDotLayout(
      dotOp, facts, aOrder, bOrder, elementsKDomain.front(), options);
  if (failed(fallback))
    return failure();

  SmallVector<DotLayoutPlan, 8> result{*fallback};
  DenseSet<Attribute> seen;
  seen.insert(fallback->mma);
  uint64_t atoms[2]{
      llvm::divideCeil(static_cast<uint64_t>(facts.tileShape[0]),
                       static_cast<uint64_t>(kC500MmaTileM)),
      llvm::divideCeil(static_cast<uint64_t>(facts.tileShape[1]),
                       static_cast<uint64_t>(kC500MmaTileN))};
  auto deriveElements = [](uint64_t atomCount, unsigned warpPartitions) {
    return static_cast<unsigned>(llvm::PowerOf2Ceil(
        std::max<uint64_t>(llvm::divideCeil(atomCount,
                                            uint64_t(warpPartitions)),
                           1)));
  };

  for (unsigned colMajor : {getDefaultMACAMmaColMajor(),
                            1u - getDefaultMACAMmaColMajor()}) {
    if (!supportsC500MacaAccumulatorOrder(facts.aType.getElementType(),
                                          facts.bType.getElementType(),
                                          colMajor))
      continue;
    unsigned packedDimension = colMajor ? 0 : 1;
    unsigned replicaDimension = 1 - packedDimension;
    for (unsigned replicaWarps = 1; replicaWarps <= facts.numWarps;
         replicaWarps <<= 1) {
      if (facts.numWarps % replicaWarps ||
          static_cast<uint64_t>(replicaWarps) > atoms[replicaDimension])
        continue;
      unsigned packedWarps = facts.numWarps / replicaWarps;
      unsigned maximumReplicaElements =
          deriveElements(atoms[replicaDimension], replicaWarps);
      if (static_cast<uint64_t>(replicaWarps) *
              maximumReplicaElements >
          atoms[replicaDimension])
        continue;
      unsigned maximumPackedElements =
          deriveElements(atoms[packedDimension], packedWarps);
      for (unsigned replicaElements :
           getPowerOfTwoGroupings(maximumReplicaElements)) {
        for (unsigned packedElements :
             getPowerOfTwoGroupings(maximumPackedElements)) {
          for (unsigned elementsK : elementsKDomain) {
            SmallVector<unsigned, 2> warps(2);
            SmallVector<unsigned, 3> elements(3);
            warps[replicaDimension] = replicaWarps;
            warps[packedDimension] = packedWarps;
            elements[replicaDimension] = replicaElements;
            elements[packedDimension] = packedElements;
            elements[2] = elementsK;
            auto encoding = buildMmaEncoding(
                dotOp.getContext(), options.computeCapability,
                facts.aType.getElementType(), facts.bType.getElementType(),
                aOrder, bOrder, warps, elements, colMajor);
            if (!seen.insert(encoding).second)
              continue;
            DotLayoutPlan plan = makeDotLayoutPlan(dotOp, encoding);
            if (!validatePlan(facts, plan))
              result.push_back(std::move(plan));
          }
        }
      }
    }
  }
  return result;
}

} // namespace

LogicalResult inferC500DotMemoryFacts(
    ModuleOp module, tt::ModuleAxisInfoAnalysis &axisInfo,
    SmallVectorImpl<DotPipelineFacts> &pipelines) {
  GluonMemDescAliasAnalysis aliases(module);
  if (failed(aliases.initialize()))
    return failure();
  FailureOr<SmallVector<AsyncWriterAccess, 8>> writers =
      collectAsyncWriters(module, aliases);
  if (failed(writers))
    return failure();

  for (DotPipelineFacts &pipeline : pipelines) {
    for (auto [operandIndex, operand] :
         llvm::enumerate(pipeline.operands)) {
      DotOperandMemoryFacts &facts = operand.memory;
      facts.hasOrderSources = !operand.terminals.empty();
      std::optional<unsigned> contiguousDimension;
      unsigned maxContiguousElements =
          std::numeric_limits<unsigned>::max();
      bool hasConsensus = true;
      bool allSourcesProveAddressContiguity = true;

      auto meet = [&](unsigned dimension, unsigned contiguousElements,
                      int64_t extent) {
        if (dimension >= 2 || extent <= 0 ||
            (contiguousDimension && *contiguousDimension != dimension)) {
          hasConsensus = false;
          return;
        }
        contiguousDimension = dimension;
        if (contiguousElements == 0) {
          allSourcesProveAddressContiguity = false;
          return;
        }
        maxContiguousElements =
            std::min(maxContiguousElements,
                     std::min(contiguousElements,
                              static_cast<unsigned>(extent)));
      };

      if (operand.hasDotProducer) {
        Value operandValue =
            operandIndex == 0 ? pipeline.dot.getA() : pipeline.dot.getB();
        auto operandType =
            dyn_cast<RankedTensorType>(operandValue.getType());
        if (!operandType ||
            operand.logicalKDimension >=
                static_cast<unsigned>(operandType.getRank()))
          return pipeline.dot.emitError()
                 << "dot-produced operand has no valid semantic K dimension";
        facts.hasOrderSources = true;
        facts.contiguousDimension = operand.logicalKDimension;
        facts.maxContiguousElements = 0;
        LDBG("[dot-memory-facts] dot="
             << pipeline.dot.getLoc() << ", operand=" << operandIndex
             << ", source=dot-producer, contiguous-dim="
             << operand.logicalKDimension);
        continue;
      }

      auto meetValue = [&](Value value, Operation *anchor) {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        FailureOr<c500::Rank2ContiguousOrderInfo> order =
            c500::inferRank2ContiguousOrder(value, &axisInfo);
        if (!type || failed(order) || order->order.empty()) {
          hasConsensus = false;
          return;
        }
        unsigned dimension = order->order.front();
        meet(dimension, order->maxContiguousElements,
             type.getShape()[dimension]);
        LLVM_DEBUG({
          DBGS() << "[dot-order-source] dot=" << pipeline.dot.getLoc()
                 << ", operand=" << operandIndex
                 << ", source=" << anchor->getName() << ", order=[";
          llvm::interleaveComma(order->order, llvm::dbgs());
          llvm::dbgs()
              << "], contiguous-elements=" << order->maxContiguousElements
              << "\n";
        });
      };

      for (Value terminal : operand.terminals) {
        auto localLoad = terminal.getDefiningOp<ttg::LocalLoadOp>();
        if (!localLoad) {
          if (auto load = terminal.getDefiningOp<tt::LoadOp>())
            meetValue(load.getPtr(), load);
          else if (Operation *def = terminal.getDefiningOp())
            meetValue(terminal, def);
          else
            hasConsensus = false;
          continue;
        }

        auto loadFacts = llvm::find_if(
            operand.localLoads,
            [&](const DotPipelineLocalLoadFacts &candidate) {
              return candidate.localLoad == localLoad;
            });
        if (loadFacts == operand.localLoads.end())
          return localLoad.emitError()
                 << "dot local_load is missing its shared view path";

        bool foundWriter = false;
        for (AsyncWriterAccess &writer : *writers) {
          if (writer.destination.root != loadFacts->sharedView.root)
            continue;
          foundWriter = true;
          auto sourceType =
              dyn_cast<RankedTensorType>(writer.copy.getSrc().getType());
          FailureOr<c500::Rank2ContiguousOrderInfo> order =
              c500::inferRank2ContiguousOrder(writer.copy.getSrc(), &axisInfo,
                                              writer.copy);
          if (!sourceType || failed(order) || order->order.empty()) {
            hasConsensus = false;
            continue;
          }
          unsigned writerDimension = order->order.front();
          std::optional<unsigned> rootDimension =
              projectDimensionToRoot(writer.destination, writerDimension);
          std::optional<unsigned> consumerDimension =
              rootDimension
                  ? projectDimensionFromRoot(loadFacts->sharedView,
                                             *rootDimension)
                  : std::nullopt;
          auto consumerType =
              dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
          if (!consumerDimension || !consumerType ||
              *consumerDimension >=
                  static_cast<unsigned>(consumerType.getRank())) {
            hasConsensus = false;
            continue;
          }
          meet(*consumerDimension, order->maxContiguousElements,
               consumerType.getShape()[*consumerDimension]);
        }
        if (!foundWriter) {
          auto sharedType =
              dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
          if (!sharedType || sharedType.getRank() != 2) {
            hasConsensus = false;
            continue;
          }
          Attribute encoding =
              unwrapNoVerifyEncoding(sharedType.getEncoding());
          if (!encoding || isa<gd::AutoEncodingAttr>(encoding)) {
            hasConsensus = false;
            continue;
          }
          SmallVector<unsigned> order = ttg::getOrder(sharedType);
          if (order.size() != 2) {
            hasConsensus = false;
            continue;
          }
          meet(order.front(), /*contiguousElements=*/0,
               sharedType.getShape()[order.front()]);
        }
      }

      if (facts.hasOrderSources && hasConsensus && contiguousDimension) {
        facts.contiguousDimension = contiguousDimension;
        if (allSourcesProveAddressContiguity &&
            maxContiguousElements != std::numeric_limits<unsigned>::max())
          facts.maxContiguousElements = maxContiguousElements;
      }
      LDBG("[dot-memory-facts] dot="
           << pipeline.dot.getLoc() << ", operand=" << operandIndex
           << ", has-sources=" << facts.hasOrderSources
           << ", consensus=" << hasConsensus << ", contiguous-dim="
           << (facts.contiguousDimension
                   ? std::to_string(*facts.contiguousDimension)
                   : "unknown")
           << ", max-contiguous-elements=" << facts.maxContiguousElements);
    }
  }
  return success();
}

LogicalResult checkBsmPermPhysicalContract(RankedTensorType sourceType,
                                           RankedTensorType resultType) {
  return success(
      succeeded(analyzeBsmPermPhysicalContract(sourceType, resultType)));
}

LogicalResult verifyBsmPermPhysicalContract(ttg::BsmPermOp op) {
  FailureOr<BsmPermPhysicalContractInfo> contract =
      analyzeBsmPermPhysicalContract(op);
  if (failed(contract)) {
    auto sourceType = dyn_cast<RankedTensorType>(op.getSrc1().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    return op.emitError()
           << "bsm_perm does not satisfy the C500 physical permutation "
              "contract: source/result must be matching MACA B operands, "
              "the split-LDS carrier and result must each contain 32 "
              "per-thread LinearLayout element slots, and the target "
              "permutation must write every result slot exactly once; "
              "source="
           << sourceType << ", result=" << resultType;
  }
  LDBG("[bsm-perm-contract] verified source-element-slots="
       << contract->sourceElementSlots
       << ", result-element-slots=" << contract->resultElementSlots
       << ", elements-N=" << contract->elementsN
       << ", elements-K=" << contract->elementsK
       << ", location=" << op.getLoc());
  return success();
}

static FailureOr<SmallVector<DotLayoutPlan, 8>> inferDotLayoutDomain(
    tt::DotOp dotOp,
    const std::array<DotOperandMemoryFacts, 2> &operandMemory,
    const std::array<DotOperandPathFacts, 2> &operandPaths,
    int computeCapability) {
  auto aType = dyn_cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(dotOp.getB().getType());
  auto cType = dyn_cast<RankedTensorType>(dotOp.getC().getType());
  auto dType = dyn_cast<RankedTensorType>(dotOp.getD().getType());
  if (!aType || !bType || !cType || !dType || aType.getRank() != 2 ||
      bType.getRank() != 2 || cType.getRank() != 2 || dType.getRank() != 2)
    return dotOp.emitError()
           << "C500 dot layout inference requires rank-2 A/B/C/D tensors";
  if (getMACAMmaVersionMajor(computeCapability) != 2)
    return dotOp.emitError()
           << "C500 dot layout inference requires MACA MMA version 2";

  auto operandElements = getOperandElementsPerMma(
      aType.getElementType(), dType.getElementType(),
      dotOp.getInputPrecision(), computeCapability);
  int numWarps = ttg::lookupNumWarps(dotOp);
  if (!operandElements || numWarps <= 0 ||
      static_cast<uint64_t>(numWarps) >
          std::numeric_limits<unsigned>::max())
    return dotOp.emitError()
           << "cannot derive the C500 MMA atom or warp count";

  FailureOr<int64_t> logicalK = getLogicalK(dotOp);
  if (operandPaths[0].passesThroughBsmPermutation)
    return dotOp.emitError()
           << "C500 BSM permutation is only defined for dot operand B";
  if (failed(logicalK))
    return dotOp.emitError()
           << "cannot derive a unique semantic dot tiling extent from "
              "the dot reduction contract";
  DotPhysicalFacts facts{aType,
                       bType,
                       dType,
                       SmallVector<int64_t, 2>(dType.getShape()),
                       *logicalK,
                       static_cast<unsigned>(numWarps),
                       *operandElements,
                       operandPaths[1].passesThroughBsmPermutation};
  if (facts.tileShape.size() != 2 || facts.tileShape[0] <= 0 ||
      facts.tileShape[1] <= 0 || facts.logicalK <= 0 ||
      llvm::any_of(facts.tileShape, [](int64_t extent) {
        return extent > std::numeric_limits<int>::max();
      }) ||
      facts.logicalK > std::numeric_limits<int>::max())
    return dotOp.emitError()
           << "C500 dot layout inference requires positive static M/N/K";

  auto fixedFragment = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
      unwrapNoVerifyEncoding(cType.getEncoding()));
  if (fixedFragment) {
    DotLayoutPlan plan = makeDotLayoutPlan(dotOp, fixedFragment);
    if (auto error = validatePlan(facts, plan))
      return dotOp.emitError()
             << "explicit C500 accumulator layout is not lowerable: "
             << *error << "; fragment encoding=" << fixedFragment;
    return SmallVector<DotLayoutPlan, 8>{std::move(plan)};
  }

  auto aOrder = getEffectiveDotOperandOrder(operandMemory[0]);
  auto bOrder = getEffectiveDotOperandOrder(operandMemory[1]);
  if (!aOrder || !bOrder || !isSupportedRank2Order(*aOrder) ||
      !isSupportedRank2Order(*bOrder))
    return dotOp.emitError()
           << "cannot prove the physical contiguous order of both dot "
              "operands";

  if (!llvm::isPowerOf2_64(facts.numWarps))
    return dotOp.emitError()
           << "C500 rank-2 MMA requires a power-of-two warp count";

  // Instruction-atom coverage is a hard lowerability constraint. The BSM path
  // below additionally fixes its register carrier; non-BSM candidates are
  // delegated to the finite domain builder.
  uint64_t atomsM = llvm::divideCeil(
      static_cast<uint64_t>(facts.tileShape[0]),
      static_cast<uint64_t>(kC500MmaTileM));
  uint64_t atomsN = llvm::divideCeil(
      static_cast<uint64_t>(facts.tileShape[1]),
      static_cast<uint64_t>(kC500MmaTileN));
  unsigned warpsM = 0;
  unsigned warpsN = 0;
  unsigned elementsM = 0;
  unsigned elementsN = 0;
  unsigned elementsK = facts.operandElementsPerMma;
  SmallVector<unsigned, 4> elementsKDomain{elementsK};
  if (!facts.usesBsm) {
    FailureOr<SmallVector<unsigned, 4>> groupings =
        buildMmaKGroupingDomain(facts, *aOrder, *bOrder);
    if (failed(groupings))
      return dotOp.emitError()
             << "cannot derive an exact C500 MMA K-grouping domain";
    elementsKDomain = std::move(*groupings);
    elementsK = elementsKDomain.front();
  }
  if (facts.usesBsm) {
    unsigned packedDimension = getDefaultMACAMmaColMajor() ? 0 : 1;
    unsigned replicaDimension = 1 - packedDimension;
    uint64_t replicaAtomSize =
        replicaDimension == 0 ? kC500MmaTileM : kC500MmaTileN;
    uint64_t minReplicaAtoms = llvm::divideCeil(
        static_cast<uint64_t>(facts.tileShape[replicaDimension]),
        replicaAtomSize);
    if (minReplicaAtoms == 0)
      return dotOp.emitError()
             << "C500 accumulator has no finite replica domain";
    auto deriveElements = [](uint64_t atoms, unsigned warpPartitions,
                             unsigned structuralFactor) {
      uint64_t required = llvm::divideCeil(atoms, uint64_t(warpPartitions));
      return static_cast<unsigned>(llvm::PowerOf2Ceil(std::max<uint64_t>(
          {required, static_cast<uint64_t>(structuralFactor), 1})));
    };

    // The BSM lowering consumes and produces exactly 32 registers per thread.
    // Its permutation loop writes elementsN * elementsK distinct results, so
    // bijectivity gives elementsN * elementsK = 32. Any K repetition would
    // multiply that carrier, hence the full logical reduction must fit exactly
    // in one MMA K fragment. These equations determine the layout directly;
    // no layout candidate enumeration or kernel-specific shape table is used.
    if (facts.logicalK % kC500MmaTileK != 0)
      return dotOp.emitError()
             << "C500 BSM requires K to be an exact multiple of the native "
                "MMA K atom "
             << kC500MmaTileK;
    uint64_t reductionAtoms =
        static_cast<uint64_t>(facts.logicalK) / kC500MmaTileK;
    if (!llvm::isPowerOf2_64(reductionAtoms) ||
        reductionAtoms < facts.operandElementsPerMma ||
        reductionAtoms > kC500BsmPhysicalRegisters ||
        kC500BsmPhysicalRegisters % reductionAtoms != 0)
      return dotOp.emitError()
             << "C500 BSM cannot represent K=" << facts.logicalK
             << " with its " << kC500BsmPhysicalRegisters
             << "-register permutation carrier";

    elementsK = static_cast<unsigned>(reductionAtoms);
    elementsN = kC500BsmPhysicalRegisters / elementsK;

    uint64_t requiredWarpsN = llvm::divideCeil(
        atomsN, static_cast<uint64_t>(elementsN));
    uint64_t roundedWarpsN = llvm::PowerOf2Ceil(requiredWarpsN);
    if (roundedWarpsN == 0 || roundedWarpsN > facts.numWarps ||
        facts.numWarps % roundedWarpsN != 0)
      return dotOp.emitError()
             << "C500 BSM N fragment requires " << roundedWarpsN
             << " warp partitions, incompatible with num-warps="
             << facts.numWarps;
    warpsN = static_cast<unsigned>(roundedWarpsN);
    warpsM = facts.numWarps / warpsN;
    if (replicaDimension == 0) {
      if (warpsM > minReplicaAtoms)
        return dotOp.emitError()
               << "C500 BSM warp grid consumes the accumulator replica "
                  "slice domain";
      elementsM = deriveElements(minReplicaAtoms, warpsM, 1);
      if (static_cast<uint64_t>(warpsM) * elementsM > minReplicaAtoms)
        return dotOp.emitError()
               << "C500 BSM elementsM consumes the accumulator replica "
                  "slice domain";
    } else {
      if (static_cast<uint64_t>(warpsN) * elementsN > minReplicaAtoms)
        return dotOp.emitError()
               << "C500 BSM N carrier consumes the accumulator replica "
                  "slice domain";
      elementsM = deriveElements(atomsM, warpsM, 1);
    }
  } else {
    return buildNonBsmDotLayoutDomain(dotOp, facts, *aOrder, *bOrder,
                                      elementsKDomain,
                                      C500DotLayoutOptions{computeCapability});
  }

  auto buildPlan = [&](ArrayRef<unsigned> warps,
                       ArrayRef<unsigned> elements)
      -> FailureOr<DotLayoutPlan> {
    auto encoding = buildMmaEncoding(
        dotOp.getContext(), computeCapability, aType.getElementType(),
        bType.getElementType(), *aOrder, *bOrder, warps, elements,
        getDefaultMACAMmaColMajor());
    DotLayoutPlan plan = makeDotLayoutPlan(dotOp, encoding);
    if (std::optional<std::string> error = validatePlan(facts, plan))
      return failure();
    return plan;
  };

  SmallVector<unsigned, 2> warps{warpsM, warpsN};
  SmallVector<unsigned, 3> elements{elementsM, elementsN, elementsK};
  FailureOr<DotLayoutPlan> plan = buildPlan(warps, elements);
  if (failed(plan))
    return dotOp.emitError() << "derived C500 BSM layout is not lowerable";
  return SmallVector<DotLayoutPlan, 8>{std::move(*plan)};
}

FailureOr<SmallVector<DotLayoutPlan, 8>>
inferC500DotLayoutDomain(const DotPipelineFacts &facts,
                         int computeCapability) {
  if (!facts.dot)
    return failure();
  tt::DotOp dot = facts.dot;
  std::array<DotOperandMemoryFacts, 2> memory{
      facts.operands[0].memory, facts.operands[1].memory};
  std::array<DotOperandPathFacts, 2> paths{
      facts.operands[0].path, facts.operands[1].path};
  FailureOr<SmallVector<DotLayoutPlan, 8>> domain =
      inferDotLayoutDomain(facts.dot, memory, paths, computeCapability);
  if (failed(domain))
    return failure();

  SmallVector<DotLayoutPlan, 8> compatible;
  for (const DotLayoutPlan &plan : *domain) {
    bool accepted = true;
    for (unsigned operandIndex : {0u, 1u}) {
      for (const DotPipelineLocalLoadFacts &load :
           facts.operands[operandIndex].localLoads) {
        ttg::LocalLoadOp localLoad = load.localLoad;
        auto sharedType =
            dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
        if (!sharedType)
          return localLoad.emitError()
                 << "dot local_load source must be a memdesc";
        if (isCompilerManagedSharedFamilyRoot(load.sharedView.root))
          continue;

        auto resultType =
            dyn_cast<RankedTensorType>(localLoad.getType());
        if (!resultType)
          return localLoad.emitError()
                 << "dot local_load result must be a ranked tensor";
        RankedTensorType candidateType = resultType.cloneWithEncoding(
            plan.getOperand(operandIndex));
        FailureOr<RegisterToSharedContractInfo> contract =
            checkRegisterToSharedContract(candidateType, sharedType);
        if (failed(contract) || !contract->lowerable) {
          accepted = false;
          LDBG("[candidate-reject] dot="
               << dot.getLoc() << ", operand=" << operandIndex
               << ", fixed-shared=" << sharedType.getEncoding()
               << ", candidate-register="
               << candidateType.getEncoding());
          break;
        }
      }
      if (!accepted)
        break;
    }
    if (accepted)
      compatible.push_back(plan);
  }

  if (compatible.empty())
    return dot.emitError()
           << "no C500 dot layout candidate is compatible with every fixed "
              "local-load shared view";
  return compatible;
}

FailureOr<Attribute>
inferC500DotSharedEncoding(Attribute dotOperandEncoding,
                           ttg::LocalLoadOp localLoad,
                           const DotOperandPathFacts &path) {
  auto dotEncoding =
      dyn_cast_or_null<ttg::DotOperandEncodingAttr>(dotOperandEncoding);
  auto resultType = dyn_cast<RankedTensorType>(localLoad.getType());
  auto sharedType = dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
  if (!dotEncoding ||
      !isa<ttg::MACAMmaEncodingAttr>(dotEncoding.getParent()) ||
      !resultType || !sharedType || resultType.getRank() != 2 ||
      sharedType.getRank() != 2 ||
      resultType.getShape() != sharedType.getShape())
    return localLoad.emitError()
           << "C500 dot shared-layout inference requires matching rank-2 "
              "local_load source/result types";
  if (path.passesThroughBsmPermutation && dotEncoding.getOpIdx() != 1)
    return localLoad.emitError()
           << "C500 BSM shared-load contract is valid only for operand B";

  if (auto trans =
          localLoad.getSrc().getDefiningOp<ttg::MemDescTransOp>()) {
    if (path.passesThroughBsmPermutation)
      return localLoad.emitError()
             << "C500 BSM does not support an uncomposed memdesc transpose";
    auto rootType = dyn_cast<ttg::MemDescType>(trans.getSrc().getType());
    std::optional<SmallVector<unsigned>> rootOrder =
        rootType ? getSharedMemoryOrder(rootType) : std::nullopt;
    if (rootType && rootOrder) {
      Attribute rootEncoding = ttg::SwizzledSharedEncodingAttr::get(
          localLoad.getContext(), dotEncoding, rootType.getShape(),
          *rootOrder, ttg::getCTALayout(dotEncoding),
          rootType.getElementType(), /*needTrans=*/true);
      if (FailureOr<Attribute> view = inferTransEncoding(
              rootEncoding, rootType.getShape(), trans.getOrder(),
              trans.getLoc());
          succeeded(view))
        return *view;
    }
  }

  SmallVector<unsigned> order = ttg::getOrderForDotOperand(
      dotEncoding.getOpIdx(), resultType.getRank(),
      /*kContig=*/!path.passesThroughBsmPermutation);
  return ttg::SwizzledSharedEncodingAttr::get(
      localLoad.getContext(), dotEncoding, resultType.getShape(), order,
      ttg::getCTALayout(dotEncoding), sharedType.getElementType(),
      /*needTrans=*/false);
}

bool haveSameC500AccumulatorProfile(Attribute lhs, Attribute rhs) {
  auto left = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
      unwrapNoVerifyEncoding(lhs));
  auto right = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
      unwrapNoVerifyEncoding(rhs));
  if (!left || !right)
    return false;
  return left.getVersionMajor() == right.getVersionMajor() &&
         left.getVersionMinor() == right.getVersionMinor() &&
         left.getWarpsPerCTA() == right.getWarpsPerCTA() &&
         left.getElementsMNK() == right.getElementsMNK() &&
         left.getColMajor() == right.getColMajor() &&
         left.getCTALayout() == right.getCTALayout();
}

} // namespace mlir::triton::gpu::metax::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
