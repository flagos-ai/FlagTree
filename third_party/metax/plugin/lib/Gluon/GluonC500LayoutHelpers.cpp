#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"
#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"
#include "Gluon/Targets/GluonC500Layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <iterator>
#include <limits>
#include <tuple>

#define DEBUG_TYPE "metax-gluon-c500-layout-helpers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon::c500 {

static LogicalResult verifyC500SubviewChain(Value value);

Value getMemDescViewSource(Operation *op) {
  if (!op)
    return {};
  if (auto index = dyn_cast<ttg::MemDescIndexOp>(op))
    return index.getSrc();
  if (auto slice = dyn_cast<ttg::MemDescSubsliceOp>(op))
    return slice.getSrc();
  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op))
    return trans.getSrc();
  if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op))
    return reshape.getSrc();
  if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(op))
    return reinterpret.getSrc();
  if (auto require = dyn_cast<gluon_dialect::RequireLayoutOp>(op))
    if (isa<ttg::MemDescType>(require.getSrc().getType()))
      return require.getSrc();
  return {};
}

Value getMemDescRoot(Value value) {
  while (value) {
    Value source = getMemDescViewSource(value.getDefiningOp());
    if (!source)
      return value;
    value = source;
  }
  return {};
}

void collectMemDescAliasValues(Value root, llvm::SetVector<Value> &aliases) {
  if (!root || aliases.contains(root))
    return;
  aliases.insert(root);
  for (Operation *user : root.getUsers()) {
    if (getMemDescViewSource(user) == root && user->getNumResults() == 1)
      collectMemDescAliasValues(user->getResult(0), aliases);
  }
}

unsigned getTensorElementOrPointeeBitWidth(RankedTensorType tensorTy) {
  Type elemTy = tensorTy.getElementType();
  if (isa<tt::PointerType>(elemTy))
    return tt::getPointeeBitWidth(tensorTy);
  if (elemTy.isIntOrFloat())
    return elemTy.getIntOrFloatBitWidth();
  return 0;
}

StringRef stringifyRank2OrderProvenance(Rank2OrderProvenance provenance) {
  switch (provenance) {
  case Rank2OrderProvenance::ConcreteEncoding:
    return "concrete-encoding";
  case Rank2OrderProvenance::AxisInfo:
    return "axis-info";
  case Rank2OrderProvenance::ContiguityHint:
    return "contiguity-hint";
  case Rank2OrderProvenance::PointerExpression:
    return "pointer-expression";
  }
  llvm_unreachable("unknown C500 rank-2 order provenance");
}

static FailureOr<Rank2ContiguousOrderInfo>
getRank2OrderFromStrictContiguity(Value value, ArrayRef<int64_t> contiguity,
                                  Rank2OrderProvenance provenance) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getRank() != 2 || contiguity.size() != 2 ||
      contiguity[0] == contiguity[1])
    return failure();
  SmallVector<unsigned, 2> order =
      contiguity[0] > contiguity[1] ? SmallVector<unsigned, 2>{0, 1}
                                    : SmallVector<unsigned, 2>{1, 0};
  int64_t proven = contiguity[order.front()];
  if (proven <= 0 || static_cast<uint64_t>(proven) >
                         std::numeric_limits<unsigned>::max())
    return failure();
  return Rank2ContiguousOrderInfo{
      order, static_cast<unsigned>(proven), provenance};
}

static std::optional<int64_t> getConstantSplatInteger(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
    Attribute attr = constant.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    return std::nullopt;
  }
  if (auto splat = dyn_cast<tt::SplatOp>(def))
    return getConstantSplatInteger(splat.getSrc());
  return std::nullopt;
}

static constexpr int64_t kUnknownRangeCoefficient =
    std::numeric_limits<int64_t>::min();

static bool isUnknownRangeCoefficient(int64_t coeff) {
  return coeff == kUnknownRangeCoefficient;
}

static SmallVector<int64_t, 2> getZeroRangeCoefficients(unsigned resultRank) {
  return SmallVector<int64_t, 2>(resultRank, 0);
}

static SmallVector<int64_t, 2> getUnknownRangeCoefficients(unsigned resultRank) {
  return SmallVector<int64_t, 2>(resultRank, kUnknownRangeCoefficient);
}

static bool hasOnlyZeroRangeCoefficients(ArrayRef<int64_t> coeffs) {
  return llvm::all_of(coeffs, [](int64_t coeff) { return coeff == 0; });
}

static bool hasOnlyUnknownRangeCoefficients(ArrayRef<int64_t> coeffs) {
  return llvm::all_of(coeffs, isUnknownRangeCoefficient);
}

struct RangeCoefficientFacts {
  SmallVector<int64_t, 2> coefficients;
  bool dependsOnCycle = false;
  bool preservesCycle = true;
};

static RangeCoefficientFacts getZeroRangeCoefficientFacts(unsigned rank) {
  return {getZeroRangeCoefficients(rank), false, true};
}

static RangeCoefficientFacts getUnknownRangeCoefficientFacts(
    unsigned rank, bool dependsOnCycle = false) {
  return {getUnknownRangeCoefficients(rank), dependsOnCycle, true};
}

static SmallVector<int64_t, 2>
addRangeCoefficients(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  SmallVector<int64_t, 2> result(lhs.begin(), lhs.end());
  for (auto [dim, rhsCoeff] : llvm::enumerate(rhs)) {
    if (isUnknownRangeCoefficient(result[dim]) ||
        isUnknownRangeCoefficient(rhsCoeff)) {
      result[dim] = kUnknownRangeCoefficient;
      continue;
    }
    result[dim] += rhsCoeff;
  }
  return result;
}

static SmallVector<int64_t, 2>
subtractRangeCoefficients(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  SmallVector<int64_t, 2> result(lhs.begin(), lhs.end());
  for (auto [dim, rhsCoeff] : llvm::enumerate(rhs)) {
    if (isUnknownRangeCoefficient(result[dim]) ||
        isUnknownRangeCoefficient(rhsCoeff)) {
      result[dim] = kUnknownRangeCoefficient;
      continue;
    }
    result[dim] -= rhsCoeff;
  }
  return result;
}

static SmallVector<int64_t, 2>
scaleRangeCoefficients(ArrayRef<int64_t> coeffs, int64_t scale) {
  if (scale == 0)
    return getZeroRangeCoefficients(coeffs.size());
  SmallVector<int64_t, 2> result(coeffs.begin(), coeffs.end());
  for (int64_t &coeff : result)
    if (!isUnknownRangeCoefficient(coeff))
      coeff *= scale;
  return result;
}

static SmallVector<int64_t, 2>
scaleRangeCoefficientsBySymbol(ArrayRef<int64_t> coeffs) {
  SmallVector<int64_t, 2> result(coeffs.begin(), coeffs.end());
  for (int64_t &coeff : result)
    if (coeff != 0)
      coeff = kUnknownRangeCoefficient;
  return result;
}

static std::optional<RangeCoefficientFacts>
getRangeCoefficientsFromOffsetImpl(Value value, unsigned resultRank,
                                   DenseSet<Value> &active) {
  if (!value)
    return std::nullopt;
  if (std::optional<int64_t> constant = getConstantSplatInteger(value))
    return getZeroRangeCoefficientFacts(resultRank);
  if (!active.insert(value).second)
    return getUnknownRangeCoefficientFacts(resultRank,
                                            /*dependsOnCycle=*/true);
  auto activeGuard = llvm::make_scope_exit([&] { active.erase(value); });

  if (isa<BlockArgument, OpResult>(value)) {
    SmallVector<Value> predecessors;
    if (appendRegionCarrierPredecessors(value, predecessors)) {
      std::optional<RangeCoefficientFacts> consensus;
      for (Value predecessor : predecessors) {
        std::optional<RangeCoefficientFacts> facts =
            getRangeCoefficientsFromOffsetImpl(predecessor, resultRank, active);
        if (!facts)
          return std::nullopt;

        // A pure cyclic value is the bottom fact of this recursive query. It
        // contributes no information until an entry edge seeds the carrier.
        // Symbolic coefficients remain ordinary unknowns and must participate
        // in branch consensus.
        if (facts->dependsOnCycle &&
            hasOnlyUnknownRangeCoefficients(facts->coefficients)) {
          if (!facts->preservesCycle)
            return RangeCoefficientFacts{
                getUnknownRangeCoefficients(resultRank),
                /*dependsOnCycle=*/true,
                /*preservesCycle=*/false};
          continue;
        }
        if (consensus && consensus->coefficients != facts->coefficients)
          return getUnknownRangeCoefficientFacts(resultRank);
        if (!consensus) {
          consensus = std::move(facts);
          continue;
        }
        consensus->dependsOnCycle |= facts->dependsOnCycle;
        consensus->preservesCycle &= facts->preservesCycle;
      }
      return consensus ? consensus
                       : std::optional<RangeCoefficientFacts>(
                             getUnknownRangeCoefficientFacts(
                                 resultRank, /*dependsOnCycle=*/true));
    }
  }

  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return std::nullopt;

  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto require = dyn_cast<gluon_dialect::RequireLayoutOp>(def))
    return getRangeCoefficientsFromOffsetImpl(require.getSrc(), resultRank,
                                              active);
  if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(def))
    return getRangeCoefficientsFromOffsetImpl(convert.getSrc(), resultRank,
                                              active);
  if (auto extract = dyn_cast<gluon_dialect::ExtractSliceOp>(def))
    return getRangeCoefficientsFromOffsetImpl(extract.getSource(), resultRank,
                                              active);
  if (auto extract = dyn_cast<ttg::ExtractTensorOp>(def))
    return getRangeCoefficientsFromOffsetImpl(extract.getSource(), resultRank,
                                              active);

  if (isa<tt::SplatOp>(def))
    return getZeroRangeCoefficientFacts(resultRank);

  if (isa<tt::MakeRangeOp>(def)) {
    if (tensorTy.getRank() != 1 || resultRank != 1)
      return std::nullopt;
    RangeCoefficientFacts facts = getZeroRangeCoefficientFacts(resultRank);
    facts.coefficients[0] = 1;
    return facts;
  }

  if (auto expand = dyn_cast<tt::ExpandDimsOp>(def)) {
    if (resultRank == 0)
      return std::nullopt;
    std::optional<RangeCoefficientFacts> srcFacts =
        getRangeCoefficientsFromOffsetImpl(expand.getSrc(), resultRank - 1,
                                           active);
    if (!srcFacts)
      return std::nullopt;
    RangeCoefficientFacts facts = getZeroRangeCoefficientFacts(resultRank);
    facts.dependsOnCycle = srcFacts->dependsOnCycle;
    facts.preservesCycle =
        srcFacts->preservesCycle && !srcFacts->dependsOnCycle;
    unsigned axis = expand.getAxis();
    if (axis >= resultRank)
      return std::nullopt;
    for (auto [dim, coeff] : llvm::enumerate(srcFacts->coefficients)) {
      unsigned resultDim = dim < axis ? dim : dim + 1;
      facts.coefficients[resultDim] = coeff;
    }
    return facts;
  }

  if (auto broadcast = dyn_cast<tt::BroadcastOp>(def)) {
    std::optional<RangeCoefficientFacts> srcFacts =
        getRangeCoefficientsFromOffsetImpl(broadcast.getSrc(), resultRank,
                                           active);
    if (!srcFacts)
      return std::nullopt;
    auto srcTy = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
    if (!srcTy || srcTy.getRank() != static_cast<int64_t>(resultRank))
      return std::nullopt;

    RangeCoefficientFacts facts = getZeroRangeCoefficientFacts(resultRank);
    facts.dependsOnCycle = srcFacts->dependsOnCycle;
    facts.preservesCycle =
        srcFacts->preservesCycle && !srcFacts->dependsOnCycle;
    ArrayRef<int64_t> srcShape = srcTy.getShape();
    ArrayRef<int64_t> dstShape = tensorTy.getShape();
    for (unsigned dim = 0; dim < resultRank; ++dim) {
      if (srcFacts->coefficients[dim] == 0)
        continue;
      if (ShapedType::isDynamic(srcShape[dim]) ||
          ShapedType::isDynamic(dstShape[dim]) ||
          srcShape[dim] != dstShape[dim])
        continue;
      facts.coefficients[dim] = srcFacts->coefficients[dim];
    }
    return facts;
  }

  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::IndexCastOp>(def))
    return getRangeCoefficientsFromOffsetImpl(def->getOperand(0), resultRank,
                                              active);

  if (isa<arith::AddIOp>(def)) {
    RangeCoefficientFacts facts = getZeroRangeCoefficientFacts(resultRank);
    unsigned cyclicOperands = 0;
    bool hasNonCyclicRangeTerm = false;
    for (Value operand : def->getOperands()) {
      std::optional<RangeCoefficientFacts> operandFacts =
          getRangeCoefficientsFromOffsetImpl(operand, resultRank, active);
      if (!operandFacts)
        return std::nullopt;
      facts.coefficients = addRangeCoefficients(
          facts.coefficients, operandFacts->coefficients);
      facts.dependsOnCycle |= operandFacts->dependsOnCycle;
      facts.preservesCycle &= operandFacts->preservesCycle;
      if (operandFacts->dependsOnCycle)
        ++cyclicOperands;
      else
        hasNonCyclicRangeTerm |=
            !hasOnlyZeroRangeCoefficients(operandFacts->coefficients);
    }
    if (facts.dependsOnCycle)
      facts.preservesCycle &=
          cyclicOperands == 1 && !hasNonCyclicRangeTerm;
    return facts;
  }

  if (isa<arith::SubIOp>(def)) {
    std::optional<RangeCoefficientFacts> lhs =
        getRangeCoefficientsFromOffsetImpl(def->getOperand(0), resultRank,
                                           active);
    std::optional<RangeCoefficientFacts> rhs =
        getRangeCoefficientsFromOffsetImpl(def->getOperand(1), resultRank,
                                           active);
    if (!lhs || !rhs)
      return std::nullopt;
    bool dependsOnCycle = lhs->dependsOnCycle || rhs->dependsOnCycle;
    bool preservesCycle = lhs->preservesCycle && rhs->preservesCycle;
    if (dependsOnCycle)
      preservesCycle &= lhs->dependsOnCycle && !rhs->dependsOnCycle &&
                        hasOnlyZeroRangeCoefficients(rhs->coefficients);
    return RangeCoefficientFacts{
        subtractRangeCoefficients(lhs->coefficients, rhs->coefficients),
        dependsOnCycle, preservesCycle};
  }

  if (isa<arith::MulIOp>(def)) {
    Value lhs = def->getOperand(0);
    Value rhs = def->getOperand(1);
    std::optional<int64_t> lhsConstant = getConstantSplatInteger(lhs);
    std::optional<int64_t> rhsConstant = getConstantSplatInteger(rhs);
    if (lhsConstant || rhsConstant) {
      int64_t scale = lhsConstant ? *lhsConstant : *rhsConstant;
      Value rangeOperand = lhsConstant ? rhs : lhs;
      std::optional<RangeCoefficientFacts> facts =
          getRangeCoefficientsFromOffsetImpl(rangeOperand, resultRank, active);
      if (!facts)
        return std::nullopt;
      if (scale == 0)
        return getZeroRangeCoefficientFacts(resultRank);
      facts->coefficients =
          scaleRangeCoefficients(facts->coefficients, scale);
      if (facts->dependsOnCycle && scale != 1)
        facts->preservesCycle = false;
      return facts;
    }

    std::optional<RangeCoefficientFacts> lhsFacts =
        getRangeCoefficientsFromOffsetImpl(lhs, resultRank, active);
    std::optional<RangeCoefficientFacts> rhsFacts =
        getRangeCoefficientsFromOffsetImpl(rhs, resultRank, active);
    if (!lhsFacts || !rhsFacts)
      return std::nullopt;
    bool dependsOnCycle =
        lhsFacts->dependsOnCycle || rhsFacts->dependsOnCycle;
    if (hasOnlyZeroRangeCoefficients(lhsFacts->coefficients))
      return RangeCoefficientFacts{
          scaleRangeCoefficientsBySymbol(rhsFacts->coefficients),
          dependsOnCycle,
          !dependsOnCycle && lhsFacts->preservesCycle &&
              rhsFacts->preservesCycle};
    if (hasOnlyZeroRangeCoefficients(rhsFacts->coefficients))
      return RangeCoefficientFacts{
          scaleRangeCoefficientsBySymbol(lhsFacts->coefficients),
          dependsOnCycle,
          !dependsOnCycle && lhsFacts->preservesCycle &&
              rhsFacts->preservesCycle};
    RangeCoefficientFacts facts =
        getUnknownRangeCoefficientFacts(resultRank, dependsOnCycle);
    facts.preservesCycle = !dependsOnCycle;
    return facts;
  }

  if (auto insert = dyn_cast<gluon_dialect::InsertSliceOp>(def)) {
    std::optional<RangeCoefficientFacts> facts =
        getRangeCoefficientsFromOffsetImpl(insert.getBase(), resultRank, active);
    std::optional<RangeCoefficientFacts> updateFacts =
        getRangeCoefficientsFromOffsetImpl(insert.getUpdate(), resultRank,
                                           active);
    if (!facts || !updateFacts)
      return std::nullopt;
    bool dependsOnCycle =
        facts->dependsOnCycle || updateFacts->dependsOnCycle;
    if (facts->coefficients != updateFacts->coefficients)
      return getUnknownRangeCoefficientFacts(resultRank, dependsOnCycle);
    facts->dependsOnCycle = dependsOnCycle;
    facts->preservesCycle &= updateFacts->preservesCycle;
    return facts;
  }

  if (auto insert = dyn_cast<ttg::InsertTensorOp>(def)) {
    // Register-slice legalization preserves the logical tensor values while
    // replacing gluon.insert_slice with ttg.insert_tensor.  Re-establish the
    // same address proof from both physical operands: replacing a sub-tile
    // cannot change a dimension's affine range coefficient when the full
    // tensor and inserted values agree on that coefficient.  A disagreement
    // widens to unknown instead of using register ownership as an address
    // fact.
    std::optional<RangeCoefficientFacts> facts =
        getRangeCoefficientsFromOffsetImpl(insert.getInserted(), resultRank,
                                           active);
    std::optional<RangeCoefficientFacts> updateFacts =
        getRangeCoefficientsFromOffsetImpl(insert.getInsert(), resultRank,
                                           active);
    if (!facts || !updateFacts)
      return std::nullopt;
    bool dependsOnCycle =
        facts->dependsOnCycle || updateFacts->dependsOnCycle;
    if (facts->coefficients != updateFacts->coefficients)
      return getUnknownRangeCoefficientFacts(resultRank, dependsOnCycle);
    facts->dependsOnCycle = dependsOnCycle;
    facts->preservesCycle &= updateFacts->preservesCycle;
    LLVM_DEBUG({
      DBGS() << "Preserved pointer-expression range coefficients across "
                "ttg.insert_tensor: [";
      llvm::interleaveComma(facts->coefficients, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });
    return facts;
  }

  if (isUniformTensorValue(value))
    return getZeroRangeCoefficientFacts(resultRank);
  return getUnknownRangeCoefficientFacts(resultRank);
}

static std::optional<SmallVector<int64_t, 2>>
getRangeCoefficientsFromOffset(Value value, unsigned resultRank) {
  DenseSet<Value> active;
  std::optional<RangeCoefficientFacts> facts =
      getRangeCoefficientsFromOffsetImpl(value, resultRank, active);
  if (!facts)
    return std::nullopt;
  return std::move(facts->coefficients);
}

static FailureOr<Rank2ContiguousOrderInfo>
inferRank2OrderFromPointerExpression(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getRank() != 2)
    return failure();
  if (!hasContiguousSubviewIndices(value))
    return failure();

  Operation *def = value.getDefiningOp();
  if (!def)
    return failure();

  if (auto require = dyn_cast<gluon_dialect::RequireLayoutOp>(def))
    return inferRank2OrderFromPointerExpression(require.getSrc());
  if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(def))
    return inferRank2OrderFromPointerExpression(convert.getSrc());
  if (auto extract = dyn_cast<gluon_dialect::ExtractSliceOp>(def))
    return inferRank2OrderFromPointerExpression(extract.getSource());
  if (auto extract = dyn_cast<ttg::ExtractTensorOp>(def))
    return inferRank2OrderFromPointerExpression(extract.getSource());

  auto addPtr = dyn_cast<tt::AddPtrOp>(def);
  if (!addPtr)
    return failure();

  std::optional<SmallVector<int64_t, 2>> coeffs =
      getRangeCoefficientsFromOffset(addPtr.getOffset(), /*resultRank=*/2);
  if (!coeffs)
    return failure();

  SmallVector<unsigned, 2> candidates;
  for (auto [dim, coeff] : llvm::enumerate(*coeffs)) {
    if (coeff == 1)
      candidates.push_back(dim);
    else if (coeff == -1)
      return failure();
  }
  if (candidates.size() != 1)
    return failure();

  unsigned contiguousDim = candidates.front();
  int64_t contiguousExtent = tensorTy.getShape()[contiguousDim];
  if (ShapedType::isDynamic(contiguousExtent) || contiguousExtent <= 0 ||
      static_cast<uint64_t>(contiguousExtent) >
          std::numeric_limits<unsigned>::max())
    return failure();
  SmallVector<unsigned, 2> order{contiguousDim, 1 - contiguousDim};
  return Rank2ContiguousOrderInfo{order,
                                  static_cast<unsigned>(contiguousExtent),
                                  Rank2OrderProvenance::PointerExpression};
}

FailureOr<Rank2ContiguousOrderInfo>
inferRank2ContiguousOrder(Value value,
                          tt::ModuleAxisInfoAnalysis *axisInfoAnalysis,
                          Operation *diagnosticOp) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getRank() != 2) {
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "C500 rank-2 contiguous order requires a rank-2 tensor value";
    return failure();
  }
  if (failed(verifyC500SubviewChain(value)))
    return failure();

  if (axisInfoAnalysis) {
    tt::AxisInfo *axisInfo = axisInfoAnalysis->getAxisInfo(value);
    if (axisInfo && axisInfo->getRank() == tensorTy.getRank()) {
      ArrayRef<int64_t> contiguity = axisInfo->getContiguity();
      FailureOr<Rank2ContiguousOrderInfo> info =
          getRank2OrderFromStrictContiguity(
              value, contiguity, Rank2OrderProvenance::AxisInfo);
      if (succeeded(info)) {
        LDBG("C500 rank-2 order " << info->order[0] << ","
                                  << info->order[1] << " inferred from "
                                  << stringifyRank2OrderProvenance(
                                         info->provenance));
        return info;
      }
    }
  }

  std::optional<unsigned> dim0Contiguity = getContiguityHint(value, 0);
  std::optional<unsigned> dim1Contiguity = getContiguityHint(value, 1);
  if (dim0Contiguity && dim1Contiguity && *dim0Contiguity != *dim1Contiguity) {
    SmallVector<unsigned, 2> order = *dim0Contiguity > *dim1Contiguity
                                        ? SmallVector<unsigned, 2>{0, 1}
                                        : SmallVector<unsigned, 2>{1, 0};
    unsigned proven = std::max(*dim0Contiguity, *dim1Contiguity);
    Rank2ContiguousOrderInfo info{order, proven,
                                  Rank2OrderProvenance::ContiguityHint};
    LDBG("C500 rank-2 order " << info.order[0] << "," << info.order[1]
                              << " inferred from "
                              << stringifyRank2OrderProvenance(
                                     info.provenance));
    return info;
  }

  FailureOr<Rank2ContiguousOrderInfo> pointerInfo =
      inferRank2OrderFromPointerExpression(value);
  if (succeeded(pointerInfo)) {
    LDBG("C500 rank-2 order " << pointerInfo->order[0] << ","
                              << pointerInfo->order[1] << " inferred from "
                              << stringifyRank2OrderProvenance(
                                     pointerInfo->provenance));
    return pointerInfo;
  }

  // A concrete tensor encoding proves how pointer elements are distributed
  // across registers, lanes, and warps. It does not prove that the pointer
  // values themselves address adjacent global elements. Keep it only as an
  // ownership-order fallback for non-copy layout selection and mark the
  // vector width unknown.
  Attribute encoding = tensorTy.getEncoding();
  if (encoding && !isa<gluon_dialect::AutoEncodingAttr>(encoding)) {
    SmallVector<unsigned> order = ttg::getOrder(tensorTy);
    if (order.size() == 2) {
      Rank2ContiguousOrderInfo info{
          SmallVector<unsigned, 2>{order[0], order[1]},
          /*maxContiguousElements=*/0,
          Rank2OrderProvenance::ConcreteEncoding};
      LDBG("C500 rank-2 ownership order " << info.order[0] << ","
                                           << info.order[1]
                                           << " inferred from "
                                           << stringifyRank2OrderProvenance(
                                                  info.provenance)
                                           << " without address-contiguity "
                                              "evidence");
      return info;
    }
  }

  if (diagnosticOp)
    diagnosticOp->emitError()
        << "cannot infer C500 rank-2 contiguous order; expected a concrete "
           "rank-2 encoding, unequal AxisInfo contiguity, or explicit "
           "tt.contiguity hints, or a simple pointer expression with exactly "
           "one unit-stride tt.make_range dimension";
  return failure();
}

std::optional<ttg::BlockedEncodingAttr>
getC500GlobalBlockedEncoding(Value value, Operation *anchor,
                             ArrayRef<unsigned> order,
                             unsigned transactionWidthCap) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!anchor || !tensorTy || !tensorTy.hasStaticShape() ||
      llvm::any_of(tensorTy.getShape(),
                   [](int64_t extent) { return extent <= 0; }))
    return std::nullopt;

  OpBuilder builder(anchor);
  int numWarps = ttg::lookupNumWarps(anchor);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(anchor);
  if (numWarps <= 0 || threadsPerWarp <= 0 || numCTAs <= 0)
    return std::nullopt;

  if (tensorTy.getRank() == 2 && order.size() == 2 && order[0] < 2 &&
      order[1] < 2 && order[0] != order[1]) {
    unsigned elemBits = getTensorElementOrPointeeBitWidth(tensorTy);
    if (elemBits == 0)
      return std::nullopt;
    uint64_t hardwareVec = std::max<uint64_t>(1, 128 / elemBits);
    uint64_t warps = static_cast<uint64_t>(numWarps);
    uint64_t threads = static_cast<uint64_t>(threadsPerWarp);
    if (warps > std::numeric_limits<uint64_t>::max() / threads)
      return std::nullopt;
    uint64_t threadsPerCTA = warps * threads;

    uint64_t numElements = 1;
    for (int64_t extent : tensorTy.getShape()) {
      uint64_t unsignedExtent = static_cast<uint64_t>(extent);
      if (numElements >
          std::numeric_limits<uint64_t>::max() / unsignedExtent)
        return std::nullopt;
      numElements *= unsignedExtent;
    }
    uint64_t requiredElemsPerThread =
        llvm::divideCeil(numElements, threadsPerCTA);
    uint64_t coverageVec =
        llvm::PowerOf2Ceil(std::max<uint64_t>(1, requiredElemsPerThread));
    uint64_t selectedVec = std::min(hardwareVec, coverageVec);
    if (transactionWidthCap > 0)
      selectedVec = std::min<uint64_t>(selectedVec, transactionWidthCap);
    if (selectedVec == 0 ||
        selectedVec > std::numeric_limits<unsigned>::max())
      return std::nullopt;
    unsigned vecElems = static_cast<unsigned>(selectedVec);
    LDBG("C500 global blocked vector width "
         << vecElems << " for " << tensorTy
         << ", hardware width=" << hardwareVec
         << ", coverage width=" << coverageVec
         << ", proven transaction cap="
         << (transactionWidthCap ? transactionWidthCap : vecElems));
    SmallVector<unsigned> sizePerThread(tensorTy.getRank(), 1);
    sizePerThread[order[0]] = vecElems;
    ttg::BlockedEncodingAttr shapeEncoding = ttg::BlockedEncodingAttr::get(
        value.getContext(), tensorTy.getShape(), sizePerThread, order,
        numWarps, threadsPerWarp, numCTAs);

    // C500's four-warp B128 register/shared copy atom is balanced across one
    // warp: eight lanes select rows and eight lanes select contiguous
    // eight-half vectors. The generic shape constructor instead saturates the
    // contiguous dimension first and produces [4, 16] for a D128 tile. That
    // layout is globally coalesced, but it breaks the target copy atom into a
    // slower physical mapping. This is one deterministic target boundary, not
    // an autotune axis: retain the generic shape-aware construction unless the
    // complete 4 x 64-lane, eight-element C500 atom is representable.
    constexpr unsigned kC500CopyLanesPerDimension = 8;
    if (numWarps == 4 && threadsPerWarp == 64 && vecElems == 8 &&
        static_cast<uint64_t>(tensorTy.getShape()[order[0]]) >=
            static_cast<uint64_t>(vecElems) *
                kC500CopyLanesPerDimension &&
        static_cast<uint64_t>(tensorTy.getShape()[order[1]]) >=
            kC500CopyLanesPerDimension) {
      SmallVector<unsigned> threads(2, kC500CopyLanesPerDimension);
      SmallVector<unsigned> warps(2, 1);
      warps[order[1]] = static_cast<unsigned>(numWarps);
      return ttg::BlockedEncodingAttr::get(
          value.getContext(), sizePerThread, threads, warps, order,
          ttg::getCTALayout(shapeEncoding));
    }
    return shapeEncoding;
  }

  return ttg::getDefaultBlockedEncoding(value.getContext(), tensorTy.getShape(),
                                        numWarps, threadsPerWarp, numCTAs);
}

std::optional<ttg::BlockedEncodingAttr>
getC500GlobalBlockedEncodingLike(Value value, Operation *anchor,
                                 ttg::BlockedEncodingAttr model) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!anchor || !tensorTy || !tensorTy.hasStaticShape() ||
      llvm::any_of(tensorTy.getShape(),
                   [](int64_t extent) { return extent <= 0; }) ||
      static_cast<int64_t>(model.getOrder().size()) != tensorTy.getRank())
    return std::nullopt;

  OpBuilder builder(anchor);
  int numWarps = ttg::lookupNumWarps(anchor);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(anchor);
  if (numWarps <= 0 || threadsPerWarp <= 0 || numCTAs <= 0)
    return std::nullopt;
  return ttg::BlockedEncodingAttr::get(
      value.getContext(), tensorTy.getShape(), model.getSizePerThread(),
      model.getOrder(), numWarps, threadsPerWarp, numCTAs);
}

static std::optional<unsigned> getDenseContiguityForDim(Attribute attr,
                                                        unsigned dim) {
  if (!attr)
    return std::nullopt;
  auto getUnsignedWidth = [](const APInt &value) -> std::optional<unsigned> {
    if (value.isNegative() ||
        value.getActiveBits() > std::numeric_limits<unsigned>::digits)
      return std::nullopt;
    return static_cast<unsigned>(value.getZExtValue());
  };
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return getUnsignedWidth(intAttr.getValue());
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (dense.getNumElements() <= dim)
      return std::nullopt;
    auto values = dense.getValues<APInt>();
    auto it = values.begin();
    std::advance(it, dim);
    return getUnsignedWidth(*it);
  }
  return std::nullopt;
}

std::optional<unsigned> getContiguityHint(Value value, unsigned dim) {
  Value current = value;
  while (Operation *def = current.getDefiningOp()) {
    // Address contiguity describes the logical pointer values, independently
    // of their register ownership. Preserve that fact across layout-only
    // wrappers which keep both shape and element identity. In particular, the
    // C500 legalizer may insert convert_layout after selecting an issue plan;
    // re-verification must still see the proof attached to the original SSA
    // value instead of inferring contiguity from the new blocked encoding.
    Value source = TypeSwitch<Operation *, Value>(def)
                       .Case<gluon_dialect::RequireLayoutOp,
                             gluon_dialect::ReleaseLayoutOp,
                             ttg::ConvertLayoutOp>(
                           [](auto op) { return op.getSrc(); })
                       .Default(Value{});
    if (!source)
      break;
    LDBG("Following address-contiguity evidence through layout-only op "
         << def->getName());
    current = source;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(current)) {
    if (auto func =
            dyn_cast_or_null<tt::FuncOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getArgNumber() < func.getNumArguments()) {
        if (Attribute attr =
                func.getArgAttr(blockArg.getArgNumber(), "tt.contiguity"))
          return getDenseContiguityForDim(attr, dim);
      }
    }
  }
  if (Operation *def = current.getDefiningOp())
    return getDenseContiguityForDim(def->getAttr("tt.contiguity"), dim);
  return std::nullopt;
}

static bool isConsecutive(ArrayRef<int64_t> values) {
  if (values.empty())
    return false;
  for (auto [index, value] : llvm::enumerate(values))
    if (value != values.front() + static_cast<int64_t>(index))
      return false;
  return true;
}

static bool hasWholeTileGluonSlice(gluon_dialect::ExtractSliceOp extract) {
  auto sourceTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!sourceTy || !resultTy || sourceTy.getRank() != resultTy.getRank() ||
      extract.getOffsets().size() != static_cast<size_t>(sourceTy.getRank()))
    return false;

  for (auto [dim, offset] : llvm::enumerate(extract.getOffsets())) {
    int64_t sourceDim = sourceTy.getShape()[dim];
    int64_t resultDim = resultTy.getShape()[dim];
    if (ShapedType::isDynamic(sourceDim) ||
        ShapedType::isDynamic(resultDim) || resultDim <= 0 || offset < 0 ||
        offset + resultDim > sourceDim || offset % resultDim != 0)
      return false;
  }
  return true;
}

static LogicalResult verifyC500SubviewChain(Value value) {
  while (value) {
    if (auto require = value.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
      value = require.getSrc();
      continue;
    }
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = convert.getSrc();
      continue;
    }
    if (auto extract = value.getDefiningOp<ttg::ExtractTensorOp>()) {
      if (!isConsecutive(extract.getCtaIdx()))
        return extract.emitError()
               << "C500 async_copy requires contiguous ttg.extract_tensor "
                  "ctaIdx; actual="
               << extract.getCtaIdx();
      if (!isConsecutive(extract.getElemIdx()))
        return extract.emitError()
               << "C500 async_copy requires contiguous ttg.extract_tensor "
                  "elemIdx; actual="
               << extract.getElemIdx();
      value = extract.getSource();
      continue;
    }
    if (auto extract = value.getDefiningOp<gluon_dialect::ExtractSliceOp>()) {
      if (!hasWholeTileGluonSlice(extract)) {
        auto sourceTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
        auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
        InFlightDiagnostic diag = extract.emitError()
                                  << "C500 async_copy Gluon slice must select a "
                                     "complete logical tile";
        if (sourceTy && resultTy)
          diag << "; source-shape=" << sourceTy.getShape()
               << ", result-shape=" << resultTy.getShape()
               << ", offsets=" << extract.getOffsets();
        diag << "; each offset must be non-negative, in bounds, and divisible "
                "by the corresponding result dimension";
        return failure();
      }
      value = extract.getSource();
      continue;
    }
    return success();
  }
  return success();
}

bool hasContiguousSubviewIndices(Value value) {
  while (value) {
    if (auto require = value.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
      value = require.getSrc();
      continue;
    }
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = convert.getSrc();
      continue;
    }
    if (auto extract = value.getDefiningOp<ttg::ExtractTensorOp>()) {
      if (!isConsecutive(extract.getCtaIdx()) ||
          !isConsecutive(extract.getElemIdx()))
        return false;
      value = extract.getSource();
      continue;
    }
    if (auto extract = value.getDefiningOp<gluon_dialect::ExtractSliceOp>()) {
      if (!hasWholeTileGluonSlice(extract))
        return false;
      value = extract.getSource();
      continue;
    }
    return true;
  }
  return true;
}

static bool isUniformTensorValueImpl(Value value,
                                     llvm::DenseSet<Value> &visited) {
  if (!value)
    return true;
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return true;
  if (!visited.insert(value).second)
    return true;

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;

  if (auto require = dyn_cast<gluon_dialect::RequireLayoutOp>(def))
    return isUniformTensorValueImpl(require.getSrc(), visited);
  if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(def))
    return isUniformTensorValueImpl(convert.getSrc(), visited);
  if (auto extract = dyn_cast<ttg::ExtractTensorOp>(def))
    return isUniformTensorValueImpl(extract.getSource(), visited);
  if (auto extract = dyn_cast<gluon_dialect::ExtractSliceOp>(def))
    return isUniformTensorValueImpl(extract.getSource(), visited);
  if (auto broadcast = dyn_cast<tt::BroadcastOp>(def))
    return isUniformTensorValueImpl(broadcast.getSrc(), visited);
  if (auto expand = dyn_cast<tt::ExpandDimsOp>(def))
    return isUniformTensorValueImpl(expand.getSrc(), visited);
  if (isa<tt::SplatOp>(def))
    return true;
  if (auto constant = dyn_cast<arith::ConstantOp>(def))
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
      return dense.isSplat();

  if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(def))
    return llvm::all_of(def->getOperands(), [&](Value operand) {
      return isUniformTensorValueImpl(operand, visited);
    });
  if (auto cmp = dyn_cast<arith::CmpIOp>(def))
    return isUniformTensorValueImpl(cmp.getLhs(), visited) &&
           isUniformTensorValueImpl(cmp.getRhs(), visited);

  return false;
}

bool isUniformTensorValue(Value value) {
  llvm::DenseSet<Value> visited;
  return isUniformTensorValueImpl(value, visited);
}

static bool isSupportedC500AsyncCopyByteWidth(unsigned byteWidth) {
  return byteWidth == 4 || byteWidth == 8 || byteWidth == 16;
}

static unsigned getElementByteWidth(unsigned elemBits) {
  return std::max<unsigned>(1, elemBits / 8);
}

SmallVector<ttg::BlockedEncodingAttr> getC500AsyncCopyLegalSourceEncodings(
    ttg::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> order,
    AsyncCopyAddressContiguity addressContiguity,
    ttg::SwizzledSharedEncodingAttr sharedEncoding) {
  SmallVector<ttg::BlockedEncodingAttr> encodings;
  auto sourceType = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  if (!sourceType || sourceType.getRank() != 2 || order.size() != 2 ||
      !addressContiguity.isProven() ||
      order.front() != addressContiguity.contiguousDim)
    return encodings;

  auto appendEncoding = [&](ttg::BlockedEncodingAttr encoding) {
    if (encoding && llvm::none_of(encodings, [&](Attribute seen) {
          return seen == encoding;
        }))
      encodings.push_back(encoding);
  };

  unsigned elementBits = getTensorElementOrPointeeBitWidth(sourceType);
  if (elementBits == 0)
    return encodings;
  unsigned elementBytes = getElementByteWidth(elementBits);
  unsigned contiguousDim = order.front();
  unsigned maxElements = std::min(
      std::max<unsigned>(1, 16 / elementBytes),
      addressContiguity.maxContiguousElements);
  if (sharedEncoding.getMaxPhase() > 1)
    maxElements = std::min(maxElements, sharedEncoding.getVec());

  OpBuilder builder(copyOp);
  int numWarps = ttg::lookupNumWarps(copyOp);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(copyOp);
  for (unsigned copyBytes : {16u, 8u, 4u}) {
    if (copyBytes % elementBytes != 0)
      continue;
    unsigned copyElements = copyBytes / elementBytes;
    if (copyElements == 0 || copyElements > maxElements)
      continue;
    SmallVector<unsigned> sizePerThread(sourceType.getRank(), 1);
    sizePerThread[contiguousDim] = copyElements;
    appendEncoding(ttg::BlockedEncodingAttr::get(
        copyOp.getContext(), sourceType.getShape(), sizePerThread, order,
        numWarps, threadsPerWarp, numCTAs));
  }
  return encodings;
}

FailureOr<AsyncCopyIssuePlan> analyzeC500AsyncCopyIssuePlan(
    ttg::AsyncCopyGlobalToLocalOp copyOp,
    ttg::BlockedEncodingAttr sourceEncoding,
    AsyncCopyAddressContiguity addressContiguity,
    ttg::SwizzledSharedEncodingAttr sharedEncoding) {
  auto sourceType = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto destinationType =
      dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!sourceType || !destinationType || sourceType.getRank() != 2 ||
      destinationType.getRank() != 2 || !sourceEncoding || !sharedEncoding)
    return failure();

  ArrayRef<unsigned> sourceOrder = sourceEncoding.getOrder();
  ArrayRef<unsigned> sharedOrder = sharedEncoding.getOrder();
  ArrayRef<unsigned> sizePerThread = sourceEncoding.getSizePerThread();
  if (sourceOrder.size() != 2 || sharedOrder.size() != 2 ||
      sizePerThread.size() != 2 || sourceOrder.front() != sharedOrder.front())
    return failure();

  AsyncCopyIssuePlan plan;
  plan.sourceEncoding = sourceEncoding;
  plan.sharedEncoding = sharedEncoding;
  plan.contiguousDim = sourceOrder.front();
  if (!addressContiguity.isProven() ||
      addressContiguity.contiguousDim != plan.contiguousDim)
    return failure();
  plan.maxContiguousElements = addressContiguity.maxContiguousElements;
  plan.copyElements = sizePerThread[plan.contiguousDim];
  unsigned elementBits = getTensorElementOrPointeeBitWidth(sourceType);
  if (elementBits == 0 || plan.copyElements == 0)
    return failure();
  plan.copyBytes = plan.copyElements * getElementByteWidth(elementBits);
  if (!isSupportedC500AsyncCopyByteWidth(plan.copyBytes))
    return failure();
  if (plan.copyElements > plan.maxContiguousElements)
    return failure();
  if (!hasContiguousSubviewIndices(copyOp.getSrc()))
    return failure();
  if (sharedEncoding.getMaxPhase() > 1 &&
      plan.copyElements > sharedEncoding.getVec())
    return failure();

  auto plannedSourceType = RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType(), sourceEncoding);
  FailureOr<triton::LinearLayout> linearSource =
      getC500LinearLayout(plannedSourceType);
  if (failed(linearSource))
    return failure();
  triton::LinearLayout sourceLayout = std::move(*linearSource);
  auto freeVariables = sourceLayout.getFreeVariableMasks();
  MLIRContext *context = copyOp.getContext();
  StringAttr reg = StringAttr::get(context, "register");
  StringAttr lane = StringAttr::get(context, "lane");
  StringAttr warp = StringAttr::get(context, "warp");
  StringAttr block = StringAttr::get(context, "block");
  plan.freeLaneMask = freeVariables.lookup(lane);
  plan.freeWarpMask = freeVariables.lookup(warp);
  plan.freeBlockMask = freeVariables.lookup(block);

  sourceLayout =
      triton::actionRemoveBroadcastedRegs(sourceLayout).apply(sourceLayout);
  auto restrictToCanonicalIssuers = [&](StringAttr dim, uint32_t freeMask,
                                        bool allowPredication) {
    if (freeMask == 0)
      return success();
    if (!allowPredication)
      return failure();
    sourceLayout = sourceLayout.removeZeroBasesAlongDim(dim);
    return success();
  };

  // Predicating redundant lane/warp/block issuers is a target capability, not
  // a generic layout rule. If support is enabled later, the physical proof is
  // performed on the canonical representative of each broadcast class.
  if (failed(restrictToCanonicalIssuers(
          lane, plan.freeLaneMask,
          kC500AsyncCopyCapabilities.allowPredicatedLaneIssuer)) ||
      failed(restrictToCanonicalIssuers(
          warp, plan.freeWarpMask,
          kC500AsyncCopyCapabilities.allowPredicatedWarpIssuer)) ||
      failed(restrictToCanonicalIssuers(
          block, plan.freeBlockMask,
          kC500AsyncCopyCapabilities.allowPredicatedBlockIssuer)))
    return failure();
  auto noSwizzleEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, sharedEncoding.getVec(), /*perPhase=*/1, /*maxPhase=*/1,
      sharedOrder, sharedEncoding.getCTALayout());
  auto noSwizzleDestinationType = ttg::MemDescType::get(
      destinationType.getShape(), destinationType.getElementType(),
      noSwizzleEncoding, destinationType.getMemorySpace(),
      destinationType.getMutableMemory(), destinationType.getAllocShape());
  FailureOr<triton::LinearLayout> linearShared =
      getC500LinearLayout(noSwizzleDestinationType);
  if (failed(linearShared))
    return failure();
  triton::LinearLayout sharedLayout = std::move(*linearShared);
  triton::LinearLayout conversion =
      sourceLayout.invertAndCompose(sharedLayout);
  if (!conversion.isTrivialOver({block}))
    return failure();

  StringAttr offset = StringAttr::get(context, "offset");
  conversion = conversion.sublayout({reg, lane, warp}, {offset});
  if (!conversion.isInjective() || !conversion.isSurjective() ||
      conversion.getNumConsecutiveInOut() <
          static_cast<int32_t>(plan.copyElements))
    return failure();

  unsigned issuedElementsPerThread = sourceLayout.getInDimSize(reg);
  if (issuedElementsPerThread == 0 ||
      issuedElementsPerThread % plan.copyElements != 0)
    return failure();
  plan.instructionsPerThread = issuedElementsPerThread / plan.copyElements;
  plan.registerRepeats = plan.instructionsPerThread;
  plan.requiresSourceSideMaskSwizzle =
      copyOp.getMask() && sharedEncoding.getMaxPhase() > 1 &&
      !isUniformTensorValue(copyOp.getMask());
  return plan;
}

FailureOr<AsyncCopyIssuePlan> selectC500AsyncCopyIssuePlan(
    ttg::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> sourceOrder,
    AsyncCopyAddressContiguity addressContiguity,
    ttg::SwizzledSharedEncodingAttr sharedEncoding) {
  // The legal encodings are ordered by the C500 transaction widths
  // (16/8/4 bytes). The first lowerable encoding therefore maximizes useful
  // bytes per issue under the proven address and shared-write contracts.
  for (ttg::BlockedEncodingAttr sourceEncoding :
       getC500AsyncCopyLegalSourceEncodings(
           copyOp, sourceOrder, addressContiguity, sharedEncoding)) {
    FailureOr<AsyncCopyIssuePlan> plan = analyzeC500AsyncCopyIssuePlan(
        copyOp, sourceEncoding, addressContiguity, sharedEncoding);
    if (failed(plan)) {
      LDBG("[async-copy-issue-reject] source="
           << sourceEncoding << " shared=" << sharedEncoding);
      continue;
    }
    LDBG("[async-copy-issue-select] source="
         << sourceEncoding << " shared=" << sharedEncoding
         << " contiguous-dim=" << plan->contiguousDim
         << " max-contiguous-elements=" << plan->maxContiguousElements
         << " copy-bytes=" << plan->copyBytes
         << " instructions-per-thread=" << plan->instructionsPerThread);
    return plan;
  }
  return failure();
}

FailureOr<AsyncCopyIssuePlan>
getC500AsyncCopyIssuePlan(ttg::AsyncCopyGlobalToLocalOp copyOp) {
  auto sourceType = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto destinationType =
      dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!sourceType || !destinationType)
    return failure();
  auto sourceEncoding = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
      sourceType.getEncoding());
  auto sharedEncoding = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      destinationType.getEncoding());
  if (!sourceEncoding || !sharedEncoding)
    return failure();
  FailureOr<Rank2ContiguousOrderInfo> addressInfo =
      inferRank2ContiguousOrder(copyOp.getSrc(),
                                /*axisInfoAnalysis=*/nullptr);
  if (failed(addressInfo) || addressInfo->maxContiguousElements == 0)
    return failure();
  AsyncCopyAddressContiguity addressContiguity{
      addressInfo->order.front(), addressInfo->maxContiguousElements};
  return analyzeC500AsyncCopyIssuePlan(
      copyOp, sourceEncoding, addressContiguity, sharedEncoding);
}

LogicalResult
verifyC500AsyncCopyIssuePlan(ttg::AsyncCopyGlobalToLocalOp copyOp) {
  FailureOr<AsyncCopyIssuePlan> plan = getC500AsyncCopyIssuePlan(copyOp);
  if (succeeded(plan)) {
    LDBG("[async-copy-plan] source=" << plan->sourceEncoding
                                     << " shared=" << plan->sharedEncoding
                                     << " contiguous-dim="
                                     << plan->contiguousDim
                                     << " max-contiguous-elements="
                                     << plan->maxContiguousElements
                                     << " copy-bytes=" << plan->copyBytes
                                     << " instructions-per-thread="
                                     << plan->instructionsPerThread);
    return success();
  }
  return copyOp.emitError()
         << "C500 async_copy has no valid physical issue plan; expected a "
            "shape-aware blocked source and shared layout with matching "
            "contiguous order, supported 4/8/16-byte segments, unique "
            "unpredicated lane/warp issuers, complete shared coverage, and "
            "proven global address contiguity from AxisInfo, tt.contiguity, "
            "or pointer-expression facts, plus contiguous extract/subview "
            "indices; source="
         << copyOp.getSrc().getType() << ", destination="
         << copyOp->getOperand(1).getType();
}

FailureOr<AsyncCopyLegalizationInfo> getC500AsyncCopyLegalizationInfo(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto dstTy = dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!srcTy || !dstTy || srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return failure();

  auto blockedEnc =
      dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  auto sharedEnc = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      sharedEncoding ? sharedEncoding : dstTy.getEncoding());
  if (!blockedEnc || !sharedEnc)
    return failure();

  ArrayRef<unsigned> srcOrder = blockedEnc.getOrder();
  ArrayRef<unsigned> sharedOrder = sharedEnc.getOrder();
  if (srcOrder.size() != 2 || sharedOrder.size() != 2 ||
      srcOrder[0] != sharedOrder[0])
    return failure();

  ArrayRef<unsigned> sizePerThread = blockedEnc.getSizePerThread();
  if (sizePerThread.size() != 2)
    return failure();

  AsyncCopyLegalizationInfo info;
  info.contiguousDim = srcOrder[0];
  info.currentVec = sizePerThread[info.contiguousDim];
  info.elemBits = getTensorElementOrPointeeBitWidth(srcTy);
  if (info.currentVec == 0 || info.elemBits == 0)
    return failure();

  unsigned elemBytes = getElementByteWidth(info.elemBits);
  FailureOr<Rank2ContiguousOrderInfo> addressInfo =
      inferRank2ContiguousOrder(copyOp.getSrc(),
                                /*axisInfoAnalysis=*/nullptr);
  if (failed(addressInfo) || addressInfo->maxContiguousElements == 0 ||
      addressInfo->order.front() != info.contiguousDim)
    return failure();
  unsigned boundVec =
      std::min(std::max<unsigned>(1, 16 / elemBytes),
               addressInfo->maxContiguousElements);
  if (sharedEnc.getMaxPhase() > 1)
    boundVec = std::min(boundVec, sharedEnc.getVec());

  info.legalVec = 0;
  for (unsigned vec = boundVec; vec >= 1; --vec) {
    if (isSupportedC500AsyncCopyByteWidth(vec * elemBytes)) {
      info.legalVec = vec;
      break;
    }
    if (vec == 1)
      break;
  }
  if (info.legalVec == 0)
    return failure();
  return info;
}

bool isC500AsyncCopyContiguousSharedWrite(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  auto sourceType = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto destinationType =
      dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!sourceType || !destinationType)
    return false;
  auto source = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
      sourceType.getEncoding());
  auto shared = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      sharedEncoding ? sharedEncoding : destinationType.getEncoding());
  if (!source || !shared)
    return false;
  FailureOr<Rank2ContiguousOrderInfo> addressInfo =
      inferRank2ContiguousOrder(copyOp.getSrc(),
                                /*axisInfoAnalysis=*/nullptr);
  if (failed(addressInfo) || addressInfo->maxContiguousElements == 0)
    return false;
  AsyncCopyAddressContiguity addressContiguity{
      addressInfo->order.front(), addressInfo->maxContiguousElements};
  return succeeded(analyzeC500AsyncCopyIssuePlan(
      copyOp, source, addressContiguity, shared));
}

bool isC500AsyncCopyContiguousSharedWriteLegalizable(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  auto destinationType =
      dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!destinationType)
    return false;
  auto shared = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      sharedEncoding ? sharedEncoding : destinationType.getEncoding());
  if (!shared)
    return false;
  FailureOr<Rank2ContiguousOrderInfo> addressInfo =
      inferRank2ContiguousOrder(copyOp.getSrc(),
                                /*axisInfoAnalysis=*/nullptr);
  if (failed(addressInfo) || addressInfo->maxContiguousElements == 0)
    return false;
  AsyncCopyAddressContiguity addressContiguity{
      addressInfo->order.front(), addressInfo->maxContiguousElements};
  return llvm::any_of(
      getC500AsyncCopyLegalSourceEncodings(copyOp, shared.getOrder(),
                                           addressContiguity, shared),
      [&](ttg::BlockedEncodingAttr source) {
        return succeeded(analyzeC500AsyncCopyIssuePlan(
            copyOp, source, addressContiguity, shared));
      });
}

LogicalResult verifyC500AsyncCopyContiguousSharedWrite(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  if (isC500AsyncCopyContiguousSharedWrite(copyOp, sharedEncoding))
    return success();
  return copyOp.emitError()
         << "C500 async_copy source/shared pair does not satisfy the physical "
            "issue plan contract";
}

LogicalResult verifyC500AsyncCopyContiguousSharedWriteLegalizable(
    ttg::AsyncCopyGlobalToLocalOp copyOp, Attribute sharedEncoding) {
  if (isC500AsyncCopyContiguousSharedWriteLegalizable(copyOp, sharedEncoding))
    return success();
  return copyOp.emitError()
         << "C500 async_copy has no legal blocked source encoding for the "
            "requested shared layout and proven global address contiguity";
}

} // namespace mlir::triton::gpu::metax::gluon::c500
