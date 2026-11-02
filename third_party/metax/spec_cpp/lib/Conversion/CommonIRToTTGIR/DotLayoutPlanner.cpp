// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "triton/Conversion/CommonIRToTTGIR/BsmLayout.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::triton::metax {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// C500 MACA v2 instruction shape used to bound the search domain.
constexpr unsigned kMmaTileM = 16;
constexpr unsigned kMmaTileN = 16;
constexpr unsigned kMmaTileK = 4;

std::string hashDescription(StringRef description) {
  auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(description));
  return llvm::toHex(ArrayRef<uint8_t>(digest), /*LowerCase=*/true);
}

int getMmaVersionMajor(int computeCapability) {
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

ttg::MACAMmaEncodingAttr buildMmaEncoding(MLIRContext *ctx,
                                          int computeCapability,
                                          ArrayRef<unsigned> warpsPerCTA,
                                          ArrayRef<unsigned> elementsMNK,
                                          unsigned colMajor) {
  SmallVector<unsigned, 2> elementsStride = {1, 1};
  return ttg::MACAMmaEncodingAttr::get(
      ctx, getMmaVersionMajor(computeCapability), computeCapability % 10,
      warpsPerCTA, elementsMNK, colMajor, /*isATrans=*/false,
      /*isBTrans=*/false, elementsStride);
}

// Assemble the atomic {mma, operandA, operandB} contract for a dot.
DotLayoutPlan makePlan(MLIRContext *ctx, ttg::MACAMmaEncodingAttr mma,
                       Type aElementType, Type bElementType) {
  DotLayoutPlan plan;
  plan.mma = mma;
  plan.operandA = ttg::DotOperandEncodingAttr::get(ctx, 0, mma, aElementType);
  plan.operandB = ttg::DotOperandEncodingAttr::get(ctx, 1, mma, bElementType);
  return plan;
}

struct DotLayoutConstraints {
  int64_t m, n, k;
  bool supportsColMajor;
  std::optional<BsmLayoutChain> bsmB;

  unsigned maxElements(unsigned dim, unsigned warps) const {
    uint64_t atoms = llvm::divideCeil<uint64_t>(
        dim == 0 ? m : n, dim == 0 ? kMmaTileM : kMmaTileN);
    return llvm::PowerOf2Ceil(
        std::max<uint64_t>(llvm::divideCeil(atoms, uint64_t(warps)), 1));
  }
};

// Powers of two in [1, maximum], e.g. maximum=4 -> {1, 2, 4}.
SmallVector<unsigned, 4> powerOfTwoGroupings(unsigned maximum) {
  SmallVector<unsigned, 4> result;
  for (unsigned value = 1; value <= maximum; value <<= 1)
    result.push_back(value);
  if (result.empty())
    result.push_back(1);
  return result;
}

FailureOr<DotLayoutConstraints>
getDotLayoutConstraints(tt::DotOp dot, int computeCapability, int numWarps) {
  auto bsmA = getBsmLayoutChain(dot.getA());
  auto bsmB = getBsmLayoutChain(dot.getB());
  if (failed(bsmA) || failed(bsmB))
    return failure();
  if (*bsmA) {
    dot.emitError("BSM permutation is supported only for dot operand B");
    return failure();
  }
  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());
  // Reject unsupported queries before toLinearLayout's target/rank assertions.
  if (getMmaVersionMajor(computeCapability) != 2) {
    dot.emitError("dot layout enumeration requires MACA MMA version 2");
    return failure();
  }
  if (numWarps <= 0 || !llvm::isPowerOf2_64(numWarps)) {
    dot.emitError(
        "dot layout enumeration requires a positive power-of-two warp count");
    return failure();
  }
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || dTy.getRank() != 2) {
    dot.emitError("dot layout enumeration requires rank-2 tensors");
    return failure();
  }
  for (RankedTensorType type : {aTy, bTy, dTy}) {
    if (llvm::any_of(type.getShape(), [](int64_t dim) {
          return dim <= 0 || dim > std::numeric_limits<int>::max() ||
                 !llvm::isPowerOf2_64(dim);
        })) {
      dot.emitError("dot layout enumeration requires static power-of-two "
                    "dimensions representable as i32");
      return failure();
    }
  }
  if (!mlir::supportMMA(dot, /*major=*/2, computeCapability % 10)) {
    dot.emitError(
        "dot dtype/precision is not supported by the target MMA instructions");
    return failure();
  }

  int64_t m = dTy.getShape()[0];
  int64_t n = dTy.getShape()[1];
  int64_t k = aTy.getShape()[1];
  Type aElem = aTy.getElementType();
  Type bElem = bTy.getElementType();
  const bool supportsColMajor =
      (aElem.isF16() || aElem.isBF16()) && (bElem.isF16() || bElem.isBF16());
  return DotLayoutConstraints{m, n, k, supportsColMajor, *bsmB};
}

bool matchesExplicitDotHints(tt::DotOp dot, const DotLayoutPlan &plan) {
  auto matches = [](Value value, Attribute expected) {
    auto hint = value.getDefiningOp<tile::SetLayoutOp>();
    if (!hint)
      return true;
    Attribute layout = hint.getLayout();
    // A blocked producer may feed a dot through a layout conversion. Only
    // explicit MMA/dot-operand anchors constrain the candidate's dot ports.
    return !isa<ttg::MACAMmaEncodingAttr, ttg::DotOperandEncodingAttr>(layout) ||
           layout == expected;
  };
  if (!matches(dot.getA(), plan.operandA) ||
      !matches(dot.getB(), plan.operandB) || !matches(dot.getC(), plan.mma))
    return false;
  for (Operation *user : dot.getResult().getUsers()) {
    auto hint = dyn_cast<tile::SetLayoutOp>(user);
    if (hint && isa<ttg::MACAMmaEncodingAttr>(hint.getLayout()) &&
        hint.getLayout() != plan.mma)
      return false;
  }
  return true;
}

// Enumeration and application must use the same function-local dot IDs.
FailureOr<SmallVector<DotLayoutDomain>> collectDots(ModuleOp module) {
  SmallVector<DotLayoutDomain> domains;
  DenseMap<Operation *, unsigned> nextIndex;
  WalkResult result = module.walk<WalkOrder::PreOrder>([&](tt::DotOp dot) {
    auto function = dot->getParentOfType<tt::FuncOp>();
    if (!function) {
      dot.emitError("dot layout enumeration requires an enclosing tt.func");
      return WalkResult::interrupt();
    }
    domains.push_back({dot,
                       function.getSymName().str(),
                       nextIndex[function.getOperation()]++,
                       {}});
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return domains;
}

LogicalResult validateDotLayoutCandidate(tt::DotOp dot,
                                        const DotLayoutPlan &plan,
                                        int computeCapability, int numWarps) {
  auto constraints = getDotLayoutConstraints(dot, computeCapability, numWarps);
  if (failed(constraints))
    return failure();
  if (auto error = checkMacaDotLayout(dot, plan.mma, plan.operandA,
                                     plan.operandB, computeCapability, numWarps))
    return dot.emitError("MMA layout is not legal for this dot and target: ")
           << *error;
  return success();
}

} // namespace

FailureOr<SmallVector<DotLayoutPlan>>
generateDotLayoutPlans(tt::DotOp dot, int computeCapability, int numWarps) {
  auto constraints = getDotLayoutConstraints(dot, computeCapability, numWarps);
  if (failed(constraints))
    return failure();
  MLIRContext *ctx = dot.getContext();
  Type aElem = dot.getA().getType().getElementType();
  Type bElem = dot.getB().getType().getElementType();

  SmallVector<DotLayoutPlan> plans;
  std::optional<std::string> bsmRejection;
  DenseSet<Attribute> seen;
  auto tryAppend = [&](ArrayRef<unsigned> warps, ArrayRef<unsigned> elements,
                       unsigned colMajor) {
    auto mma =
        buildMmaEncoding(ctx, computeCapability, warps, elements, colMajor);
    if (!seen.insert(mma).second)
      return;
    DotLayoutPlan plan = makePlan(ctx, mma, aElem, bElem);
    if (checkMacaDotLayout(dot, plan.mma, plan.operandA, plan.operandB,
                          computeCapability, numWarps))
      return;
    if (constraints->bsmB) {
      if (auto error = checkBsmLayout(*constraints->bsmB, plan.operandB)) {
        bsmRejection = std::move(error);
        return;
      }
    }
    plans.push_back(std::move(plan));
  };

  // The column-major accumulator lowering supports FP16/BF16 operands only.
  for (unsigned colMajor : {0u, 1u}) {
    if (colMajor && !constraints->supportsColMajor)
      continue;
    const unsigned packedDim = colMajor ? 0 : 1;
    const unsigned replicaDim = 1 - packedDim;
    for (unsigned replicaWarps = 1; replicaWarps <= unsigned(numWarps);
         replicaWarps <<= 1) {
      // M/N coverage may exceed the tensor shape. Let the common legality
      // check handle every warp grid, including the former [numWarps,1]
      // fallback, instead of excluding it from the enumeration.
      if (numWarps % replicaWarps)
        continue;
      unsigned packedWarps = numWarps / replicaWarps;
      unsigned maxReplicaElements =
          constraints->maxElements(replicaDim, replicaWarps);
      unsigned maxPackedElements =
          constraints->maxElements(packedDim, packedWarps);
      for (unsigned replicaElements : powerOfTwoGroupings(maxReplicaElements)) {
        for (unsigned packedElements : powerOfTwoGroupings(maxPackedElements)) {
          // Instruction packing limits are checked by checkMacaDotLayout.
          for (unsigned elementsK = 1;
               static_cast<uint64_t>(elementsK) * kMmaTileK <=
               static_cast<uint64_t>(constraints->k);
               elementsK <<= 1) {
            SmallVector<unsigned, 2> warps(2);
            SmallVector<unsigned, 3> elements(3);
            warps[replicaDim] = replicaWarps;
            warps[packedDim] = packedWarps;
            elements[replicaDim] = replicaElements;
            elements[packedDim] = packedElements;
            elements[2] = elementsK;
            tryAppend(warps, elements, colMajor);
          }
        }
      }
    }
  }

  if (plans.empty()) {
    if (constraints->bsmB)
      dot.emitError("no MMA layout supports the split load/BSM chain: ")
          << bsmRejection.value_or("M/N/K geometry is not supported");
    else
      dot.emitError("no dot layout plan covers this M/N/K geometry");
    return failure();
  }

  // The default is the first enumerated plan after ordering. Keep row-major
  // accumulators first and prefer balanced warp grids for square dots. Stable
  // ties preserve the native-K-first grouping order without pruning candidates.
  llvm::stable_sort(
      plans, [&](const DotLayoutPlan &lhs, const DotLayoutPlan &rhs) {
        auto left = cast<ttg::MACAMmaEncodingAttr>(lhs.mma);
        auto right = cast<ttg::MACAMmaEncodingAttr>(rhs.mma);
        if (left.getColMajor() != right.getColMajor())
          return left.getColMajor() < right.getColMajor();
        if (constraints->m != constraints->n)
          return false;
        auto imbalance = [](ttg::MACAMmaEncodingAttr mma) {
          auto warps = mma.getWarpsPerCTA();
          return std::max(warps[0], warps[1]) - std::min(warps[0], warps[1]);
        };
        return imbalance(left) < imbalance(right);
      });
  return plans;
}

FailureOr<SmallVector<DotLayoutDomain>>
enumerateDotLayoutPlans(ModuleOp module, int computeCapability, int numWarps) {
  auto domains = collectDots(module);
  if (failed(domains))
    return failure();
  for (auto &domain : *domains) {
    auto plans =
        generateDotLayoutPlans(domain.dot, computeCapability, numWarps);
    if (failed(plans))
      return failure();
    llvm::erase_if(*plans, [&](const DotLayoutPlan &plan) {
      return !matchesExplicitDotHints(domain.dot, plan);
    });
    if (plans->empty()) {
      domain.dot.emitError(
          "no dot layout candidate matches explicit set_layout requirements");
      return failure();
    }
    domain.plans = std::move(*plans);
  }
  return domains;
}

LogicalResult insertDotLayoutHints(
    ModuleOp module, int capability, int numWarps,
    const llvm::StringMap<DotLayoutPlan> &selection) {
  auto domains = collectDots(module);
  if (failed(domains))
    return failure();
  if (selection.size() != domains->size())
    return module.emitError(
        "candidate must select exactly one plan for every dot");

  SmallVector<DotLayoutChoice> choices;
  for (const auto &domain : *domains) {
    auto dot = domain.dot;
    for (Value value : {dot.getA(), dot.getB(), dot.getC(), dot.getD()})
      if (cast<RankedTensorType>(value.getType()).getEncoding())
        return dot.emitError(
            "layout hint injection requires unencoded TTIR tensors");
    std::string id = domain.functionName + "/dot/" +
                     std::to_string(domain.dotIndex);
    auto selected = selection.find(id);
    if (selected == selection.end())
      return dot.emitError("candidate is missing a plan for ") << id;
    const auto &plan = selected->second;
    if (failed(validateDotLayoutCandidate(dot, plan, capability, numWarps)))
      return failure();
    if (!matchesExplicitDotHints(dot, plan))
      return dot.emitError(
          "candidate conflicts with explicit set_layout requirements");
    choices.push_back({dot, plan});
  }

  // Validate the complete selection before mutating any dot. Clones share the
  // same snapshot and context, so the function-local IDs still identify them.
  for (auto [dot, plan] : choices) {
    OpBuilder builder(dot);
    for (auto [index, encoding] :
         llvm::enumerate(SmallVector<Attribute>{plan.operandA, plan.operandB})) {
      Value source = dot->getOperand(index);
      auto existing = source.getDefiningOp<tile::SetLayoutOp>();
      if (existing && existing.getLayout() == encoding)
        continue;
      auto hint = builder.create<tile::SetLayoutOp>(
          dot.getLoc(), source.getType(), source, encoding);
      dot->setOperand(index, hint.getResult());
    }
    builder.setInsertionPointAfter(dot);
    auto hint = builder.create<tile::SetLayoutOp>(
        dot.getLoc(), dot.getType(), dot.getResult(), plan.mma);
    dot.getResult().replaceAllUsesExcept(hint.getResult(), hint);
  }
  return success();
}

std::string getDotLayoutPlanId(const DotLayoutPlan &plan) {
  std::string description;
  llvm::raw_string_ostream os(description);
  os << "metax-commonir-dot-plan-v1\n";
  for (Attribute encoding : {plan.mma, plan.operandA, plan.operandB}) {
    encoding.print(os);
    os << '\n';
  }
  os.flush();
  return hashDescription(description);
}

std::string getDotLayoutProfileId(const DotLayoutPlan &plan) {
  auto mma = cast<ttg::MACAMmaEncodingAttr>(plan.mma);
  std::string description;
  llvm::raw_string_ostream os(description);
  os << "metax-commonir-accumulator-profile-v1\n"
     << mma.getVersionMajor() << ':' << mma.getVersionMinor() << '\n';
  llvm::interleaveComma(mma.getWarpsPerCTA(), os);
  os << '\n';
  llvm::interleaveComma(mma.getElementsMNK(), os);
  os << '\n' << mma.getColMajor() << '\n';
  os << mma.getCTALayout().getLinearLayout() << '\n';
  os.flush();
  return hashDescription(description);
}

} // namespace mlir::triton::metax
