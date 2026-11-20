// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <limits>

namespace mlir::triton::metax {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct DotLayoutChoice {
  triton::DotOp dot;
  DotLayoutPlan plan;
};

DotLayoutPlan makeDotLayoutPlan(triton::DotOp dot,
                                ttg::MACAMmaEncodingAttr mma) {
  auto *ctx = dot.getContext();
  return {mma,
          ttg::DotOperandEncodingAttr::get(
              ctx, 0, mma, dot.getA().getType().getElementType()),
          ttg::DotOperandEncodingAttr::get(
              ctx, 1, mma, dot.getB().getType().getElementType())};
}

std::string hashDescription(StringRef description) {
  auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(description));
  return llvm::toHex(ArrayRef<uint8_t>(digest), /*LowerCase=*/true);
}

struct DotLayoutConstraints {
  std::array<unsigned, 2> atomsMN;
  unsigned versionMinor;
  unsigned minElementsK, maxElementsK;
  bool isSquare;
  bool supportsColMajor;

  unsigned maxElementsMN(unsigned dim, unsigned warps) const {
    return llvm::divideCeil(atomsMN[dim], warps);
  }
};

FailureOr<DotLayoutConstraints>
getDotLayoutConstraints(tt::DotOp dot, int computeCapability, int numWarps) {
  auto reject = [&](StringRef reason) -> FailureOr<DotLayoutConstraints> {
    dot.emitError("cannot enumerate dot layouts: ") << reason;
    return failure();
  };
  if (computeCapability < 80 || computeCapability >= 90)
    return reject("requires MACA MMA version 2");
  if (numWarps <= 0 || !llvm::isPowerOf2_64(numWarps))
    return reject("requires a positive power-of-two warp count");
  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  auto dTy = dot.getType();
  for (auto type : {aTy, bTy, dTy}) {
    if (type.getRank() != 2)
      return reject("requires rank-2 tensors");
    if (llvm::any_of(type.getShape(), [](int64_t dim) {
          return dim <= 0 || dim > std::numeric_limits<int>::max() ||
                 !llvm::isPowerOf2_64(dim);
        }))
      return reject(
          "requires static power-of-two dimensions representable as i32");
  }
  const unsigned versionMinor = computeCapability % 10;
  if (!mlir::supportMMA(dot, 2, versionMinor))
    return reject("dot dtype/precision is not supported by the target MMA "
                  "instructions");

  unsigned m = static_cast<unsigned>(dTy.getShape()[0]);
  unsigned n = static_cast<unsigned>(dTy.getShape()[1]);
  unsigned k = static_cast<unsigned>(aTy.getShape()[1]);
  auto threadShape = ttg::getMmaThreadShape();
  Type aElem = aTy.getElementType();
  Type bElem = bTy.getElementType();
  const bool supportsColMajor =
      (aElem.isF16() || aElem.isBF16()) && (bElem.isF16() || bElem.isBF16());
  return DotLayoutConstraints{
      {llvm::divideCeil(m, threadShape[0]),
       llvm::divideCeil(n, threadShape[1])},
      versionMinor,
      getMinMmaElementsPerK(dot, versionMinor),
      k / threadShape[2],
      m == n,
      supportsColMajor,
  };
}

void appendPlansForWarpGrid(tt::DotOp dot,
                            const DotLayoutConstraints &constraints,
                            ArrayRef<unsigned> warps, bool colMajor,
                            SmallVectorImpl<DotLayoutPlan> &plans) {
  // Keep candidate order M, N, K for row-major and N, M, K for column-major.
  unsigned maxOuter = constraints.maxElementsMN(0, warps[0]);
  unsigned maxInner = constraints.maxElementsMN(1, warps[1]);
  if (colMajor)
    std::swap(maxOuter, maxInner);
  for (unsigned outer = 1; outer <= maxOuter; outer <<= 1) {
    for (unsigned inner = 1; inner <= maxInner; inner <<= 1) {
      unsigned elementsM = colMajor ? inner : outer;
      unsigned elementsN = colMajor ? outer : inner;
      for (unsigned elementsK = constraints.minElementsK;
           elementsK <= constraints.maxElementsK; elementsK <<= 1) {
        auto mma = ttg::MACAMmaEncodingAttr::get(
            dot.getContext(), 2, constraints.versionMinor, warps,
            {elementsM, elementsN, elementsK}, colMajor, /*isATrans=*/false,
            /*isBTrans=*/false, {1, 1});
        plans.push_back(makeDotLayoutPlan(dot, mma));
      }
    }
  }
}

Attribute getSetLayout(Operation *op) {
  auto hint = dyn_cast_or_null<tile::SetLayoutOp>(op);
  return hint ? hint.getLayout() : Attribute{};
}

bool hasExplicitMmaLayout(tt::DotOp dot) {
  for (Value operand : {dot.getA(), dot.getB()})
    if (!isa_and_nonnull<ttg::DotOperandEncodingAttr>(
            getSetLayout(operand.getDefiningOp())))
      return false;
  auto hasMmaLayout = [](Operation *op) {
    return isa_and_nonnull<ttg::MACAMmaEncodingAttr>(getSetLayout(op));
  };
  return hasMmaLayout(dot.getC().getDefiningOp()) ||
         llvm::any_of(dot.getResult().getUsers(), hasMmaLayout);
}

bool matchesExplicitDotHints(tt::DotOp dot, const DotLayoutPlan &plan) {
  auto matches = [](Value value, Attribute expected) {
    Attribute layout = getSetLayout(value.getDefiningOp());
    // A blocked producer may feed a dot through a layout conversion. Only
    // explicit MMA/dot-operand anchors constrain the candidate's dot ports.
    return !isa_and_nonnull<ttg::MACAMmaEncodingAttr,
                            ttg::DotOperandEncodingAttr>(layout) ||
           layout == expected;
  };
  if (!matches(dot.getA(), plan.operandA) ||
      !matches(dot.getB(), plan.operandB) || !matches(dot.getC(), plan.mma))
    return false;
  for (Operation *user : dot.getResult().getUsers()) {
    Attribute layout = getSetLayout(user);
    if (isa_and_nonnull<ttg::MACAMmaEncodingAttr>(layout) && layout != plan.mma)
      return false;
  }
  return true;
}

FailureOr<SmallVector<DotLayoutDomain>> collectTunableDots(ModuleOp module) {
  SmallVector<DotLayoutDomain> domains;
  DenseMap<Operation *, unsigned> nextIndex;
  WalkResult result = module.walk<WalkOrder::PreOrder>([&](tt::DotOp dot) {
    auto function = dot->getParentOfType<tt::FuncOp>();
    if (!function) {
      dot.emitError("dot layout enumeration requires an enclosing tt.func");
      return WalkResult::interrupt();
    }
    unsigned index = nextIndex[function.getOperation()]++;
    if (!hasExplicitMmaLayout(dot))
      domains.push_back(
          {dot,
           function.getSymName().str() + "/dot/" + std::to_string(index),
           {}});
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return domains;
}

} // namespace

FailureOr<SmallVector<DotLayoutPlan>>
generateDotLayoutPlans(tt::DotOp dot, int computeCapability, int numWarps) {
  auto constraints = getDotLayoutConstraints(dot, computeCapability, numWarps);
  if (failed(constraints))
    return failure();

  SmallVector<DotLayoutPlan> plans;

  // The column-major accumulator lowering supports FP16/BF16 operands only.
  for (bool colMajor : {false, true}) {
    if (colMajor && !constraints->supportsColMajor)
      continue;
    for (unsigned outerWarps = 1; outerWarps <= unsigned(numWarps);
         outerWarps <<= 1) {
      unsigned warpsM = outerWarps;
      unsigned warpsN = numWarps / outerWarps;
      if (colMajor)
        std::swap(warpsM, warpsN);
      appendPlansForWarpGrid(dot, *constraints, {warpsM, warpsN}, colMajor,
                             plans);
    }
  }

  if (plans.empty()) {
    dot.emitError("no dot layout plan covers this M/N/K geometry");
    return failure();
  }

  // The first plan is the default. Prefer row-major accumulators and balanced
  // warp grids for square dots; preserve enumeration order for ties.
  llvm::stable_sort(
      plans, [&](const DotLayoutPlan &lhs, const DotLayoutPlan &rhs) {
        auto left = cast<ttg::MACAMmaEncodingAttr>(lhs.mma);
        auto right = cast<ttg::MACAMmaEncodingAttr>(rhs.mma);
        if (left.getColMajor() != right.getColMajor())
          return left.getColMajor() < right.getColMajor();
        if (!constraints->isSquare)
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
  auto domains = collectTunableDots(module);
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

static FailureOr<SmallVector<DotLayoutChoice>>
collectValidatedChoices(ModuleOp module,
                        const llvm::StringMap<DotLayoutPlan> &selection) {
  auto domains = collectTunableDots(module);
  if (failed(domains))
    return failure();
  if (selection.size() != domains->size())
    return module.emitError(
        "candidate must select exactly one plan for every tunable dot");

  SmallVector<DotLayoutChoice> choices;
  for (const auto &domain : *domains) {
    auto dot = domain.dot;
    for (Value value : {dot.getA(), dot.getB(), dot.getC(), dot.getD()})
      if (cast<RankedTensorType>(value.getType()).getEncoding())
        return dot.emitError(
            "layout hint injection requires unencoded TTIR tensors");
    auto selected = selection.find(domain.dotId);
    if (selected == selection.end())
      return dot.emitError("candidate is missing a plan for ") << domain.dotId;
    const auto &plan = selected->second;
    for (Attribute layout : {plan.mma, plan.operandA, plan.operandB})
      if (!layout || layout.getContext() != module.getContext())
        return dot.emitError(
            "candidate layouts must belong to the module's context");
    if (!matchesExplicitDotHints(dot, plan))
      return dot.emitError(
          "candidate conflicts with explicit set_layout requirements");
    choices.push_back({dot, plan});
  }
  return choices;
}

LogicalResult
insertDotLayoutHints(ModuleOp module,
                     const llvm::StringMap<DotLayoutPlan> &selection) {
  auto choices = collectValidatedChoices(module, selection);
  if (failed(choices))
    return failure();

  for (auto [dot, plan] : *choices) {
    OpBuilder builder(dot);
    auto setOperandHint = [&](OpOperand &operand, Attribute encoding) {
      Value source = operand.get();
      if (getSetLayout(source.getDefiningOp()) == encoding)
        return;
      auto hint = builder.create<tile::SetLayoutOp>(
          dot.getLoc(), source.getType(), source, encoding);
      operand.set(hint.getResult());
    };
    setOperandHint(dot.getAMutable(), plan.operandA);
    setOperandHint(dot.getBMutable(), plan.operandB);

    Value result = dot.getResult();
    if (result.hasOneUse() && getSetLayout(*result.user_begin()) == plan.mma)
      continue;
    builder.setInsertionPointAfter(dot);
    auto hint = builder.create<tile::SetLayoutOp>(
        dot.getLoc(), result.getType(), result, plan.mma);
    result.replaceAllUsesExcept(hint.getResult(), hint);
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
  return hashDescription(description);
}

std::string getDotLayoutProfileId(const DotLayoutPlan &plan) {
  // Omit isATrans, isBTrans and elementsStride from the profile because every
  // candidate disables LDS transpose and uses unit strides for both operands.
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
  return hashDescription(description);
}

} // namespace mlir::triton::metax
