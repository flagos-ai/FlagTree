// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"
#include "triton/Conversion/CommonIRToTTGIR/BsmLayout.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

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

// C500 MACA v2 per-instruction MMA tiling atom (M, N, K). Mirrors the plugin's
// getMmaThreadShape()/kC500MmaTile* constants; the geometry enumeration and the
// coverage check below are expressed in multiples of these atoms.
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

unsigned getNativeElementsK(Type elementType, bool enableTf32,
                            int computeCapability) {
  if (elementType.isF32() && enableTf32)
    return 2;
  if (elementType.isF32() || elementType.isF64())
    return 1;
  if (elementType.isF16() || elementType.isBF16())
    return 4;
  if (elementType.isSignlessInteger(8))
    return computeCapability >= 86 ? 8 : 4;
  if (isa<Float8E4M3FNType, Float8E5M2Type>(elementType))
    return 8;
  return 1;
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
  unsigned nativeK;
  bool supportsColMajor;
  std::optional<BsmLayoutChain> bsmB;

  unsigned maxElements(unsigned dim, unsigned warps) const {
    uint64_t atoms = llvm::divideCeil<uint64_t>(
        dim == 0 ? m : n, dim == 0 ? kMmaTileM : kMmaTileN);
    return llvm::PowerOf2Ceil(
        std::max<uint64_t>(llvm::divideCeil(atoms, uint64_t(warps)), 1));
  }

  bool contains(ArrayRef<unsigned> warps, ArrayRef<unsigned> elements,
                unsigned colMajor, int numWarps) const {
    if (warps.size() != 2 || elements.size() != 3 || colMajor > 1 ||
        (colMajor && !supportsColMajor))
      return false;
    if (llvm::any_of(warps,
                     [](unsigned v) { return !llvm::isPowerOf2_64(v); }) ||
        llvm::any_of(elements,
                     [](unsigned v) { return !llvm::isPowerOf2_64(v); }) ||
        uint64_t(warps[0]) * warps[1] != unsigned(numWarps))
      return false;
    return elements[0] <= maxElements(0, warps[0]) &&
           elements[1] <= maxElements(1, warps[1]) && elements[2] >= nativeK &&
           uint64_t(elements[2]) * kMmaTileK <= uint64_t(k);
  }
};

// A plan is legal when its warp grid matches numWarps, its MMA geometry exactly
// covers the dot's M/N/K domain (M/N may pad up, K must divide exactly), and
// the resulting tensor types are representable by the LinearLayout model. Pure
// arithmetic runs first so we never hand an ill-formed geometry to
// toLinearLayout. Shared by enumeration and explicit candidate application.
bool isPlanValid(tt::DotOp dot, const DotLayoutPlan &plan,
                 const DotLayoutConstraints &constraints, int computeCapability,
                 int numWarps) {
  for (Attribute encoding : {plan.mma, plan.operandA, plan.operandB})
    if (!encoding || encoding.getContext() != dot.getContext())
      return false;
  auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(plan.mma);
  if (!mma)
    return false;
  ArrayRef<unsigned> warps = mma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  if (!constraints.contains(warps, elements, mma.getColMajor(), numWarps))
    return false;

  // Compare the entire triple, including target version, CTA layout,
  // transpose/stride fields and operand parents/widths, before LinearLayout.
  auto expectedMma = buildMmaEncoding(dot.getContext(), computeCapability,
                                      warps, elements, mma.getColMajor());
  auto expected = makePlan(dot.getContext(), expectedMma,
                           dot.getA().getType().getElementType(),
                           dot.getB().getType().getElementType());
  if (plan.mma != expected.mma || plan.operandA != expected.operandA ||
      plan.operandB != expected.operandB)
    return false;

  int64_t m = constraints.m, n = constraints.n, k = constraints.k;
  int repM = ttg::getNumRepM(m, warps, elements[0]);
  int repN = ttg::getNumRepN(n, warps, elements[1]);
  int repK = ttg::getNumRepK(k, elements[2]);
  int64_t coveredM = int64_t(repM) * warps[0] * kMmaTileM * elements[0];
  int64_t coveredN = int64_t(repN) * warps[1] * kMmaTileN * elements[1];
  int64_t coveredK = int64_t(repK) * kMmaTileK * elements[2];
  if (coveredM < m || coveredN < n || coveredK != k)
    return false;

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());
  auto dotType =
      RankedTensorType::get(dTy.getShape(), dTy.getElementType(), plan.mma);
  auto aType = RankedTensorType::get(aTy.getShape(), aTy.getElementType(),
                                     plan.operandA);
  auto bType = RankedTensorType::get(bTy.getShape(), bTy.getElementType(),
                                     plan.operandB);
  // toLinearLayout asserts only on rank!=2 / verMajor!=2 / LDS-trans, all of
  // which the enumeration excludes, so these calls are crash-safe.
  (void)ttg::toLinearLayout(dotType);
  (void)ttg::toLinearLayout(aType);
  (void)ttg::toLinearLayout(bType);
  return true;
}

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
  bool enableTf32 = dot.getInputPrecision() == tt::InputPrecision::TF32;

  // K packing starts from the native elementsK and doubles while it keeps
  // dividing K into whole MMA operands.
  unsigned nativeK = getNativeElementsK(aElem, enableTf32, computeCapability);

  const bool supportsColMajor =
      (aElem.isF16() || aElem.isBF16()) && (bElem.isF16() || bElem.isBF16());
  return DotLayoutConstraints{m, n, k, nativeK, supportsColMajor, *bsmB};
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
    if (!isPlanValid(dot, plan, *constraints, computeCapability, numWarps))
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
          for (unsigned elementsK = constraints->nativeK;
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
    domain.plans = std::move(*plans);
  }
  return domains;
}

FailureOr<SmallVector<DotLayoutChoice>>
selectDotLayouts(ModuleOp module, int computeCapability, int numWarps,
                 const llvm::StringMap<DotLayoutPlan> &selectedPlans) {
  auto moduleWarps = module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  if (!moduleWarps || moduleWarps.getInt() != numWarps) {
    module.emitError("dot plan injection requires prepared TTGIR with the "
                     "requested warp count");
    return failure();
  }
  auto domains = collectDots(module);
  if (failed(domains))
    return failure();
  if (selectedPlans.size() != domains->size()) {
    module.emitError("candidate must select exactly one plan for every dot");
    return failure();
  }

  SmallVector<DotLayoutChoice> choices;
  for (const auto &domain : *domains) {
    auto dot = domain.dot;
    for (Value value : {dot.getA(), dot.getB(), dot.getC(), dot.getD()}) {
      if (!cast<RankedTensorType>(value.getType()).getEncoding()) {
        dot.emitError("dot plan injection requires encoded TTGIR tensors");
        return failure();
      }
    }
    if (isa<ttg::MACAMmaEncodingAttr>(dot.getType().getEncoding())) {
      dot.emitError("dot layout plan has already been injected");
      return failure();
    }
    std::string dotId =
        domain.functionName + "/dot/" + std::to_string(domain.dotIndex);
    auto it = selectedPlans.find(dotId);
    if (it == selectedPlans.end()) {
      dot.emitError("candidate is missing a plan for ") << dotId;
      return failure();
    }
    const auto &choice = it->second;
    auto constraints =
        getDotLayoutConstraints(dot, computeCapability, numWarps);
    if (failed(constraints))
      return failure();
    if (!isPlanValid(dot, choice, *constraints, computeCapability, numWarps)) {
      dot.emitError("selected layout is not a legal enumerated plan for ")
          << dotId;
      return failure();
    }
    choices.push_back({dot, choice});
  }
  return choices;
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
