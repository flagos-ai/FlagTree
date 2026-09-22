// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

SmallVector<unsigned, 3> defaultElementsMNK(Type elementType, bool enableTf32,
                                            int computeCapability) {
  if (elementType.isF32() && enableTf32)
    return {1, 1, 2};
  if (elementType.isF32() || elementType.isF64())
    return {1, 1, 1};
  if (elementType.isF16() || elementType.isBF16())
    return {1, 1, 4};
  if (elementType.isSignlessInteger(8))
    return computeCapability >= 86 ? SmallVector<unsigned, 3>{1, 1, 8}
                                   : SmallVector<unsigned, 3>{1, 1, 4};
  if (isa<Float8E4M3FNType, Float8E5M2Type>(elementType))
    return {1, 1, 8};
  return {1, 1, 1};
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

// A plan is legal when its warp grid matches numWarps, its MMA geometry exactly
// covers the dot's M/N/K domain (M/N may pad up, K must divide exactly), and
// the resulting tensor types are representable by the LinearLayout model. Pure
// arithmetic runs first so we never hand an ill-formed geometry to
// toLinearLayout. Mirrors the plugin's validatePlan without its BSM/pipeline
// concerns.
bool isPlanValid(tt::DotOp dot, const DotLayoutPlan &plan, int64_t m, int64_t n,
                 int64_t k, int numWarps) {
  auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(plan.mma);
  if (!mma)
    return false;
  ArrayRef<unsigned> warps = mma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  if (warps.size() != 2 || elements.size() != 3)
    return false;
  if (warps[0] == 0 || warps[1] == 0 || elements[0] == 0 || elements[1] == 0 ||
      elements[2] == 0)
    return false;
  if (static_cast<int64_t>(warps[0]) * warps[1] != numWarps)
    return false;

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

} // namespace

FailureOr<SmallVector<DotLayoutPlan>>
generateDotLayoutPlans(tt::DotOp dot, int computeCapability, int numWarps) {
  MLIRContext *ctx = dot.getContext();
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
  unsigned nativeK =
      defaultElementsMNK(aElem, enableTf32, computeCapability)[2];

  // Number of MMA atoms along M and N; warps and per-warp element groups are
  // enumerated as factorizations of these counts.
  uint64_t atomM = llvm::divideCeil<uint64_t>(m, kMmaTileM);
  uint64_t atomN = llvm::divideCeil<uint64_t>(n, kMmaTileN);

  SmallVector<DotLayoutPlan> plans;
  DenseSet<Attribute> seen;
  auto tryAppend = [&](ArrayRef<unsigned> warps, ArrayRef<unsigned> elements,
                       unsigned colMajor) {
    auto mma =
        buildMmaEncoding(ctx, computeCapability, warps, elements, colMajor);
    if (!seen.insert(mma).second)
      return;
    DotLayoutPlan plan = makePlan(ctx, mma, aElem, bElem);
    if (isPlanValid(dot, plan, m, n, k, numWarps))
      plans.push_back(std::move(plan));
  };

  // Try the conservative [numWarps,1]/native-elements fallback first.
  // Enumerated alternatives follow; legality does not imply performance.
  {
    SmallVector<unsigned, 2> warps = {unsigned(numWarps), 1};
    SmallVector<unsigned, 3> elements =
        defaultElementsMNK(aElem, enableTf32, computeCapability);
    tryAppend(warps, elements, /*colMajor=*/0);
  }

  // colMajor=0: N is the packed dimension, M the replica dimension.
  const unsigned colMajor = 0;
  const unsigned packedDim = colMajor ? 0 : 1;
  const unsigned replicaDim = 1 - packedDim;
  uint64_t atoms[2] = {atomM, atomN};

  for (unsigned replicaWarps = 1; replicaWarps <= unsigned(numWarps);
       replicaWarps <<= 1) {
    if (numWarps % replicaWarps ||
        static_cast<uint64_t>(replicaWarps) > atoms[replicaDim])
      continue;
    unsigned packedWarps = numWarps / replicaWarps;
    unsigned maxReplicaElements =
        static_cast<unsigned>(llvm::PowerOf2Ceil(std::max<uint64_t>(
            llvm::divideCeil<uint64_t>(atoms[replicaDim], replicaWarps), 1)));
    unsigned maxPackedElements =
        static_cast<unsigned>(llvm::PowerOf2Ceil(std::max<uint64_t>(
            llvm::divideCeil<uint64_t>(atoms[packedDim], packedWarps), 1)));
    for (unsigned replicaElements : powerOfTwoGroupings(maxReplicaElements)) {
      for (unsigned packedElements : powerOfTwoGroupings(maxPackedElements)) {
        for (unsigned elementsK = nativeK;
             static_cast<uint64_t>(elementsK) * kMmaTileK <=
             static_cast<uint64_t>(k);
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

  if (plans.empty()) {
    dot.emitError("no dot layout plan covers this M/N/K geometry");
    return failure();
  }
  return plans;
}

FailureOr<SmallVector<DotLayoutDomain>>
enumerateDotLayoutPlans(ModuleOp module, int computeCapability, int numWarps) {
  SmallVector<DotLayoutDomain> domains;
  DenseMap<Operation *, unsigned> nextIndex;
  WalkResult result = module.walk<WalkOrder::PreOrder>([&](tt::DotOp dot) {
    auto function = dot->getParentOfType<tt::FuncOp>();
    if (!function) {
      dot.emitError("dot layout enumeration requires an enclosing tt.func");
      return WalkResult::interrupt();
    }
    auto plans = generateDotLayoutPlans(dot, computeCapability, numWarps);
    if (failed(plans))
      return WalkResult::interrupt();
    domains.push_back({dot, function.getSymName().str(),
                       nextIndex[function.getOperation()]++,
                       std::move(*plans)});
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return domains;
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
