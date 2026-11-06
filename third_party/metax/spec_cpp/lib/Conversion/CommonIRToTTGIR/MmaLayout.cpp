// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/MmaLayout.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <cstdlib>
#include <limits>

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

namespace {
constexpr unsigned kBsmSlots = 32;
} // namespace

unsigned getMmaRegisterGroupLimit(int64_t size, unsigned atomSize,
                                  unsigned warps) {
  uint64_t atoms = llvm::divideCeil<uint64_t>(size, atomSize);
  return llvm::PowerOf2Ceil(
      std::max<uint64_t>(llvm::divideCeil(atoms, uint64_t(warps)), 1));
}

ttg::MACAMmaEncodingAttr buildMmaEncoding(MLIRContext *ctx,
                                         int computeCapability,
                                         ArrayRef<unsigned> warpsPerCTA,
                                         ArrayRef<unsigned> elementsMNK,
                                         unsigned colMajor) {
  return ttg::MACAMmaEncodingAttr::get(
      ctx, 2, computeCapability % 10, warpsPerCTA, elementsMNK, colMajor,
      /*isATrans=*/false, /*isBTrans=*/false, {1, 1});
}

DotLayoutPlan makeDotLayoutPlan(triton::DotOp dot,
                                ttg::MACAMmaEncodingAttr mma) {
  auto *ctx = dot.getContext();
  return {mma,
          ttg::DotOperandEncodingAttr::get(ctx, 0, mma,
                                         dot.getA().getType().getElementType()),
          ttg::DotOperandEncodingAttr::get(ctx, 1, mma,
                                         dot.getB().getType().getElementType())};
}

std::optional<StringRef> checkMacaDotPreconditions(triton::DotOp dot,
                                                int computeCapability,
                                                int numWarps) {
  if (computeCapability < 80 || computeCapability >= 90)
    return "requires MACA MMA version 2";
  if (numWarps <= 0 || !llvm::isPowerOf2_64(numWarps))
    return "requires a positive power-of-two warp count";
  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  auto dTy = dot.getType();
  for (auto type : {aTy, bTy, dTy}) {
    if (type.getRank() != 2)
      return "requires rank-2 tensors";
    if (llvm::any_of(type.getShape(), [](int64_t dim) {
          return dim <= 0 || dim > std::numeric_limits<int>::max() ||
                 !llvm::isPowerOf2_64(dim);
        }))
      return "requires static power-of-two dimensions representable as i32";
  }
  if (!mlir::supportMMA(dot, 2, computeCapability % 10))
    return "dot dtype/precision is not supported by the target MMA instructions";
  return std::nullopt;
}

std::optional<StringRef> checkMacaDotLayout(triton::DotOp dot, Attribute encoding,
                                         Attribute operandA, Attribute operandB,
                                         int computeCapability, int numWarps) {
  if (auto error = checkMacaDotPreconditions(dot, computeCapability, numWarps))
    return error;
  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  auto dTy = dot.getType();
  for (Attribute attr : {encoding, operandA, operandB})
    if (!attr || attr.getContext() != dot.getContext())
      return "requires layouts from the dot's context";
  auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(encoding);
  if (!mma)
    return "requires a MACA MMA result layout";
  ArrayRef<unsigned> warps = mma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  auto powersOfTwo = [](ArrayRef<unsigned> values) {
    return llvm::all_of(values,
                        [](unsigned v) { return llvm::isPowerOf2_64(v); });
  };
  if (warps.size() != 2 || elements.size() != 3 || !powersOfTwo(warps) ||
      !powersOfTwo(elements) ||
      uint64_t(warps[0]) * warps[1] != unsigned(numWarps))
    return "requires power-of-two MMA geometry matching num-warps";
  Type aElem = aTy.getElementType(), bElem = bTy.getElementType();
  if (mma.getColMajor() > 1 ||
      (mma.getColMajor() &&
       (!(aElem.isF16() || aElem.isBF16()) ||
        !(bElem.isF16() || bElem.isBF16()))))
    return "column-major MMA supports FP16/BF16 operands only";

  // Check target fields and A/B layouts against the expected encoding.
  auto expectedMma = buildMmaEncoding(dot.getContext(), computeCapability, warps,
                                     elements, mma.getColMajor());
  if (mma != expectedMma ||
      operandA != ttg::DotOperandEncodingAttr::get(dot.getContext(), 0, mma, aElem) ||
      operandB != ttg::DotOperandEncodingAttr::get(dot.getContext(), 1, mma, bElem))
    return "MMA target fields or A/B operand layouts do not match";

  int64_t m = dTy.getShape()[0], n = dTy.getShape()[1], k = aTy.getShape()[1];
  auto atom = mma.getThreadShape(false);
  // Keep the existing M/N bounds until larger register groups are validated.
  auto maxElements = [&](int64_t size, unsigned axis) {
    return getMmaRegisterGroupLimit(size, atom[axis], warps[axis]);
  };
  if (elements[0] > maxElements(m, 0) || elements[1] > maxElements(n, 1))
    return "MMA M/N register groups exceed the currently supported shape bounds";
  unsigned nativeK = 1;
  if (aElem.isF32() && dot.getInputPrecision() == InputPrecision::TF32)
    nativeK = 2;
  else if (aElem.isF16() || aElem.isBF16())
    nativeK = 4;
  else if (aElem.isSignlessInteger(8))
    nativeK = computeCapability >= 86 ? 8 : 4;
  else if (isa<Float8E4M3FNType, Float8E5M2Type>(aElem))
    nativeK = 8;
  if (elements[2] < nativeK || uint64_t(elements[2]) * atom[2] > uint64_t(k))
    return "MMA K register group must fit the instruction and dot K dimension";

  int repM = ttg::getNumRepM(m, warps, elements[0]);
  int repN = ttg::getNumRepN(n, warps, elements[1]);
  int repK = ttg::getNumRepK(k, elements[2]);
  if (int64_t(repM) * warps[0] * atom[0] * elements[0] < m ||
      int64_t(repN) * warps[1] * atom[1] * elements[1] < n ||
      int64_t(repK) * atom[2] * elements[2] != k)
    return "MMA geometry must cover M/N and exactly cover K";

  // All geometry guards precede conversion to the LinearLayout model.
  (void)ttg::toLinearLayout(dTy.cloneWithEncoding(mma));
  (void)ttg::toLinearLayout(aTy.cloneWithEncoding(operandA));
  (void)ttg::toLinearLayout(bTy.cloneWithEncoding(operandB));
  return std::nullopt;
}

FailureOr<DotLayoutPlan> inferBsmDotLayout(triton::DotOp dot,
                                         int computeCapability, int numWarps) {
  auto a = dot.getA().getType();
  auto b = dot.getB().getType();
  auto d = dot.getType();
  if (auto error = checkMacaDotPreconditions(dot, computeCapability, numWarps)) {
    dot.emitError("BSM layout preparation failed: ") << *error;
    return failure();
  }

  auto mma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(d.getEncoding());
  if (!mma) {
    // Gluon's C500 BSM rule: one K fragment, 32 carrier slots, then allocate
    // enough N warps to avoid repeating the B fragment. The remaining warps
    // and elements cover M. The physical checker below validates the mapping.
    auto atom = ttg::getMmaThreadShape();
    unsigned atomM = atom[0], atomN = atom[1], atomK = atom[2];
    int64_t k = a.getShape()[1];
    if (k % atomK || k / atomK > kBsmSlots || kBsmSlots % (k / atomK)) {
      dot.emitError("BSM K must fit one 32-slot MMA fragment");
      return failure();
    }
    unsigned elementsK = k / atomK;
    unsigned elementsN = kBsmSlots / elementsK;
    unsigned atomsN = llvm::divideCeil(b.getShape()[1], int64_t(atomN));
    unsigned warpsN = llvm::PowerOf2Ceil(llvm::divideCeil(atomsN, elementsN));
    if (warpsN > unsigned(numWarps) || numWarps % warpsN) {
      dot.emitError("BSM N fragment needs more warp partitions than num-warps");
      return failure();
    }
    unsigned warpsM = numWarps / warpsN;
    unsigned atomsM = llvm::divideCeil(d.getShape()[0], int64_t(atomM));
    unsigned elementsM = llvm::PowerOf2Ceil(llvm::divideCeil(atomsM, warpsM));
    if (warpsM * elementsM > atomsM) {
      dot.emitError("BSM warp grid cannot exactly cover the M fragment");
      return failure();
    }
    mma = buildMmaEncoding(dot.getContext(), computeCapability, {warpsM, warpsN},
                           {elementsM, elementsN, elementsK}, /*colMajor=*/0);
  }
  DotLayoutPlan plan = makeDotLayoutPlan(dot, mma);
  if (auto error = checkMacaDotLayout(dot, plan.mma, plan.operandA,
                                     plan.operandB, computeCapability,
                                     numWarps)) {
    dot.emitError("MMA layout is not legal for this dot and target: ") << *error;
    return failure();
  }
  return plan;
}

FailureOr<std::optional<BsmLayoutChain>> getBsmLayoutChain(Value operand) {
  BsmLayoutChain chain;
  auto skipConversions = [&](Value value) {
    while (true) {
      if (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
        chain.conversions.push_back(cvt);
        value = cvt.getSrc();
      } else if (auto hint = value.getDefiningOp<tile::SetLayoutOp>()) {
        value = hint.getSource();
      } else {
        break;
      }
    }
    return value;
  };
  Value value = skipConversions(operand);
  chain.perm = value.getDefiningOp<ttg::BsmPermOp>();
  if (!chain.perm)
    return std::optional<BsmLayoutChain>();
  chain.load =
      skipConversions(chain.perm.getSrc1()).getDefiningOp<ttg::LocalLoadOp>();
  if (!chain.load || chain.load.getMmaMode() != 2) {
    chain.perm.emitError("BSM requires a local_load with mmaMode=2");
    return failure();
  }
  return std::optional<BsmLayoutChain>(std::move(chain));
}

namespace {

// Each BSM output selects one 16-bit half of an i32 carrier slot. This is the
// same index/coverage check as Gluon's analyzeBsmPermPhysicalContract,
// retaining the mapping so the fixed-address LDS implementation can also be
// checked.
using BsmMapping = std::array<std::pair<unsigned, unsigned>, kBsmSlots>;
std::optional<BsmMapping> getBsmMapping(unsigned n, unsigned k) {
  if (!n || !k || n % 2 || k % 2 || uint64_t(n) * k != kBsmSlots)
    return std::nullopt;
  BsmMapping mapping;
  std::array<unsigned, kBsmSlots> writes{};
  for (unsigned j = 0; j < n * k / 2; j += 2 * n) {
    for (unsigned v = 0; v < n / 2; ++v) {
      unsigned source = (j / n + v) * k;
      unsigned output = 2 * v + n * (j / k);
      for (unsigned upper : {0u, 1u}) {
        for (unsigned half : {0u, 1u}) {
          for (unsigned second : {0u, 1u}) {
            unsigned src = source + upper * n + second;
            unsigned dst = output + (2 * upper + half) * k + second;
            if (src >= kBsmSlots || dst >= kBsmSlots || ++writes[dst] != 1)
              return std::nullopt;
            mapping[dst] = {src, half};
          }
        }
      }
    }
  }
  if (llvm::any_of(writes, [](unsigned count) { return count != 1; }))
    return std::nullopt;
  return mapping;
}

// Mirror the address-only part of lowerLocalLdswithoutPerm. In addition to
// avoiding its fallback/assertion paths, prove that its v2i32 loads produce
// the selected logical B layout after BSM.
bool checkSplitLoadAddresses(LinearLayout cvt, ttg::MemDescType shared,
                             unsigned elementsN, const BsmMapping &mapping) {
  auto *ctx = shared.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  auto offset = StringAttr::get(ctx, "offset");
  if (!cvt.isTrivialOver({block}))
    return false;
  cvt = cvt.sublayout({reg, lane, warp}, {offset});
  if (!actionRemoveBroadcastedRegs(cvt).isIdentity())
    return false;

  unsigned vec = std::min(elementsN, 8u); // 128-bit maximum / 16-bit elements
  // The native implementation always emits v2i32, i.e. four 16-bit elements.
  if (vec != 4)
    return false;
  int32_t columns = shared.getShape()[ttg::getOrder(shared)[0]];
  std::vector<std::vector<int32_t>> tileBases = {{columns}, {1}, {2}};
  LinearLayout tile({{reg, tileBases}}, {{offset, llvm::NextPowerOf2(columns)}},
                    false);
  auto permutation = regPermForDivide(cvt, tile, /*left=*/true);
  if (!permutation)
    return false;
  auto permuted = permutation->apply(cvt);
  auto bases = permuted.getBases();
  auto &registerBases = bases[reg];
  for (unsigned i = 0; i < tileBases.size(); ++i) {
    if (registerBases[i] != tileBases[i])
      return false;
    registerBases[i] = {0};
  }
  for (auto dim : cvt.getInDimNames())
    for (auto basis : bases[dim])
      for (auto tileBasis : tileBases)
        if (basis[0] & tileBasis[0])
          return false;
  LinearLayout reps(bases, permuted.getOutDims(), false);
  LinearLayout addresses({{lane, bases[lane]}, {warp, bases[warp]}},
                         reps.getOutDims(), false);
  // Full buffers and slot views have no offset bits within this rank-2 shape.
  auto [additive, strides] = actionAdditiveStrides(reps, addresses, 0);
  if (!strides.isIdentity() || additive <= kBsmSlots / 2 ||
      additive < 4 * vec || additive > kBsmSlots ||
      kBsmSlots % additive)
    return false;
  int32_t upperOffset =
      reps.apply({{reg, kBsmSlots / 2}, {lane, 0}, {warp, 0}})[0].second;
  // The native split-LDS lowering still uses a fixed 512-element offset.
  if (upperOffset != 512)
    return false;

  for (int w = 0; w < cvt.getInDimSize(warp); ++w) {
    for (int l = 0; l < cvt.getInDimSize(lane); ++l) {
      std::array<int32_t, kBsmSlots> raw;
      raw.fill(-1);
      int32_t base = addresses.apply({{lane, l}, {warp, w}})[0].second;
      for (unsigned i = 0; i < kBsmSlots; i += additive) {
        int32_t outer =
            base ^ reps.apply({{reg, i}, {lane, 0}, {warp, 0}})[0].second;
        for (unsigned j = 0; j < additive / 2; j += 2 * vec) {
          int32_t inner =
              outer + reps.apply({{reg, j}, {lane, 0}, {warp, 0}})[0].second;
          for (unsigned v = 0; v < vec / 2; ++v) {
            for (unsigned upper : {0u, 1u}) {
              for (unsigned duplicate : {0u, 1u}) {
                for (unsigned second : {0u, 1u}) {
                  unsigned index = i + j + 4 * v + 2 * duplicate + second +
                                   upper * (kBsmSlots / 2);
                  if (index >= kBsmSlots || raw[index] != -1)
                    return false;
                  raw[index] =
                      inner + 2 * v + second * columns + upper * upperOffset;
                }
              }
            }
          }
        }
      }
      for (unsigned r = 0; r < kBsmSlots; ++r) {
        auto [source, half] = mapping[r];
        int32_t expected =
            cvt.apply({{reg, r}, {lane, l}, {warp, w}})[0].second;
        if (raw[source] < 0 || raw[source] + half != expected)
          return false;
      }
    }
  }
  return true;
}

} // namespace

std::optional<std::string> checkBsmLayout(BsmLayoutChain chain,
                                         Attribute encoding,
                                         ttg::MemDescType shared) {
  auto dot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(encoding);
  auto mma =
      dot ? dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent()) : nullptr;
  if (!mma || dot.getOpIdx() != 1 || mma.getVersionMajor() != 2 ||
      mma.getIsATrans() || mma.getIsBTrans())
    return "BSM requires a non-transposed MACA v2 B operand";
  if (std::getenv("TRITON_DISABLE_SPLIT_PERM"))
    return "BSM split load is disabled by TRITON_DISABLE_SPLIT_PERM";
  auto raw = cast<RankedTensorType>(chain.load.getType());
  auto result = cast<RankedTensorType>(chain.perm.getType());
  if (!shared)
    shared = cast<ttg::MemDescType>(chain.load.getSrc().getType());
  if (raw.getRank() != 2 || result.getRank() != 2 || shared.getRank() != 2 ||
      raw.getShape() != result.getShape() ||
      raw.getShape() != shared.getShape() ||
      !raw.getElementType().isInteger(32) ||
      (!result.getElementType().isF16() && !result.getElementType().isBF16()) ||
      result.getElementType() != shared.getElementType())
    return "BSM requires matching rank-2 f16/bf16 shared storage, i32 load "
           "and f16/bf16 result";
  if (!isa<ttg::SwizzledSharedEncodingAttr>(shared.getEncoding()) ||
      shared.getAllocShape().take_back(2) != shared.getShape() ||
      chain.load.getSrc().getDefiningOp<ttg::MemDescTransOp>())
    return "BSM requires an untransposed swizzled shared buffer or slot view";
  if (mma.getRepOrderForOperand(1) == ttg::getOrder(shared))
    return "BSM requires shared order different from the MMA B register order";
  auto elements = mma.getElementsMNK();
  if (elements.size() != 3)
    return "BSM requires rank-2 MMA elementsMNK";
  auto mapping = getBsmMapping(elements[1], elements[2]);
  if (!mapping)
    return "BSM permutation must cover its 32 element slots exactly once";
  auto rawLayout = ttg::toLinearLayout(raw.cloneWithEncoding(encoding));
  auto resultLayout = ttg::toLinearLayout(result.cloneWithEncoding(encoding));
  auto reg = StringAttr::get(raw.getContext(), "register");
  if (rawLayout.getInDimSize(reg) != kBsmSlots ||
      resultLayout.getInDimSize(reg) != kBsmSlots)
    return "BSM source and result must each have 32 per-thread element slots";
  if (!checkSplitLoadAddresses(
          rawLayout.invertAndCompose(ttg::toLinearLayout(shared)), shared,
          elements[1], *mapping))
    return "split LDS addresses do not implement the selected BSM layout with "
           "the native 512-element upper-group offset";
  return std::nullopt;
}

} // namespace mlir::triton::metax
