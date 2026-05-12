#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

inline constexpr llvm::StringLiteral kDisableGenericDotPipelineAttr =
    "tt.disable_generic_dot_pipeline";

static int getMusaComputeCapability(ModuleOp mod) {
  StringAttr targetAttr = mod->getAttrOfType<StringAttr>(ttg::AttrTargetName);
  if (!targetAttr)
    return -1;
  StringRef ref = targetAttr.strref();
  if (!ref.starts_with("musa:"))
    return -1;
  StringRef arch = ref.drop_front(5);
  if (arch.starts_with("ph1"))
    return 31;
  int computeCapability = -1;
  if (arch.getAsInteger(10, computeCapability))
    return -1;
  return computeCapability;
}

static std::optional<triton::musa::SQMMAEltType> toSqmmaEltType(Type elemTy) {
  if (elemTy.isF16())
    return triton::musa::SQMMAEltType::f16;
  if (elemTy.isBF16())
    return triton::musa::SQMMAEltType::bf16;
  if (elemTy.isF32())
    return triton::musa::SQMMAEltType::f32;
  if (elemTy.isInteger(32))
    return triton::musa::SQMMAEltType::s32;
  if (elemTy.isInteger(8))
    return triton::musa::SQMMAEltType::s8;
  if (llvm::isa<Float8E4M3FNType, Float8E4M3FNUZType>(elemTy))
    return triton::musa::SQMMAEltType::e4m3;
  if (llvm::isa<Float8E5M2Type, Float8E5M2FNUZType>(elemTy))
    return triton::musa::SQMMAEltType::e5m2;
  return std::nullopt;
}

static std::optional<triton::musa::SQMMAEltType>
toSqmmaOperandEltType(Type elemTy, bool allowTF32) {
  if (elemTy.isF32() && allowTF32)
    return triton::musa::SQMMAEltType::tf32;
  return toSqmmaEltType(elemTy);
}

static triton::musa::SQMMALayout inferSqmmaLayout(Value v) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(v.getType())) {
    auto order = ttg::getOrderForMemory(tensorTy);
    bool isRowMajor = !order.empty() && order.front() + 1 == tensorTy.getRank();
    return isRowMajor ? triton::musa::SQMMALayout::row
                      : triton::musa::SQMMALayout::col;
  }
  if (auto memDescTy = dyn_cast<ttg::MemDescType>(v.getType())) {
    auto order = ttg::getOrder(memDescTy);
    bool isRowMajor =
        !order.empty() && order.front() + 1 == memDescTy.getRank();
    return isRowMajor ? triton::musa::SQMMALayout::row
                      : triton::musa::SQMMALayout::col;
  }
  return triton::musa::SQMMALayout::row;
}

static bool isSupportedWmmaOperandType(Type elemTy, bool allowTF32) {
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(8) ||
      tt::type::isFloat8(elemTy))
    return true;
  return elemTy.isF32() && allowTF32;
}

static SmallVector<SmallVector<unsigned, 3>>
getWmmaCandidateInstrShapes(Type elemTy, bool allowTF32) {
  if (elemTy.isF32() && allowTF32)
    return {{16, 8, 4}, {16, 8, 8}, {16, 16, 16}};
  if (elemTy.isF16() || elemTy.isBF16()) {
    return {
        {8, 16, 16}, {16, 8, 8}, {16, 8, 16}, {16, 16, 16}, {16, 16, 32},
    };
  }
  return {
      {8, 16, 16}, {16, 8, 16}, {16, 16, 16}, {16, 16, 32}, {16, 16, 64},
  };
}

struct SelectedConfig {
  SmallVector<unsigned, 3> instrShape;
  SmallVector<unsigned, 2> warpsPerCTA;
};

static bool isKnownBrokenSqmmaConfig(Type elemTy, bool allowTF32,
                                     ArrayRef<unsigned> instrShape) {
  auto eltTypeA = toSqmmaOperandEltType(elemTy, allowTF32);
  if (!eltTypeA || instrShape.size() != 3)
    return false;

  triton::musa::SQMMAEltType eltTypeC = elemTy.isInteger(8)
                                            ? triton::musa::SQMMAEltType::s32
                                            : triton::musa::SQMMAEltType::f32;
  return !triton::musa::isSupportedSqmma(*eltTypeA, *eltTypeA, eltTypeC,
                                         instrShape[0], instrShape[1],
                                         instrShape[2]);
}

static SmallVector<unsigned, 2>
selectWarpsPerCTAForPH1(unsigned m, unsigned n, unsigned numWarps,
                        ArrayRef<unsigned> instrShape) {
  assert(instrShape.size() == 3 && "Unexpected instrShape rank");
  SmallVector<unsigned, 2> ret{1, 1};
  unsigned maxWarpsM = std::max(1u, m / instrShape[0]);
  while (ret[0] * ret[1] < numWarps) {
    bool growM =
        (m / instrShape[0] / ret[0]) >= (n / (instrShape[1] * 2) / ret[1]);
    if (growM) {
      if (ret[0] < maxWarpsM)
        ret[0] *= 2;
      else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  }
  return ret;
}

static bool shouldUseSqmmaCOperand(Type aElemTy, Type dElemTy, unsigned m,
                                   unsigned n, uint32_t maxNumImpreciseAcc,
                                   const SelectedConfig &config) {
  if (!tt::type::isFloat8(aElemTy) || !dElemTy.isF32() ||
      maxNumImpreciseAcc != 0)
    return true;

  unsigned instM = config.instrShape[0];
  unsigned instN = config.instrShape[1];
  unsigned squadsM = std::max(1u, config.warpsPerCTA[0] / 4);
  unsigned squadsN = std::max(1u, config.warpsPerCTA[1]);
  unsigned tileM = instM * squadsM;
  unsigned tileN = instN * squadsN;
  auto ceilDiv = [](unsigned x, unsigned y) { return (x + y - 1) / y; };
  unsigned numRepM = ceilDiv(m, tileM);
  unsigned numRepN = ceilDiv(n, tileN);
  bool keepSoftwareAccumFamily =
      numRepM == 1 && m <= 64 && n >= 32 && !(m >= 32 && n >= 256);
  return !keepSoftwareAccumFamily;
}

struct SqmmaAccumulationContract {
  bool useCOperand = true;
  triton::musa::SQMMAAccumulationMode mode =
      triton::musa::SQMMAAccumulationMode::hardware;
};

static SqmmaAccumulationContract selectSqmmaAccumulationContract(
    Type aElemTy, Type dElemTy, unsigned m, unsigned n, unsigned k,
    bool accIsZero, uint32_t maxNumImpreciseAcc, const SelectedConfig &config) {
  SqmmaAccumulationContract contract;
  contract.useCOperand =
      !accIsZero || shouldUseSqmmaCOperand(aElemTy, dElemTy, m, n,
                                           maxNumImpreciseAcc, config);
  if (!tt::type::isFloat8(aElemTy) || !dElemTy.isF32())
    return contract;

  unsigned instM = config.instrShape[0];
  unsigned instN = config.instrShape[1];
  unsigned instK = config.instrShape[2];
  unsigned squadsM = std::max(1u, config.warpsPerCTA[0] / 4);
  unsigned squadsN = std::max(1u, config.warpsPerCTA[1]);
  unsigned tileM = instM * squadsM;
  unsigned tileN = instN * squadsN;
  auto ceilDiv = [](unsigned x, unsigned y) { return (x + y - 1) / y; };
  unsigned numRepM = ceilDiv(m, tileM);
  unsigned numRepK = std::max(1u, ceilDiv(k, instK));

  bool softwareAccumulate =
      !accIsZero &&
      ((!contract.useCOperand) ||
       (contract.useCOperand && maxNumImpreciseAcc == 0 && numRepM == 1 &&
        numRepK == 1 && m <= 64 && n >= 32 && !(m >= 32 && n >= 256)));
  if (softwareAccumulate) {
    contract.mode = triton::musa::SQMMAAccumulationMode::software;
    return contract;
  }

  if (maxNumImpreciseAcc > 0 && maxNumImpreciseAcc <= k) {
    contract.mode = triton::musa::SQMMAAccumulationMode::partial;
    return contract;
  }

  return contract;
}

static std::optional<SelectedConfig>
selectWmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                 Type elemTy, bool allowTF32) {
  if (numWarps == 0 || (numWarps & (numWarps - 1)) != 0)
    return std::nullopt;

  auto candidates = getWmmaCandidateInstrShapes(elemTy, allowTF32);

  bool found = false;
  SmallVector<unsigned, 3> bestInstrShape = {0, 0, 0};
  unsigned bestInstCount = 0;

  for (unsigned tileM = 1; tileM <= numWarps; tileM *= 2) {
    if (numWarps % tileM)
      continue;
    unsigned tileN = numWarps / tileM;
    if (m % tileM != 0 || n % tileN != 0)
      continue;
    unsigned warpM = m / tileM;
    unsigned warpN = n / tileN;

    for (const auto &shape : candidates) {
      unsigned instM = shape[0];
      unsigned instN = shape[1];
      unsigned instK = shape[2];
      if (warpM % instM != 0 || warpN % instN != 0 || k % instK != 0)
        continue;
      unsigned instCount = (warpM / instM) * (warpN / instN) * (k / instK);
      if (!found || instCount < bestInstCount) {
        bestInstCount = instCount;
        bestInstrShape = shape;
        found = true;
      }
    }
  }

  if (!found)
    return std::nullopt;

  SelectedConfig best;
  best.instrShape = bestInstrShape;
  best.warpsPerCTA = selectWarpsPerCTAForPH1(m, n, numWarps, best.instrShape);
  return best;
}

static bool isSupportedSqmmaOperandType(Type elemTy, bool allowTF32) {
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(8) ||
      tt::type::isFloat8(elemTy))
    return true;
  return elemTy.isF32() && allowTF32;
}

static SmallVector<unsigned> getSqmmaCandidateM(Type elemTy, bool allowTF32) {
  if (elemTy.isF32() && allowTF32)
    return {128, 64, 32, 16};
  return {128, 64, 32, 16};
}

static SmallVector<unsigned> getSqmmaCandidateN(Type elemTy, bool allowTF32) {
  if (elemTy.isF32() && allowTF32)
    return {128, 64, 32, 16};
  return {128, 64, 32, 16};
}

static SmallVector<unsigned> getSqmmaCandidateK(Type elemTy, bool allowTF32) {
  if (elemTy.isF16() || elemTy.isBF16())
    return {128, 64, 32, 16};
  if (elemTy.isF32() && allowTF32)
    return {32, 16, 8};
  if (tt::type::isFloat8(elemTy) || elemTy.isInteger(8))
    return {128, 64, 32};
  return {};
}

static bool shouldAllowSqmmaTranspose(Type elemTy) {
  return elemTy.isF16() || elemTy.isBF16() || tt::type::isFloat8(elemTy);
}

enum class SqmmaTransLoadKind {
  None,       // No transpose exists on the operand load chain.
  PlainLoad,  // A LSU-fed load chain contains a tt.trans.
  Descriptor, // A descriptor-fed load chain contains a tt.trans.
};

static SqmmaTransLoadKind classifySqmmaTransLoad(Value v) {
  Value cur = v;
  while (true) {
    if (auto cvtOp = cur.getDefiningOp<ttg::ConvertLayoutOp>()) {
      cur = cvtOp.getSrc();
      continue;
    }
    if (auto bitcastOp = cur.getDefiningOp<tt::BitcastOp>()) {
      cur = bitcastOp.getSrc();
      continue;
    }
    auto transOp = cur.getDefiningOp<tt::TransOp>();
    if (!transOp)
      return SqmmaTransLoadKind::None;

    Value transSrc = transOp.getSrc();
    while (auto bitcastOp = transSrc.getDefiningOp<tt::BitcastOp>())
      transSrc = bitcastOp.getSrc();
    return transSrc.getDefiningOp<tt::DescriptorLoadOp>()
               ? SqmmaTransLoadKind::Descriptor
               : SqmmaTransLoadKind::PlainLoad;
  }
}

static Value promoteDotOperand(OpBuilder &builder, Location loc, Value operand,
                               Type promoteElemTy) {
  auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorTy)
    return operand;
  Type srcElemTy = tensorTy.getElementType();
  if (srcElemTy == promoteElemTy)
    return operand;

  auto dstTy = tensorTy.cloneWith(std::nullopt, promoteElemTy);
  if (tt::type::isFloat8(srcElemTy))
    return tt::FpToFpOp::create(builder, loc, dstTy, operand);

  if (isa<FloatType>(srcElemTy) && isa<FloatType>(promoteElemTy))
    return arith::ExtFOp::create(builder, loc, dstTy, operand);
  return operand;
}

static void promoteResidualFp8DotForFma(ModuleOp mod) {
  mod.walk([&](tt::DotOp dotOp) {
    auto aTy = dyn_cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(dotOp.getB().getType());
    auto dTy = dyn_cast<RankedTensorType>(dotOp.getType());
    if (!aTy || !bTy || !dTy)
      return;

    Type aElemTy = aTy.getElementType();
    Type bElemTy = bTy.getElementType();
    Type dElemTy = dTy.getElementType();
    if (!tt::type::isFloat8(aElemTy) && !tt::type::isFloat8(bElemTy))
      return;
    if (aElemTy == dElemTy && bElemTy == dElemTy)
      return;

    OpBuilder builder(dotOp);
    Location loc = dotOp.getLoc();
    Value newA = promoteDotOperand(builder, loc, dotOp.getA(), dElemTy);
    Value newB = promoteDotOperand(builder, loc, dotOp.getB(), dElemTy);
    dotOp.setOperand(0, newA);
    dotOp.setOperand(1, newB);
  });
}

static SmallVector<int64_t> getSqmmaPaddedAllocShape(RankedTensorType argType,
                                                     ArrayRef<unsigned> order) {
  auto shape = argType.getShape();
  SmallVector<int64_t> allocShape(shape.begin(), shape.end());
  if (allocShape.empty() || order.empty())
    return allocShape;

  unsigned leadingDim = order.front();
  if (leadingDim >= allocShape.size())
    return allocShape;

  int elemBitWidth = argType.getElementType().getIntOrFloatBitWidth();
  int64_t elemBytes = std::max<int64_t>(1, (elemBitWidth + 7) / 8);
  int64_t leadingBytes = allocShape[leadingDim] * elemBytes;
  if (leadingBytes <= 0)
    return allocShape;

  int64_t paddedLeadingBytes = leadingBytes;
  if (leadingBytes <= 256) {
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(leadingBytes)))
      paddedLeadingBytes = static_cast<int64_t>(
          llvm::PowerOf2Ceil(static_cast<uint64_t>(leadingBytes)));
  } else {
    paddedLeadingBytes = llvm::alignTo(leadingBytes, int64_t{256});
  }

  if (paddedLeadingBytes > leadingBytes &&
      (paddedLeadingBytes % elemBytes) == 0)
    allocShape[leadingDim] = paddedLeadingBytes / elemBytes;
  return allocShape;
}

static Value getSharedMemorySqmmaOperand(Value v, PatternRewriter &rewriter,
                                         int opIdx,
                                         ttg::MUSASqmmaEncodingAttr mmaEnc,
                                         bool allowTranspose) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  bool forceFreshRestage = false;
  while (true) {
    if (auto cvtOp = arg.getDefiningOp<ttg::ConvertLayoutOp>()) {
      auto srcTy = dyn_cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstTy = dyn_cast<RankedTensorType>(cvtOp.getType());
      if (srcTy && dstTy && isa<ttg::MmaEncodingTrait>(srcTy.getEncoding()) &&
          !isa<ttg::MmaEncodingTrait>(dstTy.getEncoding())) {
        forceFreshRestage = true;
        break;
      }
      arg = cvtOp.getSrc();
      continue;
    }
    if (auto bitcastOp = arg.getDefiningOp<tt::BitcastOp>()) {
      arg = bitcastOp.getSrc();
      continue;
    }
    if (auto transOp = arg.getDefiningOp<tt::TransOp>()) {
      if (allowTranspose ||
          transOp.getSrc().getDefiningOp<tt::DescriptorLoadOp>())
        break;
      arg = transOp.getSrc();
      continue;
    }
    break;
  }

  auto argType = dyn_cast<RankedTensorType>(arg.getType());
  if (!argType || !argType.getEncoding())
    return {};
  if (isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(
          argType.getEncoding()))
    return {};
  if (argType.getRank() != 2)
    return {};
  int elemBitWidth = argType.getElementType().getIntOrFloatBitWidth();
  int elemBytes = std::max(1, (elemBitWidth + 7) / 8);

  Value descSeed = arg;
  while (auto bitcastOp = descSeed.getDefiningOp<tt::BitcastOp>())
    descSeed = bitcastOp.getSrc();

  tt::DescriptorLoadOp descLoad;
  if (auto transOp = descSeed.getDefiningOp<tt::TransOp>())
    descLoad = transOp.getSrc().getDefiningOp<tt::DescriptorLoadOp>();
  else
    descLoad = descSeed.getDefiningOp<tt::DescriptorLoadOp>();

  SmallVector<unsigned> newOrder = ttg::getOrderForMemory(argType);
  if (!allowTranspose) {
    newOrder = SmallVector<unsigned>{1, 0};
  }
  bool isRowMajor =
      !newOrder.empty() && (newOrder.front() + 1 == argType.getRank());
  auto setSqmmaAttrs = [&](Operation *targetOp) {
    triton::musa::setSqmmaAttrs(targetOp, opIdx, elemBytes, isRowMajor);
  };
  auto propagateSqmmaAttrsFromLocalAlloc = [&](ttg::LocalAllocOp localAlloc) {
    SmallVector<Value> pending{localAlloc.getResult()};
    llvm::SmallPtrSet<Operation *, 16> visited;
    while (!pending.empty()) {
      Value cur = pending.pop_back_val();
      for (Operation *user : cur.getUsers()) {
        if (!visited.insert(user).second)
          continue;
        if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
          setSqmmaAttrs(indexOp.getOperation());
          pending.push_back(indexOp.getResult());
          continue;
        }
        if (auto subslice = dyn_cast<ttg::MemDescSubsliceOp>(user)) {
          setSqmmaAttrs(subslice.getOperation());
          pending.push_back(subslice.getResult());
          continue;
        }
        if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(user)) {
          pending.push_back(reinterpretOp.getResult());
          continue;
        }
        if (auto transOp = dyn_cast<ttg::MemDescTransOp>(user)) {
          pending.push_back(transOp.getResult());
          continue;
        }
      }
    }
  };
  auto propagateSqmmaAttrsToMemDescChain = [&](Value memDesc) {
    Value cur = memDesc;
    while (cur) {
      Operation *defOp = cur.getDefiningOp();
      if (!defOp)
        break;
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(defOp)) {
        setSqmmaAttrs(localAlloc.getOperation());
        propagateSqmmaAttrsFromLocalAlloc(localAlloc);
        break;
      }
      if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(defOp)) {
        setSqmmaAttrs(indexOp.getOperation());
        cur = indexOp.getSrc();
        continue;
      }
      if (auto subslice = dyn_cast<ttg::MemDescSubsliceOp>(defOp)) {
        setSqmmaAttrs(subslice.getOperation());
        cur = subslice.getSrc();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(defOp)) {
        cur = reinterpretOp.getSrc();
        continue;
      }
      if (auto transOp = dyn_cast<ttg::MemDescTransOp>(defOp)) {
        cur = transOp.getSrc();
        continue;
      }
      break;
    }
  };
  auto cgaLayout = ttg::getCGALayout(argType.getEncoding());
  auto sharedLayout = mmaEnc.composeSharedLayoutForOperand(
      cgaLayout, opIdx, argType.getShape(), newOrder,
      /*kWidth=*/0, argType.getElementType().getIntOrFloatBitWidth(),
      /*needTrans=*/false);
  auto allocShape = getSqmmaPaddedAllocShape(argType, newOrder);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(argType.getContext());
  auto memDescTy =
      ttg::MemDescType::get(argType.getShape(), argType.getElementType(),
                            sharedLayout, sharedMemorySpace,
                            /*mutableMemory=*/true, allocShape);

  if (!forceFreshRestage) {
    if (auto localLoad = arg.getDefiningOp<ttg::LocalLoadOp>()) {
      auto srcMemDescTy =
          dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
      auto samePhysicalLayout = [&](ttg::MemDescType srcTy) {
        return triton::musa::areMemDescTypesLayoutEquivalent(srcTy, memDescTy);
      };
      if (srcMemDescTy && samePhysicalLayout(srcMemDescTy)) {
        propagateSqmmaAttrsToMemDescChain(localLoad.getSrc());
        if (srcMemDescTy == memDescTy)
          return localLoad.getSrc();

        rewriter.setInsertionPointAfterValue(localLoad.getSrc());
        Value adapted = ttg::MemDescReinterpretOp::create(
            rewriter, localLoad.getLoc(), memDescTy, localLoad.getSrc());
        setSqmmaAttrs(adapted.getDefiningOp());
        return adapted;
      }
    }
  }
  if (descLoad) {
    setSqmmaAttrs(descLoad.getOperation());
  }

  Value reusedMemDesc =
      forceFreshRestage
          ? Value()
          : triton::musa::findReusableLocalAllocForSource(arg, memDescTy);
  if (reusedMemDesc)
    if (auto localAlloc = reusedMemDesc.getDefiningOp<ttg::LocalAllocOp>())
      setSqmmaAttrs(localAlloc.getOperation());

  if (reusedMemDesc)
    return reusedMemDesc;

  rewriter.setInsertionPointAfterValue(arg);
  auto localAlloc =
      ttg::LocalAllocOp::create(rewriter, arg.getLoc(), memDescTy, arg);
  setSqmmaAttrs(localAlloc.getOperation());
  return localAlloc.getResult();
}

static std::optional<SelectedConfig>
selectSqmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                  Type elemTy, bool allowTF32) {
  if (numWarps < 4 || (numWarps % 4) != 0)
    return std::nullopt;
  auto sqmmaEltType = toSqmmaOperandEltType(elemTy, allowTF32);
  if (!sqmmaEltType)
    return std::nullopt;

  auto candidateM = getSqmmaCandidateM(elemTy, allowTF32);
  auto candidateN = getSqmmaCandidateN(elemTy, allowTF32);
  auto candidateK = getSqmmaCandidateK(elemTy, allowTF32);
  if (candidateM.empty() || candidateN.empty() || candidateK.empty())
    return std::nullopt;

  bool found = false;
  SelectedConfig best;
  unsigned bestInstCount = std::numeric_limits<unsigned>::max();
  unsigned bestVolume = 0;
  unsigned bestRepM = std::numeric_limits<unsigned>::max();
  unsigned bestRepN = std::numeric_limits<unsigned>::max();

  for (unsigned instM : candidateM) {
    if (m < instM || (m % instM) != 0)
      continue;
    for (unsigned instN : candidateN) {
      if (n < instN || (n % instN) != 0)
        continue;
      if (!triton::musa::isSupportedSqmmaInstrMN(*sqmmaEltType, instM, instN))
        continue;
      for (unsigned instK : candidateK) {
        if (k < instK || (k % instK) != 0)
          continue;
        if ((instM % 4) != 0)
          continue;
        if (isKnownBrokenSqmmaConfig(elemTy, allowTF32, {instM, instN, instK}))
          continue;

        for (unsigned warpsM = 4; warpsM <= numWarps; warpsM *= 2) {
          if (numWarps % warpsM != 0)
            continue;
          unsigned warpsN = numWarps / warpsM;

          unsigned squadsM = warpsM / 4;
          unsigned tileM = instM * squadsM;
          unsigned tileN = instN * warpsN;
          if ((m % tileM) != 0 || (n % tileN) != 0)
            continue;

          unsigned instCount = (m / tileM) * (n / tileN) * (k / instK);
          unsigned repM = m / tileM;
          unsigned repN = n / tileN;
          unsigned volume = instM * instN * instK;
          if (!found || instCount < bestInstCount ||
              (instCount == bestInstCount &&
               (volume > bestVolume ||
                (volume == bestVolume &&
                 (repM < bestRepM ||
                  (repM == bestRepM && repN < bestRepN)))))) {
            found = true;
            bestInstCount = instCount;
            bestVolume = volume;
            bestRepM = repM;
            bestRepN = repN;
            best.instrShape = {instM, instN, instK};
            best.warpsPerCTA = {warpsM, warpsN};
          }
        }
      }
    }
  }

  if (!found)
    return std::nullopt;
  return best;
}

class BlockedToMUSAWmma : public RewritePattern {
public:
  explicit BlockedToMUSAWmma(MLIRContext *context, int computeCapability)
      : RewritePattern(tt::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (computeCapability != 31)
      return failure();

    auto dotOp = dyn_cast<tt::DotOp>(op);
    if (!dotOp)
      return failure();
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());
    auto oldEncoding = oldRetType.getEncoding();
    if (!oldEncoding || !isa<ttg::BlockedEncodingAttr>(oldEncoding))
      return failure();
    if (isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(oldEncoding))
      return failure();
    auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = cast<RankedTensorType>(dotOp.getB().getType());
    auto aElemTy = aTy.getElementType();
    auto bElemTy = bTy.getElementType();
    bool allowTF32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;
    if (aElemTy != bElemTy)
      return failure();
    if (!isSupportedWmmaOperandType(aElemTy, allowTF32))
      return failure();

    auto shapePerCTA = ttg::getShapePerCTA(oldRetType);
    if (shapePerCTA.size() != 2)
      return failure();
    if (shapePerCTA[0] <= 0 || shapePerCTA[1] <= 0)
      return failure();

    if (aTy.getRank() < 2)
      return failure();
    auto kDim = aTy.getShape().back();
    if (kDim <= 0)
      return failure();

    unsigned m = static_cast<unsigned>(shapePerCTA[0]);
    unsigned n = static_cast<unsigned>(shapePerCTA[1]);
    unsigned k = static_cast<unsigned>(kDim);
    unsigned numWarps = ttg::lookupNumWarps(dotOp);

    auto config = selectWmmaConfig(m, n, k, numWarps, aElemTy, allowTF32);
    if (!config)
      return failure();

    auto cgaLayout = ttg::getCGALayout(oldEncoding);
    auto mmaEnc = ttg::MUSAWmmaEncodingAttr::get(
        oldRetType.getContext(), /*versionMajor=*/3, /*versionMinor=*/1,
        config->warpsPerCTA, cgaLayout, config->instrShape);
    bool useFp32Carrier = computeCapability == 31 &&
                          oldRetType.getElementType().isF16() &&
                          aElemTy.isF16() && bElemTy.isF16();
    Type carrierElemTy =
        useFp32Carrier ? rewriter.getF32Type() : oldRetType.getElementType();
    auto newRetType =
        RankedTensorType::get(oldRetType.getShape(), carrierElemTy, mmaEnc);

    auto oldAcc = dotOp.getOperand(2);
    Value acc = useFp32Carrier ? promoteDotOperand(rewriter, dotOp.getLoc(),
                                                   oldAcc, carrierElemTy)
                               : oldAcc;
    bool accIsZero = isZeroConst(oldAcc);
    Value newAcc;
    if (accIsZero) {
      auto zeroElem = rewriter.getZeroAttr(newRetType.getElementType());
      auto zeroTensor = DenseElementsAttr::get(newRetType, zeroElem);
      newAcc = arith::ConstantOp::create(rewriter, oldAcc.getLoc(), newRetType,
                                         zeroTensor);
    } else {
      newAcc = ttg::ConvertLayoutOp::create(rewriter, oldAcc.getLoc(),
                                            newRetType, acc);
    }

    auto newAEncoding = ttg::DotOperandEncodingAttr::get(
        aTy.getContext(), 0, newRetType.getEncoding(), aElemTy);
    auto newAType =
        RankedTensorType::get(aTy.getShape(), aElemTy, newAEncoding);
    auto newA = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newAType,
                                             dotOp.getA());

    auto newBEncoding = ttg::DotOperandEncodingAttr::get(
        bTy.getContext(), 1, newRetType.getEncoding(), bElemTy);
    auto newBType =
        RankedTensorType::get(bTy.getShape(), bElemTy, newBEncoding);
    auto newB = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newBType,
                                             dotOp.getB());

    auto wmmaEltType = toSqmmaOperandEltType(aElemTy, allowTF32);
    if (!wmmaEltType)
      return failure();
    Value useC = arith::ConstantIntOp::create(rewriter, dotOp.getLoc(), 1, 1);
    auto newDot = triton::musa::WmmaDotOp::create(
        rewriter, dotOp.getLoc(), newRetType, newA, newB, newAcc, useC,
        static_cast<int32_t>(config->instrShape[0]),
        static_cast<int32_t>(config->instrShape[1]),
        static_cast<int32_t>(config->instrShape[2]), *wmmaEltType, *wmmaEltType,
        triton::musa::inferWmmaFragmentLayout(dotOp.getA(), 0),
        triton::musa::inferWmmaFragmentLayout(dotOp.getB(), 1),
        static_cast<int32_t>(dotOp.getInputPrecision()),
        /*maxNumImpreciseAcc=*/0);
    newDot->setAttr(kDisableGenericDotPipelineAttr, rewriter.getBoolAttr(true));
    if (!useFp32Carrier) {
      rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                        newDot.getResult());
      return success();
    }

    auto blockedCarrierTy = oldRetType.cloneWith(std::nullopt, carrierElemTy);
    Value blockedCarrier = ttg::ConvertLayoutOp::create(
        rewriter, dotOp.getLoc(), blockedCarrierTy, newDot.getResult());
    Value truncated = arith::TruncFOp::create(rewriter, dotOp.getLoc(),
                                              oldRetType, blockedCarrier);
    rewriter.replaceOp(dotOp, truncated);
    return success();
  }

private:
  int computeCapability;
};

class BlockedToMUSASqmma : public RewritePattern {
public:
  explicit BlockedToMUSASqmma(MLIRContext *context, int computeCapability)
      : RewritePattern(tt::DotOp::getOperationName(), 3, context),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (computeCapability != 31)
      return failure();

    auto dotOp = dyn_cast<tt::DotOp>(op);
    if (!dotOp)
      return failure();
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());
    auto oldEncoding = oldRetType.getEncoding();
    if (!oldEncoding || !isa<ttg::BlockedEncodingAttr>(oldEncoding))
      return failure();
    if (isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(oldEncoding))
      return failure();
    if (oldRetType.getRank() != 2)
      return failure();
    auto aTy = dyn_cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(dotOp.getB().getType());
    if (!aTy || !bTy)
      return failure();
    if (aTy.getRank() != 2 || bTy.getRank() != 2)
      return failure();

    auto aElemTy = aTy.getElementType();
    auto bElemTy = bTy.getElementType();
    if (aElemTy != bElemTy)
      return failure();

    bool allowTF32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;
    if (aElemTy.isF32()) {
      if (!allowTF32)
        return failure();
    }
    if (!isSupportedSqmmaOperandType(aElemTy, allowTF32))
      return failure();

    auto shapePerCTA = ttg::getShapePerCTA(oldRetType);
    if (shapePerCTA.size() != 2)
      return failure();
    if (shapePerCTA[0] <= 0 || shapePerCTA[1] <= 0)
      return failure();

    auto kDim = aTy.getShape().back();
    if (kDim <= 0)
      return failure();

    unsigned m = static_cast<unsigned>(shapePerCTA[0]);
    unsigned n = static_cast<unsigned>(shapePerCTA[1]);
    unsigned k = static_cast<unsigned>(kDim);
    unsigned numWarps = ttg::lookupNumWarps(dotOp);
    auto config = selectSqmmaConfig(m, n, k, numWarps, aElemTy, allowTF32);
    if (!config)
      return failure();

    bool useFp32Carrier = computeCapability == 31 &&
                          oldRetType.getElementType().isF16() &&
                          aElemTy.isF16() && bElemTy.isF16();
    Type carrierElemTy =
        useFp32Carrier ? rewriter.getF32Type() : oldRetType.getElementType();
    auto eltTypeC = toSqmmaEltType(carrierElemTy);
    auto eltTypeA = toSqmmaOperandEltType(aElemTy, allowTF32);
    auto eltTypeB = toSqmmaOperandEltType(bElemTy, allowTF32);
    if (!eltTypeC || !eltTypeA || !eltTypeB)
      return failure();
    if (!triton::musa::isSupportedSqmma(
            *eltTypeA, *eltTypeB, *eltTypeC, config->instrShape[0],
            config->instrShape[1], config->instrShape[2]))
      return failure();

    auto cgaLayout = ttg::getCGALayout(oldEncoding);
    auto mmaEnc = ttg::MUSASqmmaEncodingAttr::get(
        oldRetType.getContext(), /*versionMajor=*/3, /*versionMinor=*/1,
        config->warpsPerCTA, cgaLayout, config->instrShape);
    auto newRetType =
        RankedTensorType::get(oldRetType.getShape(), carrierElemTy, mmaEnc);

    auto oldAcc = dotOp.getOperand(2);
    Value acc = useFp32Carrier ? promoteDotOperand(rewriter, dotOp.getLoc(),
                                                   oldAcc, carrierElemTy)
                               : oldAcc;
    bool accIsZero = isZeroConst(oldAcc);
    Value newAcc;
    if (accIsZero) {
      auto zeroElem = rewriter.getZeroAttr(newRetType.getElementType());
      auto zeroTensor = DenseElementsAttr::get(newRetType, zeroElem);
      newAcc = arith::ConstantOp::create(rewriter, oldAcc.getLoc(), newRetType,
                                         zeroTensor);
    } else {
      newAcc = ttg::ConvertLayoutOp::create(rewriter, oldAcc.getLoc(),
                                            newRetType, acc);
    }

    SqmmaTransLoadKind transLoadKindA = classifySqmmaTransLoad(dotOp.getA());
    SqmmaTransLoadKind transLoadKindB = classifySqmmaTransLoad(dotOp.getB());
    bool allowTransposeA = transLoadKindA == SqmmaTransLoadKind::Descriptor ||
                           (transLoadKindA == SqmmaTransLoadKind::PlainLoad &&
                            shouldAllowSqmmaTranspose(aElemTy));
    bool allowTransposeB = transLoadKindB == SqmmaTransLoadKind::Descriptor ||
                           (transLoadKindB == SqmmaTransLoadKind::PlainLoad &&
                            shouldAllowSqmmaTranspose(bElemTy));
    Value newA = getSharedMemorySqmmaOperand(dotOp.getA(), rewriter, 0, mmaEnc,
                                             allowTransposeA);
    Value newB = getSharedMemorySqmmaOperand(dotOp.getB(), rewriter, 1, mmaEnc,
                                             allowTransposeB);
    if (!newA || !newB)
      return failure();

    auto accumulationContract = selectSqmmaAccumulationContract(
        aElemTy, newRetType.getElementType(), m, n, k, accIsZero,
        static_cast<uint32_t>(dotOp.getMaxNumImpreciseAcc()), *config);
    Value useC = arith::ConstantIntOp::create(
        rewriter, dotOp.getLoc(), accumulationContract.useCOperand, 1);
    auto newDot = triton::musa::SquadDotOp::create(
        rewriter, dotOp.getLoc(), newRetType, newA, newB, newAcc, useC,
        static_cast<int32_t>(config->instrShape[0]),
        static_cast<int32_t>(config->instrShape[1]),
        static_cast<int32_t>(config->instrShape[2]), *eltTypeC, *eltTypeA,
        *eltTypeB, inferSqmmaLayout(newA), inferSqmmaLayout(newB), false,
        accumulationContract.mode,
        static_cast<int32_t>(dotOp.getInputPrecision()),
        accumulationContract.mode ==
                triton::musa::SQMMAAccumulationMode::partial
            ? static_cast<int32_t>(dotOp.getMaxNumImpreciseAcc())
            : 0);
    newDot->setAttr(kDisableGenericDotPipelineAttr, rewriter.getBoolAttr(true));
    newDot->setAttr("isAsync", rewriter.getBoolAttr(false));
    if (!useFp32Carrier) {
      rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                        newDot.getResult());
      return success();
    }

    auto blockedCarrierTy = oldRetType.cloneWith(std::nullopt, carrierElemTy);
    Value blockedCarrier = ttg::ConvertLayoutOp::create(
        rewriter, dotOp.getLoc(), blockedCarrierTy, newDot.getResult());
    Value truncated = arith::TruncFOp::create(rewriter, dotOp.getLoc(),
                                              oldRetType, blockedCarrier);
    rewriter.replaceOp(dotOp, truncated);
    return success();
  }

private:
  int computeCapability;
};

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUACCELERATEMATMUL
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUAccelerateMatmulPass
    : impl::TritonMUSAGPUAccelerateMatmulBase<
          TritonMUSAGPUAccelerateMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int computeCapability = getMusaComputeCapability(mod);
    if (computeCapability < 0)
      return;

    bool disableSqmma = ::triton::tools::getBoolEnv("DISABLE_SQMMA");
    bool disableWmma = ::triton::tools::getBoolEnv("DISABLE_WMMA");

    bool sqmmaCandidate = computeCapability >= 31 && !disableSqmma;
    // Preserve the 3.6 fallback behavior: descriptor/TME modules may still
    // fall back to WMMA when SQMMA predicate matching rejects a dot.
    bool wmmaCandidate = computeCapability == 31 && !disableWmma;

    MLIRContext *context = &getContext();
    if (sqmmaCandidate || wmmaCandidate) {
      RewritePatternSet patterns(context);
      // Keep 3.2-aligned rewrite precedence: SQMMA first, then WMMA.
      if (sqmmaCandidate)
        patterns.add<BlockedToMUSASqmma>(context, computeCapability);
      if (wmmaCandidate)
        patterns.add<BlockedToMUSAWmma>(context, computeCapability);

      if (applyPatternsGreedily(mod, std::move(patterns)).failed())
        signalPassFailure();
    }

    promoteResidualFp8DotForFma(mod);
  }
};

} // namespace mlir
