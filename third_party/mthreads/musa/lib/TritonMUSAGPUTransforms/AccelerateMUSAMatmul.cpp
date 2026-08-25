#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#ifdef __TLE__
#include "TritonMUSACommon/MMAEncodingUtils.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#endif // __TLE__
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
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
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

#ifdef __TLE__
static bool isSupportedSqmmaOperandType(Type elemTy, bool allowTF32);

enum class MusaMmaValidationFailure {
  None,
  InvalidDotShape,
  MismatchedOperandType,
  UnsupportedOperandType,
  TF32PrecisionRequired,
  UnsupportedEncodingVersion,
  InvalidInstructionShape,
  UnsupportedInstruction,
  InvalidWarpLayout,
  InvalidSquadLayout,
  UnsupportedAccumulatorType,
  UnsupportedTransposeOperand,
  UnsupportedTarget,
  MismatchedExplicitResultEncoding,
  InvalidDotOperandEncoding,
  InvalidDotOperandIndex,
  InvalidDotOperandKWidth,
  MismatchedDotOperandParent,
  MismatchedAccumulatorEncoding,
  ExplicitWmmaDisabled,
  ExplicitSqmmaDisabled,
};

struct MusaMmaValidationResult {
  MusaMmaValidationFailure failure = MusaMmaValidationFailure::None;

  bool succeeded() const { return failure == MusaMmaValidationFailure::None; }
};

[[maybe_unused]] static StringRef
getMusaMmaValidationMessage(MusaMmaValidationFailure failure) {
  switch (failure) {
  case MusaMmaValidationFailure::None:
    return "";
  case MusaMmaValidationFailure::InvalidDotShape:
    return "dot operands and result must form a rank-2 or rank-3 MxK, KxN, "
           "MxN contract";
  case MusaMmaValidationFailure::MismatchedOperandType:
    return "MUSA MMA operands A and B must use the same element type";
  case MusaMmaValidationFailure::UnsupportedOperandType:
    return "MUSA MMA operand element type is unsupported";
  case MusaMmaValidationFailure::TF32PrecisionRequired:
    return "MUSA MMA f32 operands require TF32 input precision";
  case MusaMmaValidationFailure::UnsupportedEncodingVersion:
    return "MUSA MMA encoding version is unsupported by the mthreads backend";
  case MusaMmaValidationFailure::InvalidInstructionShape:
    return "MUSA MMA instruction shape must contain logical M, N and K and "
           "divide the dot shape";
  case MusaMmaValidationFailure::UnsupportedInstruction:
    return "MUSA MMA instruction shape and element types are unsupported";
  case MusaMmaValidationFailure::InvalidWarpLayout:
    return "MUSA MMA warp layout must match the module warp count";
  case MusaMmaValidationFailure::InvalidSquadLayout:
    return "MUSA SQMMA warp layout must form complete four-warp squad tiles";
  case MusaMmaValidationFailure::UnsupportedAccumulatorType:
    return "MUSA SQMMA accumulator element type is unsupported";
  case MusaMmaValidationFailure::UnsupportedTransposeOperand:
    return "MUSA SQMMA operand cannot be materialized in shared memory";
  case MusaMmaValidationFailure::UnsupportedTarget:
    return "explicit MUSA MMA requires the PH1 compute capability";
  case MusaMmaValidationFailure::MismatchedExplicitResultEncoding:
    return "the explicit result encoding does not match the MUSA MMA result "
           "type";
  case MusaMmaValidationFailure::InvalidDotOperandEncoding:
    return "explicit MUSA MMA operands A and B must use "
           "DotOperandEncodingAttr";
  case MusaMmaValidationFailure::InvalidDotOperandIndex:
    return "explicit MUSA MMA operands A and B must use dot operand indices 0 "
           "and 1";
  case MusaMmaValidationFailure::InvalidDotOperandKWidth:
    return "explicit MUSA MMA operands do not support a non-zero kWidth";
  case MusaMmaValidationFailure::MismatchedDotOperandParent:
    return "explicit MUSA MMA operand layouts must use the result encoding as "
           "their parent";
  case MusaMmaValidationFailure::MismatchedAccumulatorEncoding:
    return "explicit MUSA MMA accumulator and result must use the same "
           "encoding";
  case MusaMmaValidationFailure::ExplicitWmmaDisabled:
    return "DISABLE_WMMA conflicts with an explicit MUSA WMMA layout";
  case MusaMmaValidationFailure::ExplicitSqmmaDisabled:
    return "DISABLE_SQMMA conflicts with an explicit MUSA SQMMA layout";
  }
  llvm_unreachable("unknown MUSA MMA validation failure");
}

static bool isValidMusaMmaWarpLayout(ArrayRef<unsigned> warpsPerCTA,
                                     unsigned numWarps) {
  if (warpsPerCTA.size() != 2 && warpsPerCTA.size() != 3)
    return false;
  if (warpsPerCTA.size() == 3 && warpsPerCTA.back() != 1)
    return false;

  unsigned warpProduct = 1;
  for (unsigned warps : warpsPerCTA) {
    if (!llvm::isPowerOf2_32(warps))
      return false;
    warpProduct *= warps;
  }
  return warpProduct == numWarps;
}
#endif // __TLE__

struct DotMatrixShape {
  unsigned rank;
  unsigned batch;
  unsigned m;
  unsigned n;
  unsigned k;
};

static FailureOr<DotMatrixShape> getDotMatrixShape(tt::DotOp dotOp) {
  auto retTy = dyn_cast<RankedTensorType>(dotOp.getType());
  auto aTy = dyn_cast<RankedTensorType>(dotOp.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dotOp.getB().getType());
  if (!retTy || !aTy || !bTy)
    return failure();

  unsigned rank = retTy.getRank();
  if (rank != 2 && rank != 3)
    return failure();
  if (aTy.getRank() != rank || bTy.getRank() != rank)
    return failure();

  auto shapePerCTA = ttg::getShapePerCTA(retTy);
  if (shapePerCTA.size() != rank)
    return failure();

  int64_t m = shapePerCTA[rank - 2];
  int64_t n = shapePerCTA[rank - 1];
  int64_t k = aTy.getShape().back();
  if (m <= 0 || n <= 0 || k <= 0)
    return failure();
  int64_t batch = rank == 3 ? shapePerCTA[0] : 1;
  if (batch <= 0)
    return failure();

  return DotMatrixShape{rank, static_cast<unsigned>(batch),
                        static_cast<unsigned>(m), static_cast<unsigned>(n),
                        static_cast<unsigned>(k)};
}

#ifdef __TLE__
enum class SqmmaTransLoadKind {
  None,       // No transpose exists on the operand load chain.
  PlainLoad,  // A LSU-fed load chain contains a tt.trans.
  Descriptor, // A descriptor-fed load chain contains a tt.trans.
};

static SqmmaTransLoadKind classifySqmmaTransLoad(Value value);

struct MusaDotProblem {
  Value a;
  Value b;
  RankedTensorType aType;
  RankedTensorType bType;
  RankedTensorType cType;
  RankedTensorType dType;
  DotMatrixShape matrixShape;
  Type aElemType;
  Type bElemType;
  Type dElemType;
  bool allowTF32;
  unsigned numWarps;
  SqmmaTransLoadKind transLoadKindA;
  SqmmaTransLoadKind transLoadKindB;
};

static FailureOr<MusaDotProblem> getMusaDotProblem(tt::DotOp dotOp) {
  auto aType = dyn_cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(dotOp.getB().getType());
  auto cType = dyn_cast<RankedTensorType>(dotOp.getC().getType());
  auto dType = dyn_cast<RankedTensorType>(dotOp.getType());
  auto matrixShape = getDotMatrixShape(dotOp);
  if (!aType || !bType || !cType || !dType || failed(matrixShape))
    return failure();
  unsigned rank = matrixShape->rank;
  if (cType.getRank() != rank || dType.getRank() != rank ||
      aType.getShape()[rank - 1] != bType.getShape()[rank - 2] ||
      aType.getShape()[rank - 2] != cType.getShape()[rank - 2] ||
      bType.getShape()[rank - 1] != cType.getShape()[rank - 1] ||
      cType.getShape() != dType.getShape())
    return failure();
  if (rank == 3 && (aType.getShape()[0] != bType.getShape()[0] ||
                    aType.getShape()[0] != cType.getShape()[0]))
    return failure();

  return MusaDotProblem{
      dotOp.getA(),
      dotOp.getB(),
      aType,
      bType,
      cType,
      dType,
      *matrixShape,
      aType.getElementType(),
      bType.getElementType(),
      dType.getElementType(),
      dotOp.getInputPrecision() == tt::InputPrecision::TF32,
      static_cast<unsigned>(ttg::lookupNumWarps(dotOp)),
      classifySqmmaTransLoad(dotOp.getA()),
      classifySqmmaTransLoad(dotOp.getB()),
  };
}

static MusaMmaValidationResult
validateMusaDotProblem(const MusaDotProblem &problem, bool useSqmma) {
  if (problem.aElemType != problem.bElemType)
    return {MusaMmaValidationFailure::MismatchedOperandType};
  if (problem.aElemType.isF32() && !problem.allowTF32)
    return {MusaMmaValidationFailure::TF32PrecisionRequired};
  bool supported =
      useSqmma
          ? isSupportedSqmmaOperandType(problem.aElemType, problem.allowTF32)
          : isSupportedWmmaOperandType(problem.aElemType, problem.allowTF32);
  if (!supported)
    return {MusaMmaValidationFailure::UnsupportedOperandType};
  return {};
}

static MusaMmaValidationResult
validateWmmaConfig(const MusaDotProblem &problem, const SelectedConfig &config,
                   ttg::MUSAWmmaEncodingAttr encoding = {}) {
  if (encoding && !triton::musa::supportsMusaWmmaEncoding(encoding))
    return {MusaMmaValidationFailure::UnsupportedEncodingVersion};
  if (!isValidMusaMmaWarpLayout(config.warpsPerCTA, problem.numWarps))
    return {MusaMmaValidationFailure::InvalidWarpLayout};
  if (config.instrShape.size() != 3)
    return {MusaMmaValidationFailure::InvalidInstructionShape};
  unsigned m = problem.matrixShape.m;
  unsigned n = problem.matrixShape.n;
  unsigned k = problem.matrixShape.k;
  if (config.instrShape[0] == 0 || config.instrShape[1] == 0 ||
      config.instrShape[2] == 0 || m % config.instrShape[0] != 0 ||
      n % config.instrShape[1] != 0 || k % config.instrShape[2] != 0)
    return {MusaMmaValidationFailure::InvalidInstructionShape};
  if (!triton::musa::lookupWmmaIntrinsic(problem.aElemType, config.instrShape))
    return {MusaMmaValidationFailure::UnsupportedInstruction};
  return {};
}

static SelectedConfig
getExplicitWmmaConfig(ttg::MUSAWmmaEncodingAttr encoding) {
  SelectedConfig config;
  llvm::append_range(config.instrShape, encoding.getInstrShape());
  llvm::append_range(config.warpsPerCTA, encoding.getWarpsPerCTA());
  return config;
}

static MusaMmaValidationResult
validateExplicitWmmaContract(tt::DotOp dotOp, const MusaDotProblem &problem,
                             ttg::MUSAWmmaEncodingAttr encoding) {
  Attribute explicitEncoding =
      getTleExplicitResultEncoding(dotOp.getOperation(), 0);
  if (explicitEncoding && explicitEncoding != encoding)
    return {MusaMmaValidationFailure::MismatchedExplicitResultEncoding};

  auto aEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      problem.aType.getEncoding());
  auto bEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      problem.bType.getEncoding());
  if (!aEncoding || !bEncoding)
    return {MusaMmaValidationFailure::InvalidDotOperandEncoding};
  if (aEncoding.getOpIdx() != 0 || bEncoding.getOpIdx() != 1)
    return {MusaMmaValidationFailure::InvalidDotOperandIndex};
  if (aEncoding.getKWidth() != 0 || bEncoding.getKWidth() != 0)
    return {MusaMmaValidationFailure::InvalidDotOperandKWidth};
  if (aEncoding.getParent() != encoding || bEncoding.getParent() != encoding)
    return {MusaMmaValidationFailure::MismatchedDotOperandParent};
  if (problem.cType.getEncoding() != encoding ||
      problem.dType.getEncoding() != encoding)
    return {MusaMmaValidationFailure::MismatchedAccumulatorEncoding};

  MusaMmaValidationResult problemValidation =
      validateMusaDotProblem(problem, false);
  if (!problemValidation.succeeded())
    return problemValidation;
  return validateWmmaConfig(problem, getExplicitWmmaConfig(encoding), encoding);
}

static LogicalResult emitExplicitWmmaError(tt::DotOp dotOp,
                                           MusaMmaValidationFailure failure) {
  return dotOp.emitOpError("cannot lower explicit MUSA WMMA dot: ")
         << getMusaMmaValidationMessage(failure);
}

static LogicalResult validateExplicitWmmaDots(ModuleOp module,
                                              int computeCapability,
                                              bool disableWmma) {
  WalkResult result = module.walk([&](tt::DotOp dotOp) -> WalkResult {
    auto resultType = dyn_cast<RankedTensorType>(dotOp.getType());
    auto encoding = resultType ? dyn_cast_or_null<ttg::MUSAWmmaEncodingAttr>(
                                     resultType.getEncoding())
                               : ttg::MUSAWmmaEncodingAttr{};
    if (!encoding)
      return WalkResult::advance();
    if (computeCapability != 31) {
      emitExplicitWmmaError(dotOp, MusaMmaValidationFailure::UnsupportedTarget);
      return WalkResult::interrupt();
    }
    if (disableWmma) {
      emitExplicitWmmaError(dotOp,
                            MusaMmaValidationFailure::ExplicitWmmaDisabled);
      return WalkResult::interrupt();
    }
    auto problem = getMusaDotProblem(dotOp);
    if (failed(problem)) {
      emitExplicitWmmaError(dotOp, MusaMmaValidationFailure::InvalidDotShape);
      return WalkResult::interrupt();
    }
    MusaMmaValidationResult validation =
        validateExplicitWmmaContract(dotOp, *problem, encoding);
    if (!validation.succeeded()) {
      emitExplicitWmmaError(dotOp, validation.failure);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}
#endif // __TLE__

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
selectWmmaWarpsPerCTAForPH1(unsigned m, unsigned n, unsigned numWarps,
                            ArrayRef<unsigned> instrShape) {
  assert(instrShape.size() == 3 && "Unexpected instrShape rank");
  SmallVector<unsigned, 2> ret{1, 1};
  while (ret[0] * ret[1] < numWarps) {
    bool growM =
        (m / instrShape[0] / ret[0]) >= (n / (instrShape[1] * 2) / ret[1]);
    if (growM)
      ret[0] *= 2;
    else
      ret[1] *= 2;
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
#ifdef __TLE__
selectWmmaConfig(const MusaDotProblem &problem) {
  unsigned m = problem.matrixShape.m;
  unsigned n = problem.matrixShape.n;
  unsigned k = problem.matrixShape.k;
  unsigned numWarps = problem.numWarps;
  Type elemTy = problem.aElemType;
  bool allowTF32 = problem.allowTF32;
#else
selectWmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                 Type elemTy, bool allowTF32) {
#endif // __TLE__
  if (numWarps == 0 || (numWarps & (numWarps - 1)) != 0)
    return std::nullopt;

  auto candidates = getWmmaCandidateInstrShapes(elemTy, allowTF32);

  bool found = false;
  SmallVector<unsigned, 3> bestInstrShape = {0, 0, 0};
  unsigned bestInstCount = 0;

  for (const auto &shape : candidates) {
#ifdef __TLE__
    SelectedConfig candidate;
    candidate.instrShape = shape;
    candidate.warpsPerCTA = selectWmmaWarpsPerCTAForPH1(m, n, numWarps, shape);
    if (!validateWmmaConfig(problem, candidate).succeeded())
      continue;
#else
    if (!triton::musa::lookupWmmaIntrinsic(elemTy, shape))
      continue;
    if (m % shape[0] != 0 || n % shape[1] != 0 || k % shape[2] != 0)
      continue;
#endif // __TLE__
    unsigned instM = shape[0];
    unsigned instN = shape[1];
    unsigned instK = shape[2];
    unsigned instCount = (m / instM) * (n / instN) * (k / instK);
    if (!found || instCount < bestInstCount) {
      bestInstCount = instCount;
      bestInstrShape = shape;
      found = true;
    }
  }

  if (!found)
    return std::nullopt;

  SelectedConfig best;
  best.instrShape = bestInstrShape;
  best.warpsPerCTA =
      selectWmmaWarpsPerCTAForPH1(m, n, numWarps, best.instrShape);
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

#ifndef __TLE__
enum class SqmmaTransLoadKind {
  None,       // No transpose exists on the operand load chain.
  PlainLoad,  // A LSU-fed load chain contains a tt.trans.
  Descriptor, // A descriptor-fed load chain contains a tt.trans.
};
#endif // __TLE__

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

#ifdef __TLE__
static MusaMmaValidationResult
validateSqmmaOperandMaterialization(Value operand) {
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType)
    return {MusaMmaValidationFailure::UnsupportedTransposeOperand};
  bool needTranspose =
      classifySqmmaTransLoad(operand) != SqmmaTransLoadKind::None;
  if (auto dotEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
          operandType.getEncoding());
      dotEncoding && isa<ttg::MUSASqmmaEncodingAttr>(dotEncoding.getParent())) {
    auto sharedLayout = triton::musa::composeMusaOperandSharedLayout(
        dotEncoding, operandType.getShape(),
        ttg::getOrderForMemory(operandType),
        ttg::getCGALayout(dotEncoding.getParent()),
        operandType.getElementType(), needTranspose);
    if (!sharedLayout)
      return {MusaMmaValidationFailure::UnsupportedTransposeOperand};
  }

  Value current = operand;
  while (true) {
    if (auto convert = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = convert.getSrc();
      continue;
    }
    if (auto bitcast = current.getDefiningOp<tt::BitcastOp>()) {
      current = bitcast.getSrc();
      continue;
    }
    break;
  }

  if (auto trans = current.getDefiningOp<tt::TransOp>())
    current = trans.getSrc();
  while (auto bitcast = current.getDefiningOp<tt::BitcastOp>())
    current = bitcast.getSrc();

  auto tensorType = dyn_cast<RankedTensorType>(current.getType());
  if (!tensorType || !tensorType.getEncoding() ||
      (tensorType.getRank() != 2 && tensorType.getRank() != 3))
    return {MusaMmaValidationFailure::UnsupportedTransposeOperand};
  return {};
}

static MusaMmaValidationResult
validateSqmmaConfig(const MusaDotProblem &problem, const SelectedConfig &config,
                    ttg::MUSASqmmaEncodingAttr encoding = {}) {
  if (encoding && !triton::musa::supportsMusaSqmmaEncoding(encoding))
    return {MusaMmaValidationFailure::UnsupportedEncodingVersion};
  if (!isValidMusaMmaWarpLayout(config.warpsPerCTA, problem.numWarps))
    return {MusaMmaValidationFailure::InvalidWarpLayout};
  if (config.instrShape.size() != 3)
    return {MusaMmaValidationFailure::InvalidInstructionShape};

  unsigned m = problem.matrixShape.m;
  unsigned n = problem.matrixShape.n;
  unsigned k = problem.matrixShape.k;
  unsigned instM = config.instrShape[0];
  unsigned instN = config.instrShape[1];
  unsigned instK = config.instrShape[2];
  if (instM == 0 || instN == 0 || instK == 0 || m % instM != 0 ||
      n % instN != 0 || k % instK != 0)
    return {MusaMmaValidationFailure::InvalidInstructionShape};

  unsigned warpsM = config.warpsPerCTA[0];
  unsigned warpsN = config.warpsPerCTA[1];
  if (warpsM < 4 || warpsM % 4 != 0 || m % (instM * (warpsM / 4)) != 0 ||
      n % (instN * warpsN) != 0)
    return {MusaMmaValidationFailure::InvalidSquadLayout};

  auto eltTypeA = toSqmmaOperandEltType(problem.aElemType, problem.allowTF32);
  auto eltTypeB = toSqmmaOperandEltType(problem.bElemType, problem.allowTF32);
  bool useFp32Carrier = problem.dElemType.isF16() &&
                        problem.aElemType.isF16() && problem.bElemType.isF16();
  Type carrierElemType = useFp32Carrier
                             ? Float32Type::get(problem.dType.getContext())
                             : problem.dElemType;
  auto eltTypeC = toSqmmaEltType(carrierElemType);
  if (!eltTypeC)
    return {MusaMmaValidationFailure::UnsupportedAccumulatorType};
  if (!eltTypeA || !eltTypeB ||
      !triton::musa::isSupportedSqmma(*eltTypeA, *eltTypeB, *eltTypeC, instM,
                                      instN, instK))
    return {MusaMmaValidationFailure::UnsupportedInstruction};

  if (!validateSqmmaOperandMaterialization(problem.a).succeeded() ||
      !validateSqmmaOperandMaterialization(problem.b).succeeded())
    return {MusaMmaValidationFailure::UnsupportedTransposeOperand};
  return {};
}

static SelectedConfig
getExplicitSqmmaConfig(ttg::MUSASqmmaEncodingAttr encoding) {
  SelectedConfig config;
  llvm::append_range(config.instrShape, encoding.getInstrShape());
  llvm::append_range(config.warpsPerCTA, encoding.getWarpsPerCTA());
  return config;
}

static MusaMmaValidationResult
validateExplicitSqmmaContract(tt::DotOp dotOp, const MusaDotProblem &problem,
                              ttg::MUSASqmmaEncodingAttr encoding) {
  Attribute explicitEncoding =
      getTleExplicitResultEncoding(dotOp.getOperation(), 0);
  if (explicitEncoding && explicitEncoding != encoding)
    return {MusaMmaValidationFailure::MismatchedExplicitResultEncoding};

  auto aEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      problem.aType.getEncoding());
  auto bEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
      problem.bType.getEncoding());
  if (!aEncoding || !bEncoding)
    return {MusaMmaValidationFailure::InvalidDotOperandEncoding};
  if (aEncoding.getOpIdx() != 0 || bEncoding.getOpIdx() != 1)
    return {MusaMmaValidationFailure::InvalidDotOperandIndex};
  if (aEncoding.getKWidth() != 0 || bEncoding.getKWidth() != 0)
    return {MusaMmaValidationFailure::InvalidDotOperandKWidth};
  if (aEncoding.getParent() != encoding || bEncoding.getParent() != encoding)
    return {MusaMmaValidationFailure::MismatchedDotOperandParent};
  if (problem.cType.getEncoding() != encoding ||
      problem.dType.getEncoding() != encoding)
    return {MusaMmaValidationFailure::MismatchedAccumulatorEncoding};

  MusaMmaValidationResult problemValidation =
      validateMusaDotProblem(problem, true);
  if (!problemValidation.succeeded())
    return problemValidation;
  return validateSqmmaConfig(problem, getExplicitSqmmaConfig(encoding),
                             encoding);
}

static LogicalResult emitExplicitSqmmaError(tt::DotOp dotOp,
                                            MusaMmaValidationFailure failure) {
  return dotOp.emitOpError("cannot lower explicit MUSA SQMMA dot: ")
         << getMusaMmaValidationMessage(failure);
}

static LogicalResult validateExplicitSqmmaDots(ModuleOp module,
                                               int computeCapability,
                                               bool disableSqmma) {
  WalkResult result = module.walk([&](tt::DotOp dotOp) -> WalkResult {
    auto resultType = dyn_cast<RankedTensorType>(dotOp.getType());
    auto encoding = resultType ? dyn_cast_or_null<ttg::MUSASqmmaEncodingAttr>(
                                     resultType.getEncoding())
                               : ttg::MUSASqmmaEncodingAttr{};
    if (!encoding)
      return WalkResult::advance();
    if (computeCapability != 31) {
      emitExplicitSqmmaError(dotOp,
                             MusaMmaValidationFailure::UnsupportedTarget);
      return WalkResult::interrupt();
    }
    if (disableSqmma) {
      emitExplicitSqmmaError(dotOp,
                             MusaMmaValidationFailure::ExplicitSqmmaDisabled);
      return WalkResult::interrupt();
    }
    auto problem = getMusaDotProblem(dotOp);
    if (failed(problem)) {
      emitExplicitSqmmaError(dotOp, MusaMmaValidationFailure::InvalidDotShape);
      return WalkResult::interrupt();
    }
    MusaMmaValidationResult validation =
        validateExplicitSqmmaContract(dotOp, *problem, encoding);
    if (!validation.succeeded()) {
      emitExplicitSqmmaError(dotOp, validation.failure);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}
#endif // __TLE__

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

static bool isLowPrecisionFloatingForFma(Type elemTy) {
  return elemTy.isF16() || elemTy.isBF16();
}

static void promoteResidualDotForFma(ModuleOp mod) {
  SmallVector<tt::DotOp> dots;
  mod.walk([&](tt::DotOp dotOp) { dots.push_back(dotOp); });
  for (tt::DotOp dotOp : dots) {
    auto aTy = dyn_cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(dotOp.getB().getType());
    auto dTy = dyn_cast<RankedTensorType>(dotOp.getType());
    if (!aTy || !bTy || !dTy)
      continue;
    if (!isa_and_nonnull<ttg::BlockedEncodingAttr>(dTy.getEncoding()))
      continue;

    Type aElemTy = aTy.getElementType();
    Type bElemTy = bTy.getElementType();
    Type dElemTy = dTy.getElementType();
    OpBuilder builder(dotOp);
    Location loc = dotOp.getLoc();
    if (tt::type::isFloat8(aElemTy) || tt::type::isFloat8(bElemTy)) {
      if (aElemTy == dElemTy && bElemTy == dElemTy)
        continue;

      // Residual fp8 tt.dot paths that are not captured by SQMMA/WMMA rewrite
      // must be promoted before FMA lowering, otherwise fp8 FMA conversion
      // is unsupported and compilation fails.
      Value newA = promoteDotOperand(builder, loc, dotOp.getA(), dElemTy);
      Value newB = promoteDotOperand(builder, loc, dotOp.getB(), dElemTy);
      dotOp.setOperand(0, newA);
      dotOp.setOperand(1, newB);
      continue;
    }

    if (!isLowPrecisionFloatingForFma(aElemTy) ||
        !isLowPrecisionFloatingForFma(bElemTy) ||
        !isLowPrecisionFloatingForFma(dElemTy))
      continue;

    Type carrierElemTy = builder.getF32Type();
    auto carrierTy = dTy.cloneWith(std::nullopt, carrierElemTy);
    Value newA = promoteDotOperand(builder, loc, dotOp.getA(), carrierElemTy);
    Value newB = promoteDotOperand(builder, loc, dotOp.getB(), carrierElemTy);
    Value newC = promoteDotOperand(builder, loc, dotOp.getC(), carrierElemTy);
    auto newDot = tt::DotOp::create(builder, loc, carrierTy, newA, newB, newC,
                                    dotOp.getInputPrecision(),
                                    dotOp.getMaxNumImpreciseAcc());
    Value truncated =
        arith::TruncFOp::create(builder, loc, dTy, newDot.getResult());
    dotOp.replaceAllUsesWith(truncated);
    dotOp.erase();
  }
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
    if (arg.getDefiningOp<tt::TransOp>())
      break;
    break;
  }

  auto argType = dyn_cast<RankedTensorType>(arg.getType());
  if (!argType || !argType.getEncoding())
    return {};
  if (isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(
          argType.getEncoding()))
    return {};
  unsigned rank = argType.getRank();
  if (rank != 2 && rank != 3)
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
    newOrder.clear();
    for (int dim = static_cast<int>(rank) - 1; dim >= 0; --dim)
      newOrder.push_back(static_cast<unsigned>(dim));
  }
  bool isRowMajor =
      !newOrder.empty() && (newOrder.front() + 1 == argType.getRank());
  auto hasConflictingSqmmaAttrs = [&](Operation *targetOp) {
    if (!targetOp || !triton::musa::hasSqmmaOpIdxAttr(targetOp))
      return false;
    auto existingOpIdx = triton::musa::getSqmmaOpIdx(targetOp);
    auto existingElemBytes = triton::musa::getSqmmaElemBytes(targetOp);
    if (!existingOpIdx || !existingElemBytes)
      return true;
    bool existingRowMajor =
        triton::musa::getSqmmaRowMajor(targetOp, isRowMajor);
    return *existingOpIdx != opIdx || *existingElemBytes != elemBytes ||
           existingRowMajor != isRowMajor;
  };
  auto setSqmmaAttrs = [&](Operation *targetOp) {
    triton::musa::setSqmmaAttrs(targetOp, opIdx, elemBytes, isRowMajor);
  };
  auto setSqmmaAttrsIfCompatible = [&](Operation *targetOp) {
    if (hasConflictingSqmmaAttrs(targetOp))
      return false;
    setSqmmaAttrs(targetOp);
    return true;
  };
  auto propagateSqmmaAttrsToMemDescChain = [&](Value memDesc) {
    if (Operation *defOp = memDesc.getDefiningOp())
      return setSqmmaAttrsIfCompatible(defOp);
    return true;
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
        if (srcMemDescTy == memDescTy) {
          if (propagateSqmmaAttrsToMemDescChain(localLoad.getSrc()))
            return localLoad.getSrc();
        } else {
          (void)propagateSqmmaAttrsToMemDescChain(localLoad.getSrc());

          rewriter.setInsertionPointAfterValue(localLoad.getSrc());
          Value adapted = ttg::MemDescReinterpretOp::create(
              rewriter, localLoad.getLoc(), memDescTy, localLoad.getSrc());
          setSqmmaAttrs(adapted.getDefiningOp());
          return adapted;
        }
      }
    }
  }
  if (descLoad) {
    setSqmmaAttrsIfCompatible(descLoad.getOperation());
  }

  Value reusedMemDesc =
      forceFreshRestage
          ? Value()
          : triton::musa::findReusableLocalAllocForSource(arg, memDescTy);
  if (reusedMemDesc) {
    if (auto localAlloc = reusedMemDesc.getDefiningOp<ttg::LocalAllocOp>()) {
      if (!setSqmmaAttrsIfCompatible(localAlloc.getOperation()))
        reusedMemDesc = {};
    }
  }

  if (reusedMemDesc)
    return reusedMemDesc;

  rewriter.setInsertionPointAfterValue(arg);
  auto localAlloc =
      ttg::LocalAllocOp::create(rewriter, arg.getLoc(), memDescTy, arg);
  setSqmmaAttrs(localAlloc.getOperation());
  return localAlloc.getResult();
}

static std::optional<SelectedConfig>
#ifdef __TLE__
selectSqmmaConfig(const MusaDotProblem &problem) {
  unsigned m = problem.matrixShape.m;
  unsigned n = problem.matrixShape.n;
  unsigned k = problem.matrixShape.k;
  unsigned numWarps = problem.numWarps;
  Type elemTy = problem.aElemType;
  bool allowTF32 = problem.allowTF32;
#else
selectSqmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                  Type elemTy, bool allowTF32) {
#endif // __TLE__
  if (numWarps < 4 || (numWarps % 4) != 0)
    return std::nullopt;
#ifndef __TLE__
  auto sqmmaEltType = toSqmmaOperandEltType(elemTy, allowTF32);
  if (!sqmmaEltType)
    return std::nullopt;
#endif // __TLE__

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
#ifndef __TLE__
    if (m < instM || (m % instM) != 0)
      continue;
#endif // __TLE__
    for (unsigned instN : candidateN) {
#ifndef __TLE__
      if (n < instN || (n % instN) != 0)
        continue;
      if (!triton::musa::isSupportedSqmmaInstrMN(*sqmmaEltType, instM, instN))
        continue;
#endif // __TLE__
      for (unsigned instK : candidateK) {
#ifndef __TLE__
        if (k < instK || (k % instK) != 0)
          continue;
        if ((instM % 4) != 0)
          continue;
        if (isKnownBrokenSqmmaConfig(elemTy, allowTF32, {instM, instN, instK}))
          continue;
#endif // __TLE__

        for (unsigned warpsM = 4; warpsM <= numWarps; warpsM *= 2) {
          if (numWarps % warpsM != 0)
            continue;
          unsigned warpsN = numWarps / warpsM;

          unsigned squadsM = warpsM / 4;
          unsigned tileM = instM * squadsM;
          unsigned tileN = instN * warpsN;
#ifdef __TLE__
          SelectedConfig candidate{{instM, instN, instK}, {warpsM, warpsN}};
          if (!validateSqmmaConfig(problem, candidate).succeeded())
            continue;
#else
          if ((m % tileM) != 0 || (n % tileN) != 0)
            continue;
#endif // __TLE__

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

#ifdef __TLE__
class ExplicitToMUSAWmma : public RewritePattern {
public:
  explicit ExplicitToMUSAWmma(MLIRContext *context, int computeCapability)
      : RewritePattern(tt::DotOp::getOperationName(), 4, context),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (computeCapability != 31)
      return failure();

    auto dotOp = dyn_cast<tt::DotOp>(op);
    if (!dotOp)
      return failure();
    auto oldRetType = dyn_cast<RankedTensorType>(dotOp.getType());
    auto mmaEnc = oldRetType ? dyn_cast_or_null<ttg::MUSAWmmaEncodingAttr>(
                                   oldRetType.getEncoding())
                             : ttg::MUSAWmmaEncodingAttr{};
    if (!mmaEnc)
      return failure();

    auto problem = getMusaDotProblem(dotOp);
    if (failed(problem) ||
        !validateExplicitWmmaContract(dotOp, *problem, mmaEnc).succeeded())
      return failure();

    SelectedConfig config = getExplicitWmmaConfig(mmaEnc);
    Type aElemTy = problem->aElemType;
    Type bElemTy = problem->bElemType;
    bool useFp32Carrier = oldRetType.getElementType().isF16() &&
                          aElemTy.isF16() && bElemTy.isF16();
    Type carrierElemTy =
        useFp32Carrier ? rewriter.getF32Type() : oldRetType.getElementType();
    auto nativeRetType =
        RankedTensorType::get(oldRetType.getShape(), carrierElemTy, mmaEnc);

    Value oldAcc = dotOp.getC();
    Value promotedAcc =
        useFp32Carrier
            ? promoteDotOperand(rewriter, dotOp.getLoc(), oldAcc, carrierElemTy)
            : oldAcc;
    bool accIsZero = isZeroConst(oldAcc);
    Value nativeAcc;
    if (accIsZero) {
      auto zeroElem = rewriter.getZeroAttr(nativeRetType.getElementType());
      auto zeroTensor = DenseElementsAttr::get(nativeRetType, zeroElem);
      nativeAcc = arith::ConstantOp::create(rewriter, oldAcc.getLoc(),
                                            nativeRetType, zeroTensor);
    } else if (promotedAcc.getType() == nativeRetType) {
      nativeAcc = promotedAcc;
    } else {
      nativeAcc = ttg::ConvertLayoutOp::create(rewriter, oldAcc.getLoc(),
                                               nativeRetType, promotedAcc);
    }

    auto wmmaEltTypeA = toSqmmaOperandEltType(aElemTy, problem->allowTF32);
    auto wmmaEltTypeB = toSqmmaOperandEltType(bElemTy, problem->allowTF32);
    if (!wmmaEltTypeA || !wmmaEltTypeB)
      return failure();
    Value useC = arith::ConstantIntOp::create(rewriter, dotOp.getLoc(), 1, 1);
    auto newDot = triton::musa::WmmaDotOp::create(
        rewriter, dotOp.getLoc(), nativeRetType, dotOp.getA(), dotOp.getB(),
        nativeAcc, useC, static_cast<int32_t>(config.instrShape[0]),
        static_cast<int32_t>(config.instrShape[1]),
        static_cast<int32_t>(config.instrShape[2]), *wmmaEltTypeA,
        *wmmaEltTypeB, triton::musa::getDefaultWmmaFragmentLayout(0),
        triton::musa::getDefaultWmmaFragmentLayout(1),
        static_cast<int32_t>(dotOp.getInputPrecision()),
        /*maxNumImpreciseAcc=*/0);
    newDot->setAttr(kDisableGenericDotPipelineAttr, rewriter.getBoolAttr(true));

    Attribute explicitResultEncoding =
        getTleExplicitResultEncoding(dotOp.getOperation(), 0);
    if (!useFp32Carrier) {
      if (explicitResultEncoding)
        setTleExplicitResultEncoding(newDot.getOperation(), 0,
                                     explicitResultEncoding);
      rewriter.replaceOp(dotOp, newDot.getResult());
      return success();
    }

    Value truncated = arith::TruncFOp::create(rewriter, dotOp.getLoc(),
                                              oldRetType, newDot.getResult());
    if (explicitResultEncoding)
      setTleExplicitResultEncoding(truncated.getDefiningOp(), 0,
                                   explicitResultEncoding);
    rewriter.replaceOp(dotOp, truncated);
    return success();
  }

private:
  int computeCapability;
};

class ExplicitToMUSASqmma : public RewritePattern {
public:
  explicit ExplicitToMUSASqmma(MLIRContext *context, int computeCapability)
      : RewritePattern(tt::DotOp::getOperationName(), 5, context),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (computeCapability != 31)
      return failure();

    auto dotOp = dyn_cast<tt::DotOp>(op);
    if (!dotOp)
      return failure();
    auto oldRetType = dyn_cast<RankedTensorType>(dotOp.getType());
    auto mmaEnc = oldRetType ? dyn_cast_or_null<ttg::MUSASqmmaEncodingAttr>(
                                   oldRetType.getEncoding())
                             : ttg::MUSASqmmaEncodingAttr{};
    if (!mmaEnc)
      return failure();

    auto problem = getMusaDotProblem(dotOp);
    if (failed(problem) ||
        !validateExplicitSqmmaContract(dotOp, *problem, mmaEnc).succeeded())
      return failure();

    SelectedConfig config = getExplicitSqmmaConfig(mmaEnc);
    Type aElemTy = problem->aElemType;
    Type bElemTy = problem->bElemType;
    bool useFp32Carrier = oldRetType.getElementType().isF16() &&
                          aElemTy.isF16() && bElemTy.isF16();
    Type carrierElemTy =
        useFp32Carrier ? rewriter.getF32Type() : oldRetType.getElementType();
    auto eltTypeC = toSqmmaEltType(carrierElemTy);
    auto eltTypeA = toSqmmaOperandEltType(aElemTy, problem->allowTF32);
    auto eltTypeB = toSqmmaOperandEltType(bElemTy, problem->allowTF32);
    if (!eltTypeC || !eltTypeA || !eltTypeB)
      return failure();

    auto nativeRetType =
        RankedTensorType::get(oldRetType.getShape(), carrierElemTy, mmaEnc);
    Value oldAcc = dotOp.getC();
    Value promotedAcc =
        useFp32Carrier
            ? promoteDotOperand(rewriter, dotOp.getLoc(), oldAcc, carrierElemTy)
            : oldAcc;
    bool accIsZero = isZeroConst(oldAcc);
    Value nativeAcc;
    if (accIsZero) {
      auto zeroElem = rewriter.getZeroAttr(nativeRetType.getElementType());
      auto zeroTensor = DenseElementsAttr::get(nativeRetType, zeroElem);
      nativeAcc = arith::ConstantOp::create(rewriter, oldAcc.getLoc(),
                                            nativeRetType, zeroTensor);
    } else if (promotedAcc.getType() == nativeRetType) {
      nativeAcc = promotedAcc;
    } else {
      nativeAcc = ttg::ConvertLayoutOp::create(rewriter, oldAcc.getLoc(),
                                               nativeRetType, promotedAcc);
    }

    bool allowTransposeA = problem->transLoadKindA != SqmmaTransLoadKind::None;
    bool allowTransposeB = problem->transLoadKindB != SqmmaTransLoadKind::None;
    Value newA = getSharedMemorySqmmaOperand(dotOp.getA(), rewriter, 0, mmaEnc,
                                             allowTransposeA);
    Value newB = getSharedMemorySqmmaOperand(dotOp.getB(), rewriter, 1, mmaEnc,
                                             allowTransposeB);
    if (!newA || !newB)
      return failure();

    auto accumulationContract = selectSqmmaAccumulationContract(
        aElemTy, nativeRetType.getElementType(), problem->matrixShape.m,
        problem->matrixShape.n, problem->matrixShape.k, accIsZero,
        static_cast<uint32_t>(dotOp.getMaxNumImpreciseAcc()), config);
    Value useC = arith::ConstantIntOp::create(
        rewriter, dotOp.getLoc(), accumulationContract.useCOperand, 1);
    auto newDot = triton::musa::SquadDotOp::create(
        rewriter, dotOp.getLoc(), nativeRetType, newA, newB, nativeAcc, useC,
        static_cast<int32_t>(config.instrShape[0]),
        static_cast<int32_t>(config.instrShape[1]),
        static_cast<int32_t>(config.instrShape[2]), *eltTypeC, *eltTypeA,
        *eltTypeB, inferSqmmaLayout(newA), inferSqmmaLayout(newB), false,
        accumulationContract.mode,
        static_cast<int32_t>(dotOp.getInputPrecision()),
        accumulationContract.mode ==
                triton::musa::SQMMAAccumulationMode::partial
            ? static_cast<int32_t>(dotOp.getMaxNumImpreciseAcc())
            : 0);
    newDot->setAttr(kDisableGenericDotPipelineAttr, rewriter.getBoolAttr(true));
    newDot->setAttr("isAsync", rewriter.getBoolAttr(false));

    Attribute explicitResultEncoding =
        getTleExplicitResultEncoding(dotOp.getOperation(), 0);
    if (!useFp32Carrier) {
      if (explicitResultEncoding)
        setTleExplicitResultEncoding(newDot.getOperation(), 0,
                                     explicitResultEncoding);
      rewriter.replaceOp(dotOp, newDot.getResult());
      return success();
    }

    Value truncated = arith::TruncFOp::create(rewriter, dotOp.getLoc(),
                                              oldRetType, newDot.getResult());
    if (explicitResultEncoding)
      setTleExplicitResultEncoding(truncated.getDefiningOp(), 0,
                                   explicitResultEncoding);
    rewriter.replaceOp(dotOp, truncated);
    return success();
  }

private:
  int computeCapability;
};
#endif // __TLE__

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
#ifdef __TLE__
    Attribute explicitResultEncoding =
        getTleExplicitResultEncoding(dotOp.getOperation(), 0);
#endif // __TLE__
    auto oldEncoding = oldRetType.getEncoding();
    if (!oldEncoding || !isa<ttg::BlockedEncodingAttr>(oldEncoding))
      return failure();
    if (isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(oldEncoding))
      return failure();
#ifdef __TLE__
    auto problem = getMusaDotProblem(dotOp);
    if (failed(problem) || !validateMusaDotProblem(*problem, false).succeeded())
      return failure();
    auto aTy = problem->aType;
    auto bTy = problem->bType;
    auto aElemTy = problem->aElemType;
    auto bElemTy = problem->bElemType;
    bool allowTF32 = problem->allowTF32;
    auto config = selectWmmaConfig(*problem);
#else
    auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = cast<RankedTensorType>(dotOp.getB().getType());
    auto aElemTy = aTy.getElementType();
    auto bElemTy = bTy.getElementType();
    bool allowTF32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;
    if (aElemTy != bElemTy)
      return failure();
    if (!isSupportedWmmaOperandType(aElemTy, allowTF32))
      return failure();

    auto matrixShape = getDotMatrixShape(dotOp);
    if (failed(matrixShape))
      return failure();

    unsigned m = matrixShape->m;
    unsigned n = matrixShape->n;
    unsigned k = matrixShape->k;
    unsigned numWarps = ttg::lookupNumWarps(dotOp);

    auto config = selectWmmaConfig(m, n, k, numWarps, aElemTy, allowTF32);
#endif // __TLE__
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
        triton::musa::getDefaultWmmaFragmentLayout(0),
        triton::musa::getDefaultWmmaFragmentLayout(1),
        static_cast<int32_t>(dotOp.getInputPrecision()),
        /*maxNumImpreciseAcc=*/0);
    newDot->setAttr(kDisableGenericDotPipelineAttr, rewriter.getBoolAttr(true));
    if (!useFp32Carrier) {
#ifdef __TLE__
      auto resultCvt = rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
          dotOp, oldRetType, newDot.getResult());
      if (explicitResultEncoding)
        setTleExplicitResultEncoding(resultCvt.getOperation(), 0,
                                     explicitResultEncoding);
#else
      rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                        newDot.getResult());
#endif // __TLE__
      return success();
    }

    auto blockedCarrierTy = oldRetType.cloneWith(std::nullopt, carrierElemTy);
    Value blockedCarrier = ttg::ConvertLayoutOp::create(
        rewriter, dotOp.getLoc(), blockedCarrierTy, newDot.getResult());
    Value truncated = arith::TruncFOp::create(rewriter, dotOp.getLoc(),
                                              oldRetType, blockedCarrier);
#ifdef __TLE__
    if (explicitResultEncoding)
      setTleExplicitResultEncoding(truncated.getDefiningOp(), 0,
                                   explicitResultEncoding);
#endif // __TLE__
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
#ifdef __TLE__
    Attribute explicitResultEncoding =
        getTleExplicitResultEncoding(dotOp.getOperation(), 0);
#endif // __TLE__
    auto oldEncoding = oldRetType.getEncoding();
    if (!oldEncoding || !isa<ttg::BlockedEncodingAttr>(oldEncoding))
      return failure();
    if (isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(oldEncoding))
      return failure();
#ifdef __TLE__
    auto problem = getMusaDotProblem(dotOp);
    if (failed(problem) || !validateMusaDotProblem(*problem, true).succeeded())
      return failure();
    auto aTy = problem->aType;
    auto bTy = problem->bType;
    auto aElemTy = problem->aElemType;
    auto bElemTy = problem->bElemType;
    bool allowTF32 = problem->allowTF32;
    unsigned m = problem->matrixShape.m;
    unsigned n = problem->matrixShape.n;
    unsigned k = problem->matrixShape.k;
    auto config = selectSqmmaConfig(*problem);
#else
    auto aTy = dyn_cast<RankedTensorType>(dotOp.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(dotOp.getB().getType());
    if (!aTy || !bTy)
      return failure();
    auto matrixShape = getDotMatrixShape(dotOp);
    if (failed(matrixShape))
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

    unsigned m = matrixShape->m;
    unsigned n = matrixShape->n;
    unsigned k = matrixShape->k;
    unsigned numWarps = ttg::lookupNumWarps(dotOp);
    auto config = selectSqmmaConfig(m, n, k, numWarps, aElemTy, allowTF32);
#endif // __TLE__
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
#ifndef __TLE__
    if (!triton::musa::isSupportedSqmma(
            *eltTypeA, *eltTypeB, *eltTypeC, config->instrShape[0],
            config->instrShape[1], config->instrShape[2]))
      return failure();
#endif // __TLE__

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

#ifdef __TLE__
    SqmmaTransLoadKind transLoadKindA = problem->transLoadKindA;
    SqmmaTransLoadKind transLoadKindB = problem->transLoadKindB;
#else
    SqmmaTransLoadKind transLoadKindA = classifySqmmaTransLoad(dotOp.getA());
    SqmmaTransLoadKind transLoadKindB = classifySqmmaTransLoad(dotOp.getB());
#endif // __TLE__
    bool allowTransposeA = transLoadKindA != SqmmaTransLoadKind::None;
    bool allowTransposeB = transLoadKindB != SqmmaTransLoadKind::None;
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
#ifdef __TLE__
      auto resultCvt = rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
          dotOp, oldRetType, newDot.getResult());
      if (explicitResultEncoding)
        setTleExplicitResultEncoding(resultCvt.getOperation(), 0,
                                     explicitResultEncoding);
#else
      rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                        newDot.getResult());
#endif // __TLE__
      return success();
    }

    auto blockedCarrierTy = oldRetType.cloneWith(std::nullopt, carrierElemTy);
    Value blockedCarrier = ttg::ConvertLayoutOp::create(
        rewriter, dotOp.getLoc(), blockedCarrierTy, newDot.getResult());
    Value truncated = arith::TruncFOp::create(rewriter, dotOp.getLoc(),
                                              oldRetType, blockedCarrier);
#ifdef __TLE__
    if (explicitResultEncoding)
      setTleExplicitResultEncoding(truncated.getDefiningOp(), 0,
                                   explicitResultEncoding);
#endif // __TLE__
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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::musa::MUSADialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int computeCapability = getMusaComputeCapability(mod);
    if (computeCapability < 0)
      return;

    bool disableSqmma = ::triton::tools::getBoolEnv("DISABLE_SQMMA");
    bool disableWmma = ::triton::tools::getBoolEnv("DISABLE_WMMA");

#ifdef __TLE__
    if (failed(validateExplicitWmmaDots(mod, computeCapability, disableWmma))) {
      signalPassFailure();
      return;
    }
    if (failed(
            validateExplicitSqmmaDots(mod, computeCapability, disableSqmma))) {
      signalPassFailure();
      return;
    }
#endif // __TLE__

    bool sqmmaCandidate = computeCapability >= 31 && !disableSqmma;
    // Preserve the 3.6 fallback behavior: descriptor/TME modules may still
    // fall back to WMMA when SQMMA predicate matching rejects a dot.
    bool wmmaCandidate = computeCapability == 31 && !disableWmma;

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ttg::populateDecomposeScaledBlockedPatterns(patterns, /*benefit=*/1);
#ifdef __TLE__
    patterns.add<ExplicitToMUSASqmma>(context, computeCapability);
    patterns.add<ExplicitToMUSAWmma>(context, computeCapability);
#endif // __TLE__
    // Keep 3.2-aligned rewrite precedence: SQMMA first, then WMMA.
    if (sqmmaCandidate)
      patterns.add<BlockedToMUSASqmma>(context, computeCapability);
    if (wmmaCandidate)
      patterns.add<BlockedToMUSAWmma>(context, computeCapability);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }

#ifdef __TLE__
    WalkResult residualExplicitMma =
        mod.walk([&](tt::DotOp dotOp) -> WalkResult {
          auto resultType = dyn_cast<RankedTensorType>(dotOp.getType());
          if (!resultType || !isa_and_nonnull<ttg::MUSAWmmaEncodingAttr,
                                              ttg::MUSASqmmaEncodingAttr>(
                                 resultType.getEncoding()))
            return WalkResult::advance();
          StringRef kind =
              isa<ttg::MUSASqmmaEncodingAttr>(resultType.getEncoding())
                  ? "SQMMA"
                  : "WMMA";
          dotOp.emitOpError("failed to rewrite validated explicit MUSA ")
              << kind << " dot";
          return WalkResult::interrupt();
        });
    if (residualExplicitMma.wasInterrupted()) {
      signalPassFailure();
      return;
    }
#endif // __TLE__

    promoteResidualDotForFma(mod);
  }
};

} // namespace mlir
