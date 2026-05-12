#include "triton/Dialect/Triton/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MMAEncodingUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// clang-format off
#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.cpp.inc"
// clang-format on

#include <algorithm>

using namespace mlir;
using namespace mlir::triton::musa;
namespace ttg = mlir::triton::gpu;

void MUSADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/MUSA/IR/MUSAAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/MUSA/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/MUSA/IR/Ops.cpp.inc"
#include "Dialect/MUSA/IR/OpsEnums.cpp.inc"

namespace mlir::triton::musa {

static LogicalResult verifyAsyncBarrierId(Operation *op, Value barId,
                                          StringRef operandName) {
  APInt constant;
  if (!matchPattern(barId, m_ConstantInt(&constant)))
    return success();

  int64_t value = constant.getSExtValue();
  if (value == 0) {
    return op->emitOpError()
           << operandName << " 0 is reserved for CTA barrier on MUSA";
  }
  if (value < 0 || value > kMaxBarrierId) {
    return op->emitOpError() << operandName << " must be in [1, "
                             << kMaxBarrierId << "] when constant";
  }
  return success();
}

static LogicalResult verifyNonNegativeI32Constant(Operation *op, Value value,
                                                  StringRef operandName) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return success();
  if (constant.isNegative()) {
    return op->emitOpError()
           << operandName << " must be non-negative when constant";
  }
  return success();
}

static LogicalResult verifyDotShapeContract(Operation *op,
                                            ArrayRef<int64_t> aShape,
                                            ArrayRef<int64_t> bShape,
                                            ArrayRef<int64_t> cShape,
                                            ArrayRef<int64_t> dShape) {
  if (aShape.size() != 2 && aShape.size() != 3)
    return op->emitError("expected operands to be 2d or 3d");
  if (aShape.size() != bShape.size() || aShape.size() != cShape.size() ||
      cShape.size() != dShape.size())
    return op->emitError(
        "expected all operands and result to have the same rank");

  if (aShape[aShape.size() - 1] != bShape[bShape.size() - 2]) {
    return op->emitError("expected the last dimension of the first operand "
                         "to equal the second-to-last dimension of the "
                         "second operand");
  }

  if (aShape.size() == 3 && (aShape[0] != cShape[0] || bShape[0] != cShape[0] ||
                             cShape[0] != dShape[0])) {
    return op->emitError("expected batch dimensions to match");
  }

  if (cShape[cShape.size() - 2] != aShape[aShape.size() - 2] ||
      cShape[cShape.size() - 1] != bShape[bShape.size() - 1]) {
    return op->emitError(
        "expected accumulator shape to match dot output shape");
  }
  if (cShape != dShape)
    return op->emitError("expected result shape to match accumulator shape");
  return success();
}

static bool isFP8Type(Type type) {
  return llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                   Float8E4M3FNUZType>(type);
}

static std::optional<SQMMAEltType> getWmmaEltTypeForVerify(Type type) {
  if (type.isInteger(8))
    return SQMMAEltType::s8;
  if (type.isF16())
    return SQMMAEltType::f16;
  if (type.isBF16())
    return SQMMAEltType::bf16;
  if (type.isF32())
    return SQMMAEltType::tf32;
  if (llvm::isa<Float8E4M3FNType, Float8E4M3FNUZType>(type))
    return SQMMAEltType::e4m3;
  if (llvm::isa<Float8E5M2Type, Float8E5M2FNUZType>(type))
    return SQMMAEltType::e5m2;
  return std::nullopt;
}

static std::optional<SQMMAEltType> getSqmmaAccumEltTypeForVerify(Type type) {
  if (type.isF32())
    return SQMMAEltType::f32;
  if (type.isInteger(32))
    return SQMMAEltType::s32;
  if (type.isF16())
    return SQMMAEltType::f16;
  return std::nullopt;
}

static LogicalResult verifyNoAcceleratedFP16Accumulator(Operation *op,
                                                        Type accElemTy,
                                                        Type retElemTy,
                                                        StringRef opName) {
  if (!accElemTy.isF16() && !retElemTy.isF16())
    return success();
  return op->emitOpError()
         << opName
         << " fp16 accumulators/results are not currently supported; use an "
            "fp32 carrier and truncate after layout conversion instead";
}

LogicalResult SquadDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  auto aEnc = cast<ttg::TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc = cast<ttg::TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<triton::DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult SquadDotOp::verify() {
  auto aTy = cast<ttg::TensorOrMemDesc>(getA().getType());
  auto bTy = cast<ttg::TensorOrMemDesc>(getB().getType());
  if (aTy.getElementType() != bTy.getElementType())
    return emitError("SQMMA operands A and B must use the same element type");
  if (aTy.getElementType().getIntOrFloatBitWidth() !=
      bTy.getElementType().getIntOrFloatBitWidth())
    return emitError(
        "element types of operands A and B must have same bit width");

  auto aEncoding = aTy.getEncoding();
  auto bEncoding = bTy.getEncoding();
  if (!aEncoding && !bEncoding)
    return success();
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");

  auto accTy = cast<RankedTensorType>(getC().getType());
  auto retTy = cast<RankedTensorType>(getD().getType());
  auto retEnc = accTy.getEncoding();
  if (!retEnc)
    return emitError("miss encoding of C operand");
  auto mmaEnc = dyn_cast<ttg::MUSASqmmaEncodingAttr>(retEnc);
  if (!mmaEnc)
    return emitError("SQMMA result layout must be #ttg.musa_sqmma");
  if (!supportsMusaSqmmaEncoding(mmaEnc))
    return emitError(
        "SQMMA result encoding uses unsupported MUSA SQMMA version");
  auto instrShape = mmaEnc.getInstrShape();
  if (instrShape.size() != 3)
    return emitError("SQMMA result encoding must carry a 3D instrShape");
  if (getM() != static_cast<int32_t>(instrShape[0]) ||
      getN() != static_cast<int32_t>(instrShape[1]) ||
      getK() != static_cast<int32_t>(instrShape[2]))
    return emitError(
        "SQMMA m/n/k attrs must match the #ttg.musa_sqmma instrShape");
  if (retTy.getEncoding() != retEnc)
    return emitError(
        "SQMMA result shape and accumulator must use the same encoding");
  auto aEltType = getWmmaEltTypeForVerify(aTy.getElementType());
  auto bEltType = getWmmaEltTypeForVerify(bTy.getElementType());
  auto cEltType = getSqmmaAccumEltTypeForVerify(retTy.getElementType());
  if (!aEltType || !bEltType || !cEltType)
    return emitError("SQMMA operands/results must use supported "
                     "int8/f16/bf16/tf32/fp8/f32/s32 element types");
  if (*aEltType != getEltTypeA() || *bEltType != getEltTypeB() ||
      *cEltType != getEltTypeC())
    return emitError(
        "SQMMA eltType attrs must match operand/result element types");
  if ((aTy.getElementType().isF32() || bTy.getElementType().isF32()) &&
      getInputPrecision() != static_cast<int32_t>(triton::InputPrecision::TF32))
    return emitError("SQMMA f32 operands require TF32 input precision");
  if (!triton::musa::isSupportedSqmma(getEltTypeA(), getEltTypeB(),
                                      getEltTypeC(), getM(), getN(), getK()))
    return emitError(
        "SQMMA encoding carries an unsupported PH1 shape/type combination");
  Dialect &dialect = aEncoding.getDialect();
  auto interface = dyn_cast<triton::DialectInferLayoutInterface>(&dialect);
  if (!interface)
    return emitError(
        "SQMMA operand encoding dialect does not implement layout inference");
  if (interface->inferDotOpEncoding(aEncoding, 0, retEnc, getLoc()).failed())
    return failure();
  if (interface->inferDotOpEncoding(bEncoding, 1, retEnc, getLoc()).failed())
    return failure();
  if (failed(verifyDotShapeContract(getOperation(), aTy.getShape(),
                                    bTy.getShape(), accTy.getShape(),
                                    retTy.getShape())))
    return failure();
  auto accMode = getAccMode();
  int64_t maxNumImpreciseAcc = std::max<int64_t>(0, getMaxNumImpreciseAcc());
  bool fp8ToF32 =
      isFP8Type(aTy.getElementType()) && retTy.getElementType().isF32();
  if (usesHardwareAccumulator()) {
    if (failed(verifyNoAcceleratedFP16Accumulator(
            getOperation(), accTy.getElementType(), retTy.getElementType(),
            "SQMMA")))
      return failure();
    if (maxNumImpreciseAcc != 0)
      return emitError(
          "hardware SQMMA accumulation requires maxNumImpreciseAcc == 0");
  } else if (accMode == SQMMAAccumulationMode::partial) {
    if (!fp8ToF32)
      return emitError("partial SQMMA accumulation currently requires fp8 "
                       "inputs and f32 accumulators");
    if (maxNumImpreciseAcc <= 0)
      return emitError(
          "partial SQMMA accumulation requires maxNumImpreciseAcc > 0");
    if (maxNumImpreciseAcc > aTy.getShape().back())
      return emitError(
          "partial SQMMA accumulation requires maxNumImpreciseAcc <= K");
  } else if (accMode == SQMMAAccumulationMode::software) {
    if (!fp8ToF32)
      return emitError("software SQMMA accumulation currently requires fp8 "
                       "inputs and f32 accumulators");
    if (maxNumImpreciseAcc != 0)
      return emitError(
          "software SQMMA accumulation requires maxNumImpreciseAcc == 0");
  }
  return success();
}

bool SquadDotOp::verifyDims() {
  auto aShape = cast<ShapedType>(getA().getType()).getShape();
  auto bShape = cast<ShapedType>(getB().getType()).getShape();
  return aShape[aShape.size() - 1] == bShape[bShape.size() - 2];
}

void SquadDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<ttg::MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a,
                         ttg::SharedMemory::get());
  if (isa<ttg::MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b,
                         ttg::SharedMemory::get());
}

bool SquadDotOp::needsPartialAccumulator() {
  return getAccMode() == SQMMAAccumulationMode::partial;
}

bool SquadDotOp::usesSoftwareAccumulator() {
  return getAccMode() == SQMMAAccumulationMode::software;
}

bool SquadDotOp::usesHardwareAccumulator() {
  return getAccMode() == SQMMAAccumulationMode::hardware;
}

LogicalResult SquadDotWaitOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)context;
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return success();
}

LogicalResult SquadDotWaitOp::verify() {
  if (getInputs().empty())
    return emitOpError("expected to be waiting on at least one dependency");
  return success();
}

LogicalResult SquadDotWaitOp::canonicalize(SquadDotWaitOp op,
                                           PatternRewriter &rewriter) {
  SmallVector<Value> liveInputs;
  SmallVector<unsigned> liveResultIdxs;
  liveInputs.reserve(op.getNumResults());
  liveResultIdxs.reserve(op.getNumResults());

  for (unsigned idx = 0; idx < op.getNumResults(); ++idx) {
    bool keepForMemDesc = isa<ttg::MemDescType>(op.getInputs()[idx].getType());
    if (!keepForMemDesc && op.getResult(idx).use_empty())
      continue;
    liveInputs.push_back(op.getInputs()[idx]);
    liveResultIdxs.push_back(idx);
  }

  if (liveResultIdxs.size() == op.getNumResults())
    return failure();

  if (liveInputs.empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  rewriter.setInsertionPoint(op);
  auto newWait = SquadDotWaitOp::create(rewriter, op.getLoc(), liveInputs);
  newWait->setAttrs(op->getAttrs());
  for (unsigned newIdx = 0; newIdx < liveResultIdxs.size(); ++newIdx)
    rewriter.replaceAllUsesWith(op.getResult(liveResultIdxs[newIdx]),
                                newWait.getResult(newIdx));
  rewriter.eraseOp(op);
  return success();
}

LogicalResult WmmaDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  auto aEnc = cast<RankedTensorType>(operands[0].getType()).getEncoding();
  auto bEnc = cast<RankedTensorType>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<triton::DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult WmmaDotOp::verify() {
  auto aTy = cast<RankedTensorType>(getA().getType());
  auto bTy = cast<RankedTensorType>(getB().getType());
  if (aTy.getElementType().getIntOrFloatBitWidth() !=
      bTy.getElementType().getIntOrFloatBitWidth())
    return emitError(
        "element types of operands A and B must have same bit width");
  if ((aTy.getRank() != 2 && aTy.getRank() != 3) ||
      aTy.getRank() != bTy.getRank())
    return emitError(
        "WMMA operands must be rank-2 or rank-3 tensors with matching rank");

  auto dTy = cast<RankedTensorType>(getD().getType());
  auto mmaEnc = dyn_cast<ttg::MUSAWmmaEncodingAttr>(dTy.getEncoding());
  if (!mmaEnc)
    return emitError("WMMA result layout must be #ttg.musa_wmma");
  if (!supportsMusaWmmaEncoding(mmaEnc))
    return emitError("WMMA result encoding uses unsupported MUSA WMMA version");
  auto instrShape = mmaEnc.getInstrShape();
  if (instrShape.size() != 3)
    return emitError("WMMA result encoding must carry a 3D instrShape");
  if (std::max<int64_t>(0, getMaxNumImpreciseAcc()) != 0)
    return emitError("WMMA maxNumImpreciseAcc must be 0 until partial "
                     "accumulation is implemented");
  if (getM() != static_cast<int32_t>(instrShape[0]) ||
      getN() != static_cast<int32_t>(instrShape[1]) ||
      getK() != static_cast<int32_t>(instrShape[2]))
    return emitError(
        "WMMA m/n/k attrs must match the #ttg.musa_wmma instrShape");

  auto aEltType = getWmmaEltTypeForVerify(aTy.getElementType());
  auto bEltType = getWmmaEltTypeForVerify(bTy.getElementType());
  if (!aEltType || !bEltType)
    return emitError("WMMA operands must use supported int8/f16/bf16/tf32/fp8 "
                     "element types");
  if (*aEltType != getEltTypeA() || *bEltType != getEltTypeB())
    return emitError("WMMA eltType attrs must match operand element types");
  if ((aTy.getElementType().isF32() || bTy.getElementType().isF32()) &&
      getInputPrecision() != static_cast<int32_t>(triton::InputPrecision::TF32))
    return emitError("WMMA f32 operands require TF32 input precision");
  if (!triton::musa::lookupWmmaIntrinsic(aTy.getElementType(), instrShape))
    return emitError(
        "WMMA encoding carries an unsupported shape/type combination");

  auto aEncoding = aTy.getEncoding();
  auto bEncoding = bTy.getEncoding();
  auto aDotEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(aEncoding);
  auto bDotEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(bEncoding);
  if (!aDotEncoding || !bDotEncoding)
    return emitError("WMMA operands A and B must use DotOperandEncodingAttr");
  if (aDotEncoding.getOpIdx() != 0 || bDotEncoding.getOpIdx() != 1)
    return emitError("WMMA operands A/B must use dot operand indices 0/1");
  if (aDotEncoding.getParent() != dTy.getEncoding() ||
      bDotEncoding.getParent() != dTy.getEncoding())
    return emitError("WMMA operand dot layouts must point to the same "
                     "#ttg.musa_wmma result encoding");

  auto cTy = cast<RankedTensorType>(getC().getType());
  if (failed(verifyDotShapeContract(getOperation(), aTy.getShape(),
                                    bTy.getShape(), cTy.getShape(),
                                    dTy.getShape())))
    return failure();
  if (failed(verifyNoAcceleratedFP16Accumulator(
          getOperation(), cTy.getElementType(), dTy.getElementType(), "WMMA")))
    return failure();

  return success();
}

bool WmmaDotOp::verifyDims() {
  auto aShape = cast<ShapedType>(getA().getType()).getShape();
  auto bShape = cast<ShapedType>(getB().getType()).getShape();
  return aShape[aShape.size() - 1] == bShape[bShape.size() - 2];
}

bool WmmaDotOp::needsPartialAccumulator() { return false; }

LogicalResult BarRecordOp::verify() {
  APInt barId;
  if (!matchPattern(getBarId(), m_ConstantInt(&barId)))
    return emitOpError("bar_record expects a constant max barrier id");
  if (barId.getSExtValue() <= 0 || barId.getSExtValue() > kMaxBarrierId) {
    return emitOpError("bar_record must be in [1, ") << kMaxBarrierId << "]";
  }
  return success();
}

LogicalResult InitArrivalOp::verify() {
  if (failed(verifyAsyncBarrierId(getOperation(), getBarId(), "barId")))
    return failure();
  if (failed(verifyNonNegativeI32Constant(getOperation(), getArriveCount(),
                                          "arriveCount")))
    return failure();
  return verifyNonNegativeI32Constant(getOperation(), getPhaseId(), "phaseId");
}

LogicalResult BarrierAddTransOp::verify() {
  if (failed(verifyAsyncBarrierId(getOperation(), getBarId(), "barId")))
    return failure();
  return verifyNonNegativeI32Constant(getOperation(), getTransBytes(),
                                      "transBytes");
}

LogicalResult ArriveBarrierOp::verify() {
  return verifyAsyncBarrierId(getOperation(), getBarId(), "barId");
}

LogicalResult ArriveBarrierNoRetOp::verify() {
  return verifyAsyncBarrierId(getOperation(), getBarId(), "barId");
}

LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyAsyncBarrierId(getOperation(), getBarId(), "barId")))
    return failure();
  return verifyNonNegativeI32Constant(getOperation(), getPhaseId(), "phaseId");
}

void WaitBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBarIdMutable(),
                       ttg::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

static LogicalResult verifyTMECopyShapeContract(Operation *op, ValueRange coord,
                                                ArrayRef<int32_t> blockShape) {
  if (coord.empty() || coord.size() > 5 || coord.size() != blockShape.size())
    return op->emitOpError(
        "expects coord/blockShape rank in [1, 5] and matching");
  for (int32_t dim : blockShape) {
    if (dim <= 0)
      return op->emitOpError("expects positive blockShape dimensions");
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyTMESwizzleContract(OpTy op) {
  auto sgAttr = op->template getAttrOfType<TMESwizzleGranularityAttr>(
      "swizzleGranularity");
  auto ssAttr =
      op->template getAttrOfType<TMESwizzleStrideAttr>("swizzleStride");
  auto slAttr = op->template getAttrOfType<TMESwizzleLineAttr>("swizzleLine");
  if (!sgAttr || !ssAttr || !slAttr)
    return op->emitOpError("requires typed TME swizzle attrs");

  TMESwizzleGranularity granularity = sgAttr.getValue();
  TMESwizzleStride stride = ssAttr.getValue();
  TMESwizzleLine line = slAttr.getValue();
  if (isValidTMESwizzleConfig(granularity, stride, line))
    return success();

  return op->emitOpError()
         << "expects a valid TME swizzle config with granularity <= stride <= "
            "line, but got granularity="
         << getSwizzleGranularityBytes(granularity)
         << "B, stride=" << getSwizzleStrideBytes(stride)
         << "B, line=" << getSwizzleLineBytes(line) << "B";
}

LogicalResult AsyncTMECopyGlobalToLocalOp::verify() {
  if (failed(verifyAsyncBarrierId(getOperation(), getBarId(), "barId")))
    return failure();
  if (failed(verifyTMECopyShapeContract(getOperation(), getCoord(),
                                        getBlockShape())))
    return failure();
  return verifyTMESwizzleContract(*this);
}

LogicalResult AsyncTMECopyLocalToGlobalOp::verify() {
  if (failed(verifyTMECopyShapeContract(getOperation(), getCoord(),
                                        getBlockShape())))
    return failure();
  return verifyTMESwizzleContract(*this);
}

void AsyncTMECopyGlobalToLocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &desc = getOperation()->getOpOperand(0);
  auto &result = getOperation()->getOpOperand(getCoord().size() + 2);
  effects.emplace_back(MemoryEffects::Read::get(), &desc,
                       ::mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &result,
                       ttg::SharedMemory::get());
}

void AsyncTMECopyLocalToGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &desc = getOperation()->getOpOperand(0);
  auto &src = getOperation()->getOpOperand(getCoord().size() + 1);
  effects.emplace_back(MemoryEffects::Read::get(), &src,
                       ttg::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &desc,
                       ::mlir::triton::GlobalMemory::get());
}

} // namespace mlir::triton::musa
