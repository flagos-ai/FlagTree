#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

// clang-format off
#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MTGPU/IR/Dialect.cpp.inc"
// clang-format on

#include <algorithm>

using namespace mlir;
using namespace mlir::triton::mtgpu;
namespace ttg = mlir::triton::gpu;

void MTGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/MTGPU/IR/MTGPUTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/MTGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/MTGPU/IR/MTGPUTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/MTGPU/IR/Ops.cpp.inc"
#include "Dialect/MTGPU/IR/OpsEnums.cpp.inc"

namespace mlir::triton::mtgpu {

LogicalResult
SqmmaAccumulatorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type tensorType) {
  auto rankedTy = dyn_cast<RankedTensorType>(tensorType);
  if (!rankedTy)
    return emitError() << "expected ranked tensor accumulator type";
  if (!isa_and_nonnull<triton::gpu::MUSASqmmaEncodingAttr>(
          rankedTy.getEncoding())) {
    return emitError() << "expected tensor encoded with #ttg.musa_sqmma";
  }
  Type elemTy = rankedTy.getElementType();
  if (!elemTy.isF32() && !elemTy.isInteger(32)) {
    return emitError()
           << "expected f32 or i32 SQMMA accumulator element type, got "
           << elemTy;
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

LogicalResult SqmmaOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)context;
  (void)attributes;
  (void)properties;
  (void)regions;

  auto accTy = dyn_cast<SqmmaAccumulatorType>(operands[2].getType());
  if (!accTy)
    return failure();
  inferredReturnTypes.push_back(accTy);

  auto aEnc = cast<ttg::TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc = cast<ttg::TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getAccumulatorType().getEncoding();
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

LogicalResult SqmmaOp::verify() {
  auto aTy = cast<ttg::TensorOrMemDesc>(getA().getType());
  auto bTy = cast<ttg::TensorOrMemDesc>(getB().getType());
  if (aTy.getElementType().getIntOrFloatBitWidth() !=
      bTy.getElementType().getIntOrFloatBitWidth())
    return emitError(
        "element types of operands A and B must have same bit width");

  auto aEncoding = aTy.getEncoding();
  auto bEncoding = bTy.getEncoding();
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");

  auto accTy = dyn_cast<SqmmaAccumulatorType>(getC().getType());
  auto retTy = dyn_cast<SqmmaAccumulatorType>(getD().getType());
  if (!accTy || !retTy)
    return emitError(
        "SQMMA accumulator/result must use !mtgpu.sqmma_accumulator");
  auto accTensorTy = accTy.getAccumulatorType();
  auto retTensorTy = retTy.getAccumulatorType();
  auto retEnc = accTensorTy.getEncoding();
  Dialect &dialect = aEncoding.getDialect();
  auto interface = dyn_cast<triton::DialectInferLayoutInterface>(&dialect);
  if (!interface)
    return emitError(
        "SQMMA operand encoding dialect does not implement layout inference");
  if (interface->inferDotOpEncoding(aEncoding, 0, retEnc, getLoc()).failed())
    return failure();
  if (interface->inferDotOpEncoding(bEncoding, 1, retEnc, getLoc()).failed())
    return failure();
  if (retTensorTy.getEncoding() != retEnc)
    return emitError(
        "SQMMA result carrier must use the same encoding as the accumulator");
  if (failed(verifyDotShapeContract(getOperation(), aTy.getShape(),
                                    bTy.getShape(), accTensorTy.getShape(),
                                    retTensorTy.getShape())))
    return failure();

  auto accMode = getAccMode();
  int64_t maxNumImpreciseAcc = std::max<int64_t>(0, getMaxNumImpreciseAcc());
  bool fp8ToF32 =
      isFP8Type(aTy.getElementType()) && retTensorTy.getElementType().isF32();
  if (usesHardwareAccumulator()) {
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

void SqmmaOp::getEffects(
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

bool SqmmaOp::needsPartialAccumulator() {
  return getAccMode() == SQMMAAccumulationMode::partial;
}

bool SqmmaOp::usesSoftwareAccumulator() {
  return getAccMode() == SQMMAAccumulationMode::software;
}

bool SqmmaOp::usesHardwareAccumulator() {
  return getAccMode() == SQMMAAccumulationMode::hardware;
}

LogicalResult SqmmaWaitOp::inferReturnTypes(
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

LogicalResult SqmmaWaitOp::verify() {
  if (getInputs().empty())
    return emitOpError("expected to be waiting on at least one dependency");
  return success();
}

LogicalResult SqmmaWaitOp::canonicalize(SqmmaWaitOp op,
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
  auto newWait = SqmmaWaitOp::create(rewriter, op.getLoc(), liveInputs);
  newWait->setAttrs(op->getAttrs());
  for (unsigned newIdx = 0; newIdx < liveResultIdxs.size(); ++newIdx)
    rewriter.replaceAllUsesWith(op.getResult(liveResultIdxs[newIdx]),
                                newWait.getResult(newIdx));
  rewriter.eraseOp(op);
  return success();
}

LogicalResult PackSqmmaAccumulatorOp::verify() {
  auto carrierTy = dyn_cast<SqmmaAccumulatorType>(getCarrier().getType());
  if (!carrierTy)
    return emitError("result must be !mtgpu.sqmma_accumulator");
  if (carrierTy.getAccumulatorType() != getInput().getType())
    return emitError("carrier tensor type must match input type");
  return success();
}

LogicalResult UnpackSqmmaAccumulatorOp::verify() {
  auto carrierTy = cast<SqmmaAccumulatorType>(getCarrier().getType());
  if (carrierTy.getAccumulatorType() != getOutput().getType())
    return emitError("result tensor type must match carrier tensor type");
  return success();
}

} // namespace mlir::triton::mtgpu
