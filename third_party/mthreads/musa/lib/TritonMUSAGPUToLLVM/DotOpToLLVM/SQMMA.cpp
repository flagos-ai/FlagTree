#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "DotOpToLLVM.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

triton::musa::SQMMAEltType getMmaRetType(Value d) {
  auto dTy = cast<RankedTensorType>(d.getType()).getElementType();
  if (dTy.isF32())
    return triton::musa::SQMMAEltType::f32;
  if (dTy.isInteger(32))
    return triton::musa::SQMMAEltType::s32;
  if (dTy.isF16())
    return triton::musa::SQMMAEltType::f16;
  llvm::report_fatal_error("MUSA SQMMA: unsupported result type");
}

triton::musa::SQMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = cast<TensorOrMemDesc>(a.getType()).getElementType();
  if (aTy.isF16())
    return triton::musa::SQMMAEltType::f16;
  if (aTy.isBF16())
    return triton::musa::SQMMAEltType::bf16;
  if (aTy.isF32() && allowTF32)
    return triton::musa::SQMMAEltType::tf32;
  if (aTy.isInteger(8))
    return triton::musa::SQMMAEltType::s8;
  if (llvm::isa<Float8E5M2Type>(aTy))
    return triton::musa::SQMMAEltType::e5m2;
  if (llvm::isa<Float8E4M3FNType, Float8E4M3FNUZType>(aTy))
    return triton::musa::SQMMAEltType::e4m3;
  llvm::report_fatal_error("MUSA SQMMA: unsupported operand type");
}

bool isFP8(triton::musa::SQMMAEltType type) {
  return type == triton::musa::SQMMAEltType::e4m3 ||
         type == triton::musa::SQMMAEltType::e5m2;
}

struct SqmmaDescriptorIntrinsicSpec {
  llvm::StringRef intrinsicName;
  Type loadType;
  Type dataType;
  Value leadingStrideBytes;
  Value swizzleGranularity;
  Value swizzleStride;
};

static FailureOr<SqmmaDescriptorIntrinsicSpec>
buildSqmmaDescriptorIntrinsicSpec(Location loc, MemDescType memDescTy,
                                  ArrayRef<int64_t> physicalShape,
                                  triton::musa::SQMMAEltType eltType,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter) {
  auto order =
      triton::musa::getSharedOrder(memDescTy.getEncoding(), physicalShape);
  if (physicalShape.size() != 2 || order.size() < 2)
    return failure();

  int elemBits = memDescTy.getElementTypeBitWidth();
  if (elemBits <= 0 || (elemBits % 8) != 0)
    return failure();
  int64_t elemBytes = elemBits / 8;
  auto grouping = triton::musa::getPH1TMELeadingDimGrouping(physicalShape,
                                                            order, elemBytes);
  if (failed(grouping))
    return failure();
  int64_t groupedLeadingWidthBytes =
      grouping->elemsPerGroupInLeadingDim * elemBytes;
  if (groupedLeadingWidthBytes <= 0 ||
      !llvm::isPowerOf2_64(static_cast<uint64_t>(groupedLeadingWidthBytes)))
    return failure();

  auto swizzle = triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy);
  if (failed(swizzle))
    return failure();

  Type loadType = typeConverter->convertType(memDescTy.getElementType());
  if (!loadType)
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  llvm::StringRef intrinsicName;
  Type dataType;
  switch (eltType) {
  case triton::musa::SQMMAEltType::f16:
  case triton::musa::SQMMAEltType::bf16:
    intrinsicName = "llvm.musa.sqmma.desc.half";
    dataType = rewriter.getI16Type();
    break;
  case triton::musa::SQMMAEltType::tf32:
    intrinsicName = "llvm.musa.sqmma.desc.fp32";
    dataType = rewriter.getF32Type();
    break;
  case triton::musa::SQMMAEltType::s8:
  case triton::musa::SQMMAEltType::e4m3:
  case triton::musa::SQMMAEltType::e5m2:
    intrinsicName = "llvm.musa.sqmma.desc.i8";
    dataType = IntegerType::get(ctx, 8);
    break;
  default:
    return failure();
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return SqmmaDescriptorIntrinsicSpec{
      intrinsicName,
      loadType,
      dataType,
      b.i32_val(static_cast<int32_t>(groupedLeadingWidthBytes)),
      b.i32_val(static_cast<int32_t>(swizzle->swizzleGranularity)),
      b.i32_val(static_cast<int32_t>(swizzle->swizzleStride)),
  };
}

class SqmmaSmemLoader {
public:
  SqmmaSmemLoader() = default;
  SqmmaSmemLoader(Value tensor, Value affineBase,
                  ArrayRef<int64_t> semanticShape,
                  ArrayRef<int64_t> physicalShape, Value warpId,
                  unsigned dimWpt, bool trans, unsigned nonKTile,
                  unsigned kTile, SqmmaDescriptorIntrinsicSpec descriptorSpec,
                  ConversionPatternRewriter &rewriter, Location loc)
      : base(affineBase),
        semanticShape(semanticShape.begin(), semanticShape.end()),
        physicalShape(physicalShape.begin(), physicalShape.end()),
        warpId(warpId), dimWpt(dimWpt), trans(trans), nonKTile(nonKTile),
        kTile(kTile), descriptorSpec(descriptorSpec) {
    auto ty = cast<MemDescType>(tensor.getType());
    ord = triton::musa::getSharedOrder(ty.getEncoding(), this->physicalShape);
    elemBytes = ty.getElementTypeBitWidth() / 8;
    uint32_t widthInByte = this->physicalShape[ord[0]] * elemBytes;
    elemsPerSwizzlingRow =
        widthInByte > 256 ? 256 / elemBytes : this->physicalShape[ord[0]];
    elemsPerSwizzlingRowVal =
        TritonLLVMOpBuilder(loc, rewriter).i32_val(elemsPerSwizzlingRow);
  }

  Value smemDesc(unsigned tileNonKIdx, unsigned tileKIdx,
                 ConversionPatternRewriter &rewriter, Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value k = b.i32_val(tileKIdx * kTile);
    Value nonK = b.add(b.i32_val(tileNonKIdx * dimWpt * nonKTile),
                       b.mul(warpId, b.i32_val(nonKTile)));
    if (trans)
      std::swap(k, nonK);

    Value leadingOffset =
        b.mul(b.udiv(k, elemsPerSwizzlingRowVal),
              b.i32_val(physicalShape[ord[1]] * elemsPerSwizzlingRow));
    Value strideOffset = b.mul(nonK, elemsPerSwizzlingRowVal);
    Value offset = b.add(b.add(leadingOffset, strideOffset),
                         b.urem(k, elemsPerSwizzlingRowVal));
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value elemBase = b.bitcast(base, elemPtrTy);
    Value elemPtr = b.gep(elemPtrTy, descriptorSpec.loadType, elemBase, offset);
    Value data =
        LLVM::LoadOp::create(rewriter, loc, descriptorSpec.loadType, elemPtr);
    if (data.getType() != descriptorSpec.dataType)
      data = b.bitcast(data, descriptorSpec.dataType);
    auto desc = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, descriptorSpec.intrinsicName, TypeRange{i32_ty},
        ValueRange{data, descriptorSpec.leadingStrideBytes,
                   descriptorSpec.swizzleGranularity,
                   descriptorSpec.swizzleStride});
    return desc.getResult(0);
  }

private:
  Value base;
  SmallVector<int64_t> semanticShape;
  SmallVector<int64_t> physicalShape;
  Value warpId;
  unsigned dimWpt = 1;
  bool trans = false;
  unsigned nonKTile = 0;
  unsigned kTile = 0;
  SmallVector<unsigned> ord;
  int elemsPerSwizzlingRow = 0;
  int elemBytes = 0;
  Value elemsPerSwizzlingRowVal;
  SqmmaDescriptorIntrinsicSpec descriptorSpec;
};

SmallVector<Value> loadAccSlice(ConversionPatternRewriter &rewriter,
                                Location loc,
                                const SmallVector<Value> &elements,
                                int startIndex, int numElements,
                                Operation *insertBefore) {
  OpBuilder::InsertionGuard g(rewriter);
  if (insertBefore)
    rewriter.setInsertionPoint(insertBefore);

  SmallVector<Value> slice(numElements);
  for (int i = 0; i < numElements; ++i)
    slice[i] = elements[startIndex + i];
  return slice;
}

Value selectAccumulatorVector(Location loc, Value useC, Value acc, Value zero,
                              ConversionPatternRewriter &rewriter) {
  return triton::musa::selectAccumulatorValue(loc, useC, acc, zero, rewriter);
}

Value addAccumulate(ConversionPatternRewriter &rewriter, Location loc, Value a,
                    Value b) {
  auto vecTy = cast<VectorType>(a.getType());
  auto bld = TritonLLVMOpBuilder(loc, rewriter);
  Value res = bld.undef(vecTy);
  for (int i = 0; i < vecTy.getNumElements(); ++i) {
    Value lhs = bld.extract_element(a, bld.i32_val(i));
    Value rhs = bld.extract_element(b, bld.i32_val(i));
    Value add = vecTy.getElementType().isInteger()
                    ? LLVM::AddOp::create(rewriter, loc, lhs, rhs).getResult()
                    : LLVM::FAddOp::create(rewriter, loc, lhs, rhs).getResult();
    res = bld.insert_element(res, add, bld.i32_val(i));
  }
  return res;
}

Value addAccumulatorLane(ConversionPatternRewriter &rewriter, Location loc,
                         Value a, Value b) {
  if (a.getType().isInteger())
    return LLVM::AddOp::create(rewriter, loc, a, b).getResult();
  return LLVM::FAddOp::create(rewriter, loc, a, b).getResult();
}

static RankedTensorType getSqmmaAccumulatorTensorType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy;
  if (auto carrierTy = dyn_cast<triton::mtgpu::SqmmaAccumulatorType>(type))
    return carrierTy.getAccumulatorType();
  return RankedTensorType();
}

static bool isSqmmaAccumulatorCarrierType(Type type) {
  return isa<triton::mtgpu::SqmmaAccumulatorType>(type);
}

} // namespace

namespace mlir::triton::MUSA {

static InputPrecision decodeInputPrecisionAttr(int64_t value) {
  switch (value) {
  case 0:
    return InputPrecision::IEEE;
  case 1:
    return InputPrecision::TF32;
  case 2:
    return InputPrecision::TF32x3;
  case 3:
  case 4:
    llvm::report_fatal_error(
        "BF16x3/BF16x6 must be rewritten by TritonGPUF32DotTC before "
        "MUSA SQMMA lowering");
  default:
    llvm::report_fatal_error("Unexpected MUSA dot input precision attribute");
  }
}

static Value getDotOperandA(triton::DotOp op) { return op.getA(); }
static Value getDotOperandA(triton::musa::SquadDotOp op) { return op.getA(); }
static Value getDotOperandA(triton::mtgpu::SqmmaOp op) { return op.getA(); }
static Value getDotOperandB(triton::DotOp op) { return op.getB(); }
static Value getDotOperandB(triton::musa::SquadDotOp op) { return op.getB(); }
static Value getDotOperandB(triton::mtgpu::SqmmaOp op) { return op.getB(); }
static Value getDotOperandC(triton::DotOp op) { return op.getC(); }
static Value getDotOperandC(triton::musa::SquadDotOp op) { return op.getC(); }
static Value getDotOperandC(triton::mtgpu::SqmmaOp op) { return op.getC(); }

static Value getDotAdaptorA(triton::DotOp::Adaptor adaptor) {
  return adaptor.getA();
}
static Value getDotAdaptorA(triton::musa::SquadDotOp::Adaptor adaptor) {
  return adaptor.getA();
}
static Value getDotAdaptorA(triton::mtgpu::SqmmaOp::Adaptor adaptor) {
  return adaptor.getA();
}
static Value getDotAdaptorB(triton::DotOp::Adaptor adaptor) {
  return adaptor.getB();
}
static Value getDotAdaptorB(triton::musa::SquadDotOp::Adaptor adaptor) {
  return adaptor.getB();
}
static Value getDotAdaptorB(triton::mtgpu::SqmmaOp::Adaptor adaptor) {
  return adaptor.getB();
}
static Value getDotAdaptorC(triton::DotOp::Adaptor adaptor) {
  return adaptor.getC();
}
static Value getDotAdaptorC(triton::musa::SquadDotOp::Adaptor adaptor) {
  return adaptor.getC();
}
static Value getDotAdaptorC(triton::mtgpu::SqmmaOp::Adaptor adaptor) {
  return adaptor.getC();
}

static Value getDotUseCValue(triton::DotOp, triton::DotOp::Adaptor) {
  return Value();
}
static Value getDotUseCValue(triton::musa::SquadDotOp,
                             triton::musa::SquadDotOp::Adaptor adaptor) {
  return adaptor.getUseC();
}
static Value getDotUseCValue(triton::mtgpu::SqmmaOp,
                             triton::mtgpu::SqmmaOp::Adaptor adaptor) {
  return adaptor.getUseC();
}

static InputPrecision getDotInputPrecision(triton::DotOp op) {
  return op.getInputPrecision();
}
static InputPrecision getDotInputPrecision(triton::musa::SquadDotOp op) {
  return decodeInputPrecisionAttr(op.getInputPrecision());
}
static InputPrecision getDotInputPrecision(triton::mtgpu::SqmmaOp op) {
  return decodeInputPrecisionAttr(op.getInputPrecision());
}

static bool getDotIsAsync(triton::DotOp op) { return false; }
static bool getDotIsAsync(triton::musa::SquadDotOp op) {
  return op.getIsAsync();
}
static bool getDotIsAsync(triton::mtgpu::SqmmaOp op) { return op.getIsAsync(); }

static triton::musa::SQMMAEltType getDotEltTypeA(triton::DotOp op) {
  bool allowTF32 = getDotInputPrecision(op) == InputPrecision::TF32;
  return getMmaOperandType(op.getA(), allowTF32);
}
static triton::musa::SQMMAEltType getDotEltTypeA(triton::musa::SquadDotOp op) {
  return op.getEltTypeA();
}
static triton::musa::SQMMAEltType getDotEltTypeA(triton::mtgpu::SqmmaOp op) {
  return static_cast<triton::musa::SQMMAEltType>(
      static_cast<int>(op.getEltTypeA()));
}

static triton::musa::SQMMAEltType getDotEltTypeB(triton::DotOp op) {
  bool allowTF32 = getDotInputPrecision(op) == InputPrecision::TF32;
  return getMmaOperandType(op.getB(), allowTF32);
}
static triton::musa::SQMMAEltType getDotEltTypeB(triton::musa::SquadDotOp op) {
  return op.getEltTypeB();
}
static triton::musa::SQMMAEltType getDotEltTypeB(triton::mtgpu::SqmmaOp op) {
  return static_cast<triton::musa::SQMMAEltType>(
      static_cast<int>(op.getEltTypeB()));
}

static triton::musa::SQMMAEltType getDotEltTypeC(triton::DotOp op) {
  return getMmaRetType(op.getResult());
}
static triton::musa::SQMMAEltType getDotEltTypeC(triton::musa::SquadDotOp op) {
  return op.getEltTypeC();
}
static triton::musa::SQMMAEltType getDotEltTypeC(triton::mtgpu::SqmmaOp op) {
  return static_cast<triton::musa::SQMMAEltType>(
      static_cast<int>(op.getEltTypeC()));
}

static uint32_t getDotMaxNumImpreciseAcc(triton::DotOp op) {
  return op.getMaxNumImpreciseAcc();
}
static uint32_t getDotMaxNumImpreciseAcc(triton::musa::SquadDotOp op) {
  return static_cast<uint32_t>(
      std::max<int64_t>(0, op.getMaxNumImpreciseAcc()));
}
static uint32_t getDotMaxNumImpreciseAcc(triton::mtgpu::SqmmaOp op) {
  return static_cast<uint32_t>(
      std::max<int64_t>(0, op.getMaxNumImpreciseAcc()));
}

static triton::musa::SQMMALayout getDotLayoutA(triton::DotOp op) {
  auto aTy = cast<TensorOrMemDesc>(op.getA().getType());
  if (auto memDescTy = dyn_cast<MemDescType>(aTy))
    return triton::musa::inferSharedRowMajor(memDescTy)
               ? triton::musa::SQMMALayout::row
               : triton::musa::SQMMALayout::col;
  auto tensorTy = cast<RankedTensorType>(aTy);
  auto order = getOrderForMemory(tensorTy);
  bool isRowMajor = !order.empty() && order.front() + 1 == tensorTy.getRank();
  return isRowMajor ? triton::musa::SQMMALayout::row
                    : triton::musa::SQMMALayout::col;
}
static triton::musa::SQMMALayout getDotLayoutA(triton::musa::SquadDotOp op) {
  return op.getLayoutA();
}
static triton::musa::SQMMALayout getDotLayoutA(triton::mtgpu::SqmmaOp op) {
  return static_cast<triton::musa::SQMMALayout>(
      static_cast<int>(op.getLayoutA()));
}
static triton::musa::SQMMALayout getDotLayoutB(triton::DotOp op) {
  auto bTy = cast<TensorOrMemDesc>(op.getB().getType());
  if (auto memDescTy = dyn_cast<MemDescType>(bTy))
    return triton::musa::inferSharedRowMajor(memDescTy)
               ? triton::musa::SQMMALayout::row
               : triton::musa::SQMMALayout::col;
  auto tensorTy = cast<RankedTensorType>(bTy);
  auto order = getOrderForMemory(tensorTy);
  bool isRowMajor = !order.empty() && order.front() + 1 == tensorTy.getRank();
  return isRowMajor ? triton::musa::SQMMALayout::row
                    : triton::musa::SQMMALayout::col;
}
static triton::musa::SQMMALayout getDotLayoutB(triton::musa::SquadDotOp op) {
  return op.getLayoutB();
}
static triton::musa::SQMMALayout getDotLayoutB(triton::mtgpu::SqmmaOp op) {
  return static_cast<triton::musa::SQMMALayout>(
      static_cast<int>(op.getLayoutB()));
}

static triton::musa::SQMMAAccumulationMode
getDotAccumulationMode(triton::DotOp op) {
  auto aTy = cast<TensorOrMemDesc>(op.getA().getType());
  bool fp8 = isFP8(getDotEltTypeA(op));
  bool accFP32 =
      cast<RankedTensorType>(op.getResult().getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getDotMaxNumImpreciseAcc(op);
  if (fp8 && accFP32 && maxNumImpreciseAcc > 0 &&
      maxNumImpreciseAcc <= aTy.getShape().back())
    return triton::musa::SQMMAAccumulationMode::partial;
  return triton::musa::SQMMAAccumulationMode::hardware;
}
static triton::musa::SQMMAAccumulationMode
getDotAccumulationMode(triton::musa::SquadDotOp op) {
  return op.getAccMode();
}
static triton::musa::SQMMAAccumulationMode
getDotAccumulationMode(triton::mtgpu::SqmmaOp op) {
  return static_cast<triton::musa::SQMMAAccumulationMode>(
      static_cast<int>(op.getAccMode()));
}

template <typename DotLikeOp, typename DotLikeAdaptor>
LogicalResult convertSQMMADotImpl(DotLikeOp op, DotLikeAdaptor adaptor,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Value threadId) {
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto dTy = getSqmmaAccumulatorTensorType(op.getResult().getType());
  bool carrierMode = isSqmmaAccumulatorCarrierType(op.getResult().getType());
  if (!dTy)
    return op.emitError("MUSA SQMMA: expected tensor or SQMMA carrier result");
  auto mmaEnc = dyn_cast<MUSASqmmaEncodingAttr>(dTy.getEncoding());
  if (!mmaEnc)
    return op.emitError("MUSA SQMMA: expected #ttg.musa_sqmma result encoding");
  if (!mmaEnc.isPH1())
    return op.emitError("MUSA SQMMA: unsupported version");

  if (dTy.getRank() != 2)
    return op.emitError("MUSA SQMMA supports rank-2 tensors only");

  auto instrShape = mmaEnc.getInstrShape();
  if (instrShape.size() != 3)
    return op.emitError("MUSA SQMMA expects 3D instrShape");

  const unsigned instM = instrShape[0];
  const unsigned instN = instrShape[1];
  const unsigned instK = instrShape[2];

  Value opA = getDotOperandA(op);
  Value opB = getDotOperandB(op);
  Value opC = getDotOperandC(op);
  Value useCFlag = triton::musa::materializeUseCFlag(
      loc, getDotUseCValue(op, adaptor), rewriter);
  std::optional<bool> useCConst = getBoolFromConstant(useCFlag);

  Value adaptorA = getDotAdaptorA(adaptor);
  Value adaptorB = getDotAdaptorB(adaptor);
  Value adaptorC = getDotAdaptorC(adaptor);
  auto aOperand = triton::musa::resolveSharedOperandWithAffineBase(
      opA, adaptorA, loc, typeConverter, rewriter);
  auto bOperand = triton::musa::resolveSharedOperandWithAffineBase(
      opB, adaptorB, loc, typeConverter, rewriter);
  if (failed(aOperand))
    return op.emitError(
        "MUSA SQMMA requires operand A from ttg.local_load(shared memdesc)");
  if (failed(bOperand))
    return op.emitError(
        "MUSA SQMMA requires operand B from ttg.local_load(shared memdesc)");

  auto aMemTy = aOperand->memDescTy;
  auto bMemTy = bOperand->memDescTy;

  if (!aMemTy || !triton::musa::isSharedEncoding(aMemTy.getEncoding()))
    return op.emitError("MUSA SQMMA requires operand A in shared memory");
  if (!bMemTy || !triton::musa::isSharedEncoding(bMemTy.getEncoding()))
    return op.emitError("MUSA SQMMA requires operand B in shared memory");
  if (aMemTy.getRank() != 2 || bMemTy.getRank() != 2)
    return op.emitError("MUSA SQMMA supports rank-2 operands only");

  auto eltTypeC = getDotEltTypeC(op);
  auto eltTypeA = getDotEltTypeA(op);
  auto eltTypeB = getDotEltTypeB(op);

  if (!isSupportedSqmma(eltTypeA, eltTypeB, eltTypeC, instM, instN, instK))
    return op.emitError("MUSA SQMMA: unsupported shape or element type");
  std::string sqmmaIntrinsic =
      triton::musa::lookupSqmmaIntrinsic(eltTypeA, instM, instN, instK);
  if (sqmmaIntrinsic.empty())
    return op.emitError("MUSA SQMMA: unsupported operand type");

  auto warpsPerCTA = mmaEnc.getWarpsPerCTA();
  if (warpsPerCTA.size() < 2 || warpsPerCTA[0] % 4 != 0)
    return op.emitError("MUSA SQMMA: invalid warpsPerCTA");

  auto dShapePerCTA = getShapePerCTA(dTy);
  unsigned blockM = dShapePerCTA[0];
  unsigned blockN = dShapePerCTA[1];
  unsigned squadsM = warpsPerCTA[0] / 4;
  unsigned squadsN = warpsPerCTA[1];
  unsigned tileM = instM * squadsM;
  unsigned tileN = instN * squadsN;

  auto ceilDiv = [](unsigned x, unsigned y) { return (x + y - 1) / y; };
  unsigned numRepM = ceilDiv(blockM, tileM);
  unsigned numRepN = ceilDiv(blockN, tileN);
  int64_t kDimVal = aMemTy.getShape().back();
  if (kDimVal <= 0)
    return op.emitError("MUSA SQMMA requires static positive K dimension");
  unsigned kDim = static_cast<unsigned>(kDimVal);
  unsigned numRepK = std::max(1u, ceilDiv(kDim, instK));

  unsigned repCount = numRepM * numRepN;
  unsigned totalAccElems = mmaEnc.getTotalElemsPerThread(dTy.getShape());
  if (repCount == 0 || totalAccElems == 0 || (totalAccElems % repCount) != 0)
    return op.emitError("MUSA SQMMA: invalid accumulator partitioning");
  unsigned accElemsPerThread = totalAccElems / repCount;
  constexpr unsigned warpSize = 32;

  bool zeroAcc = isZeroConst(opC);
  SmallVector<Value> fc;
  SmallVector<Value> fcFragments;
  if (zeroAcc) {
    if (carrierMode) {
      fcFragments.resize(repCount);
    } else {
      Type accElemTy = typeConverter->convertType(dTy.getElementType());
      Value zero = LLVM::ZeroOp::create(rewriter, loc, accElemTy);
      size_t totalAcc =
          static_cast<size_t>(accElemsPerThread) * numRepM * numRepN;
      fc.assign(totalAcc, zero);
    }
  } else {
    if (carrierMode) {
      fcFragments = mlir::LLVM::MUSA::unpackSqmmaAccumulatorCarrier(
          loc, adaptorC, dTy, rewriter);
    } else {
      fc = ::mlir::unpackLLElements(loc, adaptorC, rewriter);
    }
  }

  size_t expectedAccElems =
      static_cast<size_t>(accElemsPerThread) * numRepM * numRepN;
  if (!carrierMode && fc.size() != expectedAccElems) {
    if (fc.empty() || (fc.size() % expectedAccElems) != 0)
      return op.emitError("MUSA SQMMA: accumulator size mismatch");
    size_t dupFactor = fc.size() / expectedAccElems;
    SmallVector<Value> compact;
    compact.reserve(expectedAccElems);
    for (size_t i = 0; i < expectedAccElems; ++i)
      compact.push_back(fc[i * dupFactor]);
    fc.swap(compact);
  }

  Value warp = b.udiv(threadId, b.i32_val(warpSize));
  Value warpGroup = b.and_(warp, b.i32_val(0xFFFFFFFC));
  Value squad = b.udiv(warpGroup, b.i32_val(4));
  Value squadM = b.urem(squad, b.i32_val(squadsM));
  Value squadMN = b.udiv(squad, b.i32_val(squadsM));
  Value squadN = b.urem(squadMN, b.i32_val(squadsN));

  auto layoutA = getDotLayoutA(op);
  auto layoutB = getDotLayoutB(op);
  bool loaderTransA = layoutA == triton::musa::SQMMALayout::col;
  bool loaderTransB = layoutB == triton::musa::SQMMALayout::row;
  int32_t intrinsicTransA = (layoutA == triton::musa::SQMMALayout::col) ? 1 : 0;
  int32_t intrinsicTransB = (layoutB == triton::musa::SQMMALayout::col) ? 1 : 0;

  auto aDescSpec = buildSqmmaDescriptorIntrinsicSpec(
      loc, aMemTy, aOperand->physicalShape, eltTypeA, typeConverter, rewriter);
  if (failed(aDescSpec))
    return op.emitError("MUSA SQMMA failed to derive descriptor contract for "
                        "operand A");
  auto bDescSpec = buildSqmmaDescriptorIntrinsicSpec(
      loc, bMemTy, bOperand->physicalShape, eltTypeB, typeConverter, rewriter);
  if (failed(bDescSpec))
    return op.emitError("MUSA SQMMA failed to derive descriptor contract for "
                        "operand B");

  SqmmaSmemLoader aLoader(aOperand->memDesc, aOperand->affineBase,
                          aMemTy.getShape(), aOperand->physicalShape, squadM,
                          squadsM, loaderTransA, instM, instK, *aDescSpec,
                          rewriter, loc);
  SqmmaSmemLoader bLoader(bOperand->memDesc, bOperand->affineBase,
                          bMemTy.getShape(), bOperand->physicalShape, squadN,
                          squadsN, loaderTransB, instN, instK, *bDescSpec,
                          rewriter, loc);

  auto accumulationMode = getDotAccumulationMode(op);
  uint32_t maxNumImpreciseAcc = getDotMaxNumImpreciseAcc(op);
  bool usesSoftwareAccumulator =
      accumulationMode == triton::musa::SQMMAAccumulationMode::software;
  bool needsPartialAccumulator =
      accumulationMode == triton::musa::SQMMAAccumulationMode::partial;
  Type accLaneTy = typeConverter->convertType(dTy.getElementType());
  Type accVecTy = vec_ty(accLaneTy, accElemsPerThread);

  SmallVector<Value> mmaResults;
  SmallVector<Value> mmaCarrierFragments;
  for (unsigned mRep = 0; mRep < numRepM; ++mRep) {
    for (unsigned nRep = 0; nRep < numRepN; ++nRep) {
      size_t accBase = (mRep * numRepN + nRep) * accElemsPerThread;
      unsigned fragmentIdx = mRep * numRepN + nRep;
      auto ivecTy = vec_ty(IntegerType::get(rewriter.getContext(), 32),
                           accElemsPerThread);
      if (usesSoftwareAccumulator) {
        SmallVector<Value> accumElems;
        if (zeroAcc) {
          if (carrierMode) {
            Value zeroVec = LLVM::ZeroOp::create(rewriter, loc, accVecTy);
            accumElems = unpackLLVector(loc, zeroVec, rewriter);
          } else {
            accumElems.append(fc.begin() + accBase,
                              fc.begin() + accBase + accElemsPerThread);
          }
        } else {
          Value accSliceVec;
          if (carrierMode) {
            accSliceVec = mlir::LLVM::MUSA::carrierFragmentToMathVec(
                loc, fcFragments[fragmentIdx], dTy, rewriter);
          } else {
            auto accSlice = loadAccSlice(rewriter, loc, fc, accBase,
                                         accElemsPerThread, nullptr);
            accSliceVec = packLLVector(loc, accSlice, rewriter);
          }
          // In the software-accumulation family, useC=false keeps the 3.2
          // contract of "hardware sees zero C, outer fadd still preserves the
          // running accumulator". Only dynamic first-use flags introduced by
          // accumulator-init rewriting should zero the carried accumulator
          // here.
          if (!useCConst) {
            Value zeroSliceVec =
                LLVM::ZeroOp::create(rewriter, loc, accSliceVec.getType());
            accSliceVec = selectAccumulatorVector(loc, useCFlag, accSliceVec,
                                                  zeroSliceVec, rewriter);
          }
          accumElems = unpackLLVector(loc, accSliceVec, rewriter);
        }
        for (unsigned kRep = 0; kRep < numRepK; ++kRep) {
          Value opA = aLoader.smemDesc(mRep, kRep, rewriter, loc);
          Value opB = bLoader.smemDesc(nRep, kRep, rewriter, loc);

          SmallVector<Value> args = {
              opA,
              opB,
              LLVM::ZeroOp::create(rewriter, loc, ivecTy),
              b.i32_val(intrinsicTransA),
              b.i32_val(intrinsicTransB),
              b.i32_val(0), // aNeg
              b.i32_val(0), // bNeg
              b.i32_val(1), // scale_out
              b.i32_val(0), // sat
              b.i32_val(0), // stepA
              b.i32_val(0), // stepB
              b.i32_val(0)  // stepC
          };
          auto call = LLVM::createLLVMIntrinsicCallOp(
              rewriter, loc, sqmmaIntrinsic, TypeRange{ivecTy}, args);
          Value newAcc = call.getResult(0);
          Value newMathVec = newAcc.getType() == accVecTy
                                 ? newAcc
                                 : b.bitcast(newAcc, accVecTy);
          SmallVector<Value> newElems =
              unpackLLVector(loc, newMathVec, rewriter);
          if (accumElems.empty()) {
            accumElems = std::move(newElems);
          } else {
            for (unsigned i = 0; i < newElems.size(); ++i) {
              accumElems[i] =
                  addAccumulatorLane(rewriter, loc, accumElems[i], newElems[i]);
            }
          }
        }
        if (carrierMode) {
          Value accumVec = packLLVector(loc, accumElems, rewriter);
          fcFragments[fragmentIdx] = mlir::LLVM::MUSA::mathVecToCarrierFragment(
              loc, accumVec, dTy, rewriter);
        } else {
          for (unsigned i = 0; i < accumElems.size(); ++i)
            fc[accBase + i] = accumElems[i];
        }
        continue;
      }

      Type accElemTy = accLaneTy;
      Value accVec;
      if (carrierMode) {
        accVec = zeroAcc ? LLVM::ZeroOp::create(rewriter, loc, ivecTy)
                         : fcFragments[fragmentIdx];
        if (!zeroAcc && (!useCConst || !*useCConst)) {
          Value zeroVec = LLVM::ZeroOp::create(rewriter, loc, accVec.getType());
          accVec =
              selectAccumulatorVector(loc, useCFlag, accVec, zeroVec, rewriter);
        }
      } else if (zeroAcc) {
        accVec = LLVM::ZeroOp::create(rewriter, loc,
                                      vec_ty(accElemTy, accElemsPerThread));
      } else {
        auto accSlice = loadAccSlice(rewriter, loc, fc, accBase,
                                     accElemsPerThread, nullptr);
        if (!accSlice.empty())
          accElemTy = accSlice.front().getType();
        accVec = packLLVector(loc, accSlice, rewriter);
        if (!useCConst || !*useCConst) {
          Value zeroVec = LLVM::ZeroOp::create(rewriter, loc, accVec.getType());
          accVec =
              selectAccumulatorVector(loc, useCFlag, accVec, zeroVec, rewriter);
        }
      }
      auto currentAccVecTy = accVec.getType();
      Value vecAcc =
          accVec.getType() == ivecTy ? accVec : b.bitcast(accVec, ivecTy);

      Value dAcc = zeroAcc ? Value() : vecAcc;

      uint32_t numLowPrecAcc = 0;
      Value partialAcc;
      for (unsigned kRep = 0; kRep < numRepK; ++kRep) {
        Value opA = aLoader.smemDesc(mRep, kRep, rewriter, loc);
        Value opB = bLoader.smemDesc(nRep, kRep, rewriter, loc);
        numLowPrecAcc += instK;
        bool requireAdd =
            needsPartialAccumulator &&
            (numLowPrecAcc >= maxNumImpreciseAcc || kRep == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : dAcc;
        Value inputAcc =
            mmaAcc ? mmaAcc : LLVM::ZeroOp::create(rewriter, loc, ivecTy);

        SmallVector<Value> args = {
            opA,
            opB,
            inputAcc,
            b.i32_val(intrinsicTransA),
            b.i32_val(intrinsicTransB),
            b.i32_val(0), // aNeg
            b.i32_val(0), // bNeg
            b.i32_val(1), // scale_out
            b.i32_val(0), // sat
            b.i32_val(0), // stepA
            b.i32_val(0), // stepB
            b.i32_val(0)  // stepC
        };
        auto call = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, sqmmaIntrinsic, TypeRange{ivecTy}, args);
        Value newAcc = call.getResult(0);
        if (needsPartialAccumulator) {
          partialAcc = newAcc;
        } else {
          dAcc = newAcc;
        }

        if (requireAdd) {
          Value partialFloat = partialAcc.getType() == accVecTy
                                   ? partialAcc
                                   : b.bitcast(partialAcc, accVecTy);
          if (dAcc) {
            Value dFloat =
                dAcc.getType() == accVecTy ? dAcc : b.bitcast(dAcc, accVecTy);
            dFloat = addAccumulate(rewriter, loc, dFloat, partialFloat);
            dAcc =
                dFloat.getType() == ivecTy ? dFloat : b.bitcast(dFloat, ivecTy);
          } else {
            dAcc = partialAcc;
          }
          numLowPrecAcc = 0;
          partialAcc = Value();
        }
      }

      if (needsPartialAccumulator && partialAcc) {
        Value partialFloat = partialAcc.getType() == accVecTy
                                 ? partialAcc
                                 : b.bitcast(partialAcc, accVecTy);
        if (dAcc) {
          Value dFloat =
              dAcc.getType() == accVecTy ? dAcc : b.bitcast(dAcc, accVecTy);
          dFloat = addAccumulate(rewriter, loc, dFloat, partialFloat);
          dAcc =
              dFloat.getType() == ivecTy ? dFloat : b.bitcast(dFloat, ivecTy);
        } else {
          dAcc = partialAcc;
        }
      }

      Value finalReg = dAcc ? dAcc : vecAcc;
      if (carrierMode) {
        Value finalMathVec = finalReg.getType() == accVecTy
                                 ? finalReg
                                 : b.bitcast(finalReg, accVecTy);
        Value carrierFragment = mlir::LLVM::MUSA::mathVecToCarrierFragment(
            loc, finalMathVec, dTy, rewriter);
        mmaCarrierFragments.push_back(carrierFragment);
      } else {
        Value finalAcc = finalReg.getType() == currentAccVecTy
                             ? finalReg
                             : b.bitcast(finalReg, currentAccVecTy);
        SmallVector<Value> accElems = unpackLLVector(loc, finalAcc, rewriter);
        for (unsigned i = 0; i < accElems.size(); ++i)
          mmaResults.push_back(accElems[i]);
      }
    }
  }

  if (carrierMode) {
    if (!getDotIsAsync(op)) {
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.sqmma.wait",
                                      TypeRange{}, {});
    }
    Value res = mlir::LLVM::MUSA::packSqmmaAccumulatorCarrier(
        loc,
        usesSoftwareAccumulator ? ValueRange(fcFragments)
                                : ValueRange(mmaCarrierFragments),
        dTy, rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }

  if (usesSoftwareAccumulator)
    mmaResults.assign(fc.begin(), fc.end());

  unsigned encodedAccElems = mmaEnc.getTotalElemsPerThread(dTy.getShape());
  if (mmaResults.size() != encodedAccElems)
    return op.emitError("MUSA SQMMA: result accumulator size mismatch");

  if (!getDotIsAsync(op)) {
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.sqmma.wait",
                                    TypeRange{}, {});
  }

  Value res =
      ::mlir::packLLElements(loc, typeConverter, mmaResults, rewriter, dTy);

  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertSQMMADot(triton::musa::SquadDotOp op,
                              triton::musa::SquadDotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              Value threadId) {
  return convertSQMMADotImpl(op, adaptor, typeConverter, rewriter, threadId);
}

LogicalResult convertSQMMADot(triton::mtgpu::SqmmaOp op,
                              triton::mtgpu::SqmmaOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              Value threadId) {
  return convertSQMMADotImpl(op, adaptor, typeConverter, rewriter, threadId);
}

} // namespace mlir::triton::MUSA
