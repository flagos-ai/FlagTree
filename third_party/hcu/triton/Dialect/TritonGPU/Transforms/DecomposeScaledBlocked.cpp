#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "Dialect/TritonHCUGPU/Utility/CommonUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton::gpu {

SmallVector<int, 2> DecomposeScaledBlocked::getTransposeOrder(int rank) {
  assert(rank >= 2);
  auto transOrder = llvm::to_vector<2>(llvm::seq<int>(rank - 2));
  transOrder.push_back(rank - 1);
  transOrder.push_back(rank - 2);
  return transOrder;
}

static TypedValue<RankedTensorType> decodeArg(PatternRewriter &rewriter,
                                              DotScaledOp scaledDotOp,
                                              int opIdx,
                                              FloatType computeType) {
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto isFp4 =
      ScaleDotElemType::E2M1 ==
      (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

  auto loc = v.getLoc();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // 0) Upcast value to computeType (fp16/bf16)
  if (isFp4) {
    bool kPack =
        opIdx == 0 ? scaledDotOp.getLhsKPack() : scaledDotOp.getRhsKPack();
    int packingDim = kPack ? kDim : (opIdx == 0 ? rank - 2 : rank - 1);
    v = Fp4ToFpOp::create(rewriter, loc, v, computeType, packingDim);
  } else {
    auto vType16 = v.getType().clone(computeType);
    v = cast<TypedValue<RankedTensorType>>(
        FpToFpOp::create(rewriter, loc, vType16, v).getResult());
  }
  return v;
}

LogicalResult
DecomposeScaledBlocked::matchAndRewrite(DotScaledOp scaledDotOp,
                                        PatternRewriter &rewriter) const {
  if (isa_and_nonnull<MmaEncodingTrait>(
          scaledDotOp.getResult().getType().getEncoding()))
    return failure();

  // Types
  auto computeType = getComputeType(scaledDotOp.getAElemType(),
                                    scaledDotOp.getBElemType(), rewriter);

  bool grouped = scaledDotOp.getAScale() || scaledDotOp.getBScale();
  Value result;
  if (grouped) {
    auto loc = scaledDotOp.getLoc();
    auto *ctx = rewriter.getContext();
    auto resultTy = scaledDotOp.getType();
    if (resultTy.getRank() != 2)
      return scaledDotOp.emitError(
          "HCU software E8M0 scaled dot currently requires 2D operands");
    auto a = decodeArg(rewriter, scaledDotOp, 0, computeType);
    auto b = decodeArg(rewriter, scaledDotOp, 1, computeType);
    int64_t k = a.getType().getShape()[1];
    if (k % 32)
      return scaledDotOp.emitError(
          "E8M0 scaled dot requires K divisible by 32");
    auto encoding = resultTy.getEncoding();
    auto intTy = resultTy.clone(rewriter.getI32Type());
    auto boolTy = resultTy.clone(rewriter.getI1Type());
    auto constant = [&](RankedTensorType type, int64_t value) {
      return cast<TypedValue<RankedTensorType>>(
          arith::ConstantOp::create(
              rewriter, loc, type,
              DenseElementsAttr::get(
                  type, APInt(type.getElementTypeBitWidth(), value)))
              .getResult());
    };
    // Gather a logical K group without changing global-memory accesses. The
    // constant indices are folded by the existing layout/gather optimizations.
    auto slice = [&](TypedValue<RankedTensorType> value, int axis,
                     int64_t start, int64_t length) {
      auto type = value.getType();
      if (start == 0 && type.getShape()[axis] == length)
        return value;
      auto shape = llvm::to_vector(type.getShape());
      auto parent = cast<DistributedEncodingTrait>(type.getEncoding());
      shape[axis] = length;
      auto indexTy =
          RankedTensorType::get(shape, rewriter.getI32Type(), parent);
      auto rangeTy =
          RankedTensorType::get({length}, rewriter.getI32Type(),
                                SliceEncodingAttr::get(ctx, 1 - axis, parent));
      Value range =
          MakeRangeOp::create(rewriter, loc, rangeTy, start, start + length);
      Value expanded = ExpandDimsOp::create(rewriter, loc, range, 1 - axis);
      Value indices = BroadcastOp::create(rewriter, loc, indexTy, expanded);
      return cast<TypedValue<RankedTensorType>>(
          GatherOp::create(rewriter, loc, value, indices, axis).getResult());
    };
    auto scaleExponent = [&](TypedValue<RankedTensorType> scale, int group,
                             bool transpose) {
      if (!scale)
        return std::make_pair(constant(intTy, 0), constant(boolTy, 0));
      auto codes = slice(scale, 1, group, 1);
      auto codeTy = codes.getType().clone(rewriter.getI32Type());
      codes = cast<TypedValue<RankedTensorType>>(
          arith::ExtUIOp::create(rewriter, loc, codeTy, codes).getResult());
      if (transpose)
        codes = TransOp::create(rewriter, loc, codes, getTransposeOrder(2));
      codeTy = codes.getType().cloneWithEncoding(encoding);
      codes = ConvertLayoutOp::create(rewriter, loc, codeTy, codes);
      codes = BroadcastOp::create(rewriter, loc, intTy, codes);
      auto exponent = cast<TypedValue<RankedTensorType>>(
          arith::SubIOp::create(rewriter, loc, codes, constant(intTy, 127))
              .getResult());
      auto isNan = cast<TypedValue<RankedTensorType>>(
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, codes,
                                constant(intTy, 255))
              .getResult());
      return std::make_pair(exponent, isNan);
    };
    Value zero = arith::ConstantOp::create(
        rewriter, loc, resultTy,
        DenseElementsAttr::get(resultTy, APFloat(0.0f)));
    result = scaledDotOp.getC();
    for (int group = 0; group < k / 32; ++group) {
      auto ag =
          cvtDotOperand(rewriter, scaledDotOp, 0, slice(a, 1, group * 32, 32));
      auto bg =
          cvtDotOperand(rewriter, scaledDotOp, 1, slice(b, 0, group * 32, 32));
      Value partial = DotOp::create(rewriter, loc, ag, bg, zero);
      auto lhs = scaleExponent(scaledDotOp.getAScale(), group, false);
      auto rhs = scaleExponent(scaledDotOp.getBScale(), group, true);
      Value exponent =
          arith::AddIOp::create(rewriter, loc, lhs.first, rhs.first);
      // Combining exponents before ldexp avoids BF16 input flushing and
      // separate scale overflow/underflow. ldexp rounds only the FP32 result.
      partial = ExternElementwiseOp::create(rewriter, loc, resultTy,
                                            ValueRange{partial, exponent}, "",
                                            "", "__ocml_ldexp_f32", true);
      if (!scaledDotOp.getFastMath()) {
        Value isNan =
            arith::OrIOp::create(rewriter, loc, lhs.second, rhs.second);
        Value nan = arith::ConstantOp::create(
            rewriter, loc, resultTy,
            DenseElementsAttr::get(resultTy,
                                   APFloat::getNaN(APFloat::IEEEsingle())));
        partial = arith::SelectOp::create(rewriter, loc, isNan, nan, partial);
      }
      result = arith::AddFOp::create(rewriter, loc, result, partial);
    }
  } else {
    auto a = cvtDotOperand(rewriter, scaledDotOp, 0,
                           scaleArg(rewriter, scaledDotOp, 0, computeType));
    auto b = cvtDotOperand(rewriter, scaledDotOp, 1,
                           scaleArg(rewriter, scaledDotOp, 1, computeType));
    result =
        DotOp::create(rewriter, scaledDotOp.getLoc(), a, b, scaledDotOp.getC());
    result = ConvertLayoutOp::create(rewriter, scaledDotOp.getLoc(),
                                     scaledDotOp.getType(), result);
  }

  std::string detail;
  llvm::raw_string_ostream(detail)
      << "tl.dot_scaled ("
      << stringifyScaleDotElemType(scaledDotOp.getAElemType()) << " x "
      << stringifyScaleDotElemType(scaledDotOp.getBElemType())
      << ") uses GPU software decoding"
      << (grouped ? " and groupwise FP32 exponent scaling, " : ", ")
      << computeType << " dot operands and f32 accumulation";
  HCU::recordSoftwareDot(scaledDotOp, detail);

  if (scaledDotOp.getAElemType() == ScaleDotElemType::E2M1 ||
      scaledDotOp.getBElemType() == ScaleDotElemType::E2M1) {
    if (auto loop = scaledDotOp->getParentOfType<scf::ForOp>())
      loop->setAttr(HCU::FP4DotLoop, rewriter.getUnitAttr());
  }

  rewriter.replaceOp(scaledDotOp, result);
  return success();
}

FloatType
DecomposeScaledBlocked::getComputeType(ScaleDotElemType aType,
                                       ScaleDotElemType bType,
                                       PatternRewriter &rewriter) const {
  if (aType == ScaleDotElemType::FP16 || bType == ScaleDotElemType::FP16)
    return rewriter.getF16Type();
  return rewriter.getBF16Type();
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::scaleTo16(PatternRewriter &rewriter,
                                  TypedValue<RankedTensorType> scale,
                                  FloatType computeType) const {
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  assert(computeType == rewriter.getBF16Type() ||
         computeType == rewriter.getF16Type());

  // Choose an fp type that can fit the scale value.
  FloatType largeFpType = computeType == rewriter.getF16Type()
                              ? rewriter.getF32Type()
                              : computeType;
  int intWidth = largeFpType.getIntOrFloatBitWidth();
  auto intType = rewriter.getIntegerType(intWidth);

  auto zexted =
      arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intType), scale);
  // getFpMantissaWidth() returns the number of bits in the mantissa plus the
  // sign bit!
  int shiftValue = largeFpType.getFPMantissaWidth() - 1;
  auto shiftConst =
      arith::ConstantIntOp::create(rewriter, loc, shiftValue, intWidth);
  auto shift =
      SplatOp::create(rewriter, loc, scaleTy.clone(intType), shiftConst);
  auto shlRes = arith::ShLIOp::create(rewriter, loc, zexted, shift);
  Value scaleFP =
      BitcastOp::create(rewriter, loc, scaleTy.clone(largeFpType), shlRes);
  if (largeFpType != computeType) {
    scaleFP = arith::TruncFOp::create(rewriter, loc, scaleTy.clone(computeType),
                                      scaleFP);
  }
  return cast<TypedValue<RankedTensorType>>(scaleFP);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::broadcastScale(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp, ModuleOp mod,
    TypedValue<RankedTensorType> scale, int dim) const {
  auto *ctx = rewriter.getContext();
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  auto rank = scaleTy.getRank();
  // 2.1) Expand dims along the last dimension
  {
    // 2.1.1) Find default encoding for ExpandDims
    auto shape = to_vector(scaleTy.getShape());
    shape.insert(shape.end(), 1);
    auto nWarps = lookupNumWarps(scaledDotOp);
    auto threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto numCTAs = TritonGPUDialect::getNumCTAs(mod);
    auto blockedEnc =
        getDefaultBlockedEncoding(ctx, shape, nWarps, threadsPerWarp, numCTAs);
    // 2.1.2) Cast scale16 to SliceEncoding
    auto sliceEnc = SliceEncodingAttr::get(ctx, rank, blockedEnc);
    auto sliceType = scaleTy.cloneWithEncoding(sliceEnc);
    scale = ConvertLayoutOp::create(rewriter, loc, sliceType, scale);
  }
  auto expandScale = ExpandDimsOp::create(rewriter, loc, scale, rank);
  // 2.2) Broadcast the dimension to size 32
  auto scaleShape = to_vector(scaleTy.getShape());
  scaleShape.push_back(32);
  auto broadcastScale = BroadcastOp::create(
      rewriter, loc, expandScale.getType().clone(scaleShape), expandScale);
  // 2.3) Transpose the dimension to the scaled dimension
  auto transposeOrder = llvm::to_vector(llvm::seq<int32_t>(rank));
  transposeOrder.insert(transposeOrder.begin() + dim + 1, rank);
  auto transposedScale =
      TransOp::create(rewriter, loc, broadcastScale, transposeOrder);
  // 2.4) Reshape to the shape of v
  scaleShape.pop_back();
  scaleShape[dim] *= 32;
  auto reshapeScale =
      ReshapeOp::create(rewriter, loc, scaleShape, transposedScale);
  return reshapeScale;
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::maskNan(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> mxfp, TypedValue<RankedTensorType> scale,
    int dim) const {
  // Skip NaN checks if fastMath
  if (scaledDotOp.getFastMath())
    return mxfp;

  // Implement tl.where(scale == 0xFF, float("nan"), mxfp)
  auto loc = scale.getLoc();
  auto mod = scaledDotOp->getParentOfType<ModuleOp>();

  // Scale is NaN
  auto scaleTy = scale.getType();
  auto constFF = arith::ConstantOp::create(
      rewriter, loc, scaleTy,
      DenseElementsAttr::get(scaleTy,
                             APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
  auto scaleIsNan = cast<TypedValue<RankedTensorType>>(
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, scale,
                            constFF)
          .getResult());
  auto cond = broadcastScale(rewriter, scaledDotOp, mod, scaleIsNan, dim);
  // Make scale is NaN compatible with mxfp
  auto condTy = cond.getType();
  condTy = condTy.cloneWithEncoding(mxfp.getType().getEncoding());
  cond = ConvertLayoutOp::create(rewriter, loc, condTy, cond);

  // Create NaN
  auto mxfpTy = mxfp.getType();
  auto nan = APFloat::getNaN(
      cast<FloatType>(mxfpTy.getElementType()).getFloatSemantics());
  auto constNan = arith::ConstantOp::create(
      rewriter, loc, mxfpTy, DenseElementsAttr::get(mxfpTy, nan));

  auto result = arith::SelectOp::create(rewriter, loc, cond, constNan, mxfp);
  return cast<TypedValue<RankedTensorType>>(result.getResult());
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::scaleArg(PatternRewriter &rewriter,
                                 DotScaledOp scaledDotOp, int opIdx,
                                 FloatType computeType) const {
  auto v = decodeArg(rewriter, scaledDotOp, opIdx, computeType);
  auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
  auto loc = v.getLoc();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
  if (!scale)
    return v;

  // 1) Cast scale to fp16/bf16, broadcast it and convert its layout
  auto reshapeScale = extendAndBroadcastScale(rewriter, scaledDotOp, scale,
                                              computeType, v.getType(), opIdx);

  // 2) Multiply
  auto mxfp = cast<TypedValue<RankedTensorType>>(
      arith::MulFOp::create(rewriter, loc, v, reshapeScale).getResult());

  // 3) If the scale is NaN, return NaN, else return the scaled value.
  return maskNan(rewriter, scaledDotOp, mxfp, scale, kDim);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::extendAndBroadcastScale(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> &scale, FloatType computeType,
    RankedTensorType dstType, int opIdx) const {
  auto loc = scale.getLoc();
  auto mod = scaledDotOp->getParentOfType<ModuleOp>();
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // For some weird reason, we take the scale with shape as if it were coming
  // from the lhs even when it's the rhs. In a normal world, we should accept
  // this parameter transposed, as we do with the mxfp.
  //
  // Notice: this is an inplace change.
  if (opIdx == 1) {
    auto order = getTransposeOrder(rank);
    scale = TransOp::create(rewriter, loc, scale, order);
  }

  // 1) Cast scale to compute type (fp16/bf16)
  auto scale16 = scaleTo16(rewriter, scale, computeType);

  // 2) Broadcast scale to the same shape as v and convert the layout
  auto reshapeScale = broadcastScale(rewriter, scaledDotOp, mod, scale16, kDim);
  return ConvertLayoutOp::create(rewriter, loc, dstType, reshapeScale);
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::cvtDotOperand(PatternRewriter &rewriter,
                                      DotScaledOp scaledDotOp, int opIdx,
                                      TypedValue<RankedTensorType> v) const {
  auto *ctx = rewriter.getContext();
  auto retEnc = scaledDotOp.getType().getEncoding();
  auto vType = v.getType();
  auto encoding =
      DotOperandEncodingAttr::get(ctx, opIdx, retEnc, vType.getElementType());
  auto retTy = vType.cloneWithEncoding(encoding);
  return ConvertLayoutOp::create(rewriter, v.getLoc(), retTy, v);
}

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit) {
  patterns.add<DecomposeScaledBlocked>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::gpu
