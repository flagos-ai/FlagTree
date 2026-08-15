/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TritonPPUGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

SmallVector<unsigned, 3>
PPUMmaVersionToInstrShape(int version, const ArrayRef<int64_t> &shape,
                          Type eltType, int numWarps) {
  if (version == 1 || version == 2) {
    auto rank = shape.size();
    SmallVector<unsigned, 3> ret(rank, 1);
    ret[rank - 1] = 16;
    ret[rank - 2] = 16;
    return ret;
  } else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

static bool supportPPUMMA(Value value, int version) {
  // Tell whether a DotOp support MMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2) &&
         "Unexpected PPU MMA layout version found");
  auto elemTy =
      cast<triton::gpu::TensorOrMemDesc>(value.getType()).getElementType();
  // FP8 is not natively supported on all mma versions but it can always be
  // promoted to fp16 therefore we can always support it.
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(elemTy);
  return isFP8 || elemTy.isF16() || elemTy.isBF16() ||
         (elemTy.isF32() || elemTy.isF64()) || elemTy.isInteger(8);
}

static bool supportPPUMMA(triton::DotOp op, int version) {
  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getInputPrecision() == InputPrecision::TF32;
  }
  return supportPPUMMA(op.getA(), version) && supportPPUMMA(op.getB(), version);
}

// Get the highest version supported for the hardware and the dot.
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  // List supported mma version in order of preference.
  SmallVector<int> versionsSupported;
  if (computeCapability == 80) {
    versionsSupported = {1};
  } else if (computeCapability == 89) {
    versionsSupported = {2};
  } else {
    assert(false && "computeCapability not supported");
  }
  for (int baseVersion : versionsSupported) {
    if (supportPPUMMA(op, baseVersion))
      return baseVersion;
  }
  return 0;
}

SmallVector<unsigned, 2> warpsPerTileMmaV1(DotOpInterface dotOp,
                                           const ArrayRef<int64_t> shape,
                                           int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<TransOp>(op);
  };
  auto slices = mlir::getSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;
  for (Operation *op : slices) {
    if (isa<DotOp>(op) && (op != dotOp)) {
      auto chainedDot = cast<DotOp>(op);
      auto resTy = chainedDot.getResult().getType();
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<PPUMmaEncodingAttr>(resTy.getEncoding())) {
        // return getWarpsPerCTA(mmaEncoding);
        return to_vector(mmaEncoding.getWarpsPerCTA());
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    int64_t MaxWarpM = 16 * numWarps;
    if (MaxWarpM <= shape[0]) {
      return {(unsigned)numWarps, 1};
    }
  }

  assert(rank == 2);
  SmallVector<int64_t> shapePerWarp = {16, 16};
  SmallVector<int64_t> warps = {1, 1};
  // Compute repM and repN
  SmallVector<int64_t> reps = {ceil(shape[0], shapePerWarp[0]),
                               ceil(shape[1], shapePerWarp[1])};
  // The formula for the number of registers given the reps is
  // repM * 4 * repK + repN * 2 * repK + regsC
  // where regsC = repM * repN * 4, which does not depend on the warp shape
  //
  // As such, to minimize the register pressure, we need to balance
  // repM and repN. We then untie towards M, as the lhs tile has 4 elements,
  // and the rhs tile has just 2.
  while (product(warps) < numWarps) {
    if (reps[0] >= reps[1]) {
      warps[0] *= 2;
      // Too many warps for this mma (repM == repN == 1).
      // We allocate the remaining warps to the left (arbitrary choice)
      if (reps[0] != 1) {
        reps[0] /= 2;
      }
    } else {
      warps[1] *= 2;
      reps[1] /= 2;
    }
  }
  return {(unsigned)warps[0], (unsigned)warps[1]};
}

SmallVector<unsigned> warpsPerTileMmaV2(DotOpInterface dotOp,
                                        const ArrayRef<int64_t> shape,
                                        int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<TransOp>(op);
  };
  auto slices = mlir::getSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;

  for (Operation *op : slices) {
    if (isa<DotOp>(op) && (op != dotOp)) {
      auto chainedDot = cast<DotOp>(op);
      auto resTy = chainedDot.getResult().getType();
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<PPUMmaEncodingAttr>(resTy.getEncoding())) {
        return to_vector(mmaEncoding.getWarpsPerCTA());
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    // PPUMmaV2 don't need chained dot optimization, but we still prefer to
    // assign warps to one axis to facilitate use cases like flash attention,
    // allowing reductions within the same warp.
    int64_t MaxWarpM = 16 * numWarps;
    if (MaxWarpM <= shape[0]) {
      return {(unsigned)numWarps, 1};
    }
  }

  SmallVector<unsigned> ret(rank, 1);
  SmallVector<int64_t> shapePerWarp(rank, 1);
  shapePerWarp[rank - 1] = 16;
  shapePerWarp[rank - 2] = 16;
  // TODO (@daadaada): double-check.
  // original logic in
  // https://github.com/triton-lang/triton/blob/master/lib/codegen/analysis/layout.cc#L252
  // seems buggy for shape = [32, 16] ?
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / shapePerWarp[1] / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

static LocalAllocOp
getSharedMemoryScale(Value arg, mlir::PatternRewriter &rewriter, Location loc) {
  OpBuilder::InsertionGuard g(rewriter);
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrderForMemory(argType);

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  // No swizzling for scale for now
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), /*swizzlingByteWidth=*/0,
      /*transposed=*/false,
      /*elementBitWidth=*/argType.getElementType().getIntOrFloatBitWidth(),
      /*fp4Padded=*/false, CTALayout);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return LocalAllocOp::create(rewriter, loc, newType, arg);
}

SmallVector<unsigned, 3>
getWarpsPerTile(DotOpInterface dotOp, const ArrayRef<int64_t> shape,
                int version, int numWarps,
                const SmallVector<unsigned, 3> &instrShape) {
  switch (version) {
  case 1:
    return warpsPerTileMmaV1(dotOp, shape, numWarps);
  case 2:
    return warpsPerTileMmaV2(dotOp, shape, numWarps);
  default:
    assert(false && "not supported version");
    return {0, 0};
  }
}

static bool bwdFilter(Operation *op) {
  return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
         isView(op) ||
         isa<AIULoadOp, Fp4ToFpOp, LoadOp, DescriptorLoadOp, BroadcastOp,
             ConvertLayoutOp>(op);
}

// Finds the bitwidth with which the value x is loaded
static int computeOrigBitWidth(Value x) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = bwdFilter;
  (void)getBackwardSlice(x, &slice, opt);

  // TODO: This heuristic may be a bit too coarse and may need improving
  // If the chain contains a fp4 to fp16/bf16 conversion, then the original
  // bitwidth is 4.
  if (llvm::any_of(slice, [](Operation *op) { return isa<Fp4ToFpOp>(op); }))
    return 4;

  int origBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
  for (auto op : slice) {
    if (isa<LoadOp, DescriptorLoadOp, AIULoadOp>(op)) {
      if (auto tensorTy =
              dyn_cast<RankedTensorType>(op->getResultTypes().front())) {
        origBitWidth =
            std::min<int>(origBitWidth, tensorTy.getElementTypeBitWidth());
      }
    }
  }

  // If JoinOp occurred at least once, in backward layout propagation,
  // the kWidth will be split in half as we pass through the JoinOp.
  // Hence we divide origBitWidth by 2 here to compensate for that and
  // improve our load width.
  // This won't be optimal if there is a tree of multiple JoinOps, which
  // would require counting the max number of JoinOp's along any path.
  //
  // In the future we might want to do something like trying a large kWidth,
  // run layout backpropagation and see what's the contiguity that you
  // get at the loads that feed into it.
  if (llvm::any_of(slice, [](Operation *op) { return isa<JoinOp>(op); }))
    origBitWidth /= 2;

  return origBitWidth;
}

namespace {

// Common MMA encoding creation
struct MMAEncodingResult {
  PPUMmaEncodingAttr mmaEnc;
  RankedTensorType newRetType;
  Value newAcc;
  int versionMajor;
  int versionMinor;
};

// Unified implementation for DotOpInterface
static MMAEncodingResult createMMAEncodingForDot(DotOpInterface dotOp,
                                                 PatternRewriter &rewriter,
                                                 int computeCapability,
                                                 int versionMajor) {
  auto oldRetType = cast<RankedTensorType>(dotOp.getD().getType());
  auto oldAType = cast<RankedTensorType>(dotOp.getA().getType());
  auto oldBType = cast<RankedTensorType>(dotOp.getB().getType());

  int numWarps = lookupNumWarps(dotOp);

  int versionMinor = 0;
  if (!(versionMajor == 1 || versionMajor == 2)) {
    return {nullptr, RankedTensorType(), Value(), versionMajor, versionMinor};
  }

  auto CTALayout = getCTALayout(oldRetType.getEncoding());
  auto retShapePerCTA = getShapePerCTA(oldRetType);
  auto instrShape = PPUMmaVersionToInstrShape(
      versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);
  auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                      numWarps, instrShape);

  PPUMmaEncodingAttr mmaEnc;
  if (versionMajor == 1) {
    unsigned vecSize = 1;
    if (oldRetType.getElementType().isF16()) {
      if (oldAType.getElementType().isF16() &&
          oldBType.getElementType().isF16())
        vecSize = 2;
    }
    mmaEnc = PPUMmaEncodingAttr::get(oldRetType.getContext(), versionMajor,
                                     versionMinor, warpsPerTile, CTALayout,
                                     instrShape, vecSize);
  } else if (versionMajor == 2) {
    mmaEnc = PPUMmaEncodingAttr::get(oldRetType.getContext(), versionMajor,
                                     versionMinor, warpsPerTile, CTALayout,
                                     instrShape);
  }

  auto newRetType = oldRetType.cloneWithEncoding(mmaEnc);

  auto oldAcc = dotOp->getOperand(2);
  auto newAcc =
      ConvertLayoutOp::create(rewriter, oldAcc.getLoc(), newRetType, oldAcc);

  return {mmaEnc, newRetType, newAcc, versionMajor, versionMinor};
}

// Common operand conversion
static Value convertDotOperandForMMA(Value v, int opIdx, int bitwidth,
                                     RankedTensorType newRetType,
                                     PatternRewriter &rewriter) {
  auto minType = bitwidth > 0 ? rewriter.getIntegerType(bitwidth) : v.getType();
  auto vType = cast<RankedTensorType>(v.getType());
  auto newVEncoding = DotOperandEncodingAttr::get(
      v.getContext(), opIdx, newRetType.getEncoding(), minType);
  auto newVType = vType.cloneWithEncoding(newVEncoding);
  return ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v);
}

} // namespace

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    if (computeCapability < 80) {
      dotOp.emitRemark()
          << "Dot op using MMA for compute capability " << computeCapability
          << " has been deprecated. It falls back to the FMA path.";
      return failure();
    }
    // TODO: Check data-types and SM compatibility
    auto retType = dotOp.getType();
    if (!retType.getEncoding() ||
        mlir::isa<PPUMmaEncodingAttr>(retType.getEncoding()))
      return failure();

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());

    // Enable F64 MMA only on SM80/SM90 with high performance F64 tensorcore.
    // Otherwise, fallback to F64 FMA for better performance.
    if (oldAType.getElementType().isF64() ||
        oldBType.getElementType().isF64() ||
        oldRetType.getElementType().isF64()) {
      return failure();
    }

    auto mmaVersion = getMMAVersionSafe(computeCapability, dotOp);
    auto mmaResult =
        createMMAEncodingForDot(dotOp, rewriter, computeCapability, mmaVersion);
    if (!(mmaResult.versionMajor >= 1 && mmaResult.versionMajor <= 2))
      return failure();

    Operation *newDot = nullptr;
    bool aFromLoad = comesFromLoadOrBlockArg(a);
    bool bFromLoad = comesFromLoadOrBlockArg(b);

    int minBitwidth = std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
    a = convertDotOperandForMMA(a, 0, minBitwidth, mmaResult.newRetType,
                                rewriter);
    b = convertDotOperandForMMA(b, 1, minBitwidth, mmaResult.newRetType,
                                rewriter);
    newDot = DotOp::create(rewriter, dotOp.getLoc(), mmaResult.newRetType, a, b,
                           mmaResult.newAcc, dotOp.getInputPrecision(),
                           dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                 newDot->getResult(0));
    return success();
  }
};

static DistributedEncodingTrait
replaceCTALayout(DistributedEncodingTrait layout,
                 const triton::gpu::CTAEncodingAttr &newCTALayout) {
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    return BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), newCTALayout);
  } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    return SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        replaceCTALayout(sliceLayout.getParent(), newCTALayout));
  } else {
    llvm::report_fatal_error("not implemented");
    return layout;
  }
}

static Value splitBOperand(Value b, mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  MLIRContext *ctx = b.getContext();
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  auto loadOp = b.getDefiningOp();
  assert((isa<triton::LoadOp, triton::DescriptorLoadOp,
              triton::DescriptorGatherOp>(loadOp)) &&
         "expected LoadOp");
  RankedTensorType bType = cast<RankedTensorType>(b.getType());
  auto currentLayout = cast<DistributedEncodingTrait>(bType.getEncoding());
  auto kBlock = StringAttr::get(ctx, "block");
  auto dims = standardOutDimNames(ctx, 2);
  auto newCTALayout =
      CTAEncodingAttr::get(ctx, LinearLayout({{kBlock, {{0, 1}}}}, dims));
  Attribute newLayout = replaceCTALayout(currentLayout, newCTALayout);
  rewriter.setInsertionPoint(loadOp);
  for (OpOperand &operand : loadOp->getOpOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!tensorType)
      continue;
    Value newOperand = ConvertLayoutOp::create(
        rewriter, operand.get().getLoc(),
        tensorType.cloneWithEncoding(newLayout), operand.get());
    loadOp->setOperand(operand.getOperandNumber(), newOperand);
  }
  loadOp->getResult(0).setType(bType.cloneWithEncoding(newLayout));
  Value newB = loadOp->getResult(0);
  rewriter.setInsertionPointAfter(loadOp);
  auto cvt = ConvertLayoutOp::create(rewriter, b.getLoc(), bType, newB);
  rewriter.replaceAllUsesExcept(newB, cvt.getResult(), cvt);
  return newB;
}

class ScaledBlockedToMMAv2 : public mlir::OpRewritePattern<DotScaledOp> {
  int computeCapability;
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

public:
  ScaledBlockedToMMAv2(mlir::MLIRContext *context, int computeCapability,
                       int benefit)
      : OpRewritePattern<DotScaledOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    // currently only support ppu0015 m16n16k64 mma
    if (computeCapability != 89)
      return failure();

    // TODO: Check data-types and SM compatibility
    if (!dotOp.getType().getEncoding() ||
        mlir::isa<PPUMmaEncodingAttr>(dotOp.getType().getEncoding()))
      return failure();

    // currently only support lhs&rhs kpack on ppu0015
    // TODO: support non kpack
    if (!dotOp.getLhsKPack() || !dotOp.getRhsKPack())
      return failure();

    // ppu0015 only support fp4 scaled dot
    if (dotOp.getAElemType() != ScaleDotElemType::E2M1 ||
        dotOp.getBElemType() != ScaleDotElemType::E2M1)
      return failure();

    if (!dotOp.getAScale() || !dotOp.getBScale())
      return failure();

    auto oldRetType = cast<RankedTensorType>(dotOp.getType());
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);

    auto a = dotOp.getA();
    auto b = dotOp.getB();
    auto aScale = dotOp.getAScale();
    auto bScale = dotOp.getBScale();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    auto aScaleOrder = getOrder(aScale.getType());
    auto bScaleOrder = getOrder(bScale.getType());
    // ascale, bscale row major
    if (aScaleOrder[aScaleOrder.size() - 1] != 0 ||
        bScaleOrder[bScaleOrder.size() - 1] != 0)
      return failure();

    // get MMA encoding for the given number of warps
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    int versionMajor = 2;
    int versionMinor = 0;
    auto instrShape = PPUMmaVersionToInstrShape(
        versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);

    auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                        numWarps, instrShape);
    PPUMmaEncodingAttr mmaEnc = PPUMmaEncodingAttr::get(
        oldRetType.getContext(), versionMajor, versionMinor, warpsPerTile,
        CTALayout, instrShape);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    auto getDotOperand = [&](Value v, int opIdx, int bitwidth) {
      auto minType =
          bitwidth > 0 ? rewriter.getIntegerType(bitwidth) : v.getType();
      auto vType = cast<RankedTensorType>(v.getType());
      auto newVEncoding = DotOperandEncodingAttr::get(
          v.getContext(), opIdx, newRetType.getEncoding(), minType);
      auto newVType = RankedTensorType::get(
          vType.getShape(), vType.getElementType(), newVEncoding);
      return rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);
    };

    // convert operands
    int minBitwidth = std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));

    a = getDotOperand(a, 0, minBitwidth);
    b = getDotOperand(b, 1, minBitwidth);
    ScaleDotElemType aElemType = dotOp.getAElemType();
    ScaleDotElemType bElemType = dotOp.getBElemType();

    MLIRContext *ctx = dotOp.getContext();

    auto aShape = cast<RankedTensorType>(a.getType()).getShape();
    auto bShape = cast<RankedTensorType>(b.getType()).getShape();

    int m = aShape[0];
    int k = aShape[1];
    int n = aShape[1];
    if (m < 16 || n < 16 || k < 32)
      return failure();

    auto aEncLL = LinearLayout::empty();
    auto bEncLL = LinearLayout::empty();

    StringAttr kReg = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");

    // fp4 kwidth is 32 / 4 = 8
    unsigned kwidth = 8;
    auto convertInputLayout = [&](unsigned opIdx) {
      auto newEnc = DotOperandEncodingAttr::get(ctx, opIdx, mmaEnc, kwidth / 2);

      (opIdx == 0 ? aEncLL : bEncLL) *=
          newEnc.toLinearLayout(opIdx == 0 ? aShape : bShape);
    };
    convertInputLayout(0);
    convertInputLayout(1);

    std::vector<std::vector<int>> basesLane = {
        {0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}};
    std::vector<std::vector<int>> basesReg = {{0, 1}, {8, 0}};
    StringAttr kOuter = StringAttr::get(ctx, "dim0");
    StringAttr kInner = StringAttr::get(ctx, "dim1");

    auto convertScale = [&](LinearLayout &dotLL,
                            TypedValue<RankedTensorType> &scale,
                            unsigned OpIdx) {
      auto shape = cast<RankedTensorType>(scale.getType()).getShape();

      basesLane[0] = {16 * int(warpsPerTile[OpIdx]) < shape[0]
                          ? 16 * int(warpsPerTile[OpIdx])
                          : 0,
                      0};
      basesLane[1] = {32 * int(warpsPerTile[OpIdx]) < shape[0]
                          ? 32 * int(warpsPerTile[OpIdx])
                          : 0,
                      0};
      LinearLayout::BasesT scaleBases = dotLL.getBases();
      auto &warpBases = scaleBases[kWarp];

      if (OpIdx == 1) {
        for (auto &basis : warpBases) {
          std::reverse(basis.begin(), basis.end());
        }
      }

      auto newLL = LinearLayout({{kReg, basesReg},
                                 {kLane, basesLane},
                                 {kWarp, warpBases},
                                 {kBlock, {}}},
                                {kOuter, kInner});

      // Adjust register-level layout to fill the shape
      SmallVector<int, 2> repOrder = {1, 0};
      auto dotOperandShape = scale.getType().getShape();
      SmallVector<StringAttr> standardOutDims =
          standardOutDimNames(ctx, dotOperandShape.size());
      for (auto d : repOrder) {
        auto outDim = standardOutDims[d];
        auto dimSize = newLL.getOutDimSize(outDim);
        newLL *= LinearLayout::identity1D(dotOperandShape[d] / dimSize, kReg,
                                          outDim);
      }
      Attribute newScaleEncoding = LinearEncodingAttr::get(ctx, newLL);

      auto newScaleType = RankedTensorType::get(
          scale.getType().getShape(), scale.getType().getElementType(),
          newScaleEncoding);
      auto newScale =
          rewriter.create<ConvertLayoutOp>(scale.getLoc(), newScaleType, scale);
      return newScale;
    };

    auto newAScale = convertScale(aEncLL, aScale, 0);
    auto newBScale = convertScale(bEncLL, bScale, 1);

    DotScaledOp newScaledDot = rewriter.create<triton::DotScaledOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, newAScale, newBScale,
        aElemType, bElemType, dotOp.getFastMath());
    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                 newScaledDot->getResult(0));
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type operandElType =
      cast<RankedTensorType>(operand.getType()).getElementType();
  if (type::isFloat8(operandElType)) {
    return FpToFpOp::create(builder, loc, tensorPromotedType, operand);
  }
  return arith::ExtFOp::create(builder, loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    PPUMmaEncodingAttr mmaLayout =
        dyn_cast<PPUMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      // promote to f16 unless there's hardware support for fp8 operands
      if (!isNativeFP8 || (isNativeFP8 && (mmaLayout.getVersionMajor() == 2)))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

// Transpose scaled_dot ops that have a scale on lhs.
static void transposeDotOp(DotScaledOp dotOp) {
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  Value lhsTransposed = TransOp::create(builder, lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  Value rhsTransposed = TransOp::create(builder, rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  Value cTransposed = TransOp::create(builder, c.getLoc(), c, transOrder);
  Value result = DotScaledOp::create(
      builder, dotOp.getLoc(), cTransposed.getType(), rhsTransposed,
      lhsTransposed, cTransposed, dotOp.getBScale(), dotOp.getAScale(),
      dotOp.getBElemType(), dotOp.getAElemType(), dotOp.getFastMath());
  Operation *transposedResult =
      TransOp::create(builder, result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transposedResult);
  dotOp.erase();
}

static void transposeDots(ModuleOp m) {
  // TODO: extend to regular dot when it is profitable. For instance when we may
  // want to use rhs from register for mmav3.
  SmallVector<DotScaledOp> toTranspose;
  m.walk([&](DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  for (DotScaledOp dotOp : toTranspose) {
    transposeDotOp(dotOp);
  }
}

} // namespace gpu
} // namespace triton

#define GEN_PASS_DEF_TRITONPPUGPUACCELERATEMATMUL
#include "TritonPPUGPUTransforms/Passes.h.inc"

class TritonPPUGPUAccelerateMatmulPass
    : public impl::TritonPPUGPUAccelerateMatmulBase<
          TritonPPUGPUAccelerateMatmulPass> {
public:
  using impl::TritonPPUGPUAccelerateMatmulBase<
      TritonPPUGPUAccelerateMatmulPass>::TritonPPUGPUAccelerateMatmulBase;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    auto computeCapability = getPPUComputeCapability(m);
    // We could do this generically if we manage to improve the heuristics
    // reverted in these two PRs https://github.com/triton-lang/triton/pull/5834
    // https://github.com/triton-lang/triton/pull/5837
    mlir::triton::gpu::transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    constexpr int benefitScaledMMAv2 = 10;

    patterns.add<mlir::triton::gpu::BlockedToMMA>(context, computeCapability,
                                                  benefitDefault);
    mlir::triton::gpu::populateDecomposeScaledBlockedPatterns(patterns,
                                                              benefitDefault);
    patterns.add<mlir::triton::gpu::ScaledBlockedToMMAv2>(
        context, computeCapability, benefitScaledMMAv2);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    mlir::triton::gpu::decomposeMixedModeDotOp(m, computeCapability);
  }
};

} // namespace mlir
