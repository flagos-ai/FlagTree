/**
 * Copyright 2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Utility.h"

#define DEBUG_TYPE "triton-accelerate-matmul"

// namespace mlir {
// namespace triton {
// namespace gpu {

namespace mlir {
#define GEN_PASS_DEF_TRITONGCUACCELERATEMATMUL
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// Match the fixed pattern: triton_gcu.load -> ttg.convert_layout -> tt.dot
// Returns the triton_gcu.load op if |v| is produced by a ConvertLayoutOp
// whose source is a triton_gcu::LoadOp; nullptr otherwise.
static Operation *traceBackToLoadOp(Value v) {
  auto cvt = v.getDefiningOp<ConvertLayoutOp>();
  if (!cvt)
    return nullptr;
  auto *srcOp = cvt.getSrc().getDefiningOp();
  if (srcOp && isa<triton::gcu::LoadOp>(srcOp))
    return srcOp;
  return nullptr;
}

static bool isGCU400OrGCU410Target(Operation *op) {
  auto targetAttr = op->getAttrOfType<StringAttr>("ttg.target");
  if (!targetAttr) {
    if (auto module = op->getParentOfType<mlir::ModuleOp>())
      targetAttr = module->getAttrOfType<StringAttr>("ttg.target");
  }
  if (!targetAttr)
    return false;

  StringRef target = targetAttr.getValue();
  return target.contains("gcu400") || target.contains("gcu410");
}

SmallVector<unsigned, 2>
warpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
             const SmallVector<unsigned, 3> &instrShape) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};
  assert(rank == 2 && "expected 2D tile shape");

  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  // Contains a chained dot. We prefer to assign warps to one axis
  // to facilitate use cases like flash attention, allowing reductions within
  // the same warp.
  if (llvm::find_if(slices, [](Operation *op) {
        return isa<mlir::triton::DotOpInterface>(op);
      }) != slices.end())
    return {(unsigned)numWarps, 1};

  SmallVector<int64_t> shapePerWarp = {64, 128};
  SmallVector<int64_t> warps = {1, 1};

  auto ceilDiv = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  auto product = [](const SmallVector<int64_t> &v) { return v[0] * v[1]; };

  // Compute repM and repN
  SmallVector<int64_t> reps = {ceilDiv(shape[0], shapePerWarp[0]),
                               ceilDiv(shape[1], shapePerWarp[1])};
  // The formula for the number of registers given the reps is
  // repM * 4 * repK + repN * 2 * repK + regsC
  // where regsC = repM * repN * 4, which does not depend on the warp shape.
  //
  // As such, to minimize the register pressure, we need to balance
  // repM and repN. We then untie towards M, as the lhs tile has 4 elements,
  // and the rhs tile has just 2.
  while (product(warps) < numWarps) {
    if (reps[0] >= reps[1]) {
      warps[0] *= 2;
      // Too many warps for this mma (repM == repN == 1).
      // We allocate the remaining warps to the left (arbitrary choice).
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

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                                       int opIdx, bool allowTranspose) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrderForMemory(argType);

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), argType.getShape(), newOrder, CTALayout,
      argType.getElementType(), false);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);
}

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
public:
  explicit BlockedToMMA(mlir::MLIRContext *context)
      : OpRewritePattern<DotOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isGCU400OrGCU410Target(dotOp))
      return failure();

    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    int versionMajor = 3;

    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, dotOp.getA().getType().getElementType(),
        numWarps);
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    // Check each operand independently for the fixed pattern:
    //   triton_gcu.load -> ttg.convert_layout
    // Only operands matching this pattern will be converted to SharedMemory.
    // If neither operand matches, bail out entirely.
    bool aMatchesPattern = traceBackToLoadOp(a) != nullptr;
    bool bMatchesPattern = traceBackToLoadOp(b) != nullptr;
    LLVM_DEBUG(llvm::dbgs() << "BlockedToMMA: operand A "
                            << (aMatchesPattern ? "matches" : "does NOT match")
                            << " triton_gcu.load -> convert_layout pattern\n");
    LLVM_DEBUG(llvm::dbgs() << "BlockedToMMA: operand B "
                            << (bMatchesPattern ? "matches" : "does NOT match")
                            << " triton_gcu.load -> convert_layout pattern\n");
    if (!aMatchesPattern && !bMatchesPattern) {
      LLVM_DEBUG(llvm::dbgs()
                 << "BlockedToMMA: skip dot because neither operand matches "
                    "triton_gcu.load -> convert_layout pattern\n");
      return failure();
    }

    int versionMinor = 0;
    auto warpsPerTileShape =
        warpsPerTile(dotOp, retShapePerCTA, numWarps, instrShape);

    auto mod = dotOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    Attribute mmaEnc;
    if (threadsPerWarp == 32) {
      mmaEnc = NvidiaMmaEncodingAttr::get(oldRetType.getContext(), versionMajor,
                                          versionMinor, warpsPerTileShape,
                                          CTALayout, instrShape);
    } else {
      unsigned rank = oldRetType.getRank();
      SmallVector<unsigned> sizePerThread(rank, 1);
      SmallVector<unsigned> threadsPerWarpVec(rank, 1);
      threadsPerWarpVec[rank - 1] = threadsPerWarp;
      SmallVector<unsigned> order(rank);
      for (unsigned i = 0; i < rank; ++i)
        order[i] = rank - 1 - i;
      mmaEnc = BlockedEncodingAttr::get(oldRetType.getContext(), sizePerThread,
                                        threadsPerWarpVec, warpsPerTileShape,
                                        order, CTALayout);
    }

    // PatternRewriterWithAsyncTaskIds taskIdRewriter(rewriter, dotOp);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    auto eltType = dotOp.getA().getType().getElementType();

    bool allowTranspose = eltType.isF16() || eltType.isBF16();
    auto convertToDotOpEncoding = [&](Value operand, int opIdx) -> Value {
      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto dotOpEnc = DotOperandEncodingAttr::get(
          dotOp.getContext(), opIdx, mmaEnc, tensorTy.getElementType());
      auto newTy = RankedTensorType::get(tensorTy.getShape(),
                                         tensorTy.getElementType(), dotOpEnc);
      if (tensorTy == newTy)
        return operand;
      return rewriter.create<ConvertLayoutOp>(dotOp.getLoc(), newTy, operand);
    };
    if (aMatchesPattern)
      a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
    else
      a = convertToDotOpEncoding(a, 0);
    if (bMatchesPattern)
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
    else
      b = convertToDotOpEncoding(b, 1);
    auto newDot = rewriter.create<triton::gcu::WarpGroupDotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);

    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType,
                                                 newDot->getResult(0));
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
} // namespace

// Check if GCU400/410 hardware natively supports the given input/output
// element type combination for matmul.
// Based on MATMUL_IMPL instantiations in builtins_gcu4xx_base.h.inc.
static bool isGCU4xxNativeMatmulTypes(Type inputElType, Type outputElType) {
  if (inputElType.isF32() && outputElType.isF32())
    return true;
  if (inputElType.isF16() && (outputElType.isF16() || outputElType.isF32()))
    return true;
  if (inputElType.isBF16() && outputElType.isF32())
    return true;
  if (inputElType.isInteger(8) &&
      (outputElType.isInteger(8) || outputElType.isInteger(32)))
    return true;
  if (llvm::isa<Float8E4M3FNType>(inputElType) && outputElType.isF32())
    return true;
  if (llvm::isa<Float8E5M2Type>(inputElType) && outputElType.isF32())
    return true;
  return false;
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(mlir::gpu::GPUModuleOp mod) {
  const bool isGCU400OrGCU410 = isGCU400OrGCU410Target(mod);
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type DElType = D.getType().getElementType();

    // GCU400/410 natively supports mixed-precision matmul for many type
    // combinations. Skip promotion when the hardware can handle it directly.
    if (isGCU400OrGCU410 && isGCU4xxNativeMatmulTypes(AElType, DElType))
      return;

    Type promoteType;
    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      if (!isNativeFP8)
        return;
      promoteType = builder.getF16Type();
    } else {
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    auto promoteWithCvtAware = [&](Value operand, unsigned opIdx) {
      if (auto cvt = operand.getDefiningOp<ConvertLayoutOp>()) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(cvt);
        Value promotedSrc =
            promoteOperand(builder, cvt.getLoc(), cvt.getSrc(), promoteType);
        auto newCvtType = cast<RankedTensorType>(cvt.getType())
                              .cloneWith(std::nullopt, promoteType);
        Value newCvt = builder.create<ConvertLayoutOp>(cvt.getLoc(), newCvtType,
                                                       promotedSrc);
        dotOp.setOperand(opIdx, newCvt);
      } else {
        Value promoted = promoteOperand(builder, loc, operand, promoteType);
        dotOp.setOperand(opIdx, promoted);
      }
    };
    promoteWithCvtAware(dotOp.getA(), 0);
    promoteWithCvtAware(dotOp.getB(), 1);
  });
}

namespace {
class TritonGCUAccelerateMatmulPass
    : public mlir::impl::TritonGCUAccelerateMatmulBase<
          TritonGCUAccelerateMatmulPass> {
public:
  using mlir::impl::TritonGCUAccelerateMatmulBase<
      TritonGCUAccelerateMatmulPass>::TritonGCUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::gpu::GPUModuleOp m = getOperation();
    // skip if num_warps is 1
    auto builtinModule = m->getParentOfType<ModuleOp>();
    if (builtinModule && builtinModule->hasAttr("ttg.num-warps")) {
      auto numWarps =
          cast<IntegerAttr>(builtinModule->getAttr("ttg.num-warps")).getInt();
      if (numWarps == 1)
        return;
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<BlockedToMMA>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m);
  }
};
} // namespace

// } // namespace gpu
// } // namespace triton
// } // namespace mlir
