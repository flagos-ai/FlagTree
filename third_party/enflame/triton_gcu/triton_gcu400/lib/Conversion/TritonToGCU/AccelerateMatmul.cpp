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

#include "Utils/TritonVersionCompat.h"
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

#include "Constants.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

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

// Validate that all users of a LoadOp result follow the SAME pattern to DotOp.
// Only two pure patterns are accepted:
//   Pattern A (all ConvertLayoutOp): LoadOp -> ConvertLayoutOp -> DotOp
//   Pattern B (all TransOp):        LoadOp -> TransOp -> ConvertLayoutOp ->
//   DotOp
// Mixed patterns (some A, some B) are REJECTED.
// LocalAllocOp users (created by getSharedMemoryMMAOperand during prior dot
// rewrites) are valid and do not affect the pattern classification.
static bool allUsersLeadToDotOp(Value loadResult) {
  // First pass: classify users — only ConvertLayoutOp and TransOp determine
  // the pattern. LocalAllocOp is a valid byproduct of prior rewrites and
  // does not affect the classification.
  // Dead ops (use_empty()) are skipped — they are leftovers from prior
  // dot rewrites whose ConvertLayoutOp/TransOp results are no longer consumed.
  bool hasConvertLayoutUsers = false;
  bool hasTransUsers = false;

  for (auto *user : loadResult.getUsers()) {
    if (isa<ConvertLayoutOp>(user)) {
      if (!cast<ConvertLayoutOp>(user).getResult().use_empty())
        hasConvertLayoutUsers = true;
    } else if (isa<triton::TransOp>(user)) {
      if (!cast<triton::TransOp>(user).getResult().use_empty())
        hasTransUsers = true;
    } else if (!isa<triton::gpu::LocalAllocOp>(user)) {
      // Reject any user that is not ConvertLayoutOp, TransOp, or LocalAllocOp
      return false;
    }
  }

  // Mixed patterns: some users are ConvertLayoutOp, some are TransOp — reject
  if (hasConvertLayoutUsers && hasTransUsers)
    return false;

  // Second pass: validate each user according to the determined mode.
  // LocalAllocOp users are valid without further validation — they were
  // created by getSharedMemoryMMAOperand to materialize shared memory.
  if (hasConvertLayoutUsers) {
    // Pure Pattern A: every live ConvertLayoutOp user must feed ONLY DotOps.
    // Dead ConvertLayoutOps (use_empty) are expected leftovers from prior dot
    // rewrites in the greedy rewriter. They must be SKIPPED — rejecting them
    // would incorrectly block subsequent dots sharing this LoadOp.
    for (auto *user : loadResult.getUsers()) {
      auto cvtOp = dyn_cast<ConvertLayoutOp>(user);
      if (!cvtOp)
        continue; // LocalAllocOp or dead TransOp — valid, no further check
                  // needed
      if (cvtOp.getResult().use_empty())
        continue; // Dead — leftover from prior dot rewrite, no further check
                  // needed
      // Every user of a live ConvertLayoutOp must be a DotOp.
      // LocalAllocOp is never a user of ConvertLayoutOp in Pattern A
      // (getSharedMemoryMMAOperand creates it on the LoadOp, not on
      // ConvertLayoutOp).
      if (!llvm::all_of(cvtOp.getResult().getUsers(), [](Operation *u) {
            return isa<mlir::triton::DotOp>(u);
          }))
        return false;
    }
  } else {
    // Pure Pattern B: every live TransOp user must feed ConvertLayoutOp ->
    // DotOp. Non-TransOp users of the LoadOp result are rejected — in Pattern
    // B, getSharedMemoryMMAOperand creates LocalAllocOp on the TransOp result
    // (not the LoadOp result), so LocalAllocOp cannot appear here.
    // Dead TransOps (use_empty) are leftovers from prior dot rewrites.
    for (auto *user : loadResult.getUsers()) {
      auto transOp = dyn_cast<triton::TransOp>(user);
      if (!transOp)
        return false; // Unexpected user type in Pattern B — reject
      if (transOp.getResult().use_empty())
        continue; // Dead — leftover from prior dot rewrite, no further check
                  // needed
      bool hasValidUser = false;
      for (auto *transUser : transOp.getResult().getUsers()) {
        auto cvtOp = dyn_cast<ConvertLayoutOp>(transUser);
        if (!cvtOp) {
          if (!isa<triton::gpu::LocalAllocOp>(transUser))
            return false;      // Unexpected user type — reject
          hasValidUser = true; // LocalAllocOp from prior dot rewrite — valid
          continue;
        }
        if (cvtOp.getResult().use_empty())
          continue; // Dead — leftover from prior dot rewrite
        // All users of a live ConvertLayoutOp must be DotOps (consistent with
        // Pattern A)
        if (!llvm::all_of(cvtOp.getResult().getUsers(), [](Operation *u) {
              return isa<mlir::triton::DotOp>(u);
            }))
          return false;
        hasValidUser = true;
      }
      if (!hasValidUser)
        return false;
    }
  }

  return true;
}

// Match patterns leading to tt.dot through ConvertLayoutOp:
//   tt.load -> ttg.convert_layout -> tt.dot
//   tt.load -> tt.trans -> ttg.convert_layout -> tt.dot
// Returns the tt.load op if matched; nullptr otherwise.
// Note: userNumber constraint is intentionally relaxed —
// getSharedMemoryMMAOperand creates a shared-memory copy via local_alloc,
// leaving the original register value available for other consumers. Multiple
// users of the source (e.g. flash-attention patterns where Q/K/V share a load)
// are correctly handled because LocalAllocOp only reads src to initialize the
// new smem buffer without consuming it.
static Operation *traceBackToLoadOp(Value v) {
  auto cvt = v.getDefiningOp<ConvertLayoutOp>();
  if (!cvt)
    return nullptr;

  auto *srcOp = cvt.getSrc().getDefiningOp();
  if (srcOp && isa<triton::LoadOp>(srcOp))
    return srcOp; // Pattern A: tt.load -> ttg.convert_layout -> tt.dot

  if (srcOp && isa<triton::TransOp>(srcOp)) {
    // Pattern B: tt.load -> tt.trans -> ttg.convert_layout -> tt.dot
    auto transOp = cast<triton::TransOp>(srcOp);
    auto *transSrc = transOp.getSrc().getDefiningOp();
    if (transSrc && isa<triton::LoadOp>(transSrc))
      return transSrc;
  }

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
warpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
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
    if (reps[0] > reps[1] || reps[1] == 1) {
      warps[0] *= 2;
      // Too many warps for this mma (repM == repN == 1).
      // We allocate the remaining warps to the left (arbitrary choice).
      if (reps[0] != 1) {
        reps[0] /= 2;
      }
    } else {
      warps[1] *= 2;
      if (reps[1] != 1) {
        reps[1] /= 2;
      }
    }
  }
  return {(unsigned)warps[0], (unsigned)warps[1]};
}

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value getSharedMemoryMMAOperand(Value v,
                                       mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrderForMemory(argType);

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = triton_gcu::compat::getCGALayout(argType.getEncoding());
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), argType.getShape(), newOrder, CTALayout,
      argType.getElementType(), false);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);
}

void setLoadAsyncAttr(Value v, mlir::PatternRewriter &rewriter) {
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  if (auto loadOp = arg.getDefiningOp<triton::LoadOp>())
    loadOp->setAttr(kLoadAsync, rewriter.getBoolAttr(true));
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
    if (oldRetType.getRank() >= 3)
      return failure();

    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    // Skip dot ops that are in a function with num_warps == 1 (e.g. a
    // producer-default TLE WS kernel where the producer uses 1 warp but
    // the consumer worker uses >1 warps). The num_warps here reflects
    // the dot op's enclosing function, not the module-level value.
    if (numWarps <= 1)
      return failure();
    auto CTALayout = triton_gcu::compat::getCGALayout(oldRetType.getEncoding());

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    // Hard rejection: if either operand traces to a LoadOp whose users
    // include both TransOp and non-TransOp (ConvertLayoutOp) paths, the
    // entire dot conversion is aborted — mixed patterns are not supported.
    {
      auto tracesToMixedLoadOp = [](Value v) -> bool {
        auto *loadOp = traceBackToLoadOp(v);
        if (!loadOp)
          return false;
        auto loadResult = loadOp->getResult(0);
        return !loadResult.hasOneUse() && !allUsersLeadToDotOp(loadResult);
      };
      if (tracesToMixedLoadOp(a) || tracesToMixedLoadOp(b))
        return failure();
    }

    // Check each operand independently for the fixed pattern:
    //   tt.load -> ttg.convert_layout
    bool aMatchesPattern = traceBackToLoadOp(a) != nullptr;
    bool bMatchesPattern = traceBackToLoadOp(b) != nullptr;

    setLoadAsyncAttr(a, rewriter);
    setLoadAsyncAttr(b, rewriter);
    // setLoadAsyncAttr(dotOp.getC(), rewriter);
    LLVM_DEBUG(llvm::dbgs() << "BlockedToMMA: operand A "
                            << (aMatchesPattern ? "matches" : "does NOT match")
                            << " tt.load -> convert_layout pattern\n");
    LLVM_DEBUG(llvm::dbgs() << "BlockedToMMA: operand B "
                            << (bMatchesPattern ? "matches" : "does NOT match")
                            << " tt.load -> convert_layout pattern\n");

    // ttg.local_load (TLE WS after pipe lowering)
    // ttg.convert_layout -> ttg.local_load (TLE WS after pipe lowering)
    auto traceBackToLocalLoad = [](Value v) -> bool {
      if (isa_and_nonnull<LocalLoadOp>(v.getDefiningOp()))
        return true;
      if (auto cvt = v.getDefiningOp<ConvertLayoutOp>()) {
        auto src = cvt.getSrc();
        if (isa_and_nonnull<LocalLoadOp>(src.getDefiningOp()))
          return true;
      }
      return false;
    };

    bool aFromLocalLoad = traceBackToLocalLoad(a);
    bool bFromLocalLoad = traceBackToLocalLoad(b);
    LLVM_DEBUG(llvm::dbgs()
               << "BlockedToMMA: operand A " << (aFromLocalLoad ? "" : "not")
               << " from local_load\n");
    LLVM_DEBUG(llvm::dbgs()
               << "BlockedToMMA: operand B " << (bFromLocalLoad ? "" : "not")
               << " from local_load\n");

    if (!(aMatchesPattern || bMatchesPattern || aFromLocalLoad ||
          bFromLocalLoad)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "BlockedToMMA: skip dot because neither operand matches "
                    "tt.load -> convert_layout pattern\n");
      return failure();
    }

    auto warpsPerTileShape = warpsPerTile(dotOp, retShapePerCTA, numWarps);

    auto mod = dotOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    unsigned rank = oldRetType.getRank();
    SmallVector<unsigned> sizePerThread(rank, 1);
    SmallVector<unsigned> threadsPerWarpVec(rank, 1);
    threadsPerWarpVec[rank - 1] = threadsPerWarp;
    SmallVector<unsigned> order(rank);
    for (unsigned i = 0; i < rank; ++i)
      order[i] = rank - 1 - i;
    Attribute mmaEnc = BlockedEncodingAttr::get(
        oldRetType.getContext(), sizePerThread, threadsPerWarpVec,
        warpsPerTileShape, order, CTALayout);

    // PatternRewriterWithAsyncTaskIds taskIdRewriter(rewriter, dotOp);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    auto convertToDotOpEncoding = [&](Value operand, int opIdx,
                                      bool columnMajor) -> Value {
      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto dotOpEnc = DotOperandEncodingAttr::get(
          dotOp.getContext(), opIdx, mmaEnc, /*kWidth=*/0, columnMajor);
      auto newTy = RankedTensorType::get(tensorTy.getShape(),
                                         tensorTy.getElementType(), dotOpEnc);
      if (tensorTy == newTy)
        return operand;
      return rewriter.create<ConvertLayoutOp>(dotOp.getLoc(), newTy, operand);
    };

    bool rhsColMajor = false;
    if (auto origBType = dyn_cast<RankedTensorType>(dotOp.getB().getType())) {
      if (auto origBEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
              origBType.getEncoding()))
        rhsColMajor = origBEnc.getColumnMajor();
    }

    if (aMatchesPattern)
      a = getSharedMemoryMMAOperand(a, rewriter);
    else
      a = convertToDotOpEncoding(a, 0, false);
    if (bMatchesPattern)
      b = getSharedMemoryMMAOperand(b, rewriter);
    else
      b = convertToDotOpEncoding(b, 1, rhsColMajor);
    auto newDot = rewriter.create<triton::gcu::WarpGroupDotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);

    if (rhsColMajor)
      newDot->setAttr(kRhsColumnMajor, rewriter.getUnitAttr());

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

    m.walk([&](DotOp) {
      m->setAttr(kGcuSinkImplicitDef, UnitAttr::get(context));
      return WalkResult::interrupt();
    });

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
