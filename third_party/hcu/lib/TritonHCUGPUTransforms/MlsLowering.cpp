#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Utility.h"
#include "MatrixDescriptorLayout.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tta = mlir::triton::hcugpu;

namespace {

// ===----------------------------------------------------------------------===//
// Utility functions
// ===----------------------------------------------------------------------===//

/*
 * lowering:
 *     %48 = matrix_load %47
 *     %49 = xxx_ops %48
 *
 * to:
 *     %40 = ttg.local_alloc %39
 *     %token = rocl.matrix_load_to_local %47, %40
 *     %commit = ttg.async_commit_group %token
 *     %wait = ttg.async_wait %commit
 *     %50 = ttg.local_load %40
 *
 **/
class MlsMatrixLoadLowering : public OpRewritePattern<tt::MatrixLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::MatrixLoadOp matrixOp,
                                PatternRewriter &rewriter) const override {
    if (matrixOp->hasAttr("hcu.structured_register"))
      return failure();
    auto matrixTy = cast<RankedTensorType>(matrixOp.getType());
    auto mlsAttr = matrixOp->getAttrOfType<tta::MlsEncodingAttr>(
        tta::MlsEncodingAttr::getMnemonic());
    auto descriptorShared = triton::HCU::getMatrixDescriptorSharedLayout(matrixOp);
    if (!mlsAttr && !descriptorShared)
      return rewriter.notifyMatchFailure(matrixOp, "no supported matrix shared chart");

    // 1. create a local alloc op
    auto loc = matrixOp.getLoc();
    rewriter.setInsertionPoint(matrixOp);
    auto sharedMemorySpace =
        ttg::SharedMemorySpaceAttr::get(matrixTy.getContext());
    auto ctaLayout = ttg::getCTALayout(matrixTy.getEncoding());
    Attribute sharedEncoding = descriptorShared;
    if (!sharedEncoding)
      sharedEncoding = ttg::HCUMlsSharedEncodingAttr::get(
          matrixOp.getContext(), mlsAttr.getOpIdx(), mlsAttr.getMlsTile(),
          mlsAttr.getElemBitWidth(), mlsAttr.getAlt2Kind(), mlsAttr.getVersion(),
          mlsAttr.getOrder(), ctaLayout);
    auto sharedMemDescType =
        ttg::MemDescType::get(matrixTy.getShape(), matrixTy.getElementType(),
                              sharedEncoding, sharedMemorySpace,
                              /*mutableMemory=*/true);
    auto localAllocOp =
        rewriter.create<ttg::LocalAllocOp>(loc, sharedMemDescType);
    Value destination = localAllocOp.getResult();
    // 2. create matrix_load_to_local load op and insert synchronization
    // primitives
    SmallVector<Value> indices(matrixOp.getIndices());
    SmallVector<int32_t> boundary(matrixOp.getBoundaryCheck());
    if (descriptorShared) {
      indices.push_back(rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
      boundary.clear();
    }
    auto matrixLoadToLocalOp = rewriter.create<tta::MatrixLoadToLocalOp>(
        loc, matrixOp.getBase(), matrixOp.getShape(), matrixOp.getStrides(),
        matrixOp.getTensorShape(), indices,
        boundary, matrixOp.getCache(), matrixOp.getEvict(),
        matrixOp.getIsVolatile(), destination);
    if (mlsAttr)
      matrixLoadToLocalOp->setAttr(tta::MlsEncodingAttr::getMnemonic(), mlsAttr);

    auto commitOp = rewriter.create<ttg::AsyncCommitGroupOp>(
        loc, matrixLoadToLocalOp->getResult(0));
    ttg::AsyncWaitOp waitOp =
        rewriter.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

    // 3. create local load op
    // Completion and retirement are distinct. Keep the explicit wait token
    // for RAW and LLVM alias analysis; Membar still retains read->copy WAR.
    auto localLoadResultType = RankedTensorType::get(
        matrixTy.getShape(), matrixTy.getElementType(), matrixTy.getEncoding());

    auto localLoadOp = rewriter.create<ttg::LocalLoadOp>(
        matrixOp.getLoc(), localLoadResultType, destination,
        waitOp.getResult());

    // 4. replace the original op
    rewriter.replaceOp(matrixOp, localLoadOp.getResult());

    return success();
  }
};

// Follow value-only layout views; they do not change the identity of the
// loop-carried tile. No launch shape, stage count or memory-layout ABI is used.
static Value stripTensorViews(Value value) {
  while (auto *op = value.getDefiningOp()) {
    if (!isa<ttg::ConvertLayoutOp, tt::ReshapeOp, tt::TransOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

// Pipelining carries an already-read tile into the current dot and reads its
// replacement for the next iteration. Keep that replacement read ahead of the
// independent current dot, rather than serializing both on one register set.
// Move only the pure dot: all memory accesses, completion tokens and barriers
// stay in their original order. A later use or a synchronization boundary
// prevents motion. No additional hardware or scheduling fence is necessary.
static void orderPrefetchedLocalReads(ModuleOp module) {
  auto target = module->getAttrOfType<StringAttr>("ttg.target");
  auto prefetch = module->getAttrOfType<IntegerAttr>("ttg.hcu.local_prefetch");
  if (!target || target.getValue() != "hip:gfx936" || !prefetch ||
      prefetch.getInt() <= 0)
    return;
  module.walk([](scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    auto dots = llvm::to_vector(loop.getBody()->getOps<tt::DotOp>());
    // Process accumulator chains from their last user backward.
    for (auto dot : llvm::reverse(dots)) {
      Operation *lastRead = nullptr;
      for (Value input : {dot.getA(), dot.getB()}) {
        auto argument = dyn_cast<BlockArgument>(stripTensorViews(input));
        if (!argument || argument.getOwner() != loop.getBody() ||
            argument.getArgNumber() == 0)
          continue;
        Value next = yield.getOperand(argument.getArgNumber() - 1);
        auto read = stripTensorViews(next).getDefiningOp<ttg::LocalLoadOp>();
        if (read && read->getBlock() == loop.getBody() &&
            (!lastRead || lastRead->isBeforeInBlock(read)))
          lastRead = read;
      }
      if (!lastRead || !dot->isBeforeInBlock(lastRead) ||
          !isMemoryEffectFree(dot))
        continue;
      bool legal = llvm::all_of(dot->getUsers(), [&](Operation *user) {
        auto *ancestor = loop.getBody()->findAncestorOpInBlock(*user);
        return ancestor && lastRead->isBeforeInBlock(ancestor);
      });
      for (auto *op = dot->getNextNode(); legal && op != lastRead;
           op = op->getNextNode()) {
        // Preserve existing synchronization/priority fences, calls and
        // control-flow boundaries. Only known pipeline memory ops may lie
        // between the old dot and its prefetched replacement tile.
        legal = op->getNumRegions() == 0 &&
                (isMemoryEffectFree(op) ||
                 isa<ttg::LocalLoadOp, ttg::LocalStoreOp,
                     ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp,
                     tta::MatrixLoadToLocalOp, tta::BufferLoadToLocalOp>(op));
      }
      if (legal)
        dot->moveAfter(lastRead);
    }
  });
}

// Fold a single-user chain of pure tensor views into its shared read. The
// existing layout inference proves each inverse view; the original memory
// operation and its completion token stay at the same program point. This is
// also needed for pipeline-carried tail values, where a view conversion can otherwise
// allocate an extra shared-memory transpose that the steady loop avoids.
class FoldLocalLoadViewConversion
    : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp convert,
                                PatternRewriter &rewriter) const override {
    Attribute encoding = convert.getType().getEncoding();
    if (!isa<ttg::DistributedEncodingTrait>(encoding))
      return failure();
    SmallVector<std::pair<Operation *, RankedTensorType>> views;
    Value value = convert.getSrc();
    while (auto *view = value.getDefiningOp()) {
      if (!isa<tt::ReshapeOp, tt::TransOp>(view))
        break;
      if (!value.hasOneUse())
        return failure();
      auto type = cast<RankedTensorType>(value.getType());
      views.emplace_back(view, type.cloneWithEncoding(encoding));
      encoding = mlir::inferSrcEncoding(view, encoding);
      if (!encoding)
        return failure();
      value = view->getOperand(0);
    }
    if (views.empty() || !value.hasOneUse() ||
        !isa<ttg::DistributedEncodingTrait>(encoding))
      return failure();
    auto eligibleRead = [](Value input) {
      auto load = input.getDefiningOp<ttg::LocalLoadOp>();
      return load && input.hasOneUse() &&
             isa<ttg::SharedLinearEncodingAttr>(
                 load.getSrc().getType().getEncoding());
    };
    SmallVector<Value> reads;
    scf::ForOp loop;
    unsigned resultIndex = 0;
    if (eligibleRead(value)) {
      reads.push_back(value);
    } else if (auto result = dyn_cast<OpResult>(value)) {
      loop = dyn_cast<scf::ForOp>(result.getOwner());
      resultIndex = result.getResultNumber();
      // A pipeline can carry a prefetched tile only for its peeled tail.
      // Its body argument is unused, so retargeting both possible reaching
      // reads also covers zero iterations without changing a body consumer.
      if (!loop || !loop.getRegionIterArgs()[resultIndex].use_empty())
        return failure();
      Value initial = loop.getInitArgs()[resultIndex];
      Value yielded = cast<scf::YieldOp>(loop.getBody()->getTerminator())
                          .getOperand(resultIndex);
      if (!eligibleRead(initial) || !eligibleRead(yielded))
        return failure();
      reads = {initial, yielded};
    } else {
      return failure();
    }
    auto type = cast<RankedTensorType>(value.getType()).cloneWithEncoding(encoding);
    for (Value read : reads)
      rewriter.modifyOpInPlace(read.getDefiningOp(), [&] { read.setType(type); });
    if (loop)
      rewriter.modifyOpInPlace(loop, [&] {
        loop.getRegionIterArgs()[resultIndex].setType(type);
        loop.getResult(resultIndex).setType(type);
      });
    for (auto &[view, resultType] : llvm::reverse(views))
      rewriter.modifyOpInPlace(view, [&] { view->getResult(0).setType(resultType); });
    rewriter.replaceOp(convert, convert.getSrc());
    return success();
  }
};

/*
 * print(cvt(dot encoding, blocked))) -> print(dot encoding)
 * this can save lds use when debug and try not affect the behavior(code gen) of
 *the kernel.
 **/
class MlsPrintOpCanonicalization : public OpRewritePattern<tt::PrintOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::PrintOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getArgs().size() != 1)
      return failure();
    auto convert = op.getArgs().front().getDefiningOp<ttg::ConvertLayoutOp>();
    if (!convert ||
        !isa<DotOperandEncodingAttr>(convert.getSrc().getType().getEncoding()))
      return failure();
    rewriter.replaceOpWithNewOp<triton::PrintOp>(
        op, op.getPrefix(), op.getHex(), convert.getSrc(), op.getIsSigned());
    return success();
  }
};

/*
 * lowering:
 *     %48 = convert_layout %47 : i1ty blocked -> i1ty dot
 *
 * to:
 *     %extuiOp = extui %47 : i1ty blocked -> i8ty blocked
 *     %48      = convert_layout %47 : i8ty blocked -> i8ty dot
 *     %truncOp = trunc %48 : i8ty dot -> i1ty dot
 **/
class MlsConvertOpCanonicalization
    : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (!isa<DotOperandEncodingAttr>(dstType.getEncoding()) ||
        !dstType.getElementType().isInteger(1))
      return failure();

    auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    auto convertedSrcType = RankedTensorType::get(
        srcType.getShape(), type::i8Ty(op.getContext()), srcType.getEncoding());
    auto convertedDstType = RankedTensorType::get(
        dstType.getShape(), type::i8Ty(op.getContext()), dstType.getEncoding());
    auto extuiOp =
        rewriter.create<arith::ExtUIOp>(loc, convertedSrcType, op.getSrc());
    auto convertOp = rewriter.create<ttg::ConvertLayoutOp>(
        loc, convertedDstType, extuiOp.getResult());
    auto truncOp =
        rewriter.create<arith::TruncIOp>(loc, dstType, convertOp.getResult());
    rewriter.replaceOp(op, truncOp.getResult());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUMLSLOWERING
#include "TritonHCUGPUTransforms/Passes.h.inc"

class TritonHCUGPUMlsLoweringPass
    : public impl::TritonHCUGPUMlsLoweringBase<TritonHCUGPUMlsLoweringPass> {
public:
  using impl::TritonHCUGPUMlsLoweringBase<
      TritonHCUGPUMlsLoweringPass>::TritonHCUGPUMlsLoweringBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::RewritePatternSet patterns(context);
    patterns.add<MlsMatrixLoadLowering, MlsConvertOpCanonicalization,
                 MlsPrintOpCanonicalization, FoldLocalLoadViewConversion>(context);
    if (applyPatternsGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }
    orderPrefetchedLocalReads(mod);
  }
};

} // namespace mlir
