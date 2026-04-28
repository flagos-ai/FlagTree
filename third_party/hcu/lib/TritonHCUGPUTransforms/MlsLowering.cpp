#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Utility.h"

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
    auto matrixTy = cast<RankedTensorType>(matrixOp.getType());
    auto mlsAttr = matrixOp->getAttrOfType<tta::MlsEncodingAttr>(tta::MlsEncodingAttr::getMnemonic());
    assert(mlsAttr && "mlsAttr should not be null!");

    // 1. create a local alloc op
    auto loc = matrixOp.getLoc();
    rewriter.setInsertionPoint(matrixOp);
    auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(matrixTy.getContext());
    auto ctaLayout = ttg::getCTALayout(matrixTy.getEncoding());
    auto mlsSharedEncoding = ttg::HCUMlsSharedEncodingAttr::get(
                                  matrixOp.getContext(), mlsAttr.getOpIdx(), mlsAttr.getMlsTile(),
                                  mlsAttr.getElemBitWidth(), mlsAttr.getAlt2Kind(),
                                  mlsAttr.getVersion(), mlsAttr.getOrder(),
                                  ctaLayout);
    auto sharedMemDescType = ttg::MemDescType::get(
                                  matrixTy.getShape(),
                                  matrixTy.getElementType(),
                                  mlsSharedEncoding,
                                  sharedMemorySpace,
                                  /*mutableMemory=*/true);
    auto localAllocOp = rewriter.create<ttg::LocalAllocOp>(loc, sharedMemDescType);


    // 2. create matrix_load_to_local load op and insert synchronization primitives
    auto matrixLoadToLocalOp = rewriter.create<tta::MatrixLoadToLocalOp>(
        loc, matrixOp.getBase(), matrixOp.getShape(), matrixOp.getStrides(),
        matrixOp.getTensorShape(), matrixOp.getIndices(), matrixOp.getBoundaryCheck(), matrixOp.getCache(),
        matrixOp.getEvict(), matrixOp.getIsVolatile(), localAllocOp);
    matrixLoadToLocalOp->setAttr(tta::MlsEncodingAttr::getMnemonic(), mlsAttr);

    auto commitOp = rewriter.create<ttg::AsyncCommitGroupOp>(loc,
                                                             matrixLoadToLocalOp->getResult(0));
    ttg::AsyncWaitOp waitOp = rewriter.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

    // 3. create local load op
    // note: if enter here, meaning num_stages = 1 and cur don't need local_load with token due to
    // we need MembarPass to insert gpu_barrier to handle WAR to make sure local_load finished.
    // use Encoding of convert_layout, but elementType need be same with matrixOp
    auto localLoadResultType = RankedTensorType::get(
        matrixTy.getShape(),
        matrixTy.getElementType(),
        matrixTy.getEncoding());

    auto localLoadOp = rewriter.create<ttg::LocalLoadOp>(
        matrixOp.getLoc(), localLoadResultType, localAllocOp/*, waitOp*/);

    // 4. replace the original op
    rewriter.replaceOp(matrixOp, localLoadOp.getResult());

    return success();
  }
};


/*
 * print(cvt(dot encoding, blocked))) -> print(dot encoding)
 * this can save lds use when debug and try not affect the behavior(code gen) of the kernel.
 **/
class MlsPrintOpCanonicalization : public OpRewritePattern<tt::PrintOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::PrintOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getArgs().size() != 1)
      return failure();
    auto convert = op.getArgs().front().getDefiningOp<ttg::ConvertLayoutOp>();
    if (!convert || !isa<DotOperandEncodingAttr>(convert.getSrc().getType().getEncoding()))
      return failure();
    rewriter.replaceOpWithNewOp<triton::PrintOp>(op,
                                                 op.getPrefix(),
                                                 op.getHex(),
                                                 convert.getSrc(),
                                                 op.getIsSigned());
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
class MlsConvertOpCanonicalization : public OpRewritePattern<ttg::ConvertLayoutOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ttg::ConvertLayoutOp op,
                                  PatternRewriter &rewriter) const override {
      auto srcType = op.getSrc().getType();
      auto dstType = op.getType();
      if (!isa<DotOperandEncodingAttr>(dstType.getEncoding()) || !dstType.getElementType().isInteger(1))
        return failure();

      auto loc = op.getLoc();
      rewriter.setInsertionPoint(op);

      auto convertedSrcType = RankedTensorType::get(srcType.getShape(),
                                                      type::i8Ty(op.getContext()), srcType.getEncoding());
      auto convertedDstType = RankedTensorType::get(dstType.getShape(),
                                                      type::i8Ty(op.getContext()), dstType.getEncoding());
      auto extuiOp = rewriter.create<arith::ExtUIOp>(loc, convertedSrcType, op.getSrc());
      auto convertOp = rewriter.create<ttg::ConvertLayoutOp>(loc, convertedDstType, extuiOp.getResult());
      auto truncOp = rewriter.create<arith::TruncIOp>(loc, dstType, convertOp.getResult());
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
    patterns.add<MlsMatrixLoadLowering,
                 MlsConvertOpCanonicalization,
                 MlsPrintOpCanonicalization>( context);
    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
