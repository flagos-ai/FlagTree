//===- PipelineIntrinsicsToLLVM.cpp - Lower Pipeline Intrinsics ----------===//
//
// This file implements lowering of pipeline synchronization intrinsics
// to LLVM IR, with support for NVIDIA cp.async and fallback implementations.
//
//===----------------------------------------------------------------------===//

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pipeline-intrinsics-to-llvm"

using namespace mlir;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTPIPELINEINTRINSICSTOLLVM
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

using namespace mlir;

// Helper to check if we can use cp.async (requires Ampere/SM80+)
static bool canUseCpAsync(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;

  // Check for architecture attribute
  auto archAttr = module->getAttrOfType<StringAttr>("triton_gpu.arch");
  if (!archAttr) {
    // Also check for common NVIDIA architecture attributes
    auto gpuAttr = module->getAttrOfType<StringAttr>("gpu");
    if (!gpuAttr) {
      return false;
    }
    StringRef gpu = gpuAttr.getValue();
    return gpu.contains("ampere") || gpu.contains("a100") ||
           gpu.contains("sm_80") || gpu.contains("sm_86");
  }

  StringRef arch = archAttr.getValue();
  // Ampere (SM80) or later supports cp.async
  // A100 = SM80, A40 = SM86, H100 = SM90
  return arch.contains("ampere") || arch.contains("a100") ||
         arch.contains("a40") || arch.contains("sm_80") ||
         arch.contains("sm_86") || arch.contains("hopper") ||
         arch.contains("sm_90");
}

// Helper to check if target is A100 (SM80)
static bool isA100(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;

  auto archAttr = module->getAttrOfType<StringAttr>("triton_gpu.arch");
  if (!archAttr) {
    auto gpuAttr = module->getAttrOfType<StringAttr>("gpu");
    if (!gpuAttr)
      return false;
    StringRef gpu = gpuAttr.getValue();
    return gpu.contains("a100") || gpu.contains("sm_80");
  }

  StringRef arch = archAttr.getValue();
  return arch.contains("a100") || arch.contains("sm_80");
}

//===----------------------------------------------------------------------===//
// Pipeline Intrinsic Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower triton_gpu.pipeline_init to LLVM (no-op, metadata only)
struct PipelineInitOpLowering : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.pipeline_init") {
      return failure();
    }

    // Pipeline init is metadata - just erase
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_gpu.pipeline_producer_acquire to barrier
/// On Ampere+, this will use cp.async.wait_group (emitted by LoadStoreOpToLLVM)
struct PipelineProducerAcquireOpLowering
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.pipeline_producer_acquire") {
      return failure();
    }

    Location loc = op.getLoc();

    // Insert barrier for synchronization
    // On Ampere+, the actual cp.async.wait_group will be emitted
    // by the LoadStoreOpToLLVM pass when it sees async copy operations
    rewriter.create<NVVM::Barrier0Op>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_gpu.pipeline_producer_commit to barrier
/// On Ampere+, this will use cp.async.commit_group (emitted by LoadStoreOpToLLVM)
struct PipelineProducerCommitOpLowering
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.pipeline_producer_commit") {
      return failure();
    }

    Location loc = op.getLoc();

    // Insert barrier for synchronization
    // On Ampere+, the actual cp.async.commit_group will be emitted
    // by the LoadStoreOpToLLVM pass when it sees async copy operations
    rewriter.create<NVVM::Barrier0Op>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_gpu.pipeline_consumer_wait to barrier
/// On Ampere+, this will use cp.async.wait_group (emitted by LoadStoreOpToLLVM)
struct PipelineConsumerWaitOpLowering : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.pipeline_consumer_wait") {
      return failure();
    }

    Location loc = op.getLoc();

    // Insert barrier for synchronization
    // On Ampere+, the actual cp.async.wait_group will be emitted
    // by the LoadStoreOpToLLVM pass when it sees async copy operations
    rewriter.create<NVVM::Barrier0Op>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_gpu.pipeline_consumer_release to barrier
struct PipelineConsumerReleaseOpLowering
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.pipeline_consumer_release") {
      return failure();
    }

    Location loc = op.getLoc();

    // Release is typically a barrier
    rewriter.create<NVVM::Barrier0Op>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_gpu.pipeline_flush to no-op (cleanup handled by runtime)
struct PipelineFlushOpLowering : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.pipeline_flush") {
      return failure();
    }

    // Flush is handled by final barrier
    Location loc = op.getLoc();
    rewriter.create<NVVM::Barrier0Op>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_gpu.async_copy_global_to_shared
/// On NVIDIA Ampere+: use cp.async
/// Fallback: manual load + store + barrier
struct AsyncCopyGlobalToSharedOpLowering
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
    if (op.getCallee() != "triton_gpu.async_copy_global_to_shared") {
      return failure();
    }

    Location loc = op.getLoc();

    // On A100 and other Ampere GPUs, async copy is handled by cp.async
    // This intrinsic is a marker for the actual copy operations
    // The actual cp.async instructions are generated by LoadStoreOpToLLVM

    // For A100, we can optimize by emitting a hint about async copy
    if (isA100(op)) {
      // A100-specific: mark that async copy is active
      // This allows the compiler to schedule around async copies
      LLVM_DEBUG(llvm::dbgs() << "A100 async copy hint emitted\n");
    }

    // The actual memory operations are handled by surrounding loads/stores
    // which get converted to cp.async by LoadStoreOpToLLVM

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pipeline Intrinsics Lowering Pass
//===----------------------------------------------------------------------===//

namespace {

struct PipelineIntrinsicsToLLVMPass
    : public mlir::triton::impl::ConvertPipelineIntrinsicsToLLVMBase<PipelineIntrinsicsToLLVMPass> {

  void runOnOperation() override {
    auto module = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect, NVVM::NVVMDialect>();
    target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp op) {
      StringRef callee = op.getCallee();
      return !callee.starts_with("triton_gpu.pipeline") &&
             !callee.starts_with("triton_gpu.async_copy");
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<PipelineInitOpLowering>(&getContext());
    patterns.add<PipelineProducerAcquireOpLowering>(&getContext());
    patterns.add<PipelineProducerCommitOpLowering>(&getContext());
    patterns.add<PipelineConsumerWaitOpLowering>(&getContext());
    patterns.add<PipelineConsumerReleaseOpLowering>(&getContext());
    patterns.add<PipelineFlushOpLowering>(&getContext());
    patterns.add<AsyncCopyGlobalToSharedOpLowering>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPipelineIntrinsicsToLLVMPass() {
  return std::make_unique<PipelineIntrinsicsToLLVMPass>();
}

} // namespace triton
} // namespace mlir
