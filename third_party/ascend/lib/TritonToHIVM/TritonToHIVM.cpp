/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToHIVM/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOHIVM
#include "ascend/include/TritonToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

struct CoreAndPipes {
  TCoreTypeAttr core;
  PipeAttr producer;
  PipeAttr consumer;
};

static LogicalResult EmitUnknownOpError(Operation *op,
                                        llvm::StringRef opName) {
  op->emitError("Unknown custom operation: ") << opName;
  return failure();
}

static void CreateSyncBlock(PatternRewriter &rewriter, Location loc,
                            MLIRContext *ctx, Operation *op, int64_t id,
                            hivm::SyncBlockMode mode,
                            PipeAttr pipe1, PipeAttr pipe2) {
  auto syncMode = hivm::SyncBlockModeAttr::get(ctx, mode);
  auto newOp = rewriter.create<hivm::SyncBlockOp>(
      loc, syncMode, rewriter.getI16IntegerAttr(id),
      Value{}, pipe1, pipe2);
  rewriter.replaceOp(op, newOp);
}

static CoreAndPipes GetCoreAndPipes(MLIRContext *ctx,
                                    llvm::StringRef opName,
                                    llvm::StringRef sender) {
  // Step 1: Decide pipes
  PipeAttr producer;
  PipeAttr consumer = PipeAttr::get(ctx, PIPE::PIPE_MTE2);

  if (sender == "cube") {
    producer = PipeAttr::get(ctx, PIPE::PIPE_FIX);
  } else {
    producer = PipeAttr::get(ctx, PIPE::PIPE_MTE3);
  }

  // Step 2: Decide core type
  TCoreTypeAttr core;
  if (sender == "cube") {
    if (opName == "sync_block_set")
      core = TCoreTypeAttr::get(ctx, TCoreType::CUBE);
    else
      core = TCoreTypeAttr::get(ctx, TCoreType::VECTOR);
  } else {
    if (opName == "sync_block_set")
      core = TCoreTypeAttr::get(ctx, TCoreType::VECTOR);
    else
      core = TCoreTypeAttr::get(ctx, TCoreType::CUBE);
  }

  return {core, producer, consumer};
}

} // end anonymous namespace
namespace {
struct TritonToHIVMPass
    : public mlir::triton::impl::TritonToHIVMBase<
          TritonToHIVMPass> {
  void runOnOperation() override;
};
} // namespace


struct TritonCustomOpToHIVMSyncOpConversion
    : OpRewritePattern<triton::ascend::CustomOp> {
  using OpRewritePattern<triton::ascend::CustomOp>::OpRewritePattern;

LogicalResult matchAndRewrite(triton::ascend::CustomOp op,
                              PatternRewriter &rewriter) const final {
  auto *ctx = op->getContext();
  auto loc = op->getLoc();
  auto args = op.getStrArgs();
  auto argAttr = dyn_cast<StringAttr>(args[0]);
  auto id = dyn_cast<IntegerAttr>(args[1]).getInt();
  llvm::StringRef opName = op.getOpName();
  llvm::StringRef arg = argAttr.getValue();

  if (opName == "sync_block_all") {
    if (arg == "all_cube") {
      CreateSyncBlock(rewriter, loc, ctx, op, id,
                      hivm::SyncBlockMode::ALL_CUBE,
                      PipeAttr::get(ctx, PIPE::PIPE_FIX),
                      hivm::PipeAttr{});
    } else if (arg == "all_vector") {
      CreateSyncBlock(rewriter, loc, ctx, op, id,
                      hivm::SyncBlockMode::ALL_VECTOR,
                      hivm::PipeAttr{},
                      PipeAttr::get(ctx, PIPE::PIPE_MTE3));
    } else if (arg == "all") {
      CreateSyncBlock(rewriter, loc, ctx, op, id,
                      hivm::SyncBlockMode::ALL,
                      PipeAttr::get(ctx, PIPE::PIPE_FIX),
                      PipeAttr::get(ctx, PIPE::PIPE_MTE3));
    } else {
      return EmitUnknownOpError(op, opName);
    }
    return success();
  }

  if (opName == "sync_block_set") {
    auto [coreAttr, prodPipe, consPipe] = GetCoreAndPipes(ctx, opName, arg);
    rewriter.replaceOp(op, rewriter.create<hivm::SyncBlockSetOp>(
                                loc, coreAttr, prodPipe, consPipe,
                                rewriter.getIndexAttr(id)));
    return success();
  }

  if (opName == "sync_block_wait") {
    auto [coreAttr, prodPipe, consPipe] = GetCoreAndPipes(ctx, opName, arg);
    rewriter.replaceOp(op, rewriter.create<hivm::SyncBlockWaitOp>(
                                loc, coreAttr, prodPipe, consPipe,
                                rewriter.getIndexAttr(id)));
    return success();
  }

  return EmitUnknownOpError(op, opName);
}
};


void TritonToHIVMPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<hivm::HIVMDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<TritonCustomOpToHIVMSyncOpConversion>(patterns.getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createTritonToHIVMPass() {
  return std::make_unique<TritonToHIVMPass>();
}
