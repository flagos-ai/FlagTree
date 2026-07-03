/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Licensed under the MIT license.
 */

//===----------------------------------------------------------------------===//
// FoldStagingCopy -- Fold redundant staging memref.alloc + memref.copy pairs
// produced by TritonToLinalgIncubated's tt.load lowering.
//
// Two flavours of staging chains are recognized:
//
//   Flavour A (tensor-consumed staging, e.g. Q/K/V):
//     %stage  = memref.alloc() : memref<SxE>                     // default space
//     copy1   = memref.copy %src(GM, strided), %stage            // GM -> staging
//     [annotation.mark %stage {...}]
//     %tensor = bufferization.to_tensor %stage restrict writable
//     [annotation.mark %tensor {...}]
//     copy2   = memref.copy %stage, %dst(#space)                 // staging -> on-chip
//
//     =>
//
//     copy    = memref.copy %src, %dst(#space)                   // direct GM -> on-chip
//     [annotation.mark %dst {...}]
//     %cast   = memref.memory_space_cast %dst : #space -> default
//     %tensor = bufferization.to_tensor %cast restrict writable
//     [annotation.mark %tensor {...}]
//
//   Flavour B (pure memref staging, e.g. P for matmul LHS):
//     %stage  = memref.alloc() : memref<SxE>
//     copy1   = memref.copy %src(GM, strided), %stage
//     copy2   = memref.copy %stage, %dst(#space)                 // no to_tensor!
//
//     =>
//
//     copy    = memref.copy %src, %dst(#space)                   // single direct copy
//
// The fold is safe because:
//   1. %stage has no other uses beyond the listed ops.
//   2. memref.copy GM -> on-chip is supported by the HIVM backend
//      (GM->L1 via MTE2, GM->UB via MTE2).
//   3. bufferization.to_tensor on a memory_space_cast of an on-chip buffer
//      produces the same tensor value.
//===----------------------------------------------------------------------===//

#include "ascend/include/FoldStagingCopy/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace hivm = mlir::hivm;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_FOLDSTAGINGCOPY
#include "ascend/include/FoldStagingCopy/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

/// Return true if \p op is an annotation.mark operation.
static bool isAnnotationMark(Operation *op) {
  return op->getName().getStringRef() == "annotation.mark";
}

static Value castDefaultMemrefToGM(Value value, Location loc,
                                   PatternRewriter &rewriter) {
  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (!memrefTy || memrefTy.getMemorySpace())
    return value;

  auto gmSpace =
      hivm::AddressSpaceAttr::get(rewriter.getContext(), hivm::AddressSpace::GM);
  auto gmTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                              memrefTy.getLayout(), gmSpace);
  return rewriter.create<memref::MemorySpaceCastOp>(loc, gmTy, value);
}

//===----------------------------------------------------------------------===//
// Rewrite pattern: fold staging alloc + copy1 + to_tensor + copy2
//===----------------------------------------------------------------------===//
struct FoldStagingCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value stage = copyOp.getSource();
    auto stageCast = stage.getDefiningOp<memref::MemorySpaceCastOp>();
    if (stageCast)
      stage = stageCast.getSource();
    Value dst = copyOp.getTarget();

    // ① Destination must have an explicit memory space (on-chip buffer).
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType || !dstType.getMemorySpace())
      return failure();

    // ② Source must be a staging memref.alloc in default/GM address space.
    auto stageAlloc = stage.getDefiningOp<memref::AllocOp>();
    if (!stageAlloc)
      return failure();
    auto stageType = stageAlloc.getType();
    if (auto stageSpace = stageType.getMemorySpace()) {
      auto stageAddr = dyn_cast<hivm::AddressSpaceAttr>(stageSpace);
      if (!stageAddr || stageAddr.getAddressSpace() != hivm::AddressSpace::GM)
        return failure();
    }

    // ③ Enumerate the source copy: memref.copy %src, %stage.
    //    Collect all users of %stage to verify the expected pattern.
    memref::CopyOp srcCopy;
    Operation *stageMark = nullptr;
    bufferization::ToTensorOp toTensor;
    Operation *tensorMark = nullptr;
    SmallVector<Operation *> otherUses;

    for (Operation *user : stage.getUsers()) {
      if (auto c = dyn_cast<memref::CopyOp>(user)) {
        if (c.getTarget() == stage) {
          if (srcCopy)
            return failure(); // multiple incoming copies — ambiguous
          srcCopy = c;
        } else if (c == copyOp.getOperation()) {
          continue; // the outgoing copy we are folding
        } else {
          otherUses.push_back(user);
        }
      } else if (isAnnotationMark(user)) {
        if (stageMark)
          otherUses.push_back(user);
        else
          stageMark = user;
      } else if (stageCast && user == stageCast.getOperation()) {
        continue;
      } else if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (toTensor)
          otherUses.push_back(user);
        else
          toTensor = tt;
      } else {
        otherUses.push_back(user);
      }
    }

    // srcCopy is mandatory; toTensor is optional (Flavour A has it, B doesn't).
    if (!srcCopy || !otherUses.empty())
      return failure();

    // Collect annotation.mark users of the tensor value (only if toTensor
    // exists — Flavour A path).
    if (toTensor) {
      Value tensorVal = toTensor.getResult();
      for (Operation *user : tensorVal.getUsers()) {
        if (isAnnotationMark(user) && !tensorMark)
          tensorMark = user;
      }
    }

    Location loc = copyOp.getLoc();
    Value src = castDefaultMemrefToGM(srcCopy.getSource(), loc, rewriter);
    Value dstCbuf = dst;

    // ④ Create the merged copy: memref.copy %src, %dst(#space)
    rewriter.create<memref::CopyOp>(loc, src, dstCbuf);

    // ⑤ If there was an annotation.mark on %stage, move it to %dst.
    if (stageMark) {
      OperationState state(loc, "annotation.mark");
      state.addOperands(dstCbuf);
      for (auto &namedAttr : stageMark->getAttrs())
        state.addAttribute(namedAttr.getName(), namedAttr.getValue());
      rewriter.create(state);
    }

    // ⑥ For Flavour A: drop the address space via memory_space_cast and
    //    re-create the to_tensor pointing at the cbuf alloc.
    //    For Flavour B: nothing to do — the tensor path never existed.
    if (toTensor) {
      auto genericType =
          MemRefType::get(dstType.getShape(), dstType.getElementType());
      Value cast =
          rewriter.create<memref::MemorySpaceCastOp>(loc, genericType, dstCbuf);
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          toTensor.getLoc(), toTensor.getType(), cast, toTensor.getRestrict(),
          toTensor.getWritable());

      // ⑦ If there was an annotation.mark on the tensor value, re-create it.
      if (tensorMark) {
        OperationState state(loc, "annotation.mark");
        state.addOperands(tensor);
        for (auto &namedAttr : tensorMark->getAttrs())
          state.addAttribute(namedAttr.getName(), namedAttr.getValue());
        rewriter.create(state);
      }

      rewriter.replaceOp(toTensor, tensor);
    }

    // ⑧ Erase the dead ops.
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(srcCopy);
    if (stageCast)
      rewriter.eraseOp(stageCast);
    if (stageMark)
      rewriter.eraseOp(stageMark);
    if (tensorMark)
      rewriter.eraseOp(tensorMark);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct FoldStagingCopyPass
    : public triton::impl::FoldStagingCopyBase<FoldStagingCopyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldStagingCopyPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createFoldStagingCopyPass() {
  return std::make_unique<FoldStagingCopyPass>();
}
