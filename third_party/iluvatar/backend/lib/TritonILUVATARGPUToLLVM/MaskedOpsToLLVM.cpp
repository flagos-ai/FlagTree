#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonILUVATARGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <tuple>

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

static bool isGlobalPtr(Value ptr) {
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
  return ptrTy && ptrTy.getAddressSpace() == 1;
}

static int32_t getKopForCacheModifier(triton::CacheModifier cacheMod) {
  switch (cacheMod) {
  case triton::CacheModifier::CG:
    return 1;
  case triton::CacheModifier::CS:
    return 2;
  case triton::CacheModifier::CV:
  case triton::CacheModifier::WT:
    return 3;
  default:
    return 0;
  }
}

class ConvertMaskedLoadOp
    : public OpRewritePattern<triton::iluvatargpu::MaskedLoadOp> {
public:
  ConvertMaskedLoadOp(MLIRContext *context,
                      const ILUVATAR::TargetInfo &targetInfo)
      : OpRewritePattern(context), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::iluvatargpu::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto elemTy = loadOp.getResult().getType();
    auto ptr = loadOp.getPtr();
    auto mask = loadOp.getMask();
    auto falseVal = loadOp.getFalseVal();
    auto multicastMask = loadOp.getMulticastMask();
    auto cacheMod = loadOp.getCache();

    bool volatileFlag = loadOp.getIsVolatile();
    bool nonTmpFlag = false;

    auto createLoadWithAttrs = [&](Location loadLoc) -> Value {
      int vecBits = 0;
      if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
        vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      } else {
        vecBits = elemTy.getIntOrFloatBitWidth();
      }
      assert(vecBits != 0);
      if (multicastMask) {
        loadOp.emitRemark()
            << "Multicast with bit width " << vecBits << " is not supported on "
            << targetInfo.getArch() << " falling back to regular load";
      }
      // Emit a regular load
      if (isGlobalPtr(ptr)) {
        auto load = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loadLoc, "llvm.bi.load.kop", {elemTy},
            {ptr, b.i32_val(getKopForCacheModifier(cacheMod)),
             b.i1_val(volatileFlag)});
        return load.getResult(0);
      }
      auto load =
          LLVM::LoadOp::create(rewriter, loadLoc, elemTy, ptr, /*alignment*/ 0,
                               volatileFlag, nonTmpFlag);
      return load;
    };

    bool useDirectLoad = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectLoad) {
      auto loadResult = createLoadWithAttrs(loc);
      rewriter.replaceOp(loadOp, loadResult);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, ValueRange{},
                           afterLoad, ValueRange{falseVal});
    rewriter.setInsertionPointToStart(trueBlock);
    auto loadResult = createLoadWithAttrs(loc);
    LLVM::BrOp::create(rewriter, loc, ValueRange{loadResult}, afterLoad);

    rewriter.replaceOp(loadOp, afterLoad->getArgument(0));

    return success();
  }

private:
  const ILUVATAR::TargetInfo &targetInfo;
};

class ConvertMaskedStoreOp
    : public OpRewritePattern<triton::iluvatargpu::MaskedStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::iluvatargpu::MaskedStoreOp storeOp,
                                PatternRewriter &rewriter) const override {

    auto loc = storeOp.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto val = storeOp.getValue();
    auto elemTy = storeOp.getValue().getType();
    auto ptr = storeOp.getPtr();
    auto mask = storeOp.getMask();
    bool volatileFlag = false;
    bool nonTmpFlag = false;

    int alignment = 0;
    if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
      auto vecElemTy = vecTy.getElementType();
      auto elemSizeInBytes = vecElemTy.getIntOrFloatBitWidth() / 8;
      alignment = elemSizeInBytes * vecTy.getNumElements();
    }

    auto createStoreWithAttrs = [&](Location storeLoc) {
      if (isGlobalPtr(ptr)) {
        LLVM::createLLVMIntrinsicCallOp(
            rewriter, storeLoc, "llvm.bi.store.kop", {},
            {val, ptr, b.i32_val(getKopForCacheModifier(storeOp.getCache())),
             b.i1_val(volatileFlag)});
        return;
      }
      LLVM::StoreOp::create(rewriter, storeLoc, val, ptr, alignment,
                            volatileFlag, nonTmpFlag);
    };

    bool useDirectStore = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectStore) {
      createStoreWithAttrs(loc);
      rewriter.eraseOp(storeOp);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    createStoreWithAttrs(loc);
    LLVM::BrOp::create(rewriter, loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

} // namespace

namespace mlir::triton::ILUVATAR {

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo) {
  patterns.add<ConvertMaskedLoadOp>(patterns.getContext(), targetInfo);
  patterns.add<ConvertMaskedStoreOp>(patterns.getContext());
}
} // namespace mlir::triton::ILUVATAR

// namespace mlir::triton
