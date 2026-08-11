// Copyright 2026- Xcoresigma Technology Co., Ltd

#include "tle/dsa/dialect/include/Conversion/TleToHIVM/TleDistributedOpToHIVMConverter.h"

#if __has_include("bishengir/Dialect/HIVM/IR/HIVM.h")
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dsa/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <string>

using namespace mlir;

namespace {

static std::string getTypeName(Type type) {
  if (auto fpTy = llvm::dyn_cast<FloatType>(type)) {
    if (fpTy.isBF16())
      return "bfloat16";
    else if (fpTy.isF16())
      return "half";
    else if (fpTy.isF32())
      return "float";
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    switch (intTy.getWidth()) {
    case 8:
      return "int8";
    case 16:
      return "int16";
    case 32:
      return "int32";
    case 64:
      return "int64";
    }
  }
  return "unknown";
}

/// Generic template to convert TLE distributed ops to hivm.custom
template <typename TleOp>
class TleDistributedOpToHIVM : public OpRewritePattern<TleOp> {
public:
  TleDistributedOpToHIVM(MLIRContext *context, bool existDot = false,
                         PatternBenefit benefit = PatternBenefit(1))
      : OpRewritePattern<TleOp>(context, benefit), existDotFlag(existDot) {}

  void inferCoreType(TleOp op) const {
    // If kernel contains tt.dot, need both CUBE and VECTOR; otherwise VECTOR
    // only.
    auto coreType = existDotFlag ? hivm::TCoreType::CUBE_AND_VECTOR
                                 : hivm::TCoreType::VECTOR;
    op->setAttr(hivm::TCoreTypeAttr::name,
                hivm::TCoreTypeAttr::get(op->getContext(), coreType));
  }

  LogicalResult matchAndRewrite(TleOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    inferCoreType(op);

    // Determine symbol name
    std::string symbolName;
    if (auto sym = op->template getAttrOfType<StringAttr>("symbol")) {
      symbolName = sym.str();
    }
    bool hasSideEffect = !mlir::isMemoryEffectFree(op.getOperation());

    llvm::TypeSwitch<Operation *, void>(op)
        .Case([&](triton::tle::SymmAtOp symmOp) {
          auto elemTy =
              llvm::cast<triton::PointerType>(symmOp.getSymmAddr().getType())
                  .getPointeeType();
          symbolName = "aclshmem_ptr_" + getTypeName(elemTy);
        })
        .Case([&](triton::tle::GetRankOp) { symbolName = "aclshmem_my_pe"; });

    if (symbolName.empty()) {
      symbolName = op->getName().stripDialect().str();
    }

    std::string customName = "dist." + symbolName;

    // Build custom results (tensor outputs need tensor::EmptyOp)
    llvm::SmallVector<Value> customResults;
    for (auto res : op->getResults()) {
      if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(res.getType())) {
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            loc, tensorTy.getShape(), tensorTy.getElementType());
        customResults.emplace_back(emptyOp);
      }
    }

    auto customOp = rewriter.create<hivm::CustomOp>(
        loc, op->getResultTypes(), customName, op->getOperands(), customResults,
        ValueRange{});

    // Copy original attrs first, then override with controlled values
    customOp->setAttrs(op->getAttrs());
    customOp->setAttr("hivm.is_distributed", rewriter.getUnitAttr());
    customOp.setPipe(hivm::PIPE::PIPE_S);
    customOp.setVFMode(hivm::VFMode::SIMD);
    customOp->setAttr("symbol", rewriter.getStringAttr(symbolName));

    if (!hasSideEffect) {
      customOp->setAttr("no_side_effect", rewriter.getUnitAttr());
    }

    // Record GM addr argument indices for downstream passes
    llvm::SmallVector<int> gmAddrArgsIndices;
    auto funcOp = customOp->template getParentOfType<triton::FuncOp>();
    if (funcOp) {
      for (auto &&[idx, operand] : llvm::enumerate(customOp->getOperands())) {
        if (!llvm::isa<triton::PointerType>(operand.getType()))
          continue;
        if (auto arg = llvm::dyn_cast<BlockArgument>(operand)) {
          if (arg.getOwner() == &funcOp.getFunctionBody().front()) {
            gmAddrArgsIndices.emplace_back(idx);
          }
        }
      }
    }
    customOp->setAttr("gm_addr_args_indices",
                      rewriter.getDenseI32ArrayAttr(gmAddrArgsIndices));

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, customOp);
    }

    return success();
  }

private:
  bool existDotFlag;
};

/// Helper to register patterns with existDot flag
template <typename... Args>
void registerTleDistributedOpToHIVM(
    RewritePatternSet &patterns, bool existDot,
    PatternBenefit benefit = PatternBenefit(1)) {
  patterns.add<TleDistributedOpToHIVM<Args>...>(patterns.getContext(), existDot,
                                                benefit);
}

} // namespace

void mlir::triton::tle::populateTleDistributedOpToHIVMConversionPatterns(
    mlir::RewritePatternSet &patterns, PatternBenefit benefit, bool existDot) {
  registerTleDistributedOpToHIVM<
      triton::tle::SymmAtOp, triton::tle::ExternCallOp, triton::tle::GetRankOp>(
      patterns, existDot, benefit);
}
